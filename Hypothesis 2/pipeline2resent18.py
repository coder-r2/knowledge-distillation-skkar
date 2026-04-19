import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
import copy
from tqdm import tqdm


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.inplanes = 64

        self.conv1   = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64,  2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=1)

        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_with_features(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x);  feat0 = x
        x = self.layer3(x);  feat1 = x
        x = self.layer4(x);  feat2 = x
        x = self.avgpool(x)
        logits = self.classifier(x.view(x.size(0), -1))
        return logits, [feat0, feat1, feat2]

    def extract_features(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        return self.classifier(self.extract_features(x))

    def forward_from_level(self, feat, level):
        x = feat
        if level <= 0:
            x = self.layer3(x)
        if level <= 1:
            x = self.layer4(x)
        x = self.avgpool(x)
        return self.classifier(x.view(x.size(0), -1))


def logit_distillation_loss(student_logits, teacher_logits, T=2.0):
    student_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (T * T)


def intermediate_feature_distillation_loss(student_feats, teacher_feats):
    loss = 0.0
    for sf, tf in zip(student_feats, teacher_feats):
        diff = (sf - tf).view(sf.size(0), -1)
        loss = loss + diff.norm(p=2, dim=1).mean()
    return loss


@torch.no_grad()
def build_feature_memory(model, loader, device, target_classes, per_class=50):
    model.eval()
    mem_feat, mem_lbl = [], []
    mem_inter = [[], [], []]
    class_counts = {c: 0 for c in target_classes}

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, intermediates = model.forward_with_features(x)
        pooled = model.avgpool(intermediates[2])
        f = F.normalize(pooled.view(x.size(0), -1), dim=1)

        for i in range(len(y)):
            label = y[i].item()
            if label in target_classes and class_counts[label] < per_class:
                mem_feat.append(f[i].cpu())
                mem_lbl.append(label)
                for lvl in range(3):
                    mem_inter[lvl].append(intermediates[lvl][i].cpu())
                class_counts[label] += 1

        if all(class_counts[c] >= per_class for c in target_classes):
            break

    return (
        torch.stack(mem_feat),
        torch.tensor(mem_lbl),
        [torch.stack(mem_inter[l]) for l in range(3)],
    )


def student_step(
    model,
    all_teachers,
    all_teacher_num_classes,
    loader,
    memory_features,
    memory_labels,
    memory_intermediate,
    optimizer,
    device,
    num_old_classes,
    new_class_offset,
    alpha,
    beta,
    gamma,
    delta,
    batch_replay=32,
):
    model.train()
    for t in all_teachers:
        t.eval()

    memory_features = memory_features.to(device)
    memory_labels   = memory_labels.to(device)

    old_classes = sorted(memory_labels.unique().tolist())

    memory_by_class_inter = []
    for lvl in range(3):
        mem_lvl  = memory_intermediate[lvl].to(device)
        by_class = {c: mem_lvl[memory_labels == c] for c in old_classes}
        memory_by_class_inter.append(by_class)

    for x_new, y_new in loader:
        x_new, y_new = x_new.to(device), y_new.to(device)

        logits_new, student_feats = model.forward_with_features(x_new)
        loss_new = F.cross_entropy(
            logits_new[:, new_class_offset:new_class_offset + 2],
            y_new - new_class_offset,
        )

        per_class_count = max(1, batch_replay // len(old_classes))
        loss_old = 0.0
        for lvl in range(3):
            f_parts, y_parts = [], []
            for c in old_classes:
                mem_c = memory_by_class_inter[lvl][c]
                idx   = torch.randint(0, len(mem_c), (per_class_count,))
                f_parts.append(mem_c[idx])
                y_parts.append(
                    torch.full((per_class_count,), c, device=device, dtype=torch.long)
                )
            f_lvl      = torch.cat(f_parts, dim=0)
            y_lvl      = torch.cat(y_parts, dim=0)
            logits_lvl = model.forward_from_level(f_lvl, lvl)
            loss_old  += F.cross_entropy(logits_lvl[:, :num_old_classes], y_lvl)
        loss_old /= 3.0

        loss_inter = 0.0
        for t_k in all_teachers:
            with torch.no_grad():
                _, teacher_feats_k = t_k.forward_with_features(x_new)
            loss_inter += intermediate_feature_distillation_loss(student_feats, teacher_feats_k)
        loss_inter /= len(all_teachers)

        loss_kd = 0.0
        for t_k, n_k in zip(all_teachers, all_teacher_num_classes):
            with torch.no_grad():
                teacher_logits_k = t_k(x_new)
            loss_kd += logit_distillation_loss(
                logits_new[:, :n_k],
                teacher_logits_k[:, :n_k],
            )
        loss_kd /= len(all_teachers)

        f_student = F.normalize(model.extract_features(x_new), dim=1)
        loss_feat = 0.0
        for t_k in all_teachers:
            with torch.no_grad():
                f_teacher_k = F.normalize(t_k.extract_features(x_new), dim=1)
            loss_feat += (1 - F.cosine_similarity(f_student, f_teacher_k, dim=1).mean())
        loss_feat /= len(all_teachers)

        replay_batch_size = per_class_count * len(old_classes)
        scale = len(x_new) / max(1, replay_batch_size)

        loss = (loss_new
                + alpha * scale * loss_old
                + beta  * loss_inter
                + gamma * loss_kd
                + delta * loss_feat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def filter_classes(dataset, classes):
    indices = [i for i, (_, y) in enumerate(dataset) if y in classes]
    return Subset(dataset, indices)

def split_dataset(dataset, val_ratio=0.1):
    n = len(dataset)
    n_val = int(n * val_ratio)
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, [n - n_val, n_val], generator=generator)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total

def train_to_convergence(
    model, train_loader, val_loader, optimizer, device,
    train_step_fn, max_epochs=50, patience=5, eval_fn=None,
):
    best_model, best_val, patience_counter = None, 0, 0
    for epoch in tqdm(range(max_epochs), desc="Epochs", leave=False):
        train_step_fn(model, train_loader, optimizer, device)
        val_acc = eval_fn(model) if eval_fn else evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val         = val_acc
            best_model       = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    model.load_state_dict(best_model)
    return model

def teacher_step(model, loader, optimizer, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x)[:, :2], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def freeze(model):
    m = copy.deepcopy(model)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def run_pipeline(full_train, test_set, device, alpha, beta, gamma, delta):
    NUM_STEPS        = 5
    CLASSES_PER_STEP = 2

    all_memory_features     = []
    all_memory_labels       = []
    all_memory_intermediate = [[], [], []]
    all_teachers            = []
    all_teacher_num_classes = []
    current_model           = None

    for step in range(NUM_STEPS):
        new_classes = list(range(step * CLASSES_PER_STEP, (step + 1) * CLASSES_PER_STEP))
        all_classes = list(range(0, (step + 1) * CLASSES_PER_STEP))
        old_classes = list(range(0, step * CLASSES_PER_STEP))
        num_total   = len(all_classes)
        num_old     = len(old_classes)

        train_new_full = filter_classes(full_train, new_classes)
        train_new, val_new = split_dataset(train_new_full)
        train_loader   = DataLoader(train_new,      batch_size=64,  shuffle=True)
        memory_loader  = DataLoader(train_new_full, batch_size=64,  shuffle=False)
        val_loader_new = DataLoader(val_new,        batch_size=128)

        if step == 0:
            model = ResNet18(num_classes=num_total).to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
            model = train_to_convergence(
                model, train_loader, val_loader_new, opt, device, teacher_step
            )
            current_model = model

        else:
            latest_teacher = all_teachers[-1]
            student = ResNet18(num_classes=num_total).to(device)
            t_sd    = latest_teacher.state_dict()
            s_sd    = student.state_dict()
            for key in t_sd:
                if not key.startswith("classifier."):
                    s_sd[key] = t_sd[key].clone()
            s_sd["classifier.weight"][:num_old] = t_sd["classifier.weight"][:num_old]
            s_sd["classifier.bias"][:num_old]   = t_sd["classifier.bias"][:num_old]
            student.load_state_dict(s_sd)

            opt = torch.optim.Adam(student.parameters(), lr=1e-3)

            memory_features     = torch.cat(all_memory_features)
            memory_labels_cat   = torch.cat(all_memory_labels)
            memory_intermediate = [torch.cat(all_memory_intermediate[l]) for l in range(3)]

            _at  = list(all_teachers)
            _anc = list(all_teacher_num_classes)
            _mf  = memory_features
            _ml  = memory_labels_cat
            _mi  = memory_intermediate
            _no  = num_old
            _off = step * CLASSES_PER_STEP

            def step_fn(model, loader, optimizer, device,
                        at=_at, anc=_anc, mf=_mf, ml=_ml, mi=_mi,
                        no=_no, off=_off):
                student_step(
                    model,
                    all_teachers=at,
                    all_teacher_num_classes=anc,
                    loader=loader,
                    memory_features=mf,
                    memory_labels=ml,
                    memory_intermediate=mi,
                    optimizer=optimizer,
                    device=device,
                    num_old_classes=no,
                    new_class_offset=off,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    delta=delta,
                )

            val_old_sets = []
            for c in old_classes:
                _, vc = split_dataset(filter_classes(full_train, [c]))
                val_old_sets.append(vc)
            val_all        = ConcatDataset([val_new] + val_old_sets)
            val_loader_all = DataLoader(val_all, batch_size=128)

            student = train_to_convergence(
                student, train_loader, None, opt, device, step_fn,
                eval_fn=lambda m: evaluate(m, val_loader_all, device),
            )
            current_model = student

        mem_f, mem_l, mem_i = build_feature_memory(
            current_model, memory_loader, device,
            target_classes=new_classes, per_class=50,
        )
        all_memory_features.append(mem_f)
        all_memory_labels.append(mem_l)
        for lvl in range(3):
            all_memory_intermediate[lvl].append(mem_i[lvl])

        all_teachers.append(freeze(current_model))
        all_teacher_num_classes.append(num_total)

    final_loader = DataLoader(
        filter_classes(test_set, list(range(10))), batch_size=128
    )
    return evaluate(current_model, final_loader, device)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_set   = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

    # The hyper-parameters below are the default one's
    ALPHA = 1.0
    BETA  = 1.0
    GAMMA = 1.0
    DELTA = 1.0

    final_acc = run_pipeline(
        full_train, test_set, device,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA,
    )
    print(f"Pipeline 2 — 10-class test accuracy: {final_acc*100:.2f}%")
