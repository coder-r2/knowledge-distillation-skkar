import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
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
    def __init__(self, num_classes=2):
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

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return self.classifier(x.view(x.size(0), -1))


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

def train_to_convergence(model, train_loader, val_loader, optimizer, device,
                         max_epochs=50, patience=5):
    best_model, best_val, patience_counter = None, 0, 0
    for epoch in tqdm(range(max_epochs), desc="Epochs", leave=False):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc = evaluate(model, val_loader, device)
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

def freeze(model):
    m = copy.deepcopy(model)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def run_pipeline(full_train, test_set, device):
    NUM_STEPS        = 5
    CLASSES_PER_STEP = 2

    prev_model    = None
    current_model = None

    for step in range(NUM_STEPS):
        new_classes = list(range(step * CLASSES_PER_STEP, (step + 1) * CLASSES_PER_STEP))
        all_classes = list(range(0, (step + 1) * CLASSES_PER_STEP))
        num_total   = len(all_classes)
        num_old     = num_total - CLASSES_PER_STEP

        train_new_full   = filter_classes(full_train, new_classes)
        train_new, val_new = split_dataset(train_new_full)
        train_loader     = DataLoader(train_new, batch_size=64, shuffle=True)
        val_loader       = DataLoader(val_new,   batch_size=128)

        model = ResNet18(num_classes=num_total).to(device)

        if prev_model is not None:
            t_sd = prev_model.state_dict()
            s_sd = model.state_dict()
            for key in t_sd:
                if not key.startswith("classifier."):
                    s_sd[key] = t_sd[key].clone()
            s_sd["classifier.weight"][:num_old] = t_sd["classifier.weight"][:num_old]
            s_sd["classifier.bias"][:num_old]   = t_sd["classifier.bias"][:num_old]
            model.load_state_dict(s_sd)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def new_class_step(m, loader, optimizer, device, offset=step * CLASSES_PER_STEP):
            m.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                loss = F.cross_entropy(
                    m(x)[:, offset:offset + CLASSES_PER_STEP],
                    y - offset,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        best_model_sd, best_val, patience_counter = None, 0, 0
        for epoch in tqdm(range(50), desc=f"Step {step+1} Epochs", leave=False):
            new_class_step(model, train_loader, opt, device)
            val_acc = evaluate_new_only(model, val_loader, device, step * CLASSES_PER_STEP, CLASSES_PER_STEP)
            if val_acc > best_val:
                best_val         = val_acc
                best_model_sd    = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 5:
                break
        model.load_state_dict(best_model_sd)

        prev_model    = freeze(model)
        current_model = model

    final_loader = DataLoader(filter_classes(test_set, list(range(10))), batch_size=128)
    return evaluate(current_model, final_loader, device)


def evaluate_new_only(model, loader, device, offset, num_new):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            preds = model(x)[:, offset:offset + num_new].argmax(dim=1) + offset
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_set   = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

    final_acc = run_pipeline(full_train, test_set, device)
    print(f"ResNet-18 Warm-Start (no distillation) — 10-class test accuracy: {final_acc*100:.2f}%")
