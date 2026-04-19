# train_baselines.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import timm 
from tqdm.auto import tqdm 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True 

set_seed(42)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_ops=2, magnitude=12),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

print("Downloading Imagenette natively through PyTorch...")
full_trainset = torchvision.datasets.Imagenette(root='./data', split='train', size='full', download=True, transform=train_transform)
full_testset = torchvision.datasets.Imagenette(root='./data', split='val', size='full', download=True, transform=test_transform)

trainloader = DataLoader(full_trainset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
testloader = DataLoader(full_testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

class StudentResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, 10) 

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        feat = self.backbone.layer4(x) 
        out = self.backbone.avgpool(feat)
        out = torch.flatten(out, 1)
        logits = self.backbone.fc(out)
        return feat, logits

class PCAProjector(nn.Module):
    def __init__(self, s_channels, t_dim):
        super().__init__()
        self.query = nn.Conv2d(s_channels, t_dim, 1)
        self.key   = nn.Conv2d(s_channels, t_dim, 1)
        self.value = nn.Conv2d(s_channels, t_dim, 1)

    def forward(self, x):
        q = self.query(x).flatten(2).transpose(1, 2)
        k = self.key(x).flatten(2)
        v = self.value(x).flatten(2).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k) / (q.size(-1)**0.5), dim=-1)
        return torch.matmul(attn, v)

class GLProjector(nn.Module):
    def __init__(self, s_channels, t_dim):
        super().__init__()
        self.proj = nn.Conv2d(s_channels, t_dim, 1)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

def kl_divergence_loss(s_logits, t_logits, temperature=4.0):
    s_soft = F.log_softmax(s_logits / temperature, dim=1)
    t_soft = F.softmax(t_logits / temperature, dim=1)
    return F.kl_div(s_soft, t_soft, reduction='batchmean') * (temperature ** 2)

print("Initializing Models...")
teacher_cnn = models.resnet34(weights='IMAGENET1K_V1').to(device).eval()
teacher_vit = timm.create_model('vit_small_patch16_224', pretrained=True).to(device).eval()

set_seed(42); student_base = StudentResNet().to(device)
set_seed(42); student_cnn_kd = StudentResNet().to(device)
set_seed(42); student_proj_kd = StudentResNet().to(device) # Renamed for clarity (Appendix Baseline)

pca_proj = PCAProjector(512, 384).to(device)
gl_proj = GLProjector(512, 384).to(device)

EPOCHS = 100 
LR = 3e-4
opt_base = optim.AdamW(student_base.parameters(), lr=LR, weight_decay=1e-4)
opt_cnn_kd = optim.AdamW(student_cnn_kd.parameters(), lr=LR, weight_decay=1e-4)
opt_proj_kd = optim.AdamW(list(student_proj_kd.parameters()) + list(pca_proj.parameters()) + list(gl_proj.parameters()), lr=LR, weight_decay=1e-4)

sched_base = optim.lr_scheduler.CosineAnnealingLR(opt_base, T_max=EPOCHS)
sched_cnn = optim.lr_scheduler.CosineAnnealingLR(opt_cnn_kd, T_max=EPOCHS)
sched_proj = optim.lr_scheduler.CosineAnnealingLR(opt_proj_kd, T_max=EPOCHS)

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in testloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            _, logits = model(imgs)
            _, predicted = torch.max(logits, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()
    return 100 * correct / total

best_acc_base, best_acc_cnn, best_acc_proj = 0.0, 0.0, 0.0
IMAGENETTE_CLASSES = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

ALPHA = 0.9 
BETA = 20.0 
TEMP = 4.0

print(f"Starting Baseline & Projector Distillation for {EPOCHS} Epochs...")
for epoch in range(EPOCHS):
    student_base.train()
    student_cnn_kd.train()
    student_proj_kd.train()
    pca_proj.train()
    gl_proj.train()

    pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            x_c = teacher_cnn.conv1(images)
            x_c = teacher_cnn.bn1(x_c)
            x_c = teacher_cnn.relu(x_c)
            x_c = teacher_cnn.maxpool(x_c)
            x_c = teacher_cnn.layer1(x_c)
            x_c = teacher_cnn.layer2(x_c)
            x_c = teacher_cnn.layer3(x_c)
            t_feat_cnn = teacher_cnn.layer4(x_c) 
            
            out_c = teacher_cnn.avgpool(t_feat_cnn)
            out_c = torch.flatten(out_c, 1)
            t_logits_cnn = teacher_cnn.fc(out_c)[:, IMAGENETTE_CLASSES]

            t_out_vit = teacher_vit.forward_features(images)
            t_feat_vit = t_out_vit[:, 1:, :] 
            t_logits_vit = teacher_vit.forward_head(t_out_vit)[:, IMAGENETTE_CLASSES]

        # --- 1. SUPERVISED BASELINE ---
        _, s_logits_base = student_base(images)
        loss_base = F.cross_entropy(s_logits_base, labels)
        
        opt_base.zero_grad()
        loss_base.backward()
        opt_base.step()

        # --- 2. HOMOGENEOUS KD (CNN -> CNN) ---
        s_feat_cnn, s_logits_cnn = student_cnn_kd(images)
        
        loss_ce_cnn = F.cross_entropy(s_logits_cnn, labels)
        loss_kl_cnn = kl_divergence_loss(s_logits_cnn, t_logits_cnn, temperature=TEMP)
        
        s_feat_cnn_norm = F.normalize(s_feat_cnn, p=2, dim=1)
        t_feat_cnn_norm = F.normalize(t_feat_cnn, p=2, dim=1)
        loss_feat_cnn = F.mse_loss(s_feat_cnn_norm, t_feat_cnn_norm) 
        
        loss_cnn_total = ((1.0 - ALPHA) * loss_ce_cnn) + (ALPHA * loss_kl_cnn) + (BETA * loss_feat_cnn)
        
        opt_cnn_kd.zero_grad()
        loss_cnn_total.backward()
        opt_cnn_kd.step()

        # --- 3. FAILED CROSS-ARCHITECTURE BASELINE (ViT -> CNN via Projectors) ---
        s_feat_proj, s_logits_proj = student_proj_kd(images)
        s_feat_resized = F.interpolate(s_feat_proj, size=(14, 14), mode='bilinear', align_corners=False)
        
        s_pca = pca_proj(s_feat_resized)
        s_gl = gl_proj(s_feat_resized)

        loss_ce_proj = F.cross_entropy(s_logits_proj, labels)
        loss_kl_proj = kl_divergence_loss(s_logits_proj, t_logits_vit, temperature=TEMP)
        
        s_pca_norm = F.normalize(s_pca, p=2, dim=-1)
        s_gl_norm = F.normalize(s_gl, p=2, dim=-1)
        t_feat_vit_norm = F.normalize(t_feat_vit, p=2, dim=-1)
        
        # Translation Firewall happening here (Features align mathematically but don't transfer shape)
        loss_feat_proj = F.mse_loss(s_pca_norm, t_feat_vit_norm) + F.mse_loss(s_gl_norm, t_feat_vit_norm)
        
        loss_proj_total = ((1.0 - ALPHA) * loss_ce_proj) + (ALPHA * loss_kl_proj) + (BETA * loss_feat_proj)
        
        opt_proj_kd.zero_grad()
        loss_proj_total.backward()
        opt_proj_kd.step()

    sched_base.step()
    sched_cnn.step()
    sched_proj.step()

    acc_base = evaluate(student_base)
    acc_cnn = evaluate(student_cnn_kd)
    acc_proj = evaluate(student_proj_kd)
    
    print(f"Epoch {epoch+1} | Base: {acc_base:.2f}% | CNN-KD: {acc_cnn:.2f}% | Proj-KD (Appendix): {acc_proj:.2f}%")

    if acc_base > best_acc_base:
        best_acc_base = acc_base
        torch.save(student_base.state_dict(), './best_student_base.pth')
        
    if acc_cnn > best_acc_cnn:
        best_acc_cnn = acc_cnn
        torch.save(student_cnn_kd.state_dict(), './best_student_cnn.pth')
        
    if acc_proj > best_acc_proj:
        best_acc_proj = acc_proj
        torch.save(student_proj_kd.state_dict(), './best_student_proj.pth')

print("\n==============================================")
print("Baseline Training Complete!")
print(f"High Scores -> Baseline: {best_acc_base:.2f}% | CNN-KD: {best_acc_cnn:.2f}% | Proj-KD: {best_acc_proj:.2f}%")
print("==============================================")