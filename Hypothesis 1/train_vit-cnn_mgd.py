# train_mgd.py
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
print(f"Initializing on device - {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

set_seed(42)

norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_ops=2, magnitude=12), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

clean_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

blur_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.GaussianBlur(kernel_size=(15, 15), sigma=(5.0, 5.0)), 
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

trainset = torchvision.datasets.Imagenette(root='./data', split='train', size='full', download=True, transform=train_transform)
cleanset = torchvision.datasets.Imagenette(root='./data', split='val', size='full', download=True, transform=clean_transform)
blurset = torchvision.datasets.Imagenette(root='./data', split='val', size='full', download=False, transform=blur_transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
cleanloader = DataLoader(cleanset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
blurloader = DataLoader(blurset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

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
        feat_layer3 = self.backbone.layer3(x) 
        feat_layer4 = self.backbone.layer4(feat_layer3) 
        
        pooled = self.backbone.avgpool(feat_layer4)
        flat = torch.flatten(pooled, 1) 
        logits = self.backbone.fc(flat)
        return feat_layer3, logits

#MGD Projector which we are using with mask ratio parameter
class MGDProjector(nn.Module):
    def __init__(self, in_channels=256, out_channels=384, mask_ratio=0.75): 
        super().__init__()
        self.mask_ratio = mask_ratio
        self.generator = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        if self.training:
            B, C, H, W = x.shape
            mask = (torch.rand((B, 1, H, W), device=x.device) > self.mask_ratio).float()
            x = x * mask 
        return self.generator(x)

def kl_loss(s_logits, t_logits, T=4.0):
    return F.kl_div(F.log_softmax(s_logits/T, dim=1), F.softmax(t_logits/T, dim=1), reduction='batchmean') * (T**2)


print("Loading ViT-Small Teacher...")
try:
    teacher_vit = timm.create_model('vit_small_patch16_224', pretrained=True).to(device).eval()
except Exception as e:
    print("HF Hub failed. Ensure your HF_TOKEN is set or you are authenticated.")
    raise e

IMAGENETTE_CLASSES = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701] #classes from IMAGENET which correspond to the 10 classes in IMAGENETTE

set_seed(42)
student_mgd = StudentResNet().to(device)
mgd_proj = MGDProjector(in_channels=256, out_channels=384, mask_ratio=0.75).to(device)

EPOCHS = 100 
LR = 3e-4
opt_mgd = optim.AdamW(list(student_mgd.parameters()) + list(mgd_proj.parameters()), lr=LR, weight_decay=1e-4)
sched_mgd = optim.lr_scheduler.CosineAnnealingLR(opt_mgd, T_max=EPOCHS)

#Hyperparameters for MGD loss components
ALPHA = 0.95 
BETA = 40.0  
TEMP = 4.0

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            _, logits = model(imgs) 
            _, predicted = torch.max(logits, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()
    return 100 * correct / total

print(f"🔥 Starting Extreme MGD Run for {EPOCHS} Epochs...")
print(f"{'Epoch':<6} | {'MGD Clean':<12} | {'MGD Blur':<12}")
print("-" * 35)

best_acc_base = 0.0
best_sota_blur = 0.0 

for epoch in range(EPOCHS):
    student_mgd.train()
    mgd_proj.train()
    
    for images, labels in tqdm(trainloader, leave=False):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            t_out_vit = teacher_vit.forward_features(images) 
            t_feat_spatial = t_out_vit[:, 1:, :] 
            t_feat_spatial_2d = t_feat_spatial.transpose(1, 2).reshape(images.size(0), 384, 14, 14)
            t_logits = teacher_vit.forward_head(t_out_vit)[:, IMAGENETTE_CLASSES]

        opt_mgd.zero_grad()
        s_feat_L3, s_logits_mgd = student_mgd(images)
        
        loss_ce_mgd = F.cross_entropy(s_logits_mgd, labels)
        loss_kl_mgd = kl_loss(s_logits_mgd, t_logits, T=TEMP)
        
        s_gen_features = mgd_proj(s_feat_L3) 
        loss_feat_mgd = F.mse_loss(F.normalize(s_gen_features, p=2, dim=1), F.normalize(t_feat_spatial_2d, p=2, dim=1))
        
        loss_mgd_total = ((1-ALPHA)*loss_ce_mgd) + (ALPHA*loss_kl_mgd) + (BETA*loss_feat_mgd)
        loss_mgd_total.backward()
        opt_mgd.step()

    sched_mgd.step()

    acc_mgd_clean = evaluate(student_mgd, cleanloader)
    acc_mgd_blur = evaluate(student_mgd, blurloader)
    
    print(f"{epoch+1:<6} | {acc_mgd_clean:>10.2f}% | {acc_mgd_blur:>9.2f}%")

    if acc_mgd_clean > best_acc_base:  
        best_acc_base = acc_mgd_clean
        torch.save(student_mgd.state_dict(), 'student_mgd_best_clean.pth')

    if acc_mgd_clean >= 90.0 and acc_mgd_blur > best_sota_blur:
        best_sota_blur = acc_mgd_blur
        print(f"Saving model with Blur: {acc_mgd_blur:.2f}%")
        torch.save(student_mgd.state_dict(), 'best_student_vit.pth')