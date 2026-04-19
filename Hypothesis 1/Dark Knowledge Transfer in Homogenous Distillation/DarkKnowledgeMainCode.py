import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import urllib.request
import tarfile
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

#Seed the code
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(25526)

#Change path according to your environment
DATA_DIR = '/kaggle/working/data'
CIFAR_C_DIR = os.path.join(DATA_DIR, 'CIFAR-10-C')
NUM_CLASSES = 10
BATCH_SIZE = 128 

MAX_EPOCHS = 200 
PATIENCE = 5       
TOLERANCE = 1e-3   
LEARNING_RATE = 1e-3 
TEMPERATURE = 4.0
ALPHA = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {DEVICE}")

#Downloading the Dataset, to be skipped if CIFAR-10-C is already downloaded
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract_cifar_c():
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    os.makedirs(DATA_DIR, exist_ok=True)
    tar_path = os.path.join(DATA_DIR, "CIFAR-10-C.tar")

    if not os.path.exists(CIFAR_C_DIR):
        if not os.path.exists(tar_path):
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading") as t:
                urllib.request.urlretrieve(url, filename=tar_path, reporthook=t.update_to)
        with tarfile.open(tar_path, 'r') as tar:
            if sys.version_info >= (3, 12):
                tar.extractall(path=DATA_DIR, filter='data')
            else:
                tar.extractall(path=DATA_DIR)

class CIFAR10CDataset(Dataset):
    def __init__(self, data_array, labels_array, transform=None):
        self.data = data_array
        self.labels = labels_array
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = int(self.labels[idx])
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label

#Stop early if convergence
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
#Load data
def get_dataloaders():
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    val_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, transform_test

#Use a modified ResNet for processing CIFAR dataset meaningfully
def modify_resnet_for_cifar(model):
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def get_models():
    parent = modify_resnet_for_cifar(models.resnet50(weights=None)).to(DEVICE)
    baseline_student = modify_resnet_for_cifar(models.resnet18(weights=None)).to(DEVICE)
    kd_student = modify_resnet_for_cifar(models.resnet18(weights=None)).to(DEVICE)
    return parent, baseline_student, kd_student

def kd_loss_fn(student_logits, teacher_logits, labels, T, alpha):
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    return (alpha * hard_loss) + ((1. - alpha) * soft_loss)

def train_epoch(model, dataloader, optimizer, is_kd=False, teacher=None):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(inputs)
        
        if is_kd and teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            loss = kd_loss_fn(logits, teacher_logits, labels, TEMPERATURE, ALPHA)
        else:
            loss = F.cross_entropy(logits, labels)
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits = model(inputs)
            total_loss += F.cross_entropy(logits, labels).item()
    return total_loss / len(dataloader)

def train_until_convergence(model_name, model, train_loader, val_loader, optimizer, is_kd=False, teacher=None):
    print(f"\n--- Training {model_name} ---")
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=TOLERANCE)
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, is_kd, teacher)
        val_loss = validate_epoch(model, val_loader)
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Convergence reached! Stopping early at epoch {epoch+1}.")
            break

def evaluate_robustness_classwise(model, base_c_dir, transform, num_classes=10):
    model.eval()
    results = {} 
    labels_path = os.path.join(base_c_dir, 'labels.npy')
    all_labels = np.load(labels_path)
    for file in os.listdir(base_c_dir):
        if not file.endswith('.npy') or file == 'labels.npy': continue
        corruption = file.replace('.npy', '')
        results[corruption] = {
            'overall_correct': 0, 'overall_total': 0,
            'classes': {i: {'correct': 0, 'total': 0} for i in range(num_classes)}
        }
        all_data = np.load(os.path.join(base_c_dir, file))
        
        for severity in range(1, 6):
            start_idx, end_idx = (severity - 1) * 10000, severity * 10000
            sev_data, sev_labels = all_data[start_idx:end_idx], all_labels[start_idx:end_idx]
            dataset_c = CIFAR10CDataset(sev_data, sev_labels, transform=transform)
            loader_c = DataLoader(dataset_c, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            
            with torch.no_grad():
                for inputs, labels in tqdm(loader_c, desc=f"Eval {corruption} (Sev {severity})", leave=False):
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    _, predicted = torch.max(model(inputs).data, 1)
                    results[corruption]['overall_total'] += labels.size(0)
                    results[corruption]['overall_correct'] += (predicted == labels).sum().item()

        overall_acc = 100 * results[corruption]['overall_correct'] / results[corruption]['overall_total']
        print(f"[{corruption.upper()}] Overall Robustness: {overall_acc:.2f}%")
    return results

def plot_robustness_results(parent_res, base_res, kd_res):
    print("\nGenerating Robustness Comparison Chart...")
    corruptions = list(parent_res.keys())
    
    # Extract accuracies
    parent_accs = [100 * parent_res[c]['overall_correct'] / parent_res[c]['overall_total'] for c in corruptions]
    base_accs = [100 * base_res[c]['overall_correct'] / base_res[c]['overall_total'] for c in corruptions]
    kd_accs = [100 * kd_res[c]['overall_correct'] / kd_res[c]['overall_total'] for c in corruptions]

    x = np.arange(len(corruptions))  
    width = 0.25  

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(x - width, parent_accs, width, label='Teacher (ResNet-50)', color='#1f77b4', edgecolor='black', zorder=3)
    ax.bar(x, base_accs, width, label='Baseline Student (ResNet-18)', color='#ff7f0e', edgecolor='black', zorder=3)
    ax.bar(x + width, kd_accs, width, label='KD Student (ResNet-18)', color='#2ca02c', edgecolor='black', zorder=3)
    ax.set_ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('CIFAR-10-C Robustness: Knowledge Distillation Transfer Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in corruptions], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    fig.tight_layout()
    plot_path = '/kaggle/working/robustness_comparison_chart.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Chart saved successfully to: {plot_path}")
    plt.show()

if __name__ == '__main__':
    print("Initializing Kaggle Master Pipeline with Fixed Random Seed...")
    #Download data
    download_and_extract_cifar_c()
    train_loader, val_loader, transform = get_dataloaders()

    parent, baseline_student, kd_student = get_models()
    opt_parent = optim.AdamW(parent.parameters(), lr=LEARNING_RATE)
    opt_baseline = optim.AdamW(baseline_student.parameters(), lr=LEARNING_RATE)
    opt_kd = optim.AdamW(kd_student.parameters(), lr=LEARNING_RATE)
    
    #Train Models
    train_until_convergence("Parent ResNet-50", parent, train_loader, val_loader, opt_parent)
    parent.eval() 
    train_until_convergence("Baseline ResNet-18", baseline_student, train_loader, val_loader, opt_baseline)
    train_until_convergence("KD ResNet-18", kd_student, train_loader, val_loader, opt_kd, is_kd=True, teacher=parent)
    torch.save(parent.state_dict(), "/kaggle/working/parent_resnet50_cifar.pth")
    torch.save(baseline_student.state_dict(), "/kaggle/working/baseline_student_resnet18_cifar.pth")
    torch.save(kd_student.state_dict(), "/kaggle/working/kd_student_resnet18_cifar.pth")

    print("\n--- Evaluating Parent ResNet-50 on CIFAR-10-C ---")
    parent_results = evaluate_robustness_classwise(parent, CIFAR_C_DIR, transform)
    print("\n--- Evaluating Baseline Student on CIFAR-10-C ---")
    baseline_results = evaluate_robustness_classwise(baseline_student, CIFAR_C_DIR, transform)
    print("\n--- Evaluating KD Student on CIFAR-10-C ---")
    kd_results = evaluate_robustness_classwise(kd_student, CIFAR_C_DIR, transform)

    if parent_results and baseline_results and kd_results:
        print("Full Analysis")
        for corruption in parent_results.keys(): 
            parent_overall = 100 * parent_results[corruption]['overall_correct'] / parent_results[corruption]['overall_total']
            base_overall = 100 * baseline_results[corruption]['overall_correct'] / baseline_results[corruption]['overall_total']
            kd_overall = 100 * kd_results[corruption]['overall_correct'] / kd_results[corruption]['overall_total'] 
            print(f"[{corruption.upper()}]")
            print(f"  Teacher (ResNet-50):    {parent_overall:.2f}%")
            print(f"  Student (Baseline):     {base_overall:.2f}%")
            print(f"  Student (KD):           {kd_overall:.2f}%")
            print(f"  KD Transfer Shift:      {kd_overall - base_overall:+.2f}% (Improvement over Baseline)")
            print(f"  Teacher Gap:            {parent_overall - kd_overall:+.2f}% (How far KD is from Teacher)\n")

        plot_robustness_results(parent_results, baseline_results, kd_results)
