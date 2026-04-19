import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import urllib.request
import tarfile
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm

#Seeding the code
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(25526)

#Configure model paths according to your environment
DATA_DIR = '/kaggle/working/data'
CIFAR_C_DIR = os.path.join(DATA_DIR, 'CIFAR-10-C')
NUM_CLASSES = 10
BATCH_SIZE = 128 
PARENT_WEIGHTS = '/kaggle/input/models/kretikas/r50/pytorch/default/1/parent_resnet50_cifar.pth'
BASELINE_WEIGHTS = '/kaggle/input/models/kretikas/r18base/pytorch/default/1/baseline_student_resnet18_cifar.pth'
KD_WEIGHTS = '/kaggle/input/models/kretikas/r18kd/pytorch/default/1/kd_student_resnet18_cifar.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {DEVICE}")

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Downloading Data
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

def get_transforms():
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

#The ResNet is slightly modified to run CiFAR dataset without collapsing
def modify_resnet_for_cifar(model):
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def load_pretrained_models():
    print("\n--- Loading Pretrained Models ---")
    parent = modify_resnet_for_cifar(models.resnet50(weights=None)).to(DEVICE)
    baseline_student = modify_resnet_for_cifar(models.resnet18(weights=None)).to(DEVICE)
    kd_student = modify_resnet_for_cifar(models.resnet18(weights=None)).to(DEVICE)

    try:
        parent.load_state_dict(torch.load(PARENT_WEIGHTS, map_location=DEVICE))
        baseline_student.load_state_dict(torch.load(BASELINE_WEIGHTS, map_location=DEVICE))
        kd_student.load_state_dict(torch.load(KD_WEIGHTS, map_location=DEVICE))
        print("Models loaded successfully!")
    except Exception as e:
        raise ValueError(f"Error loading models. Please verify the paths point to .pth files.\nDetails: {e}")
    
    parent.eval()
    baseline_student.eval()
    kd_student.eval()
    return parent, baseline_student, kd_student

#GradCAM
def unnormalize(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)
        loss = output[0, target_class]
        loss.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()

def get_saliency_map(model, input_image, target_class):
    input_image.requires_grad_()
    model.zero_grad()
    output = model(input_image)
    loss = output[0, target_class]
    loss.backward()
    saliency, _ = torch.max(input_image.grad.data.abs(), dim=1)
    return saliency.squeeze().cpu().numpy()

def get_occlusion_map(model, input_image, target_class, window_size=4, stride=4):
    model.eval()
    _, _, h, w = input_image.shape
    heatmap = np.zeros((h, w))
    with torch.no_grad():
        base_out = F.softmax(model(input_image), dim=1)[0, target_class].item()
    
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            h_start, h_end = i, min(i + window_size, h)
            w_start, w_end = j, min(j + window_size, w)
            occ_img = input_image.clone()
            occ_img[:, :, h_start:h_end, w_start:w_end] = 0
            with torch.no_grad():
                out = F.softmax(model(occ_img), dim=1)[0, target_class].item()
            heatmap[h_start:h_end, w_start:w_end] = base_out - out
            
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

#Error finding for GradCAM explanation
def find_and_explain_edge_cases(parent, base_student, kd_student, base_c_dir, transform):
    scenarios = {
        "SCENARIO 1: KD Success (KD fixed Baseline's mistake)": 
            lambda p, b, k, t: p == t and k == t and b != t,
    }
    
    required_counts = {
        "SCENARIO 1: KD Success (KD fixed Baseline's mistake)": 3
    }
    
    found_samples = {name: [] for name in scenarios.keys()}
    
    labels_path = os.path.join(base_c_dir, 'labels.npy')
    all_labels = np.load(labels_path)
    
    target_corruption = 'gaussian_noise.npy'
    data_path = os.path.join(base_c_dir, target_corruption)
    all_data = np.load(data_path)
    
    dataset = CIFAR10CDataset(all_data, all_labels, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True) 
    
    for img, label in loader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        target = label.item()
        
        with torch.no_grad():
            p_out = parent(img)
            b_out = base_student(img)
            k_out = kd_student(img)
            
            p_logits = p_out[1] if isinstance(p_out, tuple) else p_out
            b_logits = b_out[1] if isinstance(b_out, tuple) else b_out
            k_logits = k_out[1] if isinstance(k_out, tuple) else k_out
            
            p_pred = p_logits.argmax(dim=1).item()
            b_pred = b_logits.argmax(dim=1).item()
            k_pred = k_logits.argmax(dim=1).item()
            
        for scenario_name, condition_func in scenarios.items():
            if condition_func(p_pred, b_pred, k_pred, target) and len(found_samples[scenario_name]) < required_counts[scenario_name]:
                found_samples[scenario_name].append((img, target, p_pred, b_pred, k_pred))
                
        if all(len(found_samples[s]) >= required_counts[s] for s in scenarios.keys()):
            break
            
    gcam_parent = GradCAM(parent, parent.layer4[-1].conv3)
    gcam_base = GradCAM(base_student, base_student.layer4[-1].conv2)
    gcam_kd = GradCAM(kd_student, kd_student.layer4[-1].conv2)
    
    for scenario_name, samples in found_samples.items():
        print(f"\n[{scenario_name}] - Found {len(samples)}/{required_counts[scenario_name]} requested samples.")
        if not samples: continue
            
        for idx, (img_tensor, true_label, p_pred, b_pred, k_pred) in enumerate(samples):
            disp_img = unnormalize(img_tensor.clone().detach()[0]).permute(1, 2, 0).cpu().numpy()
            
            models_data = [
                (f"Teacher\n(Pred: {CIFAR_CLASSES[p_pred]})", parent, gcam_parent, p_pred),
                (f"Baseline\n(Pred: {CIFAR_CLASSES[b_pred]})", base_student, gcam_base, b_pred), 
                (f"KD Student\n(Pred: {CIFAR_CLASSES[k_pred]})", kd_student, gcam_kd, k_pred)
            ]
            
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            fig.suptitle(f"{scenario_name} (Sample {idx+1})\nTrue Label = {CIFAR_CLASSES[true_label]}", fontsize=16, fontweight='bold')
            
            for row, (title, model, gcam, pred_target) in enumerate(models_data):
                axes[row, 0].imshow(disp_img)
                axes[row, 0].set_title(title, fontsize=12, fontweight='bold')
                axes[row, 0].axis('off')
                
                sal = get_saliency_map(model, img_tensor.clone(), pred_target)
                axes[row, 1].imshow(sal, cmap='hot')
                axes[row, 1].set_title(f"Saliency Map")
                axes[row, 1].axis('off')
                
                cam = gcam.generate(img_tensor.clone(), pred_target)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = np.uint8(255 * disp_img) * 0.5 + heatmap * 0.5
                axes[row, 2].imshow(overlay.astype(np.uint8))
                axes[row, 2].set_title(f"Grad-CAM")
                axes[row, 2].axis('off')
                
                occ = get_occlusion_map(model, img_tensor.clone(), pred_target)
                axes[row, 3].imshow(occ, cmap='plasma')
                axes[row, 3].set_title(f"Occlusion Sensitivity")
                axes[row, 3].axis('off')
                
            plt.tight_layout()
            safe_name = scenario_name.split(':')[0].replace(" ", "_")
            plot_path = f'/kaggle/working/{safe_name}_sample_{idx+1}.png'
            plt.savefig(plot_path, dpi=300)
            plt.show()

#Degradation across severity levels
def get_custom_corruption_transform(corruption_type, severity):
    transform_list = []
    
    if severity > 0:
        if corruption_type == 'blur':
            sigma = severity * 0.4
            kernel_size = int(sigma * 4) + 1
            if kernel_size % 2 == 0: kernel_size += 1
            transform_list.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma))
            
        elif corruption_type == 'color_jitter':
            factor = severity * 0.3
            transform_list.append(transforms.ColorJitter(brightness=factor, contrast=factor, saturation=factor))
            
    transform_list.append(transforms.ToTensor())
    
    if severity > 0:
        if corruption_type == 'noise':
            std = severity * 0.08
            transform_list.append(transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * std, 0, 1)))
            
    mean, std_norm = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    transform_list.append(transforms.Normalize(mean=mean, std=std_norm))
    
    return transforms.Compose(transform_list)

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits = outputs[1]
            else:
                logits = outputs
                
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100.0

def run_progressive_corruption_experiment(parent, base_student, kd_student):
    corruptions = ['noise', 'blur', 'color_jitter']
    severities = [0, 1, 2, 3, 4, 5]
    
    results = {
        cor_type: {'Teacher': [], 'Baseline': [], 'KD_Student': []} 
        for cor_type in corruptions
    }
    
    for cor_type in corruptions:
        print(f"\nEvaluating Corruption Type: [{cor_type.upper()}]")
        
        for severity in severities:
            transform = get_custom_corruption_transform(cor_type, severity)
            
            test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
            
            acc_p = evaluate_model(parent, test_loader)
            acc_b = evaluate_model(base_student, test_loader)
            acc_k = evaluate_model(kd_student, test_loader)
            
            results[cor_type]['Teacher'].append(acc_p)
            results[cor_type]['Baseline'].append(acc_b)
            results[cor_type]['KD_Student'].append(acc_k)
            
            print(f"  Sev {severity}: Teacher={acc_p:.1f}% | Baseline={acc_b:.1f}% | KD={acc_k:.1f}%")
            
    return results, severities

def plot_degradation_curves(results, severities):
    corruptions = list(results.keys())
    fig, axes = plt.subplots(1, len(corruptions), figsize=(20, 6))
    fig.suptitle('Model Performance Degradation vs. Corruption Severity', fontsize=18, fontweight='bold', y=1.05)
    
    for idx, cor_type in enumerate(corruptions):
        ax = axes[idx]
        ax.plot(severities, results[cor_type]['Teacher'], marker='o', linewidth=2, markersize=8, label='Teacher (ResNet-50)', color='#1f77b4')
        ax.plot(severities, results[cor_type]['Baseline'], marker='s', linewidth=2, markersize=8, label='Baseline (ResNet-18)', color='#ff7f0e')
        ax.plot(severities, results[cor_type]['KD_Student'], marker='^', linewidth=2, markersize=8, label='KD Student (ResNet-18)', color='#2ca02c')
        
        ax.set_title(f'Corruption: {cor_type.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Severity Level (0 = Clean)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xticks(severities)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.7)
        if idx == 0:
            ax.legend(fontsize=11)
            
    plt.tight_layout()
    plot_path = '/kaggle/working/corruption_degradation_curves.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"\nDegradation plot saved to {plot_path}")
    plt.show()

#Entropy and Calibration Analysis
def calculate_average_entropy(model, dataloader, device):
    model.eval()
    total_entropy = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Calculating Entropy", leave=False):
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[1]
            else:
                logits = outputs
            probs = F.softmax(logits, dim=1)
            log_probs = torch.log(probs + 1e-9)
            entropy = -torch.sum(probs * log_probs, dim=1)
            total_entropy += entropy.sum().item()
            num_samples += images.size(0)

    return total_entropy / num_samples

def evaluate_cifar10_entropy(parent, baseline_student, kd_student, transform):
    cifar10_test = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(cifar10_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print("Evaluating Parent (Teacher) Model")
    parent_entropy = calculate_average_entropy(parent, test_loader, DEVICE)
    
    print("Evaluating Baseline Student Model")
    baseline_entropy = calculate_average_entropy(baseline_student, test_loader, DEVICE)
    
    print("Evaluating KD Student Model")
    kd_entropy = calculate_average_entropy(kd_student, test_loader, DEVICE)
    
    print("\nAverage Predictive Entropy Results")
    print(f"1. Teacher (ResNet-50):        {parent_entropy:.4f} nats")
    print(f"2. Baseline Student (ResNet-18): {baseline_entropy:.4f} nats")
    print(f"3. KD Student (ResNet-18):     {kd_entropy:.4f} nats")

if __name__ == '__main__':
    #Download dataset
    download_and_extract_cifar_c()
    transform = get_transforms()
    #Load pretrained models
    parent, baseline_student, kd_student = load_pretrained_models()
    evaluate_cifar10_entropy(parent, baseline_student, kd_student, transform)
    find_and_explain_edge_cases(parent, baseline_student, kd_student, CIFAR_C_DIR, transform)
    results, severities = run_progressive_corruption_experiment(parent, baseline_student, kd_student)
    plot_degradation_curves(results, severities)
