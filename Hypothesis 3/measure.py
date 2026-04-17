import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

# Import your custom modules
from models import BaselineResNet50, SelfDistillationResNet50, SelfDistillationResNet18_H10k, Baseline_Resnet18_H10k
from datasets import get_medmnist_dataloaders, get_chestxray_dataloaders, get_ham10000_dataloaders

# ---------------------------------------------------------
# Calibration & Performance Metrics
# ---------------------------------------------------------

def calculate_ece_mce(y_true, y_prob, n_bins=15):
    """
    Calculates Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    ece = 0.0
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * error
            mce = max(mce, error)
            
    return ece, mce

def calculate_brier_score(y_true, y_prob, num_classes):
    """
    Calculates the Brier Score.
    """
    y_true_one_hot = np.eye(num_classes)[y_true]
    brier_score = np.mean(np.sum((y_prob - y_true_one_hot)**2, axis=1))
    return brier_score

def evaluate_model(model, dataloader, device, num_classes):
    """
    Runs inference and calculates all metrics, supporting both standard
    and multi-exit architectures.
    """
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Handle Self-Distilled multi-exit outputs
            if isinstance(outputs, tuple):
                logits = outputs[0][-1]  # Get logits from the deepest exit (Exit 4)
            else:
                logits = outputs
                
            all_logits.append(logits.cpu())
            all_targets.append(targets.squeeze().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    y_prob = F.softmax(all_logits, dim=1).numpy()
    y_true = all_targets.numpy()

    metrics = {}

    # NLL (Cross Entropy)
    metrics['NLL'] = F.cross_entropy(all_logits, all_targets).item()
    
    # Calibration Metrics
    metrics['ECE'], metrics['MCE'] = calculate_ece_mce(y_true, y_prob)
    metrics['Brier_Score'] = calculate_brier_score(y_true, y_prob, num_classes)
    
    # Standard Metrics
    predictions = np.argmax(y_prob, axis=1)
    metrics['Accuracy'] = np.mean(predictions == y_true)
    
    # AUROC
    try:
        if num_classes == 2:
            # Binary classification requires positive class probabilities
            metrics['AUROC'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # Multi-class (One-vs-Rest)
            metrics['AUROC'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except ValueError:
        metrics['AUROC'] = float('nan') 

    return metrics

# ---------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_model_dir = "Hypothesis 3/saved_models"

    # # --- 1. BLOODMNIST SECTION (8 Classes) ---
    # print("\nLoading BloodMNIST Dataset...")
    # _, _, test_loader_blood, in_channels_blood, num_classes_blood = get_medmnist_dataloaders('bloodmnist', batch_size=128)

    # model_baseline_blood = BaselineResNet50(num_classes=num_classes_blood, in_channels=in_channels_blood) 
    # model_distilled_blood = SelfDistillationResNet50(num_classes=num_classes_blood, in_channels=in_channels_blood)
    
    # # Load weights (adjust filenames if yours differ)
    # baseline_path = os.path.join(base_model_dir, 'resnet50_bloodmnist_baseline_best.pth')
    # distilled_path = os.path.join(base_model_dir, 'resnet50_bloodmnist_self_distilled_best.pth')
    
    # if os.path.exists(baseline_path):
    #     model_baseline_blood.load_state_dict(torch.load(baseline_path, map_location=device))
    # if os.path.exists(distilled_path):
    #     model_distilled_blood.load_state_dict(torch.load(distilled_path, map_location=device))

    # model_baseline_blood.to(device)
    # model_distilled_blood.to(device)

    # print("\n--- Evaluating Baseline ResNet-50 on BloodMNIST ---")
    # baseline_metrics = evaluate_model(model_baseline_blood, test_loader_blood, device, num_classes=num_classes_blood)
    # for k, v in baseline_metrics.items():
    #     print(f"{k}: {v:.4f}")

    # print("\n--- Evaluating Self-Distilled ResNet-50 on BloodMNIST ---")
    # distilled_metrics = evaluate_model(model_distilled_blood, test_loader_blood, device, num_classes=num_classes_blood)
    # for k, v in distilled_metrics.items():
    #     print(f"{k}: {v:.4f}")

    # # --- 2. CHEST X-RAY SECTION (2 Classes) ---
    # print("\nLoading Chest X-Ray Dataset...")
    # data_dir = 'Hypothesis 3/data/chest_xray/chest_xray'
    # _, _, test_loader_chest, in_channels_chest, num_classes_chest = get_chestxray_dataloaders(data_dir, batch_size=64)
    # model_baseline_chest = Baseline_Resnet18_H10k(num_classes=num_classes_chest, in_channels=in_channels_chest)
    # model_distilled_chest = SelfDistillationResNet18_H10k(num_classes=num_classes_chest, in_channels=in_channels_chest)

    # # Load weights (adjust filenames if yours differ)
    # baseline_path_chest = os.path.join(base_model_dir, 'resnet18_chestxray_baseline_best.pth')
    # distilled_path_chest = os.path.join(base_model_dir, 'resnet18_chestxray_self_distilled_best.pth')
    # if os.path.exists(baseline_path_chest):
    #     model_baseline_chest.load_state_dict(torch.load(baseline_path_chest, map_location=device))
    # if os.path.exists(distilled_path_chest):
    #     model_distilled_chest.load_state_dict(torch.load(distilled_path_chest, map_location=device))
    
    # model_baseline_chest.to(device)
    # model_distilled_chest.to(device)

    # print("\n--- Evaluating Baseline ResNet-18 on Chest X-Ray ---")
    # baseline_metrics_chest = evaluate_model(model_baseline_chest, test_loader_chest, device, num_classes=num_classes_chest)
    # for k, v in baseline_metrics_chest.items():
    #     print(f"{k}: {v:.4f}")
    
    # print("\n--- Evaluating Self-Distilled ResNet-18 on Chest X-Ray ---")
    # distilled_metrics_chest = evaluate_model(model_distilled_chest, test_loader_chest, device, num_classes=num_classes_chest)
    # for k, v in distilled_metrics_chest.items():
    #     print(f"{k}: {v:.4f}")

    # === 3. HAM10000 SECTION (7 Classes) ===
    print("\nLoading HAM10000 Dataset...")
    data_dir_ham = 'Hypothesis 3/data/HAM10000'
    _, _, test_loader_ham, in_channels_ham, num_classes_ham = get_ham10000_dataloaders(data_dir_ham, batch_size=64)
    model_baseline_ham = Baseline_Resnet18_H10k(num_classes=num_classes_ham, in_channels=in_channels_ham)
    model_distilled_ham = SelfDistillationResNet18_H10k(num_classes=num_classes_ham, in_channels=in_channels_ham)

    # Load weights (adjust filenames if yours differ)
    baseline_path_ham = os.path.join(base_model_dir, 'resnet18_ham10000_baseline_best.pth')
    distilled_path_ham = os.path.join(base_model_dir, 'resnet18_ham10000_self_distilled_best.pth')
    if os.path.exists(baseline_path_ham):
        model_baseline_ham.load_state_dict(torch.load(baseline_path_ham, map_location=device))

    if os.path.exists(distilled_path_ham):
        model_distilled_ham.load_state_dict(torch.load(distilled_path_ham, map_location=device))
    
    model_baseline_ham.to(device)
    model_distilled_ham.to(device)

    print("\n--- Evaluating Baseline ResNet-18 on HAM10000 ---")
    baseline_metrics_ham = evaluate_model(model_baseline_ham, test_loader_ham, device, num_classes=num_classes_ham)
    for k, v in baseline_metrics_ham.items():
        print(f"{k}: {v:.4f}")
    
    print("\n--- Evaluating Self-Distilled ResNet-18 on HAM10000 ---")
    distilled_metrics_ham = evaluate_model(model_distilled_ham, test_loader_ham, device, num_classes=num_classes_ham)
    for k, v in distilled_metrics_ham.items():
        print(f"{k}: {v:.4f}")
