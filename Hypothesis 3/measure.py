import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

from models import BaselineResNet50, SelfDistillationResNet50, SelfDistillationResNet18_H10k, Baseline_Resnet18_H10k
from datasets import get_medmnist_dataloaders, get_chestxray_dataloaders, get_ham10000_dataloaders

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

def evaluate_model(model, dataloader, device, num_classes, is_sd=False):
    """
    Runs inference and calculates metrics for standard baselines 
    and individually for EACH exit of a multi-exit architecture.
    """
    model.eval()
    all_logits = {}
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if is_sd:
                exits = outputs[0] if isinstance(outputs, tuple) else outputs
                for i, exit_logits in enumerate(exits):
                    exit_name = f"Exit_{i+1}"
                    if exit_name not in all_logits:
                        all_logits[exit_name] = []
                    all_logits[exit_name].append(exit_logits.cpu())
            else:
                if 'Baseline' not in all_logits:
                    all_logits['Baseline'] = []
                all_logits['Baseline'].append(outputs.cpu())
                
            all_targets.append(targets.squeeze().cpu())

    all_targets = torch.cat(all_targets, dim=0)
    y_true = all_targets.numpy()

    results = {}
    for name, logits_list in all_logits.items():
        cat_logits = torch.cat(logits_list, dim=0)
        y_prob = F.softmax(cat_logits, dim=1).numpy()
        
        metrics = {}
        
        metrics['NLL'] = F.cross_entropy(cat_logits, all_targets).item()
        
        metrics['ECE'], metrics['MCE'] = calculate_ece_mce(y_true, y_prob)
        metrics['Brier_Score'] = calculate_brier_score(y_true, y_prob, num_classes)
        
        predictions = np.argmax(y_prob, axis=1)
        metrics['Accuracy'] = np.mean(predictions == y_true)
        
        try:
            if num_classes == 2:
                metrics['AUROC'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['AUROC'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            metrics['AUROC'] = float('nan') 

        results[name] = metrics

    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_model_dir = "Hypothesis 3/saved_models"

    # --- 1. BLOODMNIST SECTION (8 Classes) ---
    print("\nLoading BloodMNIST Dataset...")
    _, _, test_loader_blood, in_channels_blood, num_classes_blood = get_medmnist_dataloaders('bloodmnist', batch_size=128)

    model_baseline_blood = BaselineResNet50(num_classes=num_classes_blood, in_channels=in_channels_blood) 
    model_distilled_blood = SelfDistillationResNet50(num_classes=num_classes_blood, in_channels=in_channels_blood)
    
    baseline_path = os.path.join(base_model_dir, 'resnet50_bloodmnist_baseline_best.pth')
    distilled_path = os.path.join(base_model_dir, 'resnet50_bloodmnist_self_distilled_best.pth')
    
    if os.path.exists(baseline_path):
        model_baseline_blood.load_state_dict(torch.load(baseline_path, map_location=device))
    if os.path.exists(distilled_path):
        model_distilled_blood.load_state_dict(torch.load(distilled_path, map_location=device))

    model_baseline_blood.to(device)
    model_distilled_blood.to(device)

    print("\n--- Evaluating Baseline ResNet-50 on BloodMNIST ---")
    baseline_res_blood = evaluate_model(model_baseline_blood, test_loader_blood, device, num_classes=num_classes_blood, is_sd=False)
    for metric, val in baseline_res_blood['Baseline'].items():
        print(f"  {metric}: {val:.4f}")

    print("\n--- Evaluating Self-Distilled ResNet-50 on BloodMNIST ---")
    distilled_res_blood = evaluate_model(model_distilled_blood, test_loader_blood, device, num_classes=num_classes_blood, is_sd=True)
    for exit_name, metrics in distilled_res_blood.items():
        print(f"  [{exit_name}]")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    # # --- 2. CHEST X-RAY SECTION (2 Classes)
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
    # baseline_res_chest = evaluate_model(model_baseline_chest, test_loader_chest, device, num_classes=num_classes_chest, is_sd=False)
    # for metric, val in baseline_res_chest['Baseline'].items():
    #     print(f"  {metric}: {val:.4f}")
    
    # print("\n--- Evaluating Self-Distilled ResNet-18 on Chest X-Ray ---")
    # distilled_res_chest = evaluate_model(model_distilled_chest, test_loader_chest, device, num_classes=num_classes_chest, is_sd=True)
    # for exit_name, metrics in distilled_res_chest.items():
    #     print(f"  [{exit_name}]")
    #     for k, v in metrics.items():
    #         print(f"    {k}: {v:.4f}")