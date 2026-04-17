import os
import torch
import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
import random
import numpy as np

# Import custom modules
# from datasets import get_chestxray_dataloaders
from models import Baseline_Resnet18_H10k, SelfDistillationResNet18_H10k
# from losses import SelfDistillationLoss
from train import (
    # ECEMetric, 
    # train_standard_epoch, evaluate_standard, 
    measure_standard_inference,
    # train_sd_epoch, evaluate_sd, 
    measure_sd_inference
)
# from calibration_curve import plot_calibration_curves

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(67)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = 'Hypothesis 3/saved_models'
    
    # Save Paths for Chest X-Ray ResNet-18
    sd_save_path = os.path.join(save_dir, 'resnet18_chestxray_self_distilled_best.pth')
    baseline_save_path = os.path.join(save_dir, 'resnet18_chestxray_baseline_best.pth')

    # Hardcoded Dataset info so we don't have to waste time loading the dataloaders
    in_channels = 3
    num_classes = 2 # NORMAL and PNEUMONIA
    batch_size = 64 

    # --- Models ---
    print("Initializing ResNet-18 models...")
    sd_model = SelfDistillationResNet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)
    baseline_model = Baseline_Resnet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)

    # --- Load Weights ---
    try:
        sd_model.load_state_dict(torch.load(sd_save_path, map_location=device))
        baseline_model.load_state_dict(torch.load(baseline_save_path, map_location=device))
        print("Successfully loaded saved weights.")
    except Exception as e:
        print(f"Warning: Could not load weights. ({e}) Measuring inference with untrained weights.")

    # ==========================================
    # TRAINING LOOPS & EVALUATION (COMMENTED OUT)
    # ==========================================
    """
    print("\n--- Starting SD Model Training ---")
    # ... training code ...
    
    print("\n--- Starting Baseline ResNet-18 Training ---")
    # ... training code ...
    
    print("\n--- Running Final Evaluation on TEST Set ---")
    # ... test evaluation code ...
    """

    # ==========================================
    # MEASURE INFERENCE TIME
    # ==========================================
    print("\n--- Measuring Inference Throughput ---")
    # Chest X-Ray uses 224x224 images.
    dummy_batch = torch.randn(batch_size, in_channels, 224, 224).to(device)
    
    print("[Self-Distilled Model Exits]")
    for exit_idx in range(4):
        time_ms = measure_sd_inference(sd_model, dummy_batch, exit_idx)
        print(f"Exit {exit_idx+1} Inference Time: {time_ms:.2f} ms")
        
    print("\n[Baseline Model]")
    baseline_time = measure_standard_inference(baseline_model, dummy_batch)
    print(f"Baseline Inference Time: {baseline_time:.2f} ms")

    # ==========================================
    # GENERATE PLOTS (COMMENTED OUT)
    # ==========================================
    """
    print("\nGenerating Calibration Curves...")
    # ... plotting code ...
    """

if __name__ == "__main__":
    main()