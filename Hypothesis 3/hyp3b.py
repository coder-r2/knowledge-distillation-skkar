import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np

# Import custom modules
from datasets import get_medmnist_dataloaders
from models import BaselineResNet18, SelfDistillationResNet18
from losses import CalibrationAwareSelfDistillationLoss
from train import (
    ECEMetric, 
    evaluate_standard, measure_standard_inference,
    train_sd_epoch, evaluate_sd, measure_sd_inference
)
from calibration_curve import plot_calibration_curves

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
    
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    save_dir = 'Hypothesis 3/saved_models'
    os.makedirs(save_dir, exist_ok=True)
    casd_save_path = os.path.join(save_dir, 'resnet18_bloodmnist_casd_best.pth')
    
    # Path to your PRE-TRAINED baseline from hyp3a
    baseline_save_path = os.path.join(save_dir, 'resnet18_bloodmnist_baseline_best.pth')

    # --- Hyperparameters ---
    epochs = 250
    patience = 30
    learning_rate = 0.001
    batch_size = 128

    # --- Data Loading ---
    print("Loading datasets...")
    train_loader, val_loader, test_loader, in_channels, num_classes = get_medmnist_dataloaders('bloodmnist', batch_size)

    # --- Instantiation ---
    ece_metric = ECEMetric(n_bins=15)

    # CASD Model 
    casd_model = SelfDistillationResNet18(num_classes=num_classes, in_channels=in_channels).to(device)
    casd_criterion = CalibrationAwareSelfDistillationLoss(alpha=0.5, lambda_weight=0.01, beta=0.5).to(device)
    casd_optimizer = optim.Adam(casd_model.parameters(), lr=learning_rate)

    # Baseline Model (For Evaluation Only)
    baseline_model = BaselineResNet18(num_classes=num_classes, in_channels=in_channels).to(device)

    # --- Telemetry Storage (CASD Only) ---
    casd_history = {
        'epochs': [], 'train_loss': [], 'val_loss': [],
        'ece': [[] for _ in range(4)], 'acc': [[] for _ in range(4)]
    }

    # ==========================================
    # 1. TRAIN CASD MODEL (Calibration-Aware)
    # ==========================================
    print("\n--- Starting CASD Model Training ---")
    best_casd_val_loss = float('inf')
    casd_patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_sd_epoch(casd_model, train_loader, casd_optimizer, casd_criterion, device, epoch, epochs)
        
        # Evaluate standard CE validation loss for early stopping parity
        val_results = evaluate_sd(casd_model, val_loader, device, ece_metric, criterion=nn.CrossEntropyLoss(), num_exits=4)
        
        deepest_val_loss = val_results[3]['loss']

        casd_history['epochs'].append(epoch)
        casd_history['train_loss'].append(train_loss)
        casd_history['val_loss'].append(deepest_val_loss)
        
        for i, res in enumerate(val_results):
            casd_history['ece'][i].append(res['ece'])
            casd_history['acc'][i].append(res['acc'])

        # Early Stopping Logic
        if deepest_val_loss < best_casd_val_loss:
            best_casd_val_loss = deepest_val_loss
            casd_patience_counter = 0
            torch.save(casd_model.state_dict(), casd_save_path)
            is_best = " (New Best!)"
        else:
            casd_patience_counter += 1
            is_best = ""

        if epoch % 5 == 0 or is_best:    
            print(f"Epoch [{epoch}/{epochs}] CASD Train Loss: {train_loss:.4f} | Exit 4 Val Loss: {deepest_val_loss:.4f}{is_best}")

        if casd_patience_counter >= patience:
            print(f"\nEarly stopping triggered for CASD model at epoch {epoch}")
            break

    # ==========================================
    # 2. FINAL EVALUATION (LOADING BEST WEIGHTS)
    # ==========================================
    print("\n--- Running Final Evaluation on TEST Set ---")
    
    # Load CASD
    casd_model.load_state_dict(torch.load(casd_save_path, map_location=device))
    
    # Load PRE-TRAINED Baseline
    try:
        baseline_model.load_state_dict(torch.load(baseline_save_path, map_location=device))
        print("Successfully loaded pre-trained baseline model.")
    except FileNotFoundError:
        print(f"\nERROR: Could not find the baseline model at {baseline_save_path}")
        print("Please ensure you have run hyp3a.py first to generate the baseline weights.")
        return

    print("\n[CASD Model Test Results]")
    casd_final_results = evaluate_sd(casd_model, test_loader, device, ece_metric, num_exits=4)
    for res in casd_final_results:
        print(f"Exit {res['exit']} - Test Acc: {res['acc']:.2f}% | Test ECE: {res['ece']:.2f}%")

    print("\n[Baseline Model Test Results]")
    baseline_results = evaluate_standard(baseline_model, test_loader, device, ece_metric)
    print(f"Baseline - Test Acc: {baseline_results['acc']:.2f}% | Test ECE: {baseline_results['ece']:.2f}%")

    print("\nMeasuring Inference Throughput...")
    dummy_batch = torch.randn(128, in_channels, 32, 32).to(device)
    for exit_idx in range(4):
        time_ms = measure_sd_inference(casd_model, dummy_batch, exit_idx)
        print(f"CASD Exit {exit_idx+1} Inference Time: {time_ms:.2f} ms")
        
    baseline_time = measure_standard_inference(baseline_model, dummy_batch)
    print(f"Baseline Inference Time: {baseline_time:.2f} ms")

    # ==========================================
    # 3. PLOT CONVERGENCE GRAPHS (CASD ONLY)
    # ==========================================
    os.makedirs('Hypothesis 3/results', exist_ok=True)
    plt.figure(figsize=(15, 6))
    plt.suptitle('CASD ResNet-18 Training Convergence', fontsize=16)

    # Graph 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(casd_history['epochs'], casd_history['train_loss'], label='CASD Train Loss', color='blue', alpha=0.6)
    plt.plot(casd_history['epochs'], casd_history['val_loss'], label='CASD Val Loss (Exit 4)', color='darkblue', linewidth=2)
    plt.title('Training & Validation Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Graph 2: ECE
    plt.subplot(1, 2, 2)
    colors = ['#c6e2ff', '#7eb6ff', '#2171ed', '#00008b']
    for i in range(4):
        plt.plot(casd_history['epochs'], casd_history['ece'][i], label=f'CASD Exit {i+1}', color=colors[i])
    plt.title('Validation ECE vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('ECE (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Hypothesis 3/results/resnet18_bloodmnist_casd_convergence.png')
    plt.show()

    # ==========================================
    # 4. GENERATE CALIBRATION CURVES
    # ==========================================
    print("\nGenerating Calibration Curves for Best Models...")
    models_to_plot = [
        {
            'path': baseline_save_path,
            'class': BaselineResNet18,
            'name': 'ResNet-18 Baseline (Best)'
        },
        {
            'path': casd_save_path,
            'class': SelfDistillationResNet18,  
            'name': 'ResNet-18 CASD (Best)'
        }
    ]
    plot_calibration_curves(models_to_plot)

if __name__ == "__main__":
    main()