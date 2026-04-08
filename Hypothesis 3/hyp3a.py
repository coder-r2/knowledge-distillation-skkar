import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np

# Import custom modules
from datasets import get_ham10000_dataloaders
from models import Baseline_Resnet18_H10k, SelfDistillationResNet18_H10k
from losses import SelfDistillationLoss
from train import (
    ECEMetric, 
    train_standard_epoch, evaluate_standard, measure_standard_inference,
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

    # Ensure save directory exists
    save_dir = 'Hypothesis 3/saved_models'
    os.makedirs(save_dir, exist_ok=True)
    sd_save_path = os.path.join(save_dir, 'resnet18_ham10000_self_distilled_best.pth')
    baseline_save_path = os.path.join(save_dir, 'resnet18_ham10000_baseline_best.pth')

    # --- Hyperparameters ---
    epochs = 250
    patience = 30
    learning_rate = 0.001
    batch_size = 128

    # --- Data Loading ---
    print("Loading datasets...")
    # Updated to handle running from the workspace root
    data_dir = 'Hypothesis 3/data/HAM10000' 
    train_loader, val_loader, test_loader, in_channels, num_classes = get_ham10000_dataloaders(data_dir, batch_size)

    # --- Instantiation ---
    ece_metric = ECEMetric(n_bins=15)

    # SD Model
    sd_model = SelfDistillationResNet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)
    # Applied your Optuna hyperparameters here
    sd_criterion = SelfDistillationLoss(alpha=0.575, lambda_weight=0.0084, temperature=4.447)
    sd_optimizer = optim.Adam(sd_model.parameters(), lr=learning_rate)

    # Baseline Model
    baseline_model = Baseline_Resnet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)
    baseline_criterion = nn.CrossEntropyLoss()
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)

    # --- Telemetry Storage ---
    sd_history = {
        'epochs': [], 'train_loss': [], 'val_loss': [],
        'ece': [[] for _ in range(4)], 'acc': [[] for _ in range(4)]
    }
    baseline_history = {
        'epochs': [], 'train_loss': [], 'val_loss': [], 'ece': [], 'acc': []
    }

    # ==========================================
    # 1. TRAIN SELF-DISTILLATION MODEL
    # ==========================================
    print("\n--- Starting SD Model Training ---")
    best_sd_val_ece = float('inf') # Changed to track ECE instead of Loss
    sd_patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_sd_epoch(sd_model, train_loader, sd_optimizer, sd_criterion, device, epoch, epochs)
        
        val_results = evaluate_sd(sd_model, val_loader, device, ece_metric, criterion=nn.CrossEntropyLoss(), num_exits=4)
        
        deepest_val_loss = val_results[3]['loss']
        deepest_val_ece = val_results[3]['ece'] # Track the ECE for early stopping

        sd_history['epochs'].append(epoch)
        sd_history['train_loss'].append(train_loss)
        sd_history['val_loss'].append(deepest_val_loss)
        
        for i, res in enumerate(val_results):
            sd_history['ece'][i].append(res['ece'])
            sd_history['acc'][i].append(res['acc'])

        # Early Stopping Logic (Now based on ECE)
        if deepest_val_ece < best_sd_val_ece:
            best_sd_val_ece = deepest_val_ece
            sd_patience_counter = 0
            torch.save(sd_model.state_dict(), sd_save_path)
            is_best = " (New Best ECE!)"
        else:
            sd_patience_counter += 1
            is_best = ""

        if epoch % 5 == 0 or is_best:    
            print(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} | Exit 4 Val ECE: {deepest_val_ece:.2f}%{is_best}")

        if sd_patience_counter >= patience:
            print(f"\nEarly stopping triggered for SD model at epoch {epoch}")
            break

    # ==========================================
    # 2. TRAIN BASELINE MODEL
    # ==========================================
    print("\n--- Starting Baseline ResNet-18 Training ---")
    best_baseline_val_ece = float('inf') # Changed to track ECE instead of Loss
    baseline_patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_standard_epoch(baseline_model, train_loader, baseline_optimizer, baseline_criterion, device, epoch, epochs)
        val_results = evaluate_standard(baseline_model, val_loader, device, ece_metric, criterion=baseline_criterion)
        
        val_loss = val_results['loss']
        val_ece = val_results['ece']

        baseline_history['epochs'].append(epoch)
        baseline_history['train_loss'].append(train_loss)
        baseline_history['val_loss'].append(val_loss)
        baseline_history['ece'].append(val_ece)
        baseline_history['acc'].append(val_results['acc'])

        # Early Stopping Logic (Now based on ECE)
        if val_ece < best_baseline_val_ece:
            best_baseline_val_ece = val_ece
            baseline_patience_counter = 0
            torch.save(baseline_model.state_dict(), baseline_save_path)
            is_best = " (New Best ECE!)"
        else:
            baseline_patience_counter += 1
            is_best = ""

        if epoch % 5 == 0 or is_best:
            print(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} | Val ECE: {val_ece:.2f}%{is_best}")

        if baseline_patience_counter >= patience:
            print(f"\nEarly stopping triggered for Baseline model at epoch {epoch}")
            break

    # ==========================================
    # 3. FINAL EVALUATION (LOADING BEST WEIGHTS)
    # ==========================================
    print("\n--- Running Final Evaluation on TEST Set ---")
    sd_model.load_state_dict(torch.load(sd_save_path, map_location=device))
    baseline_model.load_state_dict(torch.load(baseline_save_path, map_location=device))

    print("\n[Self-Distillation Model Test Results]")
    sd_final_results = evaluate_sd(sd_model, test_loader, device, ece_metric, num_exits=4)
    for res in sd_final_results:
        print(f"Exit {res['exit']} - Test Acc: {res['acc']:.2f}% | Test ECE: {res['ece']:.2f}%")

    print("\n[Baseline Model Test Results]")
    baseline_results = evaluate_standard(baseline_model, test_loader, device, ece_metric)
    print(f"Baseline - Test Acc: {baseline_results['acc']:.2f}% | Test ECE: {baseline_results['ece']:.2f}%")

    print("\nMeasuring Inference Throughput...")
    dummy_batch = torch.randn(128, in_channels, 224, 224).to(device)
    
    for exit_idx in range(4):
        time_ms = measure_sd_inference(sd_model, dummy_batch, exit_idx)
        print(f"SD Exit {exit_idx+1} Inference Time: {time_ms:.2f} ms")
        
    baseline_time = measure_standard_inference(baseline_model, dummy_batch)
    print(f"Baseline Inference Time: {baseline_time:.2f} ms")

    # ==========================================
    # 4. PLOT CONVERGENCE GRAPHS
    # ==========================================
    os.makedirs('Hypothesis 3/results', exist_ok=True)
    plt.figure(figsize=(15, 6))
    plt.suptitle('Performance Comparison: Self-Distillation vs. Standard ResNet-18', fontsize=16)

    # Graph 1: Loss vs Epoch (Train and Val)
    plt.subplot(1, 2, 1)
    plt.plot(sd_history['epochs'], sd_history['train_loss'], label='SD Train Loss', color='blue', alpha=0.6)
    plt.plot(sd_history['epochs'], sd_history['val_loss'], label='SD Val Loss (Exit 4)', color='darkblue', linewidth=2)
    plt.plot(baseline_history['epochs'], baseline_history['train_loss'], label='Base Train Loss', color='red', alpha=0.6, linestyle='--')
    plt.plot(baseline_history['epochs'], baseline_history['val_loss'], label='Base Val Loss', color='darkred', linewidth=2, linestyle='--')
    plt.title('Training & Validation Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Graph 2: ECE vs Epoch
    plt.subplot(1, 2, 2)
    colors = ['#c6e2ff', '#7eb6ff', '#2171ed', '#00008b']
    for i in range(4):
        plt.plot(sd_history['epochs'], sd_history['ece'][i], label=f'SD Exit {i+1}', color=colors[i])
    plt.plot(baseline_history['epochs'], baseline_history['ece'], label='Baseline', color='red', linewidth=2, linestyle='--')
    plt.title('Validation ECE vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('ECE (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Hypothesis 3/results/resnet18_ham10000_convergence.png')
    plt.show()

    # ==========================================
    # 5. GENERATE CALIBRATION CURVES
    # ==========================================
    print("\nGenerating Calibration Curves for Best Models...")
    models_to_plot = [
        {
            'path': baseline_save_path,
            'class': Baseline_Resnet18_H10k,
            'name': 'ResNet-18 Baseline (Best)'
        },
        {
            'path': sd_save_path,
            'class': SelfDistillationResNet18_H10k,
            'name': 'ResNet-18 Self-Distilled (Best)'
        }
    ]
    plot_calibration_curves(models_to_plot)

if __name__ == "__main__":
    main()