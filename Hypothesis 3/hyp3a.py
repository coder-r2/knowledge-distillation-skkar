import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import custom modules
from datasets import get_medmnist_dataloaders
from models import BaselineResNet18, SelfDistillationResNet18, BaselineResNet50, SelfDistillationResNet50
from losses import SelfDistillationLoss
from train import (
    ECEMetric, 
    train_standard_epoch, evaluate_standard, measure_standard_inference,
    train_sd_epoch, evaluate_sd, measure_sd_inference
)

import random
import numpy as np

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

    # --- Hyperparameters ---
    epochs = 150
    learning_rate = 0.001
    batch_size = 128

    # --- Data Loading ---
    print("Loading datasets...")
    train_loader, val_loader, test_loader, in_channels, num_classes = get_medmnist_dataloaders('bloodmnist', batch_size)

    # --- Instantiation ---
    # Global evaluation metric
    ece_metric = ECEMetric(n_bins=15)

    # SD Model
    sd_model = SelfDistillationResNet50(num_classes=num_classes, in_channels=in_channels).to(device)
    sd_criterion = SelfDistillationLoss()
    sd_optimizer = optim.Adam(sd_model.parameters(), lr=learning_rate)

    # Baseline Model
    baseline_model = BaselineResNet50(num_classes=num_classes, in_channels=in_channels).to(device)
    baseline_criterion = nn.CrossEntropyLoss()
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)

    # --- Telemetry Storage ---
    # Stores metrics every 10 epochs for plotting
    sd_history = {
        'epochs': [],
        'train_loss': [],
        'ece': [[] for _ in range(4)], # List for each of the 4 exits
        'acc': [[] for _ in range(4)]
    }
    baseline_history = {
        'epochs': [],
        'train_loss': [],
        'ece': [],
        'acc': []
    }

    # ==========================================
    # 1. TRAIN SELF-DISTILLATION MODEL
    # ==========================================
    print("\n--- Starting SD Model Training ---")
    for epoch in range(1, epochs + 1):
        loss = train_sd_epoch(sd_model, train_loader, sd_optimizer, sd_criterion, device, epoch, epochs)

        sd_history['epochs'].append(epoch)
        sd_history['train_loss'].append(loss)
        
        # Evaluate using the validation loader
        val_results = evaluate_sd(sd_model, val_loader, device, ece_metric, num_exits=4)
        for i, res in enumerate(val_results):
            sd_history['ece'][i].append(res['ece'])
            sd_history['acc'][i].append(res['acc'])

        if epoch % 10 == 0:    
            print(f"Epoch [{epoch}/{epochs}] SD Loss: {loss:.4f} | Exit 1 Val Acc: {val_results[0]['acc']:.2f}%")

    print("\nSD Training Complete. Saving weights...")
    torch.save(sd_model.state_dict(), 'Hypothesis 3/saved_models/resnet50_bloodmnist_self_distilled.pth')

    print("Running Final Evaluation on TEST Set...")
    sd_final_results = evaluate_sd(sd_model, test_loader, device, ece_metric, num_exits=4)
    print("\n--- SD Final Test Results ---")
    for res in sd_final_results:
        print(f"Exit {res['exit']} - Test Acc: {res['acc']:.2f}% | Test ECE: {res['ece']:.2f}%")

    print("\nMeasuring SD Inference Throughput...")
    dummy_batch = torch.randn(128, in_channels, 32, 32).to(device)
    sd_times = []
    for exit_idx in range(4):
        time_ms = measure_sd_inference(sd_model, dummy_batch, exit_idx)
        sd_times.append(time_ms)
        print(f"Exit {exit_idx+1} Inference Time: {time_ms:.2f} ms")


    # ==========================================
    # 2. TRAIN BASELINE MODEL
    # ==========================================
    print("\n--- Starting Baseline ResNet-50 Training ---")
    for epoch in range(1, epochs + 1):
        loss = train_standard_epoch(baseline_model, train_loader, baseline_optimizer, baseline_criterion, device, epoch, epochs)

        baseline_history['epochs'].append(epoch)
        baseline_history['train_loss'].append(loss)
        
        # Evaluate using the validation loader
        val_results = evaluate_standard(baseline_model, val_loader, device, ece_metric)
        baseline_history['ece'].append(val_results['ece'])
        baseline_history['acc'].append(val_results['acc'])

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] Baseline Loss: {loss:.4f} | Val Acc: {val_results['acc']:.2f}%")

    print("\nBaseline Training Complete. Saving weights...")
    torch.save(baseline_model.state_dict(), 'Hypothesis 3/saved_models/resnet50_bloodmnist_baseline.pth')

    print("Evaluating Baseline on Test Set...")
    baseline_results = evaluate_standard(baseline_model, test_loader, device, ece_metric)
    baseline_time = measure_standard_inference(baseline_model, dummy_batch)

    print(f"Baseline - Acc: {baseline_results['acc']:.2f}% | ECE: {baseline_results['ece']:.2f}% | Time: {baseline_time:.2f} ms")

    # ==========================================
    # 3. PLOT CONVERGENCE GRAPHS
    # ==========================================
    plt.figure(figsize=(15, 6))
    plt.suptitle('Performance Comparison: Self-Distillation vs. Standard ResNet-50 (BloodMNIST)', fontsize=16)

    # Graph 1: Training Loss vs Epoch
    plt.subplot(1, 2, 1)
    plt.plot(sd_history['epochs'], sd_history['train_loss'], label='SD Total Loss', color='blue', linewidth=2)
    plt.plot(baseline_history['epochs'], baseline_history['train_loss'], label='Baseline Loss', color='red', linestyle='--')
    plt.title('Training Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Graph 2: ECE vs Epoch
    plt.subplot(1, 2, 2)
    colors = ['#c6e2ff', '#7eb6ff', '#2171ed', '#00008b'] # Light to dark blue for exits
    for i in range(4):
        plt.plot(sd_history['epochs'], sd_history['ece'][i], label=f'SD Exit {i+1}', color=colors[i])
    
    plt.plot(baseline_history['epochs'], baseline_history['ece'], label='Baseline', color='red', linewidth=2, linestyle='--')
    plt.title('Validation ECE vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('ECE (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Hypothesis 3/results/resnet50_bloodmnist.png')
    plt.show()

if __name__ == "__main__":
    main()