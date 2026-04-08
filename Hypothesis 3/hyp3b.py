import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np

# Import custom modules
from datasets import get_chestxray_dataloaders
from models import Baseline_Resnet18_H10k, SelfDistillationResNet18_H10k
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = 'Hypothesis 3/saved_models'
    os.makedirs(save_dir, exist_ok=True)
    # Updated Save Paths
    casd_save_path = os.path.join(save_dir, 'resnet18_chestxray_casd_best.pth')
    baseline_save_path = os.path.join(save_dir, 'resnet18_chestxray_baseline_best.pth')

    epochs = 250
    patience = 30
    learning_rate = 0.001
    batch_size = 64 # Lowered to 64 for VRAM

    print("Loading datasets...")
    data_dir = 'Hypothesis 3/data/chest_xray/chest_xray'
    train_loader, val_loader, test_loader, in_channels, num_classes = get_chestxray_dataloaders(data_dir, batch_size)

    ece_metric = ECEMetric(n_bins=15)

    casd_model = SelfDistillationResNet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)
    
    # Using direct DECE penalty loss
    casd_criterion = CalibrationAwareSelfDistillationLoss(alpha=0.5, lambda_weight=0.01, beta=0.5, temperature=3.0).to(device)
    casd_optimizer = optim.Adam(casd_model.parameters(), lr=learning_rate)

    baseline_model = Baseline_Resnet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)

    casd_history = {
        'epochs': [], 'train_loss': [], 'val_loss': [],
        'ece': [[] for _ in range(4)], 'acc': [[] for _ in range(4)]
    }

    print("\n--- Starting FAST CASD Model Training ---")
    best_casd_val_ece = float('inf') 
    casd_patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_sd_epoch(casd_model, train_loader, casd_optimizer, casd_criterion, device, epoch, epochs)
        
        val_results = evaluate_sd(casd_model, val_loader, device, ece_metric, criterion=nn.CrossEntropyLoss(), num_exits=4)
        
        deepest_val_loss = val_results[3]['loss']
        deepest_val_ece = val_results[3]['ece']

        casd_history['epochs'].append(epoch)
        casd_history['train_loss'].append(train_loss)
        casd_history['val_loss'].append(deepest_val_loss)
        
        for i, res in enumerate(val_results):
            casd_history['ece'][i].append(res['ece'])
            casd_history['acc'][i].append(res['acc'])

        if deepest_val_ece < best_casd_val_ece:
            best_casd_val_ece = deepest_val_ece
            casd_patience_counter = 0
            torch.save(casd_model.state_dict(), casd_save_path)
            is_best = " (New Best ECE!)"
        else:
            casd_patience_counter += 1
            is_best = ""

        if epoch % 5 == 0 or is_best:    
            print(f"Epoch [{epoch}/{epochs}] CASD Train Loss: {train_loss:.4f} | Exit 4 Val ECE: {deepest_val_ece:.2f}%{is_best}")

        if casd_patience_counter >= patience:
            print(f"\nEarly stopping triggered for CASD model at epoch {epoch}")
            break

    print("\n--- Running Final Evaluation on TEST Set ---")
    casd_model.load_state_dict(torch.load(casd_save_path, map_location=device))
    
    try:
        baseline_model.load_state_dict(torch.load(baseline_save_path, map_location=device))
        print("Successfully loaded pre-trained baseline model.")
    except FileNotFoundError:
        print(f"\nERROR: Could not find the baseline model at {baseline_save_path}")
        print("Please ensure you have run hyp3a.py first to generate the baseline weights.")
        return

    print("\n[FAST CASD Model Test Results]")
    casd_final_results = evaluate_sd(casd_model, test_loader, device, ece_metric, num_exits=4)
    for res in casd_final_results:
        print(f"Exit {res['exit']} - Test Acc: {res['acc']:.2f}% | Test ECE: {res['ece']:.2f}%")

    print("\n[Baseline Model Test Results]")
    baseline_results = evaluate_standard(baseline_model, test_loader, device, ece_metric)
    print(f"Baseline - Test Acc: {baseline_results['acc']:.2f}% | Test ECE: {baseline_results['ece']:.2f}%")

    os.makedirs('Hypothesis 3/results', exist_ok=True)
    plt.figure(figsize=(15, 6))
    plt.suptitle('FAST CASD Training Convergence (Chest X-Ray)', fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(casd_history['epochs'], casd_history['train_loss'], label='CASD Train Loss', color='blue', alpha=0.6)
    plt.plot(casd_history['epochs'], casd_history['val_loss'], label='CASD Val Loss (Exit 4)', color='darkblue', linewidth=2)
    plt.title('Training & Validation Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

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
    plt.savefig('Hypothesis 3/results/resnet18_chestxray_fast_casd_convergence.png')
    plt.show()

    print("\nGenerating Calibration Curves for Best Models...")
    models_to_plot = [
        {
            'path': baseline_save_path,
            'class': Baseline_Resnet18_H10k,
            'name': 'ResNet-18 Baseline (Chest X-Ray)'
        },
        {
            'path': casd_save_path,
            'class': SelfDistillationResNet18_H10k,  
            'name': 'ResNet-18 FAST CASD (Chest X-Ray)'
        }
    ]
    plot_calibration_curves(models_to_plot)

if __name__ == "__main__":
    main()