import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import custom modules
from datasets import get_medmnist_dataloaders
from models import BaselineResNet18, SelfDistillationResNet18
from losses import SelfDistillationLoss
from train import (
    ECEMetric, 
    train_standard_epoch, evaluate_standard, measure_standard_inference,
    train_sd_epoch, evaluate_sd, measure_sd_inference
)

def main():
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
    print(f"Classes: {num_classes}, Input Channels: {in_channels}")

    # --- Instantiation ---
    # Global evaluation metric
    ece_metric = ECEMetric(n_bins=15)

    # SD Model
    sd_model = SelfDistillationResNet18(num_classes=num_classes, in_channels=in_channels).to(device)
    sd_criterion = SelfDistillationLoss()
    sd_optimizer = optim.Adam(sd_model.parameters(), lr=learning_rate)

    # Baseline Model
    baseline_model = BaselineResNet18(num_classes=num_classes, in_channels=in_channels).to(device)
    baseline_criterion = nn.CrossEntropyLoss()
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)

    # ==========================================
    # 1. TRAIN SELF-DISTILLATION MODEL
    # ==========================================
    print("\n--- Starting SD Model Training ---")
    for epoch in range(1, epochs + 1):
        loss = train_sd_epoch(sd_model, train_loader, sd_optimizer, sd_criterion, device, epoch, epochs)

        if epoch % 10 == 0:
            print(f"\n--- Epoch [{epoch}/{epochs}] SD Periodic Evaluation ---")
            print(f"Training Loss: {loss:.4f}")
            val_results = evaluate_sd(sd_model, val_loader, device, ece_metric, num_exits=4)
            for res in val_results:
                print(f"Exit {res['exit']} - Val Acc: {res['acc']:.2f}% | Val ECE: {res['ece']:.2f}%")
            print("-" * 40)

    print("\nSD Training Complete. Saving weights...")
    torch.save(sd_model.state_dict(), 'resnet18_self_distillation.pth')

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
    print("\n--- Starting Baseline ResNet-18 Training ---")
    for epoch in range(1, epochs + 1):
        loss = train_standard_epoch(baseline_model, train_loader, baseline_optimizer, baseline_criterion, device, epoch, epochs)

    print("\nBaseline Training Complete. Saving weights...")
    torch.save(baseline_model.state_dict(), 'resnet18_baseline.pth')

    print("Evaluating Baseline on Test Set...")
    baseline_results = evaluate_standard(baseline_model, test_loader, device, ece_metric)
    baseline_time = measure_standard_inference(baseline_model, dummy_batch)

    print(f"Baseline - Acc: {baseline_results['acc']:.2f}% | ECE: {baseline_results['ece']:.2f}% | Time: {baseline_time:.2f} ms")

    # ==========================================
    # 3. PLOT RESULTS
    # ==========================================
    sd_accuracies = [res['acc'] for res in sd_final_results]
    sd_eces = [res['ece'] for res in sd_final_results]

    plt.figure(figsize=(12, 5))

    # Accuracy vs Inference Time
    plt.subplot(1, 2, 1)
    plt.plot(sd_times, sd_accuracies, marker='o', linestyle='-', color='b', label='SD Exits (1-4)')
    for i, (x, y) in enumerate(zip(sd_times, sd_accuracies)):
        plt.annotate(f'Exit {i+1}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    plt.scatter(baseline_time, baseline_results['acc'], color='r', s=100, label='Baseline', zorder=5)
    
    plt.title('Accuracy vs. Inference Time')
    plt.xlabel('Inference Time (ms per batch)')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # ECE Plot
    plt.subplot(1, 2, 2)
    labels = ['Exit 1', 'Exit 2', 'Exit 3', 'Exit 4', 'Baseline']
    eces = sd_eces + [baseline_results['ece']]
    colors = ['skyblue', 'skyblue', 'skyblue', 'royalblue', 'tomato']

    bars = plt.bar(labels, eces, color=colors)
    plt.title('Expected Calibration Error (Lower is Better)')
    plt.ylabel('ECE (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('exp1_results.png')
    plt.show()

if __name__ == "__main__":
    main()