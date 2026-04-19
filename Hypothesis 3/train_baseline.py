import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm

from datasets import get_chestxray_dataloaders, get_medmnist_dataloaders
from models import Baseline_Resnet18_H10k, BaselineResNet50

class ECEMetric:
    """Calculates Expected Calibration Error using 15 bins."""
    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    def compute(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_boundaries[:-1], self.bin_boundaries[1:]):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item() * 100

def train_standard_epoch(model, dataloader, optimizer, criterion, device, current_epoch, total_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Standard Epoch [{current_epoch}/{total_epochs}]", leave=False)

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device).squeeze().long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(dataloader)

def evaluate_standard(model, dataloader, device, ece_metric, criterion=None):
    model.eval()
    all_targets = []
    all_logits = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device).squeeze().long()
            logits = model(inputs)

            if criterion is not None:
                loss = criterion(logits, targets)
                total_loss += loss.item()

            all_targets.append(targets)
            all_logits.append(logits)

    all_targets = torch.cat(all_targets)
    all_logits = torch.cat(all_logits)

    _, preds = torch.max(all_logits, 1)
    acc = (preds == all_targets).float().mean().item() * 100
    ece = ece_metric.compute(all_logits, all_targets)
    
    result = {"acc": acc, "ece": ece}
    if criterion is not None:
        result["loss"] = total_loss / len(dataloader)

    return result

def measure_standard_inference(model, dummy_input, num_runs=100):
    model.eval()
    for _ in range(10): # Warmup
        _ = model(dummy_input)

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / num_runs
    else:
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        end_time = time.perf_counter()
        return ((end_time - start_time) / num_runs) * 1000

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = 'Hypothesis 3/saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    num_epochs = 200
    patience = 30
    batch_size = 128
    learning_rate = 0.001
    ece_metric = ECEMetric(n_bins=15)

    # === 1. CHEST X-RAY SECTION (ResNet-18) ===
    print("\n--- Initializing Baseline ResNet-18 on Chest X-Ray ---")
    data_dir = 'Hypothesis 3/data/chest_xray/chest_xray'
    train_loader, val_loader, test_loader, in_channels, num_classes = get_chestxray_dataloaders(data_dir, batch_size=batch_size)
    model = Baseline_Resnet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)
    save_path = os.path.join(save_dir, 'resnet18_chestxray_baseline_best.pth')

    # # === 2. BLOODMNIST SECTION (ResNet-50) ===
    # print("\n--- Initializing Baseline ResNet-50 on BloodMNIST ---")
    # train_loader, val_loader, test_loader, in_channels, num_classes = get_medmnist_dataloaders('bloodmnist', batch_size=batch_size)
    # model = BaselineResNet50(num_classes=num_classes, in_channels=in_channels).to(device)
    # save_path = os.path.join(save_dir, 'resnet50_bloodmnist_baseline_best.pth')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("\nStarting Training (Early Stopping Patience: 30)...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_standard_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        val_results = evaluate_standard(model, val_loader, device, ece_metric, criterion)
        
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_results['loss']:.4f} | Val Acc: {val_results['acc']:.2f}%")
        
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved to {save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"\n[!] Early stopping triggered. Validation loss has not improved for {patience} epochs.")
            break

    print("\n========================================")
    print("        Post-Training Evaluation        ")
    print("========================================")
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    # 1. Test Set Metrics
    print("Evaluating Test Set...")
    test_results = evaluate_standard(model, test_loader, device, ece_metric)
    print(f"  Test Accuracy : {test_results['acc']:.2f}%")
    print(f"  Test ECE      : {test_results['ece']:.2f}%")
    
    # 2. Inference Speed
    print("\nMeasuring Inference Latency...")
    dummy_input = torch.randn(1, in_channels, 32, 32).to(device)
    latency = measure_standard_inference(model, dummy_input, num_runs=200)
    print(f"  Avg Inference Time : {latency:.2f} ms / image")
    print("========================================\n")

if __name__ == "__main__":
    main()