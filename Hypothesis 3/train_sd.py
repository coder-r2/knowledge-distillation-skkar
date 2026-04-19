import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm

from datasets import get_chestxray_dataloaders, get_medmnist_dataloaders
from models import SelfDistillationResNet18_H10k, SelfDistillationResNet50
from losses import SelfDistillationLoss

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

def train_sd_epoch(model, dataloader, optimizer, criterion, device, current_epoch, total_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"SD Epoch [{current_epoch}/{total_epochs}]", leave=False)

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device).squeeze().long()

        optimizer.zero_grad()
        logits_list, bottlenecks_list = model(inputs)
        loss = criterion(logits_list, bottlenecks_list, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return running_loss / len(dataloader)

def evaluate_sd(model, dataloader, device, ece_metric, criterion=None, num_exits=4):
    model.eval()
    all_targets = []
    all_logits = [[] for _ in range(num_exits)]
    total_deepest_loss = 0.0

    pbar = tqdm(dataloader, desc="Evaluating SD", leave=False)

    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device).squeeze().long()
            logits_list = model(inputs)
            
            if isinstance(logits_list, tuple):
                logits_list = logits_list[0]

            if criterion is not None:
                loss = criterion(logits_list[-1], targets)
                total_deepest_loss += loss.item()

            all_targets.append(targets)
            for i in range(num_exits):
                all_logits[i].append(logits_list[i])

    all_targets = torch.cat(all_targets)
    results = []

    for i in range(num_exits):
        cat_logits = torch.cat(all_logits[i])
        _, preds = torch.max(cat_logits, 1)
        acc = (preds == all_targets).float().mean().item() * 100
        ece = ece_metric.compute(cat_logits, all_targets)
        
        res_dict = {"exit": i+1, "acc": acc, "ece": ece}
        if i == num_exits - 1 and criterion is not None:
            res_dict["loss"] = total_deepest_loss / len(dataloader)
            
        results.append(res_dict)

    return results

def measure_sd_inference(model, dummy_input, exit_idx, num_runs=100):
    model.eval()
    for _ in range(10): # Warmup
        _ = model.early_exit_forward(dummy_input, exit_idx)

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model.early_exit_forward(dummy_input, exit_idx)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / num_runs
    else:
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model.early_exit_forward(dummy_input, exit_idx)
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
    num_exits = 4

    # === 1. CHEST X-RAY SECTION (ResNet-18) ===
    print("\n--- Initializing SD ResNet-18 on Chest X-Ray ---")
    data_dir = 'Hypothesis 3/data/chest_xray/chest_xray'
    train_loader, val_loader, test_loader, in_channels, num_classes = get_chestxray_dataloaders(data_dir, batch_size=batch_size)
    model = SelfDistillationResNet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)
    save_path = os.path.join(save_dir, 'resnet18_chestxray_self_distilled_best.pth')
    criterion = SelfDistillationLoss(alpha=0.357, lambda_weight=0.005, temperature=3.013)

    # # === 2. BLOODMNIST SECTION (ResNet-50) ===
    # print("\n--- Initializing SD ResNet-50 on BloodMNIST ---")
    # train_loader, val_loader, test_loader, in_channels, num_classes = get_medmnist_dataloaders('bloodmnist', batch_size=batch_size)
    # model = SelfDistillationResNet50(num_classes=num_classes, in_channels=in_channels).to(device)
    # save_path = os.path.join(save_dir, 'resnet50_bloodmnist_self_distilled_best.pth')
    # criterion = SelfDistillationLoss()

    val_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("\nStarting Training (Early Stopping Patience: 30)...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_sd_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        val_results = evaluate_sd(model, val_loader, device, ece_metric, criterion=val_criterion, num_exits=num_exits)
        
        deepest_val_loss = val_results[-1]['loss']
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Deepest Val Loss: {deepest_val_loss:.4f} | Teacher Acc: {val_results[-1]['acc']:.2f}%")
        
        if deepest_val_loss < best_val_loss:
            best_val_loss = deepest_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved to {save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"\n[!] Early stopping triggered. Deepest Validation loss has not improved for {patience} epochs.")
            break

    print("\n========================================")
    print("        Post-Training Evaluation        ")
    print("========================================")
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    # 1. Test Set Metrics
    print("Evaluating Test Set (All Exits)...")
    test_results = evaluate_sd(model, test_loader, device, ece_metric, num_exits=num_exits)
    
    dummy_input = torch.randn(1, in_channels, 32, 32).to(device)
    
    for i, res in enumerate(test_results):
        print(f"\n[Exit {res['exit']}]")
        print(f"  Test Accuracy : {res['acc']:.2f}%")
        print(f"  Test ECE      : {res['ece']:.2f}%")
        
        latency = measure_sd_inference(model, dummy_input, exit_idx=i, num_runs=200)
        print(f"  Inference Time: {latency:.2f} ms / image")
        
    print("\n========================================\n")

if __name__ == "__main__":
    main()