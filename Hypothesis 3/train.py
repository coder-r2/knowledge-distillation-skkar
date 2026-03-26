import time
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

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

        return ece.item() * 100 # Return as percentage

# ==========================================
# STANDARD MODEL FUNCTIONS
# ==========================================

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

# ==========================================
# SELF-DISTILLATION (SD) FUNCTIONS
# ==========================================

def train_sd_epoch(model, dataloader, optimizer, criterion, device, current_epoch, total_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"SD Epoch [{current_epoch}/{total_epochs}]", leave=False)

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

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
            logits_list, _ = model(inputs)

            if criterion is not None:
                # Calculate validation loss strictly for the deepest exit
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
        # Attach the calculated loss only to the deepest exit's results
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