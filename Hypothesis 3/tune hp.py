import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState

# Import your existing custom modules
from datasets import get_chestxray_dataloaders
from models import SelfDistillationResNet18_H10k
from losses import SelfDistillationLoss
from train import train_sd_epoch, evaluate_sd, ECEMetric

# --- Configuration ---
DATA_DIR = 'Hypothesis 3/data/chest_xray/chest_xray'
BATCH_SIZE = 64 # Lowered to 64 for VRAM with X-Rays
EPOCHS_PER_TRIAL = 30 # Keep this lower for tuning (e.g., 30-50 epochs)
N_TRIALS = 20         # How many different combinations Optuna should try

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data once (outside the loop to save time)
print("Loading datasets for tuning...")
train_loader, val_loader, _, in_channels, num_classes = get_chestxray_dataloaders(DATA_DIR, BATCH_SIZE)
ece_metric = ECEMetric(n_bins=15)

def objective(trial):
    """The function Optuna will run multiple times with different parameters."""
    
    # 1. Let Optuna suggest hyperparameters
    # Alpha usually ranges from 0.1 to 0.9
    alpha = trial.suggest_float("alpha", 0.3, 0.7)
    # Lambda is usually very small, so we use a log scale (e.g., 0.0001 to 0.1)
    lambda_weight = trial.suggest_float("lambda_weight", 1e-4, 1e-1, log=True)
    # Temperature usually ranges from 1.0 to 5.0 or 10.0
    temperature = trial.suggest_float("temperature", 1.0, 5.0)

    # 2. Instantiate Model and Optimizer
    model = SelfDistillationResNet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Instantiate Loss with Optuna's suggested parameters
    criterion = SelfDistillationLoss(alpha=alpha, lambda_weight=lambda_weight, temperature=temperature).to(device)

    best_ece = float('inf')

    # 4. Short Training Loop
    for epoch in range(1, EPOCHS_PER_TRIAL + 1):
        # Train
        train_sd_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS_PER_TRIAL)
        
        # Evaluate
        val_results = evaluate_sd(model, val_loader, device, ece_metric, criterion=nn.CrossEntropyLoss(), num_exits=4)
        
        # We care about the Teacher's (Exit 4) calibration for tuning purposes
        deepest_val_ece = val_results[3]['ece']
        
        # Track the best ECE achieved in this trial
        if deepest_val_ece < best_ece:
            best_ece = deepest_val_ece

        # 5. Report progress to Optuna
        trial.report(deepest_val_ece, epoch)

        # 6. Early Pruning
        # If this trial is performing terribly compared to previous trials, Optuna stops it early!
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_ece

def main():
    print("Starting Hyperparameter Optimization...")
    
    # Create an Optuna study that tries to MINIMIZE the returned value (ECE)
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(objective, n_trials=N_TRIALS)

    # --- Print Results ---
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\nStudy statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("\nBest trial:")
    trial = study.best_trial

    print(f"  Lowest Validation ECE: {trial.value:.2f}%")
    print("  Best Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()