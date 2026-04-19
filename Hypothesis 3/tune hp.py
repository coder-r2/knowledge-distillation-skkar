import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState

from datasets import get_chestxray_dataloaders, get_medmnist_dataloaders
from models import SelfDistillationResNet18_H10k, SelfDistillationResNet50
from losses import SelfDistillationLoss

from train_sd import train_sd_epoch, evaluate_sd, ECEMetric

EPOCHS_PER_TRIAL = 30
N_TRIALS = 20
BATCH_SIZE = 64

def create_objective(train_loader, val_loader, in_channels, num_classes, device, model_class):
    """Factory function to pass datasets and model types into the Optuna objective."""
    
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.1, 0.9)
        lambda_weight = trial.suggest_float("lambda_weight", 1e-4, 1e-1, log=True)
        temperature = trial.suggest_float("temperature", 1.0, 5.0)

        model = model_class(num_classes=num_classes, in_channels=in_channels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        criterion = SelfDistillationLoss(alpha=alpha, lambda_weight=lambda_weight, temperature=temperature).to(device)
        val_criterion = nn.CrossEntropyLoss()
        ece_metric = ECEMetric(n_bins=15)

        best_ece = float('inf')

        for epoch in range(1, EPOCHS_PER_TRIAL + 1):
            train_sd_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS_PER_TRIAL)
            
            val_results = evaluate_sd(model, val_loader, device, ece_metric, criterion=val_criterion, num_exits=4)
            
            deepest_val_ece = val_results[-1]['ece']
            
            if deepest_val_ece < best_ece:
                best_ece = deepest_val_ece

            trial.report(best_ece, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_ece

    return objective

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("\n" + "="*50)
    print("   Phase 1: Pre-Training Hyperparameter Search   ")
    print("="*50)

    # === 1. CHEST X-RAY SECTION (ResNet-18) ===
    print("\nLoading Chest X-Ray dataset for tuning...")
    data_dir_chest = 'Hypothesis 3/data/chest_xray/chest_xray'
    train_loader, val_loader, _, in_channels, num_classes = get_chestxray_dataloaders(data_dir_chest, BATCH_SIZE)
    
    objective = create_objective(
        train_loader=train_loader, 
        val_loader=val_loader, 
        in_channels=in_channels, 
        num_classes=num_classes, 
        device=device, 
        model_class=SelfDistillationResNet18_H10k
    )
    dataset_name = "Chest X-Ray"

    # # === 2. BLOODMNIST SECTION (ResNet-50) ===
    # print("\nLoading BloodMNIST dataset for tuning...")
    # train_loader, val_loader, _, in_channels, num_classes = get_medmnist_dataloaders('bloodmnist', BATCH_SIZE)
    # 
    # objective = create_objective(
    #     train_loader=train_loader, 
    #     val_loader=val_loader, 
    #     in_channels=in_channels, 
    #     num_classes=num_classes, 
    #     device=device, 
    #     model_class=SelfDistillationResNet50
    # )
    # dataset_name = "BloodMNIST"

    print(f"\nStarting Optuna Study for {dataset_name} ({N_TRIALS} trials, {EPOCHS_PER_TRIAL} epochs/trial)...")
    
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(objective, n_trials=N_TRIALS)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n" + "="*50)
    print("                 TUNING COMPLETE                  ")
    print("="*50)
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials:   {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    trial = study.best_trial
    print("\n[BEST HYPERPARAMETERS FOUND]")
    print(f"  Lowest Validation ECE: {trial.value:.2f}%")
    for key, value in trial.params.items():
        if key == "lambda_weight":
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value:.4f}")
            
    print("\n=> ACTION REQUIRED:")
    print("   Update these values in the criterion initialization inside 'train_sd.py'")
    print("   before running your final 200-epoch convergence training!")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()