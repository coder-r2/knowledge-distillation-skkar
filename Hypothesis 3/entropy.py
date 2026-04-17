import os
import torch
import torch.nn.functional as F
import numpy as np

# Import your existing custom modules
from datasets import get_chestxray_dataloaders, get_medmnist_dataloaders
from models import Baseline_Resnet18_H10k, SelfDistillationResNet18_H10k, BaselineResNet50, SelfDistillationResNet50

def calculate_mean_entropy(logits):
    """
    Calculates the mean Shannon entropy of a batch of logits.
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)
    
    # Calculate entropy: -sum(p * log(p)). 
    # We add a tiny epsilon (1e-9) to prevent log(0) which results in NaN.
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
    
    # Return the average entropy across the entire batch
    return entropy.mean().item()

def evaluate_entropy(models_to_evaluate, test_loader, device):
    print("\n=======================================================")
    print("             Predictive Entropy Evaluation             ")
    print("=======================================================")

    for model_info in models_to_evaluate:
        model_path = model_info['path']
        model_class = model_info['class']
        model_name = model_info['name']
        is_sd = model_info['is_sd']
        n_classes = model_info['n_classes']
        n_channels = model_info['n_channels']

        # Instantiate and load model
        model = model_class(num_classes=n_classes, in_channels=n_channels).to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue
            
        model.eval()
        
        # Track total entropy
        if is_sd:
            total_entropy = {f'Exit {i+1}': 0.0 for i in range(4)}
        else:
            total_entropy = {'Baseline': 0.0}
            
        total_batches = 0

        with torch.no_grad():
            # We can ignore labels (_) since we are calculating overall entropy
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                
                if is_sd:
                    logits_list, _ = outputs
                    for i, logits in enumerate(logits_list):
                        batch_entropy = calculate_mean_entropy(logits)
                        total_entropy[f'Exit {i+1}'] += batch_entropy
                else:
                    batch_entropy = calculate_mean_entropy(outputs)
                    total_entropy['Baseline'] += batch_entropy
                    
                total_batches += 1

        # Calculate and print the averages
        print(f"\n--- {model_name} ---")
        for exit_name, ent_sum in total_entropy.items():
            avg_entropy = ent_sum / total_batches
            print(f"Mean {exit_name} Entropy : {avg_entropy:.4f} nats")

    print("\n=======================================================\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = 'Hypothesis 3/saved_models'

    # =========================================================================
    # COMMENTED BLOCK: ResNet-50 on BloodMNIST
    # =========================================================================
    print("Loading BloodMNIST test dataloader...")
    _, _, test_loader, in_channels, num_classes = get_medmnist_dataloaders(data_flag='bloodmnist', batch_size=128)

    models_to_evaluate = [
        {
            'path': os.path.join(save_dir, 'resnet50_bloodmnist_baseline_best.pth'),
            'class': BaselineResNet50,
            'name': 'ResNet-50 Supervised Baseline',
            'is_sd': False,
            'n_classes': num_classes,
            'n_channels': in_channels
        },
        {
            'path': os.path.join(save_dir, 'resnet50_bloodmnist_self_distilled_best.pth'),
            'class': SelfDistillationResNet50,
            'name': 'ResNet-50 Self-Distilled Model',
            'is_sd': True,
            'n_classes': num_classes,
            'n_channels': in_channels
        }
    ]

    # =========================================================================
    # ACTIVE BLOCK: ResNet-18 on Chest X-Ray
    # =========================================================================
    # print("Loading Chest X-Ray test dataloader...")
    # data_dir = 'Hypothesis 3/data/chest_xray/chest_xray'
    # _, _, test_loader, in_channels, num_classes = get_chestxray_dataloaders(data_dir, batch_size=64)
    
    # models_to_evaluate = [
    #     {
    #         'path': os.path.join(save_dir, 'resnet18_chestxray_baseline_best.pth'),
    #         'class': Baseline_Resnet18_H10k,
    #         'name': 'Supervised Baseline (Chest X-Ray)',
    #         'is_sd': False,
    #         'n_classes': num_classes,
    #         'n_channels': in_channels
    #     },
    #     {
    #         'path': os.path.join(save_dir, 'resnet18_chestxray_self_distilled_best.pth'),
    #         'class': SelfDistillationResNet18_H10k,
    #         'name': 'Self-Distilled Model (Chest X-Ray)',
    #         'is_sd': True,
    #         'n_classes': num_classes,
    #         'n_channels': in_channels
    #     }
    # ]
    # =========================================================================

    evaluate_entropy(models_to_evaluate, test_loader, device)

if __name__ == "__main__":
    main()