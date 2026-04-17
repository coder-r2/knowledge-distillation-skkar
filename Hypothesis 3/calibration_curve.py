import os
import torch
import numpy as np
import math
import matplotlib.pyplot as plt

# Adjust imports based on your dataset (HAM10000 or Chest X-Ray)
from datasets import get_medmnist_dataloaders, get_chestxray_dataloaders
from models import Baseline_Resnet18_H10k, SelfDistillationResNet18_H10k, BaselineResNet50, SelfDistillationResNet50

def plot_model_comparison_classwise(models_to_plot, data_dir, get_dataloader_fn, num_bins=50):
    """
    Plots class-wise (One-vs-All) calibration curves, overlaying different models 
    on the same plot for direct comparison.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading dataloader from {data_dir}...")
    _, _, test_loader, n_channels, n_classes = get_dataloader_fn(data_dir, batch_size=128)
    print(f"Test data loaded. Detected {n_classes} classes.")

    # --- 1. Gather Predictions for All Models ---
    model_results = []

    for model_info in models_to_plot:
        model_path = model_info['path']
        model_class = model_info['class']
        model_name = model_info['name']

        print(f"\nEvaluating {model_name}...")
        model = model_class(num_classes=n_classes, in_channels=n_channels).to(device)

        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            try:
                # Handle DataParallel wrapper
                state_dict = torch.load(model_path, map_location=device)
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            except Exception as e_inner:
                print(f"Failed to load {model_name}: {e_inner}")
                continue 

        model.eval()
        all_probabilities = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.squeeze().to(device)
                outputs = model(images)
                
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    logits, _ = outputs
                    final_logits = logits[-1] # Deepest exit
                elif isinstance(outputs, torch.Tensor):
                    final_logits = outputs
                else:
                    raise TypeError(f"Unexpected output type: {type(outputs)}")

                probabilities = torch.softmax(final_logits, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        model_results.append({
            'name': model_name,
            'probs': np.array(all_probabilities), # Shape: (N, C)
            'labels': np.array(all_labels)        # Shape: (N,)
        })
        print(f"Finished evaluation for {model_name}.")

    # --- 2. Dynamic Subplot Grid ---
    cols = min(n_classes, 3)
    rows = math.ceil(n_classes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), sharey=True, sharex=True)
    fig.suptitle('Baseline vs. Self-Distilled: Class-wise Calibration', fontsize=18, y=1.02)
    
    # Handle the case where n_classes is 1 (axes isn't an array)
    if n_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Standard distinct colors

    # --- 3. Compute and Plot Overlaid Curves ---
    for c in range(n_classes):
        ax = axes[c]
        
        for idx, res in enumerate(model_results):
            class_probs = res['probs'][:, c]
            class_targets = (res['labels'] == c).astype(int)

            bin_fraction_of_positives = np.zeros(num_bins)
            bin_avg_confidence = np.zeros(num_bins)
            bin_counts = np.zeros(num_bins)

            for j in range(num_bins):
                in_bin = (class_probs > bin_lowers[j]) & (class_probs <= bin_uppers[j])
                bin_counts[j] = np.sum(in_bin)

                if bin_counts[j] > 0:
                    bin_fraction_of_positives[j] = np.mean(class_targets[in_bin])
                    bin_avg_confidence[j] = np.mean(class_probs[in_bin])
            
            # ECE Calculation
            ece = np.sum((bin_counts / len(class_targets)) * np.abs(bin_fraction_of_positives - bin_avg_confidence))
            
            # Plot Model Curve
            non_empty_bins = bin_counts > 0
            ax.plot(bin_avg_confidence[non_empty_bins], bin_fraction_of_positives[non_empty_bins], 
                    marker='o', linestyle='', color=colors[idx % len(colors)], 
                    label=f'{res["name"]} (ECE: {ece:.4f})')

        # Add Ideal Line
        ax.plot([0, 1], [0, 1], 'k--', label='Ideal Calibration', alpha=0.7)

        # Subplot Formatting
        ax.set_title(f'Class {c}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.6)

        if c % cols == 0:
            ax.set_ylabel('Fraction of Positives')
        if c >= (rows - 1) * cols:
            ax.set_xlabel('Mean Predicted Probability')
        
        ax.legend(fontsize=9)

    # Hide unused subplots
    for extra_ax in axes[n_classes:]:
        extra_ax.set_visible(False)

    fig.tight_layout()
    os.makedirs('Hypothesis 3/results', exist_ok=True)
    save_path = 'Hypothesis 3/results/model_comparison_classwise_calibration.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved comparison figure to {save_path}")
    plt.show()

# ==========================================
# STANDALONE EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print("\n--- Starting Head-to-Head Calibration Evaluation ---")
    
    save_dir = 'Hypothesis 3/saved_models'
    data_directory = 'Hypothesis 3/data/chest_xray' # Adjust if using HAM10000
    dataloader_function = get_chestxray_dataloaders
    
    models_to_evaluate = [
        {
            'path': os.path.join(save_dir, 'resnet18_chestxray_baseline_best.pth'),
            'class': Baseline_Resnet18_H10k,

            'name': 'Baseline'
        },
        {
            'path': os.path.join(save_dir, 'resnet18_chestxray_self_distilled_best.pth'),
            'class': SelfDistillationResNet18_H10k,
            'name': 'Self-Distilled'
        }
    ]
    
    plot_model_comparison_classwise(
        models_to_plot=models_to_evaluate, 
        data_dir=data_directory, 
        get_dataloader_fn=dataloader_function
    )