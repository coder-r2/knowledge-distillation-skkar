
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.resnet import resnet18
from torch.utils.data import DataLoader

# Assuming datasets.py and models.py are in the 'Hypothesis 3' directory
# and we can import from them.
from datasets import get_medmnist_dataloaders
from models import SelfDistillationResNet18, BaselineResNet18

def plot_calibration_curves(models_to_plot, num_bins=15):
    """
    Plots calibration curves for a list of models side-by-side on the BloodMNIST test set.

    Args:
        models_to_plot (list of dicts): A list of dictionaries, where each dict contains:
                                          'path': Path to the saved model state dictionary.
                                          'class': The model class to instantiate.
                                          'name': A name for the plot title.
        num_bins (int): The number of bins to use for the calibration curve.
    """
    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    _, _, test_loader, n_channels, n_classes = get_medmnist_dataloaders('bloodmnist', batch_size=128)
    print("Test data loaded.")

    num_models = len(models_to_plot)
    fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 6), sharey=True)
    if num_models == 1:
        axes = [axes] # Make it iterable

    for i, model_info in enumerate(models_to_plot):
        model_path = model_info['path']
        model_class = model_info['class']
        model_name = model_info['name']
        ax = axes[i]

        # --- Load Model ---
        if model_class == SelfDistillationResNet18:
            model = model_class(num_classes=n_classes, in_channels=n_channels).to(device)
        else: # BaselineResNet18
            model = model_class(num_classes=n_classes, in_channels=n_channels).to(device)

        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model '{model_name}' loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            try:
                state_dict = torch.load(model_path, map_location=device)
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                print(f"Model '{model_name}' loaded successfully after removing 'module.' prefix.")
            except Exception as e_inner:
                print(f"Failed to load model '{model_name}' even after attempting to fix DataParallel wrapper: {e_inner}")
                continue # Skip to the next model

        model.eval()

        # --- Get Predictions and Confidences ---
        all_confidences = []
        all_corrects = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.squeeze().to(device)

                outputs = model(images)
                
                if isinstance(outputs, tuple) and len(outputs) == 2: # Self-distillation model
                    logits, _ = outputs
                    final_logits = logits[-1]
                elif isinstance(outputs, torch.Tensor): # Baseline model
                    final_logits = outputs
                else:
                    raise TypeError(f"Unexpected model output type for '{model_name}': {type(outputs)}")

                probabilities = torch.softmax(final_logits, dim=1)
                confidences, predictions = torch.max(probabilities, 1)
                
                corrects = (predictions == labels).cpu().numpy()

                all_confidences.extend(confidences.cpu().numpy())
                all_corrects.extend(corrects)

        all_confidences = np.array(all_confidences)
        all_corrects = np.array(all_corrects)
        
        print(f"Total predictions for '{model_name}': {len(all_confidences)}")

        # --- Binning ---
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_fraction_of_positives = np.zeros(num_bins)
        bin_avg_confidence = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)

        for j in range(num_bins):
            in_bin = (all_confidences > bin_lowers[j]) & (all_confidences <= bin_uppers[j])
            bin_counts[j] = np.sum(in_bin)

            if bin_counts[j] > 0:
                bin_fraction_of_positives[j] = np.mean(all_corrects[in_bin])
                bin_avg_confidence[j] = np.mean(all_confidences[in_bin])
        
        # --- ECE Calculation ---
        ece = np.sum((bin_counts / len(all_corrects)) * np.abs(bin_fraction_of_positives - bin_avg_confidence))
        print(f"Expected Calibration Error (ECE) for '{model_name}': {ece:.4f}")


        # --- Plotting ---
        non_empty_bins = bin_counts > 0
        
        # Plot the calibration curve as a line with markers
        # We plot the average confidence in each bin vs. the fraction of positives in that bin
        ax.plot(bin_avg_confidence[non_empty_bins], bin_fraction_of_positives[non_empty_bins], 
                'b-o', label='Calibration Curve')

        # Ideal calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Ideal Calibration')

        ax.set_xlabel('Mean Predicted Probability (Confidence)')
        if i == 0:
            ax.set_ylabel('Fraction of Positives (Accuracy)')
        ax.set_title(f'Calibration Curve for {model_name}\nECE: {ece:.4f}')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # --- Configuration for side-by-side plotting ---
    models_to_plot = [
        {
            'path': '/home/rrg/umc203/project/Hypothesis 3/saved_models/resnet18_bloodmnist_baseline.pth',
            'class': BaselineResNet18,
            'name': 'ResNet-18 Baseline'
        },
        {
            'path': '/home/rrg/umc203/project/Hypothesis 3/saved_models/resnet18_bloodmnist_self_distilled.pth',
            'class': SelfDistillationResNet18,
            'name': 'ResNet-18 Self-Distilled'
        }
    ]

    plot_calibration_curves(models_to_plot)
