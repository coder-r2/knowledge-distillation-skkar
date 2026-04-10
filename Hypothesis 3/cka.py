import os
import torch
import numpy as np
from tqdm.auto import tqdm

from datasets import get_chestxray_dataloaders
from models import SelfDistillationResNet18_H10k

def extract_features(model, dataloader, device):
    """
    Extracts the latent features from all 4 exits of the self-distilled model.
    """
    model.eval()
    intercepted_features = {}
    
    def get_features(name):
        def hook(model, input, output):
            intercepted_features[name] = input[0].detach().cpu().numpy()
        return hook

    # Register hooks on the classifiers of the SD model
    hooks = [
        model.classifier1.register_forward_hook(get_features('exit1')),
        model.classifier2.register_forward_hook(get_features('exit2')),
        model.classifier3.register_forward_hook(get_features('exit3')),
        model.classifier4.register_forward_hook(get_features('exit4'))
    ]

    all_features = {'exit1': [], 'exit2': [], 'exit3': [], 'exit4': []}

    print("Extracting features (Test Set)...")
    with torch.no_grad():
        for images, _ in tqdm(dataloader, leave=False):
            images = images.to(device)
            _ = model(images) 
            
            for key in all_features.keys():
                all_features[key].append(intercepted_features[key])

    for h in hooks:
        h.remove()

    for key in all_features.keys():
        # Concatenate and flatten to 2D (Batch_size, Features)
        all_features[key] = np.concatenate(all_features[key], axis=0)
        all_features[key] = all_features[key].reshape(all_features[key].shape[0], -1)
        
    return all_features

def linear_cka(X, Y):
    """
    Computes Linear Centered Kernel Alignment (CKA) between two feature matrices.
    """
    # 1. Mean-center the columns of both feature spaces
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    # 2. Compute the dot product between the two spaces
    dot_prod = np.dot(X_centered.T, Y_centered)
    
    # 3. Compute Frobenius norms
    numerator = np.linalg.norm(dot_prod, ord='fro') ** 2
    den1 = np.linalg.norm(np.dot(X_centered.T, X_centered), ord='fro')
    den2 = np.linalg.norm(np.dot(Y_centered.T, Y_centered), ord='fro')
    
    # Return normalized alignment score
    return numerator / (den1 * den2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Setup Data and Model ---
    batch_size = 128
    print("Loading ChestXRay dataloaders...")
    _, _, test_loader, in_channels, num_classes = get_chestxray_dataloaders(data_dir='Hypothesis 3/data/chest_xray/chest_xray', batch_size=batch_size)

    save_dir = 'Hypothesis 3/saved_models'
    sd_path = os.path.join(save_dir, 'resnet18_chestxray_self_distilled_best.pth')

    sd_model = SelfDistillationResNet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)

    print("\nLoading trained weights...")
    sd_model.load_state_dict(torch.load(sd_path, map_location=device))

    # --- 2. Extract Features ---
    test_features = extract_features(sd_model, test_loader, device)

    # --- 3. Compute CKA against the Teacher (Exit 4) ---
    print("\n========================================================")
    print("      Representation Similarity (Linear CKA) vs Exit 4  ")
    print("========================================================")
    
    teacher_features = test_features['exit4']
    
    for i in range(1, 4):
        student_features = test_features[f'exit{i}']
        cka_score = linear_cka(student_features, teacher_features)
        print(f"  Exit {i} vs Exit 4 (Teacher) : {cka_score:.4f}")
        
    print("========================================================\n")

if __name__ == "__main__":
    main()