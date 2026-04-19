import os
import torch
import numpy as np
from tqdm.auto import tqdm

from datasets import get_chestxray_dataloaders, get_medmnist_dataloaders
from models import SelfDistillationResNet18_H10k, SelfDistillationResNet50

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
        all_features[key] = np.concatenate(all_features[key], axis=0)
        all_features[key] = all_features[key].reshape(all_features[key].shape[0], -1)
        
    return all_features

def linear_cka(X, Y):
    """
    Computes Linear Centered Kernel Alignment (CKA) between two feature matrices.
    """
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    dot_prod = np.dot(X_centered.T, Y_centered)
    
    numerator = np.linalg.norm(dot_prod, ord='fro') ** 2
    den1 = np.linalg.norm(np.dot(X_centered.T, X_centered), ord='fro')
    den2 = np.linalg.norm(np.dot(Y_centered.T, Y_centered), ord='fro')
    
    return numerator / (den1 * den2)

def compute_cka_for_model(model_name, test_features):
    """
    Helper function to print CKA scores cleanly.
    """
    print("\n========================================================")
    print(f"   Representation Similarity (Linear CKA) vs Exit 4   ")
    print(f"   Model: {model_name}                                ")
    print("========================================================")
    
    teacher_features = test_features['exit4']
    
    for i in range(1, 4):
        student_features = test_features[f'exit{i}']
        cka_score = linear_cka(student_features, teacher_features)
        print(f"  Exit {i} vs Exit 4 (Teacher) : {cka_score:.4f}")
        
    print("========================================================\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = 'Hypothesis 3/saved_models'
    batch_size = 128

    # --- 1. BLOODMNIST SECTION (ResNet-50) ---
    print("\nLoading BloodMNIST dataloaders...")
    _, _, test_loader_blood, in_channels_blood, num_classes_blood = get_medmnist_dataloaders('bloodmnist', batch_size=batch_size)
    sd_path_blood = os.path.join(save_dir, 'resnet50_bloodmnist_self_distilled_best.pth')
    
    sd_model_blood = SelfDistillationResNet50(num_classes=num_classes_blood, in_channels=in_channels_blood).to(device)
    if os.path.exists(sd_path_blood):
        print(f"Loading trained weights from {sd_path_blood}...")
        sd_model_blood.load_state_dict(torch.load(sd_path_blood, map_location=device))
    else:
        print(f"Warning: Weights not found at {sd_path_blood}")

    test_features_blood = extract_features(sd_model_blood, test_loader_blood, device)
    compute_cka_for_model("BloodMNIST (ResNet-50)", test_features_blood)


    # # --- 2. CHEST X-RAY SECTION (ResNet-18) ---
    # print("\nLoading Chest X-Ray dataloaders...")
    # chest_data_dir = 'Hypothesis 3/data/chest_xray/chest_xray'
    # _, _, test_loader_chest, in_channels_chest, num_classes_chest = get_chestxray_dataloaders(chest_data_dir, batch_size=batch_size)
    # sd_path_chest = os.path.join(save_dir, 'resnet18_chestxray_self_distilled_best.pth')
    
    # sd_model_chest = SelfDistillationResNet18_H10k(num_classes=num_classes_chest, in_channels=in_channels_chest).to(device)
    # if os.path.exists(sd_path_chest):
    #     print(f"Loading trained weights from {sd_path_chest}...")
    #     sd_model_chest.load_state_dict(torch.load(sd_path_chest, map_location=device))
    # else:
    #     print(f"Warning: Weights not found at {sd_path_chest}")

    # test_features_chest = extract_features(sd_model_chest, test_loader_chest, device)
    # compute_cka_for_model("Chest X-Ray (ResNet-18)", test_features_chest)


if __name__ == "__main__":
    main()