import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from datasets import get_medmnist_dataloaders, get_chestxray_dataloaders
from models import Baseline_Resnet18_H10k, SelfDistillationResNet18_H10k, BaselineResNet50, SelfDistillationResNet50

def extract_features(model, dataloader, device, is_sd_model=False):
    """Extracts the latent features from the model just before the linear classifiers."""
    model.eval()
    intercepted_features = {}
    
    def get_features(name):
        def hook(model, input, output):
            intercepted_features[name] = input[0].detach().cpu().numpy()
        return hook

    hooks = []
    if is_sd_model:
        hooks.append(model.classifier1.register_forward_hook(get_features('exit1')))
        hooks.append(model.classifier2.register_forward_hook(get_features('exit2')))
        hooks.append(model.classifier3.register_forward_hook(get_features('exit3')))
        hooks.append(model.classifier4.register_forward_hook(get_features('exit4')))
    else:
        hooks.append(model.fc.register_forward_hook(get_features('baseline')))

    all_features = {k: [] for k in (['exit1', 'exit2', 'exit3', 'exit4'] if is_sd_model else ['baseline'])}
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features", leave=False):
            images = images.to(device)
            labels = labels.cpu().numpy().flatten()
            
            _ = model(images) 
            
            for key in all_features.keys():
                all_features[key].append(intercepted_features[key])
            all_labels.extend(labels)

    for h in hooks:
        h.remove()

    for key in all_features.keys():
        all_features[key] = np.concatenate(all_features[key], axis=0)
        
    return all_features, np.array(all_labels)

def compute_centroid_similarities(features, labels, num_classes):
    """Computes Intra-class and Inter-class cosine similarity against class centroids."""
    centroids = []
    
    for c in range(num_classes):
        class_features = features[labels == c]
        if len(class_features) > 0:
            centroid = np.mean(class_features, axis=0)
            centroids.append(centroid)
        else:
            centroids.append(np.zeros(features.shape[1]))

    intra_sims = []
    inter_sims = []

    for i in range(len(features)):
        feat = features[i].reshape(1, -1)
        true_class = int(labels[i])

        for c in range(num_classes):
            if np.all(centroids[c] == 0): continue
            
            sim = cosine_similarity(feat, centroids[c].reshape(1, -1))[0, 0]
            if c == true_class:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    return np.mean(intra_sims), np.mean(inter_sims)

def evaluate_and_print_similarities(dataset_name, test_feat_base, test_feat_sd, test_labels, num_classes):
    """Helper function to print similarities cleanly for any dataset."""
    print("\n" + "="*60)
    print(f"      Latent Space Cosine Similarity Analysis ({dataset_name}) ")
    print("="*60)
    
    # Baseline
    base_intra, base_inter = compute_centroid_similarities(test_feat_base['baseline'], test_labels, num_classes)
    print(f"[Baseline Model]")
    print(f"  Intra-class Similarity (Cohesion)  : {base_intra:.4f}  <-- Higher is better")
    print(f"  Inter-class Similarity (Separation): {base_inter:.4f}  <-- Lower is better\n")

    # SD Exits
    for i in range(1, 5):
        key = f'exit{i}'
        sd_intra, sd_inter = compute_centroid_similarities(test_feat_sd[key], test_labels, num_classes)
        print(f"[Self-Distilled Exit {i}]")
        print(f"  Intra-class Similarity (Cohesion)  : {sd_intra:.4f}")
        print(f"  Inter-class Similarity (Separation): {sd_inter:.4f}\n")
        
    print("="*60 + "\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = 'Hypothesis 3/saved_models'
    batch_size = 128

    # --- 1. BLOODMNIST SECTION (ResNet-50) ---
    print("\nLoading BloodMNIST dataloaders...")
    _, _, test_loader_blood, in_channels_blood, num_classes_blood = get_medmnist_dataloaders('bloodmnist', batch_size=batch_size)
    
    baseline_path_blood = os.path.join(save_dir, 'resnet50_bloodmnist_baseline_best.pth')
    sd_path_blood = os.path.join(save_dir, 'resnet50_bloodmnist_self_distilled_best.pth')
    
    baseline_model_blood = BaselineResNet50(num_classes=num_classes_blood, in_channels=in_channels_blood).to(device)
    sd_model_blood = SelfDistillationResNet50(num_classes=num_classes_blood, in_channels=in_channels_blood).to(device)
    
    print("Loading trained weights...")
    try:
        baseline_model_blood.load_state_dict(torch.load(baseline_path_blood, map_location=device))
        sd_model_blood.load_state_dict(torch.load(sd_path_blood, map_location=device))
        
        print("\n--- Processing Baseline Model ---")
        test_feat_base_blood, test_labels_blood = extract_features(baseline_model_blood, test_loader_blood, device, is_sd_model=False)
        print("\n--- Processing Self-Distilled Model ---")
        test_feat_sd_blood, _ = extract_features(sd_model_blood, test_loader_blood, device, is_sd_model=True)
        
        evaluate_and_print_similarities("BloodMNIST", test_feat_base_blood, test_feat_sd_blood, test_labels_blood, num_classes_blood)
    except Exception as e:
        print(f"Error loading BloodMNIST weights: {e}")


    # # --- 2. CHEST X-RAY SECTION (ResNet-18) ---
    # print("\nLoading Chest X-Ray dataloaders...")
    # data_dir_chest = 'Hypothesis 3/data/chest_xray/chest_xray'
    # _, _, test_loader_chest, in_channels_chest, num_classes_chest = get_chestxray_dataloaders(data_dir=data_dir_chest, batch_size=batch_size)
    # 
    # baseline_path_chest = os.path.join(save_dir, 'resnet18_chestxray_baseline_best.pth')
    # sd_path_chest = os.path.join(save_dir, 'resnet18_chestxray_self_distilled_best.pth')
    # 
    # baseline_model_chest = Baseline_Resnet18_H10k(num_classes=num_classes_chest, in_channels=in_channels_chest).to(device)
    # sd_model_chest = SelfDistillationResNet18_H10k(num_classes=num_classes_chest, in_channels=in_channels_chest).to(device)
    # 
    # print("Loading trained weights...")
    # try:
    #     baseline_model_chest.load_state_dict(torch.load(baseline_path_chest, map_location=device))
    #     sd_model_chest.load_state_dict(torch.load(sd_path_chest, map_location=device))
    #     
    #     print("\n--- Processing Baseline Model ---")
    #     test_feat_base_chest, test_labels_chest = extract_features(baseline_model_chest, test_loader_chest, device, is_sd_model=False)
    #     print("\n--- Processing Self-Distilled Model ---")
    #     test_feat_sd_chest, _ = extract_features(sd_model_chest, test_loader_chest, device, is_sd_model=True)
    #     
    #     evaluate_and_print_similarities("Chest X-Ray", test_feat_base_chest, test_feat_sd_chest, test_labels_chest, num_classes_chest)
    # except Exception as e:
    #     print(f"Error loading Chest X-Ray weights: {e}")

if __name__ == "__main__":
    main()