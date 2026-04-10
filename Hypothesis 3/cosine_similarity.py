import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

# Import your custom modules
from datasets import get_chestxray_dataloaders
from models import Baseline_Resnet18_H10k, SelfDistillationResNet18_H10k

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
            labels = labels.squeeze().numpy()
            
            _ = model(images) 
            
            for key in all_features.keys():
                all_features[key].append(intercepted_features[key])
            all_labels.extend(labels)

    for h in hooks:
        h.remove()

    for key in all_features.keys():
        all_features[key] = np.concatenate(all_features[key], axis=0)
        
    return all_features, np.array(all_labels)

def compute_centroid_similarities(features, labels, num_classes=2):
    """Computes Intra-class and Inter-class cosine similarity against class centroids."""
    centroids = []
    
    # 1. Calculate the centroid (mean vector) for each class
    for c in range(num_classes):
        class_features = features[labels == c]
        centroid = np.mean(class_features, axis=0)
        centroids.append(centroid)

    intra_sims = []
    inter_sims = []

    # 2. Compute similarity of each test sample to the centroids
    for i in range(len(features)):
        feat = features[i].reshape(1, -1)
        true_class = labels[i]

        for c in range(num_classes):
            sim = cosine_similarity(feat, centroids[c].reshape(1, -1))[0, 0]
            if c == true_class:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    return np.mean(intra_sims), np.mean(inter_sims)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup Data and Models ---
    data_dir = 'Hypothesis 3/data/chest_xray/chest_xray'
    batch_size = 64
    print("Loading Chest X-Ray dataloaders...")
    _, _, test_loader, in_channels, num_classes = get_chestxray_dataloaders(data_dir, batch_size)

    save_dir = 'Hypothesis 3/saved_models'
    baseline_path = os.path.join(save_dir, 'resnet18_chestxray_baseline_best.pth')
    sd_path = os.path.join(save_dir, 'resnet18_chestxray_self_distilled_best.pth')

    baseline_model = Baseline_Resnet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)
    sd_model = SelfDistillationResNet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)

    print("Loading trained weights...")
    baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
    sd_model.load_state_dict(torch.load(sd_path, map_location=device))

    # --- Extract Features ---
    print("\n--- Processing Baseline Model ---")
    test_feat_base, test_labels = extract_features(baseline_model, test_loader, device, is_sd_model=False)

    print("\n--- Processing Self-Distilled Model ---")
    test_feat_sd, _ = extract_features(sd_model, test_loader, device, is_sd_model=True)

    # --- Calculate and Print Cosine Similarities ---
    print("\n========================================================")
    print("      Latent Space Cosine Similarity Analysis           ")
    print("========================================================")
    
    # Baseline
    base_intra, base_inter = compute_centroid_similarities(test_feat_base['baseline'], test_labels, num_classes)
    print(f"[Baseline ResNet-18]")
    print(f"  Intra-class Similarity (Cohesion)  : {base_intra:.4f}  <-- Higher is better")
    print(f"  Inter-class Similarity (Separation): {base_inter:.4f}  <-- Lower is better\n")

    # SD Exits
    for i in range(1, 5):
        key = f'exit{i}'
        sd_intra, sd_inter = compute_centroid_similarities(test_feat_sd[key], test_labels, num_classes)
        print(f"[Self-Distilled Exit {i}]")
        print(f"  Intra-class Similarity (Cohesion)  : {sd_intra:.4f}")
        print(f"  Inter-class Similarity (Separation): {sd_inter:.4f}\n")
        
    print("========================================================\n")

if __name__ == "__main__":
    main()