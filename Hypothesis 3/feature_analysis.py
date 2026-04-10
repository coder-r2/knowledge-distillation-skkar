import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

# Import your custom modules
from datasets import get_chestxray_dataloaders
from models import Baseline_Resnet18_H10k, SelfDistillationResNet18_H10k

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_features(model, dataloader, device, is_sd_model=False):
    """
    Extracts the latent features from the model just before the linear classifiers.
    """
    model.eval()
    
    # Dictionary to hold the intercepted features
    intercepted_features = {}
    
    # Hook function to grab the input to the classifier layers
    def get_features(name):
        def hook(model, input, output):
            # input is a tuple of (tensor,), we want the tensor, detached and moved to CPU
            intercepted_features[name] = input[0].detach().cpu().numpy()
        return hook

    # Register the hooks
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

    print("Extracting features...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, leave=False):
            images = images.to(device)
            labels = labels.squeeze().numpy()
            
            _ = model(images) # Forward pass triggers the hooks
            
            for key in all_features.keys():
                all_features[key].append(intercepted_features[key])
            all_labels.extend(labels)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Concatenate lists into large numpy arrays
    for key in all_features.keys():
        all_features[key] = np.concatenate(all_features[key], axis=0)
        
    return all_features, np.array(all_labels)

def main():
    set_seed(67)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Setup Data and Models ---
    data_dir = 'Hypothesis 3/data/chest_xray/chest_xray'
    batch_size = 64
    print("Loading Chest X-Ray dataloaders...")
    train_loader, _, test_loader, in_channels, num_classes = get_chestxray_dataloaders(data_dir, batch_size)

    save_dir = 'Hypothesis 3/saved_models'
    baseline_path = os.path.join(save_dir, 'resnet18_chestxray_baseline_best.pth')
    sd_path = os.path.join(save_dir, 'resnet18_chestxray_self_distilled_best.pth')

    baseline_model = Baseline_Resnet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)
    sd_model = SelfDistillationResNet18_H10k(num_classes=num_classes, in_channels=in_channels).to(device)

    print("\nLoading trained weights...")
    baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
    sd_model.load_state_dict(torch.load(sd_path, map_location=device))

    # --- 2. Extract Features ---
    print("\n--- Processing Baseline Model ---")
    train_feat_base, train_labels = extract_features(baseline_model, train_loader, device, is_sd_model=False)
    test_feat_base, test_labels = extract_features(baseline_model, test_loader, device, is_sd_model=False)

    print("\n--- Processing Self-Distilled Model ---")
    train_feat_sd, _ = extract_features(sd_model, train_loader, device, is_sd_model=True)
    test_feat_sd, _ = extract_features(sd_model, test_loader, device, is_sd_model=True)

    # --- 3. k-NN Evaluation ---
    print("\n==============================================")
    print("      k-NN Classification Results (k=5)       ")
    print("==============================================")
    
    # Baseline k-NN
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(train_feat_base['baseline'], train_labels)
    acc_base = knn.score(test_feat_base['baseline'], test_labels) * 100
    print(f"Baseline Linear Features : {acc_base:.2f}% Accuracy")

    # SD Exits k-NN
    sd_accs = []
    for i in range(1, 5):
        key = f'exit{i}'
        knn.fit(train_feat_sd[key], train_labels)
        acc_sd = knn.score(test_feat_sd[key], test_labels) * 100
        sd_accs.append(acc_sd)
        print(f"SD Model Exit {i} Features : {acc_sd:.2f}% Accuracy")
    print("==============================================\n")

    # --- 4. t-SNE Visualization ---
    print("Generating t-SNE embeddings for the Test Set (this may take a minute)...")
    
    # We will plot Baseline vs SD Exit 1 vs SD Exit 4 (Teacher)
    features_to_plot = {
        'Baseline': test_feat_base['baseline'],
        'SD Exit 1 (Shallow)': test_feat_sd['exit1'],
        'SD Exit 4 (Teacher)': test_feat_sd['exit4']
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('t-SNE Visualization of Latent Spaces (Chest X-Ray Test Set)', fontsize=16)

    class_names = ['Normal', 'Pneumonia']
    colors = ['#1f77b4', '#d62728']

    for ax, (title, features) in zip(axes, features_to_plot.items()):
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        embeddings = tsne.fit_transform(features)
        
        # Scatter plot
        for class_idx, color in zip(range(num_classes), colors):
            mask = (test_labels == class_idx)
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                       c=color, label=class_names[class_idx], alpha=0.6, s=15)
            
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if ax == axes[0]:
            ax.legend()

    plt.tight_layout()
    os.makedirs('Hypothesis 3/results/feature_analysis', exist_ok=True)
    save_path = 'Hypothesis 3/results/feature_analysis/tsne_latent_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved t-SNE plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()

# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.manifold import TSNE
# from tqdm.auto import tqdm

# # Import your custom modules for BloodMNIST and ResNet50
# from datasets import get_medmnist_dataloaders
# from models import BaselineResNet50, SelfDistillationResNet50

# def set_seed(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def extract_features(model, dataloader, device, is_sd_model=False):
#     """
#     Extracts the latent features from the model just before the linear classifiers.
#     """
#     model.eval()
    
#     # Dictionary to hold the intercepted features
#     intercepted_features = {}
    
#     # Hook function to grab the input to the classifier layers
#     def get_features(name):
#         def hook(model, input, output):
#             # input is a tuple of (tensor,), we want the tensor, detached and moved to CPU
#             intercepted_features[name] = input[0].detach().cpu().numpy()
#         return hook

#     # Register the hooks
#     hooks = []
#     if is_sd_model:
#         hooks.append(model.classifier1.register_forward_hook(get_features('exit1')))
#         hooks.append(model.classifier2.register_forward_hook(get_features('exit2')))
#         hooks.append(model.classifier3.register_forward_hook(get_features('exit3')))
#         hooks.append(model.classifier4.register_forward_hook(get_features('exit4')))
#     else:
#         # Standard ResNet uses 'fc' for the final fully connected layer
#         hooks.append(model.fc.register_forward_hook(get_features('baseline')))

#     all_features = {k: [] for k in (['exit1', 'exit2', 'exit3', 'exit4'] if is_sd_model else ['baseline'])}
#     all_labels = []

#     print("Extracting features...")
#     with torch.no_grad():
#         for images, labels in tqdm(dataloader, leave=False):
#             images = images.to(device)
#             labels = labels.squeeze().numpy()
            
#             _ = model(images) # Forward pass triggers the hooks
            
#             for key in all_features.keys():
#                 all_features[key].append(intercepted_features[key])
#             all_labels.extend(labels)

#     # Cleanup hooks
#     for h in hooks:
#         h.remove()

#     # Concatenate lists into large numpy arrays
#     for key in all_features.keys():
#         all_features[key] = np.concatenate(all_features[key], axis=0)
        
#     return all_features, np.array(all_labels)

# def main():
#     set_seed(67)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # --- 1. Setup Data and Models ---
#     batch_size = 128  # Safe for 32x32 images
#     print("Loading BloodMNIST dataloaders...")
#     # Unpack 5 values specifically matching get_medmnist_dataloaders
#     train_loader, val_loader, test_loader, in_channels, num_classes = get_medmnist_dataloaders(data_flag='bloodmnist', batch_size=batch_size)

#     save_dir = 'Hypothesis 3/saved_models'
#     baseline_path = os.path.join(save_dir, 'resnet50_bloodmnist_baseline_best.pth')
#     sd_path = os.path.join(save_dir, 'resnet50_bloodmnist_self_distilled_best.pth')

#     baseline_model = BaselineResNet50(num_classes=num_classes, in_channels=in_channels).to(device)
#     sd_model = SelfDistillationResNet50(num_classes=num_classes, in_channels=in_channels).to(device)

#     print("\nLoading trained weights...")
#     baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
#     sd_model.load_state_dict(torch.load(sd_path, map_location=device))

#     # --- 2. Extract Features ---
#     print("\n--- Processing Baseline Model ---")
#     train_feat_base, train_labels = extract_features(baseline_model, train_loader, device, is_sd_model=False)
#     test_feat_base, test_labels = extract_features(baseline_model, test_loader, device, is_sd_model=False)

#     print("\n--- Processing Self-Distilled Model ---")
#     train_feat_sd, _ = extract_features(sd_model, train_loader, device, is_sd_model=True)
#     test_feat_sd, _ = extract_features(sd_model, test_loader, device, is_sd_model=True)

#     # --- 3. k-NN Evaluation ---
#     print("\n==============================================")
#     print("      k-NN Classification Results (k=5)       ")
#     print("==============================================")
    
#     # Baseline k-NN
#     knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
#     knn.fit(train_feat_base['baseline'], train_labels)
#     acc_base = knn.score(test_feat_base['baseline'], test_labels) * 100
#     print(f"Baseline Linear Features : {acc_base:.2f}% Accuracy")

#     # SD Exits k-NN
#     sd_accs = []
#     for i in range(1, 5):
#         key = f'exit{i}'
#         knn.fit(train_feat_sd[key], train_labels)
#         acc_sd = knn.score(test_feat_sd[key], test_labels) * 100
#         sd_accs.append(acc_sd)
#         print(f"SD Model Exit {i} Features : {acc_sd:.2f}% Accuracy")
#     print("==============================================\n")

#     # --- 4. t-SNE Visualization ---
#     print("Generating t-SNE embeddings for the Test Set (this may take a minute)...")
    
#     # We will plot Baseline vs SD Exit 1 vs SD Exit 4 (Teacher)
#     features_to_plot = {
#         'Baseline': test_feat_base['baseline'],
#         'SD Exit 1 (Shallow)': test_feat_sd['exit1'],
#         'SD Exit 4 (Teacher)': test_feat_sd['exit4']
#     }
    
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     fig.suptitle('t-SNE Visualization of Latent Spaces (BloodMNIST Test Set)', fontsize=16)

#     # BloodMNIST has 8 classes, dynamically create a colormap
#     cmap = plt.get_cmap('tab10')
#     colors = [cmap(i) for i in range(num_classes)]
#     class_names = [f'Class {i}' for i in range(num_classes)] 

#     for ax, (title, features) in zip(axes, features_to_plot.items()):
#         # Compute t-SNE
#         tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
#         embeddings = tsne.fit_transform(features)
        
#         # Scatter plot
#         for class_idx, color in zip(range(num_classes), colors):
#             mask = (test_labels == class_idx)
#             ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
#                        color=color, label=class_names[class_idx], alpha=0.6, s=15)
            
#         ax.set_title(title)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         if ax == axes[0]:
#             ax.legend(loc='best', fontsize='small')

#     plt.tight_layout()
#     os.makedirs('Hypothesis 3/results/feature_analysis', exist_ok=True)
#     save_path = 'Hypothesis 3/results/feature_analysis/tsne_latent_comparison_bloodmnist.png'
#     plt.savefig(save_path, dpi=300)
#     print(f"Saved t-SNE plot to {save_path}")
#     plt.show()

# if __name__ == "__main__":
#     main()