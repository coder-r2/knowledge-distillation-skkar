import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from medmnist import INFO

# 1. Updated Imports for BloodMNIST and ResNet50s
from models import BaselineResNet50, SelfDistillationResNet50 
from datasets import get_medmnist_dataloaders

# ==========================================
# 1. Grad-CAM Wrapper
# ==========================================
class SingleOutputWrapper(nn.Module):
    """Forces the model to return only the specific logits for Grad-CAM."""
    def __init__(self, model, is_multi_exit=False, exit_idx=None):
        super().__init__()
        self.model = model
        self.is_multi_exit = is_multi_exit
        self.exit_idx = exit_idx

    def forward(self, x):
        if self.is_multi_exit:
            # Multi-exit model returns (logits, bottlenecks)
            logits, _ = self.model(x) 
            return logits[self.exit_idx]
        else:
            return self.model(x)

# ==========================================
# 2. Unified Analysis Function
# ==========================================
def find_exit4_correction_cases(baseline_model, sd_model, dataloader, target_layers, device, num_cases=1, class_map=None):
    
    def format_pred(idx):
        if class_map:
            name = class_map.get(str(idx), class_map.get(idx, ""))
            return f"{idx} ({name})" if name else str(idx)
        return str(idx)

    baseline_model.eval().to(device)
    sd_model.eval().to(device)
    cases_found = 0
    
    print("--- Searching for Exit 4 Correction Cases ---")
    print("Condition: Baseline WRONG | SD Exit 4 CORRECT")
    
    for images, labels in dataloader:
        if cases_found >= num_cases:
            break
            
        images = images.to(device)
        labels = labels.squeeze().long().to(device) 
        
        with torch.no_grad():
            base_logits = baseline_model(images)
            sd_logits, _ = sd_model(images)
            
            base_preds = torch.argmax(base_logits, dim=1)
            sd_preds = [torch.argmax(l, dim=1) for l in sd_logits] 

        for i in range(images.size(0)):
            label = labels[i].item()
            b_pred = base_preds[i].item()
            
            # Condition 1: Baseline must be wrong
            if b_pred == label:
                continue
                
            # Condition 2: SD Exit 4 must be right (Index 3 is the 4th exit)
            sd_exit4_pred = sd_preds[3][i].item()
            if sd_exit4_pred != label:
                continue
                
            print(f"Match Found! True: {format_pred(label)} | Baseline: {format_pred(b_pred)} (Wrong) | SD Exit 4: {format_pred(sd_exit4_pred)} (Correct)")
            
            img_tensor = images[i].unsqueeze(0)
            
            # --- Generate Baseline Grad-CAM ---
            base_wrapper = SingleOutputWrapper(baseline_model, is_multi_exit=False)
            cam_base = GradCAM(model=base_wrapper, target_layers=[target_layers['baseline']])
            gray_cam_base = cam_base(input_tensor=img_tensor)[0, :]
            
            # --- Generate SD Exit 4 Grad-CAM ---
            sd_wrapper = SingleOutputWrapper(sd_model, is_multi_exit=True, exit_idx=3)
            cam_sd = GradCAM(model=sd_wrapper, target_layers=[target_layers['sd_exits'][3]])
            gray_cam_sd = cam_sd(input_tensor=img_tensor)[0, :]
            
            # --- Format Image ---
            # Denormalize based on BloodMNIST
            img_display = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min()) 
            img_display = np.clip(img_display, 0, 1) # Ensure valid range for plt
            
            vis_base = show_cam_on_image(img_display, gray_cam_base, use_rgb=True)
            vis_sd = show_cam_on_image(img_display, gray_cam_sd, use_rgb=True)
            
            # --- Plotting 1x3 Grid ---
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img_display)
            axes[0].set_title(f"Original BloodMNIST\nTrue: {format_pred(label)}")
            axes[0].axis('off')
            
            axes[1].imshow(vis_base)
            axes[1].set_title(f"Baseline (Fail)\nPred: {format_pred(b_pred)}")
            axes[1].axis('off')
            
            axes[2].imshow(vis_sd)
            axes[2].set_title(f"SD Deep (Exit 4)\nPred: {format_pred(sd_exit4_pred)}")
            axes[2].axis('off')
            
            plt.suptitle("Self-Distillation (Exit 4) Correcting Baseline Overthinking (BloodMNIST)", fontsize=16)
            plt.tight_layout()
            plt.show()
            
            cases_found += 1
            if cases_found >= num_cases:
                return 
            break 

# ==========================================
# 3. Execution
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Update Dataset Loading for BloodMNIST
    data_flag = 'bloodmnist'
    info = INFO[data_flag]
    num_classes = len(info['label'])
    
    _, _, test_loader, _, _ = get_medmnist_dataloaders(
        data_flag=data_flag, 
        batch_size=32
    )

    # 3. Initialize ResNet50 Models
    baseline_model = BaselineResNet50(num_classes=num_classes)
    sd_model = SelfDistillationResNet50(num_classes=num_classes)

    # 4. Load Weights (Ensure these point to your BloodMNIST ResNet-50 weights)
    try:
        baseline_model.load_state_dict(torch.load('Hypothesis 3/saved_models/resnet50_bloodmnist_baseline_best.pth', map_location=device))
        sd_model.load_state_dict(torch.load('Hypothesis 3/saved_models/resnet50_bloodmnist_self_distilled_best.pth', map_location=device))
        print("Both models loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 5. Map Target Layers for ResNet50
    target_layers_config = {
        'baseline': baseline_model.layer4[-1],
        'sd_exits': [
            sd_model.layer1[-1],
            sd_model.layer2[-1],
            sd_model.layer3[-1],
            sd_model.layer4[-1]  
        ]
    }

    # 6. Define Class Dictionary automatically from MedMNIST INFO
    my_class_map = info['label']

    # Run Analysis
    find_exit4_correction_cases(
        baseline_model=baseline_model, 
        sd_model=sd_model, 
        dataloader=test_loader, 
        target_layers=target_layers_config, 
        device=device, 
        num_cases=25, 
        class_map=my_class_map
    )

if __name__ == '__main__':
    main()