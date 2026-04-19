import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from medmnist import INFO
from datasets import get_medmnist_dataloaders, get_chestxray_dataloaders
from models import BaselineResNet50, SelfDistillationResNet50, Baseline_Resnet18_H10k, SelfDistillationResNet18_H10k

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SingleOutputWrapper(nn.Module):
    """Forces the model to return only the specific logits for Grad-CAM."""
    def __init__(self, model, is_multi_exit=False, exit_idx=None):
        super().__init__()
        self.model = model
        self.is_multi_exit = is_multi_exit
        self.exit_idx = exit_idx

    def forward(self, x):
        if self.is_multi_exit:
            logits = self.model(x) 
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits[self.exit_idx]
        else:
            return self.model(x)

def find_exit4_correction_cases(baseline_model, sd_model, dataloader, target_layers, device, dataset_name, num_cases=1, class_map=None):
    
    def format_pred(idx):
        if class_map:
            name = class_map.get(str(idx), class_map.get(idx, ""))
            return f"{idx} ({name})" if name else str(idx)
        return str(idx)

    baseline_model.eval().to(device)
    sd_model.eval().to(device)
    cases_found = 0
    
    print(f"\n--- Searching for Exit 4 Correction Cases ({dataset_name}) ---")
    print("Condition: Baseline WRONG | SD Exit 4 CORRECT")
    
    for images, labels in dataloader:
        if cases_found >= num_cases:
            break
            
        images = images.to(device)
        labels = labels.view(-1).long().to(device) 
        
        with torch.no_grad():
            base_logits = baseline_model(images)
            sd_outputs = sd_model(images)
            sd_logits = sd_outputs[0] if isinstance(sd_outputs, tuple) else sd_outputs
            
            base_preds = torch.argmax(base_logits, dim=1)
            sd_preds = [torch.argmax(l, dim=1) for l in sd_logits] 

        for i in range(images.size(0)):
            label = labels[i].item()
            b_pred = base_preds[i].item()
            
            if b_pred == label:
                continue
                
            sd_exit4_pred = sd_preds[3][i].item()
            if sd_exit4_pred != label:
                continue
                
            print(f"Match Found! True: {format_pred(label)} | Baseline: {format_pred(b_pred)} (Wrong) | SD Exit 4: {format_pred(sd_exit4_pred)} (Correct)")
            
            img_tensor = images[i].unsqueeze(0)
            
            base_wrapper = SingleOutputWrapper(baseline_model, is_multi_exit=False)
            cam_base = GradCAM(model=base_wrapper, target_layers=[target_layers['baseline']])
            gray_cam_base = cam_base(input_tensor=img_tensor)[0, :]
            
            sd_wrapper = SingleOutputWrapper(sd_model, is_multi_exit=True, exit_idx=3)
            cam_sd = GradCAM(model=sd_wrapper, target_layers=[target_layers['sd_exits'][3]])
            gray_cam_sd = cam_sd(input_tensor=img_tensor)[0, :]
            
            img_display = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min()) 
            img_display = np.clip(img_display, 0, 1)
            
            vis_base = show_cam_on_image(img_display, gray_cam_base, use_rgb=True)
            vis_sd = show_cam_on_image(img_display, gray_cam_sd, use_rgb=True)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img_display)
            axes[0].set_title(f"Original Image\nTrue: {format_pred(label)}")
            axes[0].axis('off')
            
            axes[1].imshow(vis_base)
            axes[1].set_title(f"Baseline (Fail)\nPred: {format_pred(b_pred)}")
            axes[1].axis('off')
            
            axes[2].imshow(vis_sd)
            axes[2].set_title(f"SD Deep (Exit 4)\nPred: {format_pred(sd_exit4_pred)}")
            axes[2].axis('off')
            
            plt.suptitle(f"Self-Distillation (Exit 4) Correcting Baseline Overthinking ({dataset_name})", fontsize=16)
            plt.tight_layout()
            
            save_dir = 'Hypothesis 3/results/gradcam'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{dataset_name.replace(' ', '_').lower()}_correction_{cases_found+1}.png"), dpi=300, bbox_inches='tight')
            plt.show()
            
            cases_found += 1
            if cases_found >= num_cases:
                return 
        if cases_found >= num_cases:
            break

def main():
    set_seed(67)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = 'Hypothesis 3/saved_models'

    # === 1. CHEST X-RAY SECTION (ResNet-18) ===
    print("\nLoading Chest X-Ray dataloaders...")
    data_dir_chest = 'Hypothesis 3/data/chest_xray/chest_xray'
    _, _, test_loader_chest, in_channels_chest, num_classes_chest = get_chestxray_dataloaders(data_dir=data_dir_chest, batch_size=32)
    
    baseline_model_chest = Baseline_Resnet18_H10k(num_classes=num_classes_chest, in_channels=in_channels_chest).to(device)
    sd_model_chest = SelfDistillationResNet18_H10k(num_classes=num_classes_chest, in_channels=in_channels_chest).to(device)

    baseline_path_chest = os.path.join(save_dir, 'resnet18_chestxray_baseline_best.pth')
    sd_path_chest = os.path.join(save_dir, 'resnet18_chestxray_self_distilled_best.pth')

    try:
        baseline_model_chest.load_state_dict(torch.load(baseline_path_chest, map_location=device))
        sd_model_chest.load_state_dict(torch.load(sd_path_chest, map_location=device))
        
        target_layers_chest = {
            'baseline': baseline_model_chest.layer4[-1],
            'sd_exits': [
                sd_model_chest.layer1[-1],
                sd_model_chest.layer2[-1],
                sd_model_chest.layer3[-1],
                sd_model_chest.layer4[-1]  
            ]
        }
        
        class_map_chest = {0: "NORMAL", 1: "PNEUMONIA"}
        
        find_exit4_correction_cases(
            baseline_model=baseline_model_chest, 
            sd_model=sd_model_chest, 
            dataloader=test_loader_chest, 
            target_layers=target_layers_chest, 
            device=device,
            dataset_name="Chest X-Ray",
            num_cases=5, 
            class_map=class_map_chest
        )
    except Exception as e:
        print(f"Error evaluating Chest X-Ray: {e}")


    # # === 2. BLOODMNIST SECTION (ResNet-50) ===
    # print("\nLoading BloodMNIST dataloaders...")
    # data_flag = 'bloodmnist'
    # info = INFO[data_flag]
    # _, _, test_loader_blood, in_channels_blood, num_classes_blood = get_medmnist_dataloaders(data_flag=data_flag, batch_size=32)
    # 
    # baseline_model_blood = BaselineResNet50(num_classes=num_classes_blood, in_channels=in_channels_blood).to(device)
    # sd_model_blood = SelfDistillationResNet50(num_classes=num_classes_blood, in_channels=in_channels_blood).to(device)
    # 
    # baseline_path_blood = os.path.join(save_dir, 'resnet50_bloodmnist_baseline_best.pth')
    # sd_path_blood = os.path.join(save_dir, 'resnet50_bloodmnist_self_distilled_best.pth')
    # 
    # try:
    #     baseline_model_blood.load_state_dict(torch.load(baseline_path_blood, map_location=device))
    #     sd_model_blood.load_state_dict(torch.load(sd_path_blood, map_location=device))
    #     
    #     target_layers_blood = {
    #         'baseline': baseline_model_blood.layer4[-1],
    #         'sd_exits': [
    #             sd_model_blood.layer1[-1],
    #             sd_model_blood.layer2[-1],
    #             sd_model_blood.layer3[-1],
    #             sd_model_blood.layer4[-1]  
    #         ]
    #     }
    #     
    #     class_map_blood = info['label']
    #     
    #     find_exit4_correction_cases(
    #         baseline_model=baseline_model_blood, 
    #         sd_model=sd_model_blood, 
    #         dataloader=test_loader_blood, 
    #         target_layers=target_layers_blood, 
    #         device=device,
    #         dataset_name="BloodMNIST",
    #         num_cases=5, 
    #         class_map=class_map_blood
    #     )
    # except Exception as e:
    #     print(f"Error evaluating BloodMNIST: {e}")

if __name__ == '__main__':
    main()