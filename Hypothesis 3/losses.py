import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfDistillationLoss(nn.Module):
    def __init__(self, alpha=0.357, lambda_weight=0.001, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.lambda_weight = lambda_weight
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits_list, features_list, targets):
        teacher_logits = logits_list[-1]
        teacher_features = features_list[-1]
        
        total_loss = 0.0
        
        for i in range(len(logits_list)):
            student_logits = logits_list[i]
            student_features = features_list[i]
            
            loss_ce = self.ce_loss(student_logits, targets)
            
            if i == len(logits_list) - 1:
                total_loss += loss_ce
                continue
            loss_kl = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_logits / self.temperature, dim=1),
                F.softmax(teacher_logits.detach() / self.temperature, dim=1)
            ) * (self.temperature ** 2)
            
            loss_hint = F.mse_loss(student_features, teacher_features.detach())
            
            exit_loss = loss_ce + (self.alpha * loss_kl) + (self.lambda_weight * loss_hint)
            total_loss += exit_loss

        return total_loss / len(logits_list)