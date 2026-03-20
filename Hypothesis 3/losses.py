import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, lambda_weight=0.01, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.lambda_weight = lambda_weight
        self.T = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, logits_list, bottlenecks_list, targets):
        # targets shape might be [batch, 1] for medmnist, squeeze it
        targets = targets.squeeze().long()

        total_loss = 0
        num_classifiers = len(logits_list)

        # The deepest classifier acts as the teacher
        teacher_logits = logits_list[-1]
        teacher_features = bottlenecks_list[-1]

        # Softened teacher probabilities for KL Divergence
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / self.T, dim=1)

        for i in range(num_classifiers):
            student_logits = logits_list[i]

            # 1. Cross Entropy Loss (for all classifiers, including deepest)
            ce = self.ce_loss(student_logits, targets)

            # 2 & 3. KL Divergence & L2 Hint Loss (only for shallow classifiers)
            if i < num_classifiers - 1:
                student_features = bottlenecks_list[i]

                # KL Divergence
                student_log_probs = F.log_softmax(student_logits / self.T, dim=1)
                kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.T ** 2)

                # L2 Hint Loss
                hint = self.mse_loss(student_features, teacher_features)

                loss_i = (1 - self.alpha) * ce + self.alpha * kl + self.lambda_weight * hint
            else:
                # Deepest classifier only gets supervised by labels
                loss_i = ce

            total_loss += loss_i

        return total_loss