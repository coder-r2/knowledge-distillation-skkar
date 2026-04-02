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

class DECELoss(nn.Module):
    """
    Differentiable Expected Calibration Error (DECE).
    Approximates hard accuracy and hard binning with smooth differentiable operations.
    """
    def __init__(self, n_bins=15, tau_a=100.0, tau_b=0.01):
        super().__init__()
        self.n_bins = n_bins
        self.tau_a = tau_a
        self.tau_b = tau_b
        # Creates centers for the soft bins mathematically equivalent to standard ECE bounds
        centers = torch.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins)
        self.register_buffer('centers', centers)

    def forward(self, logits, targets):
        B, C = logits.shape
        device = logits.device # Grab the device dynamically from the model output
        
        # Model confidence (probabilities)
        probs = F.softmax(logits, dim=1)
        conf, _ = torch.max(probs, dim=1) # Shape: (B,)
        
        # --- 1. Differentiable Accuracy (All-Pairs Soft Ranking) ---
        batch_indices = torch.arange(B, device=device)
        
        # Get the logits of the true class
        true_logits = logits[batch_indices, targets]
        # Compare every class logit to the true class logit
        diff = logits - true_logits.unsqueeze(1) 
        
        # Mask out the comparison to the true class itself (j != c)
        mask = torch.ones_like(diff)
        mask[batch_indices, targets] = 0.0
        
        # Calculate soft rank using the scaled sigmoid function
        soft_rank = 1.0 + torch.sum(torch.sigmoid(diff * self.tau_a) * mask, dim=1)
        # Soft accuracy representation (max(0, 2 - rank))
        soft_acc = F.relu(2.0 - soft_rank) 
        
        # --- 2. Soft Binning ---
        # 🚨 THE FIX: Explicitly move the centers buffer to the active device 🚨
        centers_on_device = self.centers.to(device)
        
        # Calculate distance to bin centers and apply softmax with temperature
        dist = - (conf.unsqueeze(1) - centers_on_device.unsqueeze(0))**2 / self.tau_b
        o_m = F.softmax(dist, dim=1) # Probability of falling into each bin
        
        # --- 3. Compute Differentiable ECE ---
        dece = 0.0
        for m in range(self.n_bins):
            o_m_bin = o_m[:, m] 
            bin_weight = torch.sum(o_m_bin)
            
            # Prevent NaN on empty bins
            if bin_weight > 1e-6:
                # Differentiable bin accuracy and confidence
                acc_bin = torch.sum(o_m_bin * soft_acc) / bin_weight
                conf_bin = torch.sum(o_m_bin * conf) / bin_weight
                
                # Accumulate weighted gap
                dece += (bin_weight / B) * torch.abs(acc_bin - conf_bin)
                
        return dece

class CalibrationAwareSelfDistillationLoss(nn.Module):
    """
    Approach 1: Distilled Calibration.
    The Teacher (deepest classifier) trains using Cross-Entropy AND DECE loss.
    The Students train using Cross-Entropy, KL Divergence (to mimic the calibrated Teacher), 
    and L2 Hint Loss, naturally inheriting the calibration.
    """
    def __init__(self, alpha=0.5, lambda_weight=0.01, beta=0.5, temperature=3.0, n_bins=15):
        super().__init__()
        self.alpha = alpha
        self.lambda_weight = lambda_weight
        self.beta = beta  # Strength of the DECE penalty on the Teacher
        self.T = temperature
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.dece_loss = DECELoss(n_bins=n_bins)

    def forward(self, logits_list, bottlenecks_list, targets):
        targets = targets.squeeze().long()
        total_loss = 0
        num_classifiers = len(logits_list)

        # The deepest classifier acts as the calibrated teacher
        teacher_logits = logits_list[-1]
        teacher_features = bottlenecks_list[-1]

        # Softened teacher probabilities for KL Divergence
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / self.T, dim=1)

        for i in range(num_classifiers):
            student_logits = logits_list[i]
            ce = self.ce_loss(student_logits, targets)

            # If it is a SHALLOW classifier (Student)
            if i < num_classifiers - 1:
                student_features = bottlenecks_list[i]

                # KL Divergence pulls student toward calibrated teacher
                student_log_probs = F.log_softmax(student_logits / self.T, dim=1)
                kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.T ** 2)

                # L2 Hint Loss
                hint = self.mse_loss(student_features, teacher_features)

                # Standard SD Loss
                loss_i = (1 - self.alpha) * ce + self.alpha * kl + self.lambda_weight * hint
            
            # If it is the DEEPEST classifier (Teacher)
            else:
                # Apply the Differentiable ECE penalty exclusively here
                dece = self.dece_loss(teacher_logits, targets)
                loss_i = ce + (self.beta * dece)

            total_loss += loss_i

        return total_loss