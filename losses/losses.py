import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 1. Focal Loss (with label smoothing)
# ---------------------------
# class ClassificationLoss(nn.Module):
#     def __init__(self, gamma=2.0, label_smoothing=0.1):
#         super().__init__()
#         self.gamma = gamma
#         self.label_smoothing = label_smoothing

#     def forward(self, logits, targets):
#         # Calculate standard Cross Entropy loss without reducing to a mean yet
#         ce_loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        
#         # Calculate pt (the predicted probability of the true target class)
#         probs = torch.softmax(logits, dim=1)
#         pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
#         # Apply the Focal Loss focusing parameter: (1 - pt)^gamma
#         # This reduces the loss weight for examples the model is already confident about
#         focal_loss = ((1.0 - pt) ** self.gamma * ce_loss).mean()
        
#         return focal_loss


class ClassificationLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
 
    def forward(self, logits, targets):
        # Single forward pass: log_softmax covers both ce_loss and pt
        log_probs = F.log_softmax(logits, dim=1)
 
        # Cross-entropy with label smoothing (manual, to avoid a second softmax call)
        num_classes = logits.size(1)
        smooth = self.label_smoothing
        # Smooth targets: (1 - ε) * one_hot + ε / C  →  uniform mix
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, smooth / num_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smooth + smooth / num_classes)
        ce_loss = -(smooth_targets * log_probs).sum(dim=1)  # (B,)
 
        # pt from the same log_probs — no redundant softmax
        pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).exp()  # (B,)
 
        focal_loss = ((1.0 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# ---------------------------
# 2. Knowledge Distillation Loss
# ---------------------------
class KDLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        """
        KL divergence between softened distributions
        """
        T = self.T

        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)

        loss = self.kl(student_log_probs, teacher_probs) * (T * T)
        return loss


# ---------------------------
# 3. Supervised Contrastive Loss
# ---------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, embeddings, labels):
        """
        embeddings: (B, D)
        labels: (B,)
        """
        device = embeddings.device
        B = embeddings.shape[0]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.T

        # Mask for positives
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Remove self-comparisons
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        mask = mask * logits_mask

        # Issue #3: log-sum-exp trick for numerical stability under fp16/autocast.
        # With T=0.07, similarities are amplified ~14x. Without subtracting the
        # max, exp() can overflow fp16 (max ~65504) and produce NaN.
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Log-softmax with stability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positives
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        loss = -mean_log_prob_pos.mean()
        return loss


# ---------------------------
# 4. Generator Adversarial Loss
# ---------------------------
class GeneratorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, gen_logits, gen_labels):
        """
        Standard CE — GRL handles adversarial behavior.
        """
        return self.ce(gen_logits, gen_labels)