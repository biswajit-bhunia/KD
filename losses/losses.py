import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 1. Standard Cross Entropy (with label smoothing)
# ---------------------------
class ClassificationLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        # Label smoothing prevents overconfident predictions by training
        # against soft targets (e.g., 0.05/0.95 instead of 0/1).
        # This produces better-calibrated probabilities and improves recall.
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        return self.ce(logits, targets)


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
        self.ce = nn.CrossEntropyLoss()

    def forward(self, gen_logits, gen_labels):
        """
        Standard CE — GRL handles adversarial behavior.
        """
        return self.ce(gen_logits, gen_labels)