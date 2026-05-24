"""
Multi-Level Knowledge Distillation
───────────────────────────────────
Provides separate KD pathways for the dual-domain architecture:

  1. Embedding KD  — MSE between teacher and student fused embeddings
  2. Logits KD     — KL-divergence between softened logit distributions

All feature-level pathways operate on matching 256-dim vectors, so no
projection adapters are needed.

Usage:
    kd = MultiLevelKD(feat_dim=256, temperature=4.0)
    losses = kd(student_out, teacher_out)
    # losses["total"], losses["embedding"], losses["logits"]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLevelKD(nn.Module):
    """
    Multi-level knowledge distillation loss for dual-domain architectures.

    Args:
        feat_dim:    dimensionality of feature/embedding vectors (default: 256)
        temperature: softmax temperature for logits KD (default: 4.0)

        w_embedding: weight for fused embedding KD (default: 1.0)
        w_logits:    weight for logits KD (default: 1.0)
    """

    def __init__(
        self,
        feat_dim: int = 256,
        temperature: float = 4.0,
        w_embedding: float = 1.0,
        w_logits: float = 1.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.T = temperature
        self.w_embedding = w_embedding
        self.w_logits = w_logits

        # Feature-level: MSE (L2) loss — works well for normalized embeddings
        self.feat_loss = nn.MSELoss()

        # Logits-level: KL divergence on softened distributions
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def _logits_kd(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """KL divergence between temperature-softened distributions."""
        T = self.T
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        return self.kl_loss(student_log_probs, teacher_probs) * (T * T)

    def forward(
        self,
        student_out: dict,
        teacher_out: dict,
    ) -> dict:
        """
        Compute multi-level KD losses.

        Args:
            student_out: dict with keys "embedding", "logits"
            teacher_out: dict with same keys (from frozen teacher)

        Returns:
            dict with individual loss components and weighted total:
                "embedding", "logits", "total"
        """
        # Validate shapes
        for key in ("embedding", "logits"):
            assert key in student_out, f"MultiLevelKD: missing student key '{key}'"
            assert key in teacher_out, f"MultiLevelKD: missing teacher key '{key}'"

        # Feature-level KD (MSE on L2-normalized embeddings to prevent magnitude collapse)
        student_emb = F.normalize(student_out["embedding"], p=2, dim=1)
        teacher_emb = F.normalize(teacher_out["embedding"].detach(), p=2, dim=1)
        
        loss_embedding = self.feat_loss(student_emb, teacher_emb)

        # Logits KD (KL divergence)
        loss_logits = self._logits_kd(
            student_out["logits"],
            teacher_out["logits"].detach()
        )

        # Weighted total
        total = (
            self.w_embedding * loss_embedding
            + self.w_logits    * loss_logits
        )

        return {
            "embedding": loss_embedding,
            "logits":    loss_logits,
            "total":     total,
        }
