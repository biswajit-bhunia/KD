"""
Multi-Level Knowledge Distillation
───────────────────────────────────
Provides separate KD pathways for the dual-domain architecture:

  1. Semantic KD   — MSE between teacher and student semantic features
  2. Forensic KD   — MSE between teacher and student forensic features
  3. Embedding KD  — MSE between teacher and student fused embeddings
  4. Logits KD     — KL-divergence between softened logit distributions

All feature-level pathways operate on matching 256-dim vectors, so no
projection adapters are needed.

Usage:
    kd = MultiLevelKD(feat_dim=256, temperature=4.0)
    losses = kd(student_out, teacher_out)
    # losses["total"], losses["semantic"], losses["forensic"],
    # losses["embedding"], losses["logits"]
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
        w_semantic:  weight for semantic feature KD (default: 1.0)
        w_forensic:  weight for forensic feature KD (default: 1.0)
        w_embedding: weight for fused embedding KD (default: 1.0)
        w_logits:    weight for logits KD (default: 1.0)
    """

    def __init__(
        self,
        feat_dim: int = 256,
        temperature: float = 4.0,
        w_semantic: float = 1.0,
        w_forensic: float = 1.0,
        w_embedding: float = 1.0,
        w_logits: float = 1.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.T = temperature
        self.w_semantic = w_semantic
        self.w_forensic = w_forensic
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
            student_out: dict with keys "semantic_feat", "forensic_feat",
                         "embedding", "logits"
            teacher_out: dict with same keys (from frozen teacher)

        Returns:
            dict with individual loss components and weighted total:
                "semantic", "forensic", "embedding", "logits", "total"
        """
        # Validate shapes
        for key in ("semantic_feat", "forensic_feat", "embedding", "logits"):
            assert key in student_out, f"MultiLevelKD: missing student key '{key}'"
            assert key in teacher_out, f"MultiLevelKD: missing teacher key '{key}'"

        # Feature-level KD (MSE)
        loss_semantic = self.feat_loss(
            student_out["semantic_feat"],
            teacher_out["semantic_feat"].detach()
        )
        loss_forensic = self.feat_loss(
            student_out["forensic_feat"],
            teacher_out["forensic_feat"].detach()
        )
        loss_embedding = self.feat_loss(
            student_out["embedding"],
            teacher_out["embedding"].detach()
        )

        # Logits KD (KL divergence)
        loss_logits = self._logits_kd(
            student_out["logits"],
            teacher_out["logits"].detach()
        )

        # Weighted total
        total = (
            self.w_semantic  * loss_semantic
            + self.w_forensic  * loss_forensic
            + self.w_embedding * loss_embedding
            + self.w_logits    * loss_logits
        )

        return {
            "semantic":  loss_semantic,
            "forensic":  loss_forensic,
            "embedding": loss_embedding,
            "logits":    loss_logits,
            "total":     total,
        }
