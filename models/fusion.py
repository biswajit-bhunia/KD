"""
Gated Fusion Module
───────────────────
Lightweight adaptive fusion that dynamically weighs semantic vs forensic
features via a learned sigmoid gate.  Shared by both teacher and student
networks to ensure consistent fusion behavior during knowledge distillation.

Design rationale:
  - Simple concatenation treats both streams equally, losing the ability
    to prioritize one domain when it carries more discriminative signal.
  - This gated mechanism learns a per-sample, per-dimension weighting:
        alpha = sigmoid(W @ [sem; for] + b)
        fused = alpha * sem + (1 - alpha) * for
  - The gate is lightweight (a single linear layer + sigmoid) so it adds
    minimal parameters and is stable under federated aggregation.
"""

import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    """
    Gated fusion of two feature vectors with matching dimensionality.

    Args:
        feat_dim: dimensionality of each input feature vector (default: 256)

    Input:
        feat_a: (B, feat_dim) — e.g. semantic features
        feat_b: (B, feat_dim) — e.g. forensic features

    Output:
        fused:  (B, feat_dim) — gated combination
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.feat_dim = feat_dim

        # Gate network: concat → linear → sigmoid
        self.gate_net = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid()
        )

        # Initialize gate bias to 0 so initial alpha ≈ 0.5 (equal weighting)
        nn.init.zeros_(self.gate_net[0].bias)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_a: (B, feat_dim) — semantic features
            feat_b: (B, feat_dim) — forensic features

        Returns:
            fused: (B, feat_dim)
        """
        assert feat_a.shape == feat_b.shape, (
            f"GatedFusion shape mismatch: feat_a={feat_a.shape}, feat_b={feat_b.shape}"
        )
        assert feat_a.shape[1] == self.feat_dim, (
            f"GatedFusion dim mismatch: expected {self.feat_dim}, got {feat_a.shape[1]}"
        )

        # Compute gate
        concat = torch.cat([feat_a, feat_b], dim=1)  # (B, feat_dim*2)
        alpha = self.gate_net(concat)                  # (B, feat_dim), values in [0, 1]

        # Gated combination
        fused = alpha * feat_a + (1.0 - alpha) * feat_b  # (B, feat_dim)

        return fused
