"""
Dual-Domain Student Model
─────────────────────────
Architecture:
  Semantic Student  — MobileNetV2 (pretrained) on RGB (B,3,H,W)
  Forensic Student  — Lightweight 5-layer CNN on forensic stack (B,6,H,W)
  Fusion            — GatedFusion (learned sigmoid gate)
  Classifier        — Linear(embed_dim, 2)

Design decisions:
  - MobileNetV2 for semantic: pretrained ImageNet features transfer well to
    face/manipulation understanding.  ~2.5M params.
  - Lightweight CNN for forensic: the 6-ch forensic stack is hand-crafted
    (SRM, FFT) — no benefit from ImageNet pretraining.
    A custom CNN is more parameter-efficient (~500K).
  - Total student: ~3.1M params — fits within the 3-5M target for
    federated learning (consumer GPUs, efficient communication).

Forward signature:
  forward(x_rgb, x_forensic) → dict
  NOTE: Changed from the old forward(x_forensic, x_grad).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

from models.fusion import GatedFusion

# Semantic Student (MobileNetV2)
class SemanticStudentBranch(nn.Module):
    """
    MobileNetV2 backbone for RGB images.

    Input:  (B, 3, H, W) — RGB image normalized to [-1, 1]
    Output: (B, out_dim)  — semantic feature vector
    """

    def __init__(self, pretrained: bool = True, out_dim: int = 256):
        super().__init__()

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)

        # MobileNetV2 features: all conv layers (no classifier)
        self.features = model.features           # → (B, 1280, H/32, W/32)
        self.pool = nn.AdaptiveAvgPool2d(1)      # → (B, 1280, 1, 1)
        self.backbone_dim = 1280

        # Projection to shared embedding space
        self.proj = nn.Sequential(
            nn.Linear(self.backbone_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

        # ImageNet normalization buffers (dataset uses [-1, 1])
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize_for_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1, 1] (dataset) to ImageNet-normalized [0, 1]."""
        x = (x + 1.0) * 0.5  # → [0, 1]
        return (x - self.imagenet_mean) / self.imagenet_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB image in [-1, 1]
        Returns:
            feat: (B, out_dim) semantic feature vector
        """
        x = self._normalize_for_imagenet(x)
        x = self.features(x)         # (B, 1280, H', W')
        x = self.pool(x)             # (B, 1280, 1, 1)
        x = x.flatten(1)             # (B, 1280)
        x = self.proj(x)             # (B, out_dim)
        return x

# Forensic Student (Lightweight CNN)
class ForensicStudentCNN(nn.Module):
    """
    Lightweight 5-layer CNN for the 6-channel forensic feature stack.

    Rationale: The forensic stack is already composed of hand-crafted
    signal-processing features (SRM, FFT).
    MobileNetV2's value is its ImageNet pretraining, which does not
    transfer to these engineered signals.  A custom CNN is more
    parameter-efficient (~500K vs ~2.5M) while providing sufficient
    capacity to distill from the ResNet-18 forensic teacher.

    Input:  (B, 6, H, W) — forensic feature stack
    Output: (B, out_dim)   — forensic feature vector
    """

    def __init__(self, in_channels: int = 6, out_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            # Block 1: 6 → 64, downsample 2x
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 64 → 128, downsample 2x
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 128 → 256, downsample 2x
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 256 → 256, downsample 2x
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 5: 256 → 256, global pool
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),       # → (B, 256, 1, 1)
        )

        # Projection (identity dim → keeps things simple, but provides
        # a clean interface matching the semantic branch)
        self.proj = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) forensic feature stack
        Returns:
            feat: (B, out_dim) forensic feature vector
        """
        x = self.net(x)               # (B, 256, 1, 1)
        x = x.flatten(1)              # (B, 256)
        x = self.proj(x)              # (B, out_dim)
        return x

# Student Model (Dual-Domain)
class StudentModel(nn.Module):
    """
    Dual-domain student combining semantic (MobileNetV2) and forensic
    (lightweight CNN) branches with gated fusion.

    Args:
        embed_dim:   fusion/embedding output dimension (default: 256)
        num_classes: number of output classes (default: 2)
        pretrained:  use ImageNet pretrained weights for MobileNetV2 (default: True)

    Forward:
        x_rgb:      (B, 3, H, W)  — RGB image
        x_forensic: (B, 6, H, W) — forensic feature stack

    Returns:
        dict with keys: embedding, logits

    NOTE: Forward signature changed from old (x_forensic, x_grad) to (x_rgb, x_forensic).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()

        # Dual-domain branches
        self.semantic_student = SemanticStudentBranch(pretrained=pretrained, out_dim=embed_dim)
        self.forensic_student = ForensicStudentCNN(in_channels=6, out_dim=embed_dim)

        # Gated fusion (same module as teacher for consistency)
        self.fusion = GatedFusion(feat_dim=embed_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.4)

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.embed_dim = embed_dim

    def forward(self, x_rgb: torch.Tensor, x_forensic: torch.Tensor) -> dict:
        """
        Args:
            x_rgb:      (B, 3, H, W) RGB image
            x_forensic: (B, 6, H, W) forensic feature stack

        Returns:
            dict:
                embedding:     (B, embed_dim) — fused representation
                logits:        (B, num_classes)
        """
        # Dual streams
        E_sem = self.semantic_student(x_rgb)       # (B, embed_dim)
        E_for = self.forensic_student(x_forensic)  # (B, embed_dim)

        # Gated fusion
        E_fused = self.fusion(E_sem, E_for)        # (B, embed_dim)
        E_fused_dropped = self.dropout(E_fused)

        # Classification
        logits = self.classifier(E_fused_dropped)  # (B, num_classes)

        return {
            "embedding": E_fused,
            "logits": logits,
        }