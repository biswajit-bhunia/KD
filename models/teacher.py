"""
Dual-Domain Teacher Model
─────────────────────────
Architecture:
  Semantic Teacher  — ResNet-50 on RGB (B,3,H,W)
  Forensic Teacher  — ResNet-18 on forensic stack (B,12,H,W)
  Fusion            — GatedFusion (learned sigmoid gate)
  Classifier        — Linear(embed_dim, 2)

The teacher exposes per-branch features for multi-level knowledge
distillation into the student network.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from models.fusion import GatedFusion


# ---------------------------
# Semantic Teacher (ResNet-50)
# ---------------------------
class SemanticTeacher(nn.Module):
    """
    ResNet-50 backbone for RGB images.

    Purpose: semantic reasoning, facial structure consistency,
    global visual representations, manipulation semantics.

    Input:  (B, 3, H, W) — RGB image normalized to [-1, 1]
    Output: (B, out_dim)  — semantic feature vector
    """

    def __init__(self, pretrained: bool = True, out_dim: int = 256):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)

        # Remove classifier (fc layer) — keep everything up to avgpool
        self.features = nn.Sequential(*list(model.children())[:-1])  # → (B, 2048, 1, 1)
        self.backbone_dim = 2048

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
        x = self.features(x)         # (B, 2048, 1, 1)
        x = x.flatten(1)             # (B, 2048)
        x = self.proj(x)             # (B, out_dim)
        return x


# ---------------------------
# Forensic Teacher (ResNet-18)
# ---------------------------
class ForensicTeacher(nn.Module):
    """
    ResNet-18 backbone modified for 12-channel forensic input.

    Purpose: artifact learning, frequency inconsistencies,
    manipulation traces, forensic texture representations.

    Input:  (B, 12, H, W) — forensic feature stack (SRM + FFT + wavelet + Laplacian)
    Output: (B, out_dim)   — forensic feature vector

    The first conv layer is replaced with a 12-channel version.
    Pretrained weights are transferred by repeating the 3-ch weights 4x.
    """

    def __init__(self, pretrained: bool = True, out_dim: int = 256):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)

        # Replace first conv: 3 → 12 channels
        old_conv = model.conv1  # Conv2d(3, 64, 7, stride=2, padding=3)
        new_conv = nn.Conv2d(
            12, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Transfer pretrained weights: repeat 3-ch weights 4 times for 12 channels
        if pretrained and old_conv.weight is not None:
            with torch.no_grad():
                # old_conv.weight: (64, 3, 7, 7) → repeat along channel dim → (64, 12, 7, 7)
                # Scale by 1/sqrt(4) to preserve activation variance: conv sums over
                # input channels, so repeating identical weights 4× inflates variance
                # by 4×. Dividing by sqrt(4) = 2 restores the pretrained activation scale.
                new_conv.weight.copy_(old_conv.weight.repeat(1, 4, 1, 1) / (4 ** 0.5))
        model.conv1 = new_conv

        # Remove classifier — keep everything up to avgpool
        self.features = nn.Sequential(*list(model.children())[:-1])  # → (B, 512, 1, 1)
        self.backbone_dim = 512

        # Projection to shared embedding space
        self.proj = nn.Sequential(
            nn.Linear(self.backbone_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, H, W) forensic feature stack
        Returns:
            feat: (B, out_dim) forensic feature vector
        """
        x = self.features(x)         # (B, 512, 1, 1)
        x = x.flatten(1)             # (B, 512)
        x = self.proj(x)             # (B, out_dim)
        return x


# ---------------------------
# Teacher Model (Dual-Domain)
# ---------------------------
class TeacherModel(nn.Module):
    """
    Dual-domain teacher combining semantic (ResNet-50) and forensic (ResNet-18)
    branches with gated fusion.

    Args:
        pretrained:  use ImageNet pretrained weights (default: True)
        embed_dim:   fusion/embedding output dimension (default: 256)
        num_classes: number of output classes (default: 2)

    Forward:
        x_rgb:      (B, 3, H, W)  — RGB image
        x_forensic: (B, 12, H, W) — forensic feature stack

    Returns:
        dict with keys: semantic_feat, forensic_feat, embedding, logits
    """

    def __init__(
        self,
        pretrained: bool = True,
        embed_dim: int = 256,
        num_classes: int = 2,
    ):
        super().__init__()

        # Dual-domain backbones
        self.semantic = SemanticTeacher(pretrained=pretrained, out_dim=embed_dim)
        self.forensic = ForensicTeacher(pretrained=pretrained, out_dim=embed_dim)

        # Gated fusion
        self.fusion = GatedFusion(feat_dim=embed_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.embed_dim = embed_dim

    def forward(self, x_rgb: torch.Tensor, x_forensic: torch.Tensor) -> dict:
        """
        Args:
            x_rgb:      (B, 3, H, W) RGB image
            x_forensic: (B, 12, H, W) forensic feature stack

        Returns:
            dict:
                semantic_feat: (B, embed_dim)
                forensic_feat: (B, embed_dim)
                embedding:     (B, embed_dim)
                logits:        (B, num_classes)
        """
        # Dual streams
        E_sem = self.semantic(x_rgb)         # (B, embed_dim)
        E_for = self.forensic(x_forensic)    # (B, embed_dim)

        # Gated fusion
        E_fused = self.fusion(E_sem, E_for)  # (B, embed_dim)
        E_fused = self.dropout(E_fused)

        # Classification
        logits = self.classifier(E_fused)    # (B, num_classes)

        return {
            "semantic_feat": E_sem,
            "forensic_feat": E_for,
            "embedding": E_fused,
            "logits": logits,
        }
