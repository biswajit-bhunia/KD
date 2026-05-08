# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torchvision.models as models
# pyrefly: ignore [missing-import]
from torchvision.models import ResNet18_Weights, ResNet50_Weights


# ---------------------------
# Semantic Backbone (ResNet)
# ---------------------------
class ResNetBackbone(nn.Module):
    def __init__(self, name="resnet18", pretrained=True, out_dim=512):
        super().__init__()

        # Issue #5: use the modern weights API instead of deprecated pretrained=True
        if name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet18(weights=weights)
            dim = 512
        elif name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
            dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        # Remove classifier
        self.features = nn.Sequential(*list(model.children())[:-1])  # (B, dim, 1, 1)
        self.out_dim = dim

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize_for_imagenet(self, x):
        # Dataset images are normalized to [-1, 1]; pretrained ResNet expects ImageNet stats on [0, 1].
        x = (x + 1.0) * 0.5
        return (x - self.imagenet_mean) / self.imagenet_std

    def forward(self, x):
        x = self._normalize_for_imagenet(x)
        x = self.features(x)
        x = x.flatten(1)
        return x


# ---------------------------
# Forensic Backbone (light CNN)
# ---------------------------
class ForensicCNN(nn.Module):
    def __init__(self, in_channels=12, out_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(128, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ---------------------------
# Teacher Model
# ---------------------------
class TeacherModel(nn.Module):
    def __init__(
        self,
        semantic_backbone="resnet18",
        pretrained=True,
        embed_dim=512,
        num_classes=2
    ):
        super().__init__()

        # Backbones
        self.semantic = ResNetBackbone(semantic_backbone, pretrained)
        self.forensic = ForensicCNN(in_channels=12)

        fused_dim = self.semantic.out_dim + self.forensic.out_dim

        # Fusion MLP
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x_rgb, x_forensic):
        """
        Args:
            x_rgb:       (B,3,H,W)
            x_forensic:  (B,12,H,W)

        Returns:
            dict with embedding + logits
        """

        # Dual streams
        E_sem = self.semantic(x_rgb)
        E_for = self.forensic(x_forensic)

        # Fusion
        E_t = torch.cat([E_sem, E_for], dim=1)
        E_t = self.mlp(E_t)

        logits = self.classifier(E_t)

        return {
            "embedding": E_t,
            "logits": logits
        }
