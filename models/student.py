import torch
import torch.nn as nn


# ---------------------------
# Shared CNN block (reuse)
# ---------------------------
class SmallCNN(nn.Module):
    def __init__(self, in_channels, out_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(256, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ---------------------------
# Student Model (Dual Stream)
# ---------------------------
class StudentModel(nn.Module):
    def __init__(
        self,
        forensic_in_channels=12,
        gradient_in_channels=3,
        embed_dim=256,
        num_classes=2
    ):
        super().__init__()

        # Branches
        self.forensic_branch = SmallCNN(forensic_in_channels, out_dim=256)
        self.gradient_branch = SmallCNN(gradient_in_channels, out_dim=256)

        fused_dim = self.forensic_branch.out_dim + self.gradient_branch.out_dim

        # Fusion MLP
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x_forensic, x_grad):
        """
        Args:
            x_forensic: (B,12,H,W)
            x_grad:     (B,3,H,W)

        Returns:
            dict with embedding + logits
        """

        # Dual streams
        E_for = self.forensic_branch(x_forensic)
        E_grad = self.gradient_branch(x_grad)

        # Fusion
        E_s = torch.cat([E_for, E_grad], dim=1)
        E_s = self.mlp(E_s)

        logits = self.classifier(E_s)

        return {
            "embedding": E_s,
            "logits": logits
        }