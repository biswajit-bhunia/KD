import torch
import torch.nn as nn


class GeneratorClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_generators: int,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        """
        Args:
            in_dim: embedding dimension (e.g., student embed_dim)
            num_generators: number of generator classes
            hidden_dim: optional MLP hidden size
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_generators)
        )

    def forward(self, x):
        """
        Args:
            x: (B, in_dim)

        Returns:
            logits: (B, num_generators)
        """
        return self.net(x)