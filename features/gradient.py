import torch
import torch.nn as nn


class GradientExtractor(nn.Module):
    def __init__(self, model: nn.Module):
        """
        Args:
            model: frozen feature model (ResNet / CLIP / discriminator)
        """
        super().__init__()
        self.model = model
        self._freeze_model()
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _freeze_model(self):
        """
        Freeze all parameters so no updates happen.
        """
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Compute the gradient of the frozen model's output w.r.t. the input.

        The input is intentionally .detach()-ed so the gradient map is a
        fixed feature (like SRM/FFT), not a differentiable transform.
        The student learns to interpret the static gradient pattern but
        cannot back-propagate through the extraction itself.

        Args:
            x: (B,3,H,W)

        Returns:
            grad: (B,3,H,W) gradient w.r.t input
        """
        # Detach severs the graph intentionally (see docstring)
        x = x.clone().detach().requires_grad_(True)

        # Dataset images are normalized to [-1, 1]; pretrained ImageNet models expect ImageNet stats on [0, 1].
        x_model = (x + 1.0) * 0.5
        x_model = (x_model - self.imagenet_mean) / self.imagenet_std

        # Forward pass through frozen model
        output = self.model(x_model)

        # Handle different output formats
        if isinstance(output, (tuple, list)):
            output = output[0]

        # Ensure scalar loss
        loss = output.sum()

        # Compute gradients w.r.t input
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=x,
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]

        return grad
