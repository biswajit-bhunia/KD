import torch
import torch.nn as nn


class FedProxLoss(nn.Module):
    """
    Proximal regularization term for FedProx.

    Penalizes the local model for drifting too far from the global model
    during local training. This is critical for non-IID settings where
    each client sees different generators — without it, client models
    diverge and FedAvg produces a poor global model.

    Loss = (mu / 2) * ||w_local - w_global||^2
    """

    def __init__(self, mu: float = 0.01):
        """
        Args:
            mu: proximal term weight. Higher = more regularization toward
                the global model. Typical range: 0.001 - 0.1.
        """
        super().__init__()
        self.mu = mu

    def forward(self, local_model: nn.Module, global_params: dict) -> torch.Tensor:
        """
        Args:
            local_model: the client's local model being trained
            global_params: snapshot of global model state_dict (detached)

        Returns:
            proximal penalty scalar
        """
        proximal_term = torch.tensor(0.0, device=next(local_model.parameters()).device)

        for name, param in local_model.named_parameters():
            if param.requires_grad and name in global_params:
                global_weight = global_params[name].to(param.device)
                proximal_term += ((param - global_weight) ** 2).sum()

        return (self.mu / 2.0) * proximal_term
