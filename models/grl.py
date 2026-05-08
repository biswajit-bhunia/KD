# pyrefly: ignore [missing-import]
import torch
from torch.autograd import Function


class GRL(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Identity in forward pass
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Reverse gradient in backward pass
        """
        return -ctx.lambda_ * grad_output, None


# Optional wrapper for easier use
class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GRL.apply(x, self.lambda_)