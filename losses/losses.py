import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 1. Focal Loss (with label smoothing)
# ---------------------------



class ClassificationLoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        # Standard Cross-Entropy natively calibrates probabilities around 0.5
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
 
    def forward(self, logits, targets):
        return self.loss(logits, targets)



