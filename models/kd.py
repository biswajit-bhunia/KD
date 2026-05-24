import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLevelKD(nn.Module):
    """
    Multi-level Knowledge Distillation from a frozen teacher to a student.

    Computes two independent loss terms:
      - "embedding": MSE between L2-normalized feature embeddings
      - "logits":    KL divergence between temperature-softened output distributions

    Weighting between these terms is intentionally delegated to the caller
    (client.py) via lambda_kd and lambda_feat_kd, keeping this module
    responsible only for computing the raw per-level losses.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        temperature: float = 4.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.T = temperature

        # Feature-level: MSE loss on L2-normalized embeddings
        self.feat_loss = nn.MSELoss()

        # Logits-level: KL divergence on softened distributions
        # nn.KLDivLoss(input, target) computes KL(target || input)
        # With input=student and target=teacher this gives KL(teacher || student),
        # which is Hinton et al.'s original KD formulation.
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def _logits_kd(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence between temperature-softened distributions.

        Temperature T is applied to both student and teacher logits before
        softmax. The T² factor compensates for the gradient magnitude
        reduction caused by dividing logits by T (Hinton et al., 2015).
        """
        T = self.T
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        teacher_probs     = F.softmax(teacher_logits / T, dim=1)
        return self.kl_loss(student_log_probs, teacher_probs) * (T * T)

    def forward(
        self,
        student_out: dict,
        teacher_out: dict,
    ) -> dict:
        """
        Compute per-level KD losses.

        Args:
            student_out: dict with keys "embedding" and "logits" from student
            teacher_out: dict with keys "embedding" and "logits" from teacher
                         (teacher tensors are detached inside this function;
                          caller does not need to detach them beforehand)

        Returns:
            dict with keys:
              "embedding": MSE loss on L2-normalized embeddings  (scalar)
              "logits":    KL divergence on softened logits       (scalar)

        Note:
            This module does NOT combine the two terms. The caller (client.py)
            applies lambda_kd and lambda_feat_kd weights explicitly:

                loss = loss_ce
                     + lambda_kd      * kd_losses["logits"]
                     + lambda_feat_kd * kd_losses["embedding"]
        """
        for key in ("embedding", "logits"):
            assert key in student_out, f"MultiLevelKD: missing student key '{key}'"
            assert key in teacher_out, f"MultiLevelKD: missing teacher key '{key}'"

        # Feature-level KD
        # L2 normalization on both sides prevents magnitude collapse and
        # ensures MSE measures angular distance rather than scale difference.
        student_emb = F.normalize(student_out["embedding"], p=2, dim=1)
        teacher_emb = F.normalize(teacher_out["embedding"].detach(), p=2, dim=1)
        loss_embedding = self.feat_loss(student_emb, teacher_emb)

        # Logits-level KD
        loss_logits = self._logits_kd(
            student_out["logits"],
            teacher_out["logits"].detach(),
        )

        return {
            "embedding": loss_embedding,
            "logits":    loss_logits,
        }