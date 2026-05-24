"""
Runtime Defensive Validation Utilities
──────────────────────────────────────
Lightweight assertion checks for training stability.
All checks are gated by a module-level DEBUG flag so they add
negligible overhead in production runs.

Usage:
    from utils.debug_checks import DEBUG, check_loss, check_gradients, check_kd_decomposition
    DEBUG = True   # enable checks (default: False)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

# Master switch — set to True for debug runs, False for production
DEBUG = False

def check_loss(loss: torch.Tensor, name: str = "loss") -> None:
    """Assert loss is finite, scalar, and has gradients attached."""
    if not DEBUG:
        return
    assert loss.dim() == 0, f"[DEBUG] {name} is not scalar: shape={loss.shape}"
    assert torch.isfinite(loss), f"[DEBUG] {name} is not finite: {loss.item()}"
    assert loss.requires_grad, f"[DEBUG] {name} has no gradient graph (detached?)"

def check_tensor(t: torch.Tensor, name: str, expected_shape: Optional[tuple] = None) -> None:
    """Assert tensor is finite and optionally matches expected shape."""
    if not DEBUG:
        return
    assert torch.isfinite(t).all(), f"[DEBUG] {name} contains non-finite values"
    if expected_shape is not None:
        assert t.shape == expected_shape, (
            f"[DEBUG] {name} shape mismatch: expected {expected_shape}, got {t.shape}"
        )

def check_kd_decomposition(kd_losses: Dict[str, torch.Tensor], loss_kd: torch.Tensor,
                            loss_feat: torch.Tensor) -> None:
    """Assert KD losses are properly decomposed (no double-counting)."""
    if not DEBUG:
        return
    # loss_feat should NOT contain logits
    feat_only = kd_losses["embedding"]
    assert torch.allclose(loss_feat, feat_only, atol=1e-6), (
        f"[DEBUG] KD decomposition error: loss_feat={loss_feat.item():.6f} "
        f"!= embedding={feat_only.item():.6f}. "
        f"Possible double-counting of logits KD."
    )
    assert torch.allclose(loss_kd, kd_losses["logits"], atol=1e-6), (
        f"[DEBUG] loss_kd should equal kd_losses['logits']: "
        f"{loss_kd.item():.6f} != {kd_losses['logits'].item():.6f}"
    )
    for k, v in kd_losses.items():
        assert torch.isfinite(v), f"[DEBUG] KD loss '{k}' is not finite: {v.item()}"

def check_gradients(model: nn.Module, tag: str = "model") -> Dict[str, float]:
    """Check gradient health: return stats and warn on issues."""
    if not DEBUG:
        return {}
    stats = {"total_params": 0, "has_grad": 0, "zero_grad": 0, "max_grad_norm": 0.0}
    for name, p in model.named_parameters():
        if p.requires_grad:
            stats["total_params"] += 1
            if p.grad is not None:
                stats["has_grad"] += 1
                grad_norm = p.grad.data.norm().item()
                if grad_norm == 0:
                    stats["zero_grad"] += 1
                stats["max_grad_norm"] = max(stats["max_grad_norm"], grad_norm)
                assert torch.isfinite(p.grad).all(), (
                    f"[DEBUG] {tag}.{name} has non-finite gradients"
                )
    missing = stats["total_params"] - stats["has_grad"]
    if missing > 0:
        print(f"  [DEBUG] {tag}: {missing}/{stats['total_params']} parameters have NO gradient")
    return stats

def check_model_output(out: dict, embed_dim: int, batch_size: int, num_classes: int = 2,
                        tag: str = "model") -> None:
    """Validate the standard model output dictionary."""
    if not DEBUG:
        return
    required_keys = {"embedding", "logits"}
    missing = required_keys - set(out.keys())
    assert not missing, f"[DEBUG] {tag} output missing keys: {missing}"

    check_tensor(out["embedding"], f"{tag}.embedding", (batch_size, embed_dim))
    check_tensor(out["logits"], f"{tag}.logits", (batch_size, num_classes))

def check_forensic_stack(x_for: torch.Tensor, batch_size: int, H: int, W: int) -> None:
    """Validate forensic stack shape and finiteness."""
    if not DEBUG:
        return
    check_tensor(x_for, "forensic_stack", (batch_size, 6, H, W))
    assert x_for.dtype == torch.float32, (
        f"[DEBUG] forensic_stack dtype should be float32, got {x_for.dtype} "
        f"(was it computed inside autocast?)"
    )

def check_teacher_frozen(teacher: nn.Module) -> None:
    """Assert teacher parameters are frozen and model is in eval mode."""
    if not DEBUG:
        return
    assert not teacher.training, "[DEBUG] Teacher is not in eval mode"
    for name, p in teacher.named_parameters():
        assert not p.requires_grad, f"[DEBUG] Teacher parameter '{name}' is not frozen"

def print_loss_decomposition(ce: float, kd: float, feat: float,
                              total: float, lambda_kd: float, lambda_feat: float) -> None:
    """Print detailed loss breakdown for debugging."""
    if not DEBUG:
        return
    print(f"    [DEBUG] Loss decomposition:")
    print(f"      CE:     {ce:.4f} (weight: 1.0)")
    print(f"      KD:     {kd:.4f} (weight: {lambda_kd}  → contrib: {lambda_kd * kd:.4f})")
    print(f"      Feat:   {feat:.4f} (weight: {lambda_feat} → contrib: {lambda_feat * feat:.4f})")
    print(f"      Total:  {total:.4f}")
