"""
Training functions for the dual-domain deepfake detection pipeline.

Stage 1: Train Teacher (centralized)
Stage 2: Train Student with multi-level KD (centralized)
Stage 3: GRL-based generator invariance (optional, disabled by default)

All stages use the unified forward signature:
    model(x_rgb, x_forensic) → dict with "semantic_feat", "forensic_feat",
                                        "embedding", "logits"
"""

import math
import time
import torch
import torch.nn as nn

from utils.debug_checks import (
    check_loss, check_kd_decomposition, check_forensic_stack,
    check_teacher_frozen, check_gradients, DEBUG as _DEBUG,
)

from features.forensic import build_forensic_stack

from models.kd import MultiLevelKD

from losses.losses import (
    ClassificationLoss
)

def move_to_device(batch, device):
    return {
        "image":  batch["image"].to(device),
        "label":  batch["label"].to(device),
        "gen_id": batch["gen_id"].to(device)
    }

def _fmt_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"

def _quick_validate(model, val_loader, device):
    """
    Run a fast validation pass and return AUC.
    Imports evaluate lazily to avoid circular imports.
    """
    from training.validate import evaluate
    metrics = evaluate(model, val_loader, device, calibrate_threshold=True)
    return metrics

def _health_check(tag, epoch, lr, loss_history, ce, kd, supcon, base_lr):
    """
    After epoch 1, print a health report to catch misconfigurations early.
    """
    issues = []

    # Check LR is reasonable
    if lr < base_lr * 0.01:
        issues.append(f"⚠ LR={lr:.6f} is very low (expected ~{base_lr:.6f}). Scheduler may not have reset.")

    # Check loss is not NaN/Inf
    if len(loss_history) > 0 and (loss_history[-1] != loss_history[-1]):  # NaN check
        issues.append("⚠ Loss is NaN! Check for numerical instability.")

    # Check CE is reasonable for binary classification
    if ce > 0.7:
        issues.append(f"⚠ CE={ce:.4f} is high — model is near random (expected < 0.65).")

    # Check loss is decreasing (after 2+ epochs)
    if len(loss_history) >= 2:
        delta = loss_history[-1] - loss_history[-2]
        if delta > 0:
            issues.append(f"⚠ Loss INCREASED by {delta:.4f} — training may be unstable.")

    if issues:
        print(f"\n  ⚡ {tag} Health Check (Epoch {epoch}):")
        for issue in issues:
            print(f"    {issue}")
        print()
    else:
        print(f"  ✓ {tag} Health check passed (Epoch {epoch}) — LR, loss, and components look normal.\n")

# STAGE 1: Train Teacher
def train_teacher(
    model,
    dataloader,
    optimizer,
    device,
    epochs=5,
    health_check_every=3,
):
    cls_loss = ClassificationLoss()

    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    model.train()
    stage_start = time.time()
    num_batches = len(dataloader)

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            batch = move_to_device(batch, device)

            x_rgb  = batch["image"]
            labels = batch["label"]

            x_for = build_forensic_stack(x_rgb)

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                out = model(x_rgb, x_for)

                loss_ce  = cls_loss(out["logits"], labels)

                loss = loss_ce

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        lr_now = optimizer.param_groups[0]["lr"]
        remaining = epoch_time * (epochs - epoch - 1)
        print(f"  [Teacher] Epoch {epoch+1}/{epochs} done | "
              f"Loss: {avg_loss:.4f} | LR: {lr_now:.6f} | "
              f"Time: {_fmt_time(epoch_time)} | ETA: {_fmt_time(remaining)}")

        # Periodic health check (not just epoch 1)
        if (epoch + 1) % health_check_every == 0:
            _health_check("[Teacher]", epoch + 1, lr_now, [avg_loss],
                          avg_loss, 0, 0, optimizer.defaults['lr'])

    total_time = time.time() - stage_start
    print(f"  [Teacher] Training complete in {_fmt_time(total_time)}")

# STAGE 2: Train Student (Multi-Level KD)
def train_student(
    student,
    teacher,
    dataloader,
    optimizer,
    device,
    epochs=5,
    lambda_kd=0.3,
    lambda_feat_kd=0.5,
    val_loader=None,
    val_every=3,
    patience=2,
    health_check_every=3,
    temperature_kd=4.0,
):
    """
    Train student with multi-level knowledge distillation from teacher.

    KD pathways:
      - Semantic feature KD (MSE)
      - Forensic feature KD (MSE)
      - Embedding KD (MSE)
      - Logits KD (KL-div)
    """
    cls_loss = ClassificationLoss()
    multi_kd = MultiLevelKD(temperature=temperature_kd)

    # Freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    # Reset optimizer LR — previous stage's scheduler leaves lr decayed
    # and sets 'initial_lr' which corrupts new scheduler's cosine cycle
    base_lr = optimizer.defaults['lr']
    for group in optimizer.param_groups:
        group['lr'] = base_lr
        group.pop('initial_lr', None)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    student.train()
    stage_start = time.time()
    num_batches = len(dataloader)

    best_val_auc = -1.0          # -1 sentinel so first check is always "new best"
    best_model_state = None
    checks_without_improvement = 0  # counts validation rounds, not epochs
    loss_history = []

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        total_ce   = 0
        total_kd   = 0
        total_feat = 0

        student.train()

        for batch_idx, batch in enumerate(dataloader):
            batch = move_to_device(batch, device)

            x_rgb  = batch["image"]
            labels = batch["label"]

            x_for = build_forensic_stack(x_rgb)

            with torch.no_grad():
                teacher_out = teacher(x_rgb, x_for)

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                student_out = student(x_rgb, x_for)

                loss_ce  = cls_loss(student_out["logits"], labels)

                # Multi-level KD
                kd_losses = multi_kd(student_out, teacher_out)
                loss_kd   = kd_losses["logits"]      # logits KD
                # Feature-only KD (excludes logits to avoid double-counting)
                loss_feat = kd_losses["embedding"]

                loss = (
                    loss_ce
                    + lambda_kd      * loss_kd
                    + lambda_feat_kd * loss_feat
                )

            # --- Debug checks (gated by DEBUG flag, zero overhead in production) ---
            if _DEBUG and batch_idx == 0:
                check_forensic_stack(x_for, x_rgb.shape[0], x_rgb.shape[2], x_rgb.shape[3])
                check_teacher_frozen(teacher)
                check_kd_decomposition(kd_losses, loss_kd, loss_feat)
                check_loss(loss, "total_loss")

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

            # --- Debug: gradient health on first batch of each epoch ---
            if _DEBUG and batch_idx == 0:
                check_gradients(student, tag=f"student_epoch{epoch+1}")

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_ce   += loss_ce.item()
            total_kd   += loss_kd.item()
            total_feat += loss_feat.item()

        scheduler.step()

        epoch_time = time.time() - epoch_start
        n = num_batches
        avg_loss_val = total_loss / n
        avg_ce = total_ce / n
        avg_kd = total_kd / n
        avg_feat = total_feat / n
        lr_now = optimizer.param_groups[0]["lr"]
        remaining = epoch_time * (epochs - epoch - 1)
        loss_history.append(avg_loss_val)

        print(
            f"  [Student KD] Epoch {epoch+1}/{epochs} done | "
            f"CE: {avg_ce:.4f} | KD: {avg_kd:.4f} | "
            f"Feat: {avg_feat:.4f} | "
            f"Total: {avg_loss_val:.4f} | "
            f"LR: {lr_now:.6f} | Time: {_fmt_time(epoch_time)} | "
            f"ETA: {_fmt_time(remaining)}"
        )

        # Periodic health check
        if (epoch + 1) % health_check_every == 0:
            _health_check("[Student KD]", epoch + 1, lr_now, loss_history,
                          avg_ce, avg_kd, 0, base_lr)

        # Periodic validation & early stopping
        if val_loader is not None and (epoch + 1) % val_every == 0:
            print(f"  📊 Mid-training validation (Epoch {epoch+1})...")
            metrics = _quick_validate(student, val_loader, device)
            val_auc = metrics["auc"]
            print(f"  📊 AUC: {val_auc:.4f} | Acc: {metrics['accuracy']:.4f} | "
                  f"Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

            if val_auc > best_val_auc:
                is_first = best_val_auc < 0
                improvement = val_auc - max(best_val_auc, 0.0)
                best_val_auc = val_auc
                best_model_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
                checks_without_improvement = 0
                if is_first:
                    print(f"  📊 ✓ First checkpoint — AUC: {best_val_auc:.4f}")
                else:
                    print(f"  📊 ✓ Improved by {improvement:.4f} — best AUC so far: {best_val_auc:.4f}")
            else:
                checks_without_improvement += 1
                print(f"  📊 ✗ No improvement for {checks_without_improvement}/{patience} checks "
                      f"(best: {best_val_auc:.4f})")

                if checks_without_improvement >= patience:
                    print(f"  📊 ⚡ Early stopping triggered after {patience} consecutive bad checks.")
                    print(f"  📊 Stopping at epoch {epoch+1}/{epochs}")
                    break

            student.train()  # Back to training mode

    total_time = time.time() - stage_start
    print(f"  [Student KD] Training complete in {_fmt_time(total_time)}")

    if best_model_state is not None:
        print(f"  [Student KD] Restoring best weights (AUC: {best_val_auc:.4f})")
        student.load_state_dict(best_model_state)

