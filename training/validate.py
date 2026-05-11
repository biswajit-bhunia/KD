import time
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve
)
from features.forensic import build_forensic_stack


def find_optimal_threshold(labels, probs):
    """
    Find the threshold that maximises F1 using the precision-recall curve.

    Uses sklearn's precision_recall_curve which computes *exact* precision and
    recall at every operating point.  The old implementation used a ROC-based
    approximation (precision ≈ tpr/(tpr+fpr)) that is only valid for balanced
    classes — it systematically overestimated precision on skewed splits.

    ⚠ Leakage note: the threshold is searched on the *same* data it is then
    applied to, so Acc / F1 / Recall at this threshold are optimistically
    biased.  AUC (computed separately, threshold-free) is unaffected and
    remains the reliable metric for model comparison.
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(labels, probs)
    # precision_recall_curve appends a sentinel point (precision=1, recall=0)
    # with no corresponding threshold, so slice [:-1] to align arrays.
    f1_arr = (
        2 * precision_arr[:-1] * recall_arr[:-1]
        / (precision_arr[:-1] + recall_arr[:-1] + 1e-8)
    )
    if len(f1_arr) == 0:
        return 0.5, 0.0
    best_idx = int(np.argmax(f1_arr))
    return float(thresholds[best_idx]), float(f1_arr[best_idx])


def evaluate(
    model,
    dataloader,
    device,
    grad_extractor=None,
    threshold=None,
    calibrate_threshold=True,
):
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    num_batches = len(dataloader)
    eval_start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x      = batch["image"].to(device)
            labels = batch["label"].to(device)

            x_for = build_forensic_stack(x)

            if grad_extractor is None:
                out = model(x, x_for)
            else:
                with torch.enable_grad():
                    x_grad = grad_extractor(x)
                out = model(x_for, x_grad)

            logits = out["logits"]
            probs  = torch.softmax(logits, dim=1)[:, 1]

            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            # Progress at 50% and 100%
            if (batch_idx + 1) == num_batches or (batch_idx + 1) == num_batches // 2:
                elapsed = time.time() - eval_start
                pct = (batch_idx + 1) / num_batches * 100
                print(f"    [Eval] Batch {batch_idx+1}/{num_batches} ({pct:.0f}%) | "
                      f"Elapsed: {elapsed:.1f}s")

    eval_time = time.time() - eval_start

    labels_np = np.array(all_labels)
    probs_np = np.array(all_probs)

    if threshold is not None:
        active_threshold = float(threshold)
        threshold_source = "provided"
    elif calibrate_threshold:
        active_threshold, _ = find_optimal_threshold(labels_np, probs_np)
        threshold_source = "val-calibrated"
    else:
        active_threshold = 0.5
        threshold_source = "default"

    # Compute metrics at BOTH thresholds
    preds_default  = (probs_np > 0.5).astype(int)
    preds_active   = (probs_np > active_threshold).astype(int)

    # Use active threshold for reported metrics
    acc       = accuracy_score(labels_np, preds_active)
    precision = precision_score(labels_np, preds_active, zero_division=0)
    recall    = recall_score(labels_np, preds_active, zero_division=0)
    f1        = f1_score(labels_np, preds_active, zero_division=0)

    try:
        auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        auc = 0.0

    # Also compute at default 0.5 for comparison
    acc_05  = accuracy_score(labels_np, preds_default)
    rec_05  = recall_score(labels_np, preds_default, zero_division=0)
    f1_05   = f1_score(labels_np, preds_default, zero_division=0)

    print(f"    [Eval] Complete ({len(labels_np)} samples in {eval_time:.1f}s)")
    print(f"    [Eval] Threshold: {active_threshold:.3f} "
          f"({threshold_source}; default 0.5)  AUC is threshold-free")
    if abs(active_threshold - 0.5) > 0.05:
        print(f"    [Eval] At t=0.5:   Acc={acc_05:.4f} | Rec={rec_05:.4f} | F1={f1_05:.4f}")
        print(f"    [Eval] At t={active_threshold:.3f}: Acc={acc:.4f} | Rec={recall:.4f} | F1={f1:.4f} ← using this")

    return {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "auc":       auc,
        "threshold": active_threshold
    }
