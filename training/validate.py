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

import torchvision.transforms.functional as TF

def evaluate(
    model,
    dataloader,
    device,
    threshold=None,
    calibrate_threshold=False,
    use_tta=False,
):
    """
    Evaluate a model (teacher or student) on a dataloader.

    Both teacher and student now share the same forward signature:
        model(x_rgb, x_forensic) → dict with "logits"

    No gradient extractor is needed — the dual-domain architecture uses
    (RGB, forensic_stack) as inputs for both networks.

    Args:
        model:       TeacherModel or StudentModel
        dataloader:  validation/test DataLoader
        device:      torch.device
        threshold:   optional fixed threshold; if None, calibrated from data
        calibrate_threshold: whether to search for optimal threshold
        use_tta:     whether to use Test-Time Augmentation (5-crop)
    """
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []
    all_gen_names = []
    all_video_ids = []

    num_batches = len(dataloader)
    eval_start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x      = batch["image"].to(device)
            labels = batch["label"].to(device)

            if not use_tta:
                # Build forensic stack from RGB
                x_for = build_forensic_stack(x)

                # Unified forward: both teacher and student use (x_rgb, x_forensic)
                out = model(x, x_for)
                probs = torch.softmax(out["logits"], dim=1)[:, 1]
            else:
                aug_probs = []
                augs = [
                    lambda img: img,                                         # Original
                    lambda img: TF.hflip(img),                               # Horizontal flip
                    lambda img: TF.gaussian_blur(img, kernel_size=[3, 3]),   # Mild blur
                    lambda img: torch.clamp(img * 1.1, -1.0, 1.0),           # Brighter
                    lambda img: torch.clamp(img * 0.9, -1.0, 1.0),           # Darker
                ]
                for aug_fn in augs:
                    x_aug = aug_fn(x)
                    x_for_aug = build_forensic_stack(x_aug)
                    out_aug = model(x_aug, x_for_aug)
                    aug_probs.append(torch.softmax(out_aug["logits"], dim=1)[:, 1])
                
                # Average probabilities across all augmentations
                probs = torch.stack(aug_probs, dim=0).mean(dim=0)

            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_gen_names.extend(batch["gen_name"])
            all_video_ids.extend(batch["video_id"])

            # Progress at 50% and 100%
            if (batch_idx + 1) == num_batches or (batch_idx + 1) == num_batches // 2:
                elapsed = time.time() - eval_start
                pct = (batch_idx + 1) / num_batches * 100
                print(f"    [Eval] Batch {batch_idx+1}/{num_batches} ({pct:.0f}%) | "
                      f"Elapsed: {elapsed:.1f}s")

    eval_time = time.time() - eval_start

    # --- VIDEO-LEVEL AGGREGATION ---
    from collections import defaultdict
    video_to_probs = defaultdict(list)
    video_to_label = {}
    video_to_gen = {}

    for v_id, prob, lbl, gen in zip(all_video_ids, all_probs, all_labels, all_gen_names):
        video_to_probs[v_id].append(prob)
        video_to_label[v_id] = lbl
        video_to_gen[v_id] = gen

    vid_labels_list = []
    vid_probs_list = []
    vid_gen_names_list = []

    for v_id, probs_list in video_to_probs.items():
        vid_probs_list.append(np.mean(probs_list))
        vid_labels_list.append(video_to_label[v_id])
        vid_gen_names_list.append(video_to_gen[v_id])

    labels_np = np.array(vid_labels_list)
    probs_np = np.array(vid_probs_list)
    gen_names_np = np.array(vid_gen_names_list)
    num_frames = len(all_labels)
    num_videos = len(labels_np)
    # -------------------------------

    if threshold is not None:
        eval_threshold = float(threshold)
        threshold_source = "provided"
    else:
        eval_threshold = 0.5
        threshold_source = "default"

    if calibrate_threshold:
        calibrated_threshold, _ = find_optimal_threshold(labels_np, probs_np)
        eval_threshold = calibrated_threshold
        threshold_source = "calibrated"
    else:
        threshold_source = "provided"

    returned_threshold = eval_threshold

    print(f"    [Eval] Complete ({num_frames} frames -> {num_videos} videos in {eval_time:.1f}s)")
    print(f"    [Eval] Eval Threshold: {eval_threshold:.3f} ({threshold_source})")

    preds_eval = (probs_np > eval_threshold).astype(int)

    acc       = accuracy_score(labels_np, preds_eval)
    precision = precision_score(labels_np, preds_eval, zero_division=0)
    recall    = recall_score(labels_np, preds_eval, zero_division=0)
    f1        = f1_score(labels_np, preds_eval, zero_division=0)

    try:
        auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        auc = 0.0

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels_np, preds_eval)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Threshold: {eval_threshold:.3f})")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Per-generator evaluation
    print(f"\n    [Eval] --- Per-Generator Breakdown (Video-Level) ---")
    unique_gens = set(gen_names_np)
    for gen in unique_gens:
        if gen == "real":
            continue
        mask = (gen_names_np == gen) | (gen_names_np == "real")
        gen_labels = labels_np[mask]
        gen_probs = probs_np[mask]
        gen_preds = preds_eval[mask]
        
        if len(np.unique(gen_labels)) > 1:
            try:
                g_auc = roc_auc_score(gen_labels, gen_probs)
            except ValueError:
                g_auc = 0.0
            g_f1 = f1_score(gen_labels, gen_preds, zero_division=0)
            g_acc = accuracy_score(gen_labels, gen_preds)
            print(f"    [Eval] {gen:15s} | Acc: {g_acc:.4f} | F1: {g_f1:.4f} | AUC: {g_auc:.4f}")

    return {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "auc":       auc,
        "threshold": returned_threshold
    }
