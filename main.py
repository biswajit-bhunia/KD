import time
import yaml
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from collections import Counter
from sklearn.model_selection import train_test_split

# Data
from data.dataset import DeepfakeDataset
from data.loader import load_samples
from data.split import find_cross_split_duplicates, find_cross_split_near_duplicates, split_samples

# Models
from models.teacher import TeacherModel
from models.student import StudentModel

# Training
from training.train import (
    train_teacher,
    train_student,
    train_with_grl
)

# Evaluation
from training.validate import evaluate
from features.gradient import GradientExtractor

# Utils
from utils.checkpoint import save_checkpoint
from utils.reproducibility import make_generator, seed_everything, seed_worker


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


def print_metrics(tag, metrics):
    print(
        f"{tag} | "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"Prec: {metrics['precision']:.4f} | "
        f"Rec: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"AUC: {metrics['auc']:.4f}"
    )


def main():
    pipeline_start = time.time()

    # ---------------------------
    # 1. Load config
    # ---------------------------
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    batch_size    = config["batch_size"]
    lr            = config["learning_rate"]
    seed          = config.get("seed", 42)
    deterministic = config.get("deterministic", False)
    seed_everything(seed, deterministic=deterministic)

    lambda_kd      = config["lambda_kd"]
    lambda_supcon  = config["lambda_supcon"]
    lambda_grl     = config["lambda_grl"]
    max_lambda_grl = config.get("max_lambda_grl", 0.3)
    temp_kd        = config.get("temperature_kd", 4.0)
    temp_supcon    = config.get("temperature_supcon", 0.07)
    teacher_epochs = config.get("teacher_epochs", 5)
    student_epochs = config.get("student_epochs", 12)
    grl_epochs     = config.get("grl_epochs", 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Centralized Deepfake Detection Pipeline")
    print(f"  Device: {device} | Batch: {batch_size} | LR: {lr}")
    print(f"  λ_kd: {lambda_kd} | λ_sc: {lambda_supcon} | λ_grl: {lambda_grl}")
    print(f"{'='*60}")

    # ---------------------------
    # 2. Load & Split Dataset
    # ---------------------------
    print("\n  Loading dataset...")
    load_start = time.time()
    samples = load_samples("data/")

    # Previous label-stratified split allowed the same fake generators in train and validation.
    # train_samples, val_samples = train_test_split(
    #     samples,
    #     test_size=0.2,
    #     stratify=[s[1] for s in samples],
    #     random_state=42
    # )
    # Previous two-way generator holdout still used validation as the final report set.
    # train_samples, val_samples = split_samples(
    #     samples,
    #     test_size=0.2,
    #     mode="generator_holdout",
    #     random_state=42
    # )
    # Build name → ID mapping from loaded samples so config can use human-readable names
    import os as _os
    _gen_name_to_id = {}
    for path, label, gen_id in samples:
        if label == 1 and gen_id not in _gen_name_to_id.values():
            gen_name = _os.path.basename(_os.path.dirname(path))
            _gen_name_to_id[gen_name] = gen_id

    def _resolve_generator(name_key):
        name = config.get(name_key, None)
        if name is None:
            return None
        if name not in _gen_name_to_id:
            available = sorted(_gen_name_to_id.keys())
            raise ValueError(f"Config '{name_key}: {name}' not found. Available: {available}")
        return _gen_name_to_id[name]

    val_gen_id  = _resolve_generator("val_generator_name")
    test_gen_id = _resolve_generator("test_generator_name")

    train_samples, val_samples, test_samples, split_info = split_samples(
        samples,
        test_size=0.2,
        mode="generator_holdout_3way",
        holdout_generator_id=val_gen_id,
        test_generator_id=test_gen_id,
        random_state=seed
    )

    n_real = sum(1 for s in samples if s[1] == 0)
    n_fake = sum(1 for s in samples if s[1] == 1)
    n_gens = len(set(s[2] for s in samples if s[1] == 1))
    print(f"  Loaded {len(samples)} samples ({n_real} real, {n_fake} fake, {n_gens} generators) "
          f"in {time.time() - load_start:.1f}s")
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    print(f"  Train fake generators: {', '.join(split_info['train_generators'])}")
    print(f"  Val fake generator:    {split_info['val_generator']}")
    print(f"  Test fake generator:   {split_info['test_generator']}")

    duplicates = find_cross_split_duplicates({
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    })
    if duplicates:
        print(f"  ⚠ Found {len(duplicates)} exact duplicate image hashes across splits.")
        print(f"  ⚠ Example duplicate paths: {duplicates[0][2]}")
    else:
        print("  Exact duplicate check: no cross-split duplicates found.")

    real_near_duplicates = find_cross_split_near_duplicates({
        "train": [s for s in train_samples if s[1] == 0],
        "val": [s for s in val_samples if s[1] == 0],
        "test": [s for s in test_samples if s[1] == 0],
    }, max_hamming_distance=1)
    if real_near_duplicates:
        print(f"  ⚠ Found {len(real_near_duplicates)} possible near-duplicate real-image pairs across splits.")
        print(f"  ⚠ Example near-duplicate paths: {real_near_duplicates[0][1]}")
    else:
        print("  Real near-duplicate check: no likely cross-split duplicates found.")

    train_dataset = DeepfakeDataset(train_samples, augment=True)
    val_dataset   = DeepfakeDataset(val_samples,   augment=False)
    test_dataset  = DeepfakeDataset(test_samples,  augment=False)

    # ---------------------------
    # 3. Dataloaders
    # ---------------------------
    train_labels = [s[1] for s in train_samples]
    counts  = Counter(train_labels)
    weights = [1.0 / counts[l] for l in train_labels]

    train_generator = make_generator(seed)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=train_generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=make_generator(seed + 1),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=make_generator(seed + 2),
    )

    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # ---------------------------
    # 4. Initialize Models
    # ---------------------------
    print("\n  Initializing models...")
    init_start = time.time()
    teacher = TeacherModel().to(device)
    student = StudentModel().to(device)

    # MobileNetV3-Small: ~3-4x faster forward+backward than ResNet18,
    # reduces gradient extraction bottleneck while preserving feature quality
    grad_model     = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(device)
    grad_extractor = GradientExtractor(grad_model).to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"  Teacher: {teacher_params:,} params | Student: {student_params:,} params")
    print(f"  Models initialized in {time.time() - init_start:.1f}s")

    # ---------------------------
    # 5. Optimizers
    # ---------------------------
    teacher_opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    student_opt = torch.optim.Adam(student.parameters(), lr=lr)

    # ---------------------------
    # 6. Stage 1: Train Teacher
    # ---------------------------
    print(f"\n{'─'*60}")
    print(f"  STAGE 1: Training Teacher ({teacher_epochs} epochs)")
    print(f"{'─'*60}")

    stage1_start = time.time()
    train_teacher(
        model=teacher,
        dataloader=train_loader,
        optimizer=teacher_opt,
        device=device,
        epochs=teacher_epochs,
        lambda_supcon=lambda_supcon,
        temperature_supcon=temp_supcon
    )

    print("\n  Evaluating teacher...")
    metrics = evaluate(teacher, val_loader, device)
    print_metrics("  [Teacher]", metrics)

    save_checkpoint(teacher, teacher_opt, epoch=5, path="checkpoints/teacher_stage1.pth")
    best_teacher_auc = metrics["auc"]
    stage1_time = time.time() - stage1_start
    print(f"  → Saved teacher checkpoint (AUC={best_teacher_auc:.4f})")
    print(f"  → Stage 1 complete in {_fmt_time(stage1_time)}")

    # Free GPU memory from Stage 1 before loading gradient extractor
    torch.cuda.empty_cache()

    # ---------------------------
    # 7. Stage 2: Train Student (KD)
    # ---------------------------
    print(f"\n{'─'*60}")
    print(f"  STAGE 2: Training Student with KD ({student_epochs} epochs)")
    print(f"{'─'*60}")

    stage2_start = time.time()
    train_student(
        student=student,
        teacher=teacher,
        dataloader=train_loader,
        optimizer=student_opt,
        device=device,
        grad_model=grad_model,
        epochs=student_epochs,
        lambda_kd=lambda_kd,
        lambda_supcon=lambda_supcon,
        val_loader=val_loader,
        grad_extractor_eval=grad_extractor,
        val_every=3,
        patience=6,
        temperature_kd=temp_kd,
        temperature_supcon=temp_supcon
    )

    print("\n  Evaluating student...")
    metrics = evaluate(student, val_loader, device, grad_extractor)
    print_metrics("  [Student KD]", metrics)

    save_checkpoint(student, student_opt, epoch=12, path="checkpoints/student_stage2.pth")
    best_student_auc = metrics["auc"]
    stage2_time = time.time() - stage2_start
    print(f"  → Saved student checkpoint (AUC={best_student_auc:.4f})")
    print(f"  → Stage 2 complete in {_fmt_time(stage2_time)}")

    # Free GPU memory from Stage 2
    torch.cuda.empty_cache()

    # ---------------------------
    # 8. Stage 3: Generator Invariance (GRL)
    # ---------------------------
    print(f"\n{'─'*60}")
    print(f"  STAGE 3: Training with GRL ({grl_epochs} epochs)")
    print(f"{'─'*60}")

    num_generators = len(set(s[2] for s in train_samples if s[1] == 1))
    print(f"  → Number of train fake generators used for GRL: {num_generators}")

    stage3_start = time.time()
    student_opt_grl = train_with_grl(
        student=student,
        teacher=teacher,
        dataloader=train_loader,
        optimizer=student_opt,
        device=device,
        grad_model=grad_model,
        num_generators=num_generators,
        epochs=grl_epochs,
        lambda_kd=lambda_kd,
        lambda_supcon=lambda_supcon,
        base_lambda_grl=lambda_grl,
        max_lambda_grl=max_lambda_grl,
        val_loader=val_loader,
        grad_extractor_eval=grad_extractor,
        val_every=3,
        patience=2,
        temperature_kd=temp_kd,
        temperature_supcon=temp_supcon
    )

    print("\n  Evaluating student (GRL) on validation for threshold calibration...")
    metrics = evaluate(student, val_loader, device, grad_extractor, calibrate_threshold=True)
    print_metrics("  [GRL Val]", metrics)

    print("\n  Final test evaluation using validation threshold...")
    test_metrics = evaluate(
        student,
        test_loader,
        device,
        grad_extractor,
        threshold=metrics["threshold"],
        calibrate_threshold=False,
    )
    print_metrics("  [GRL Test]", test_metrics)

    save_checkpoint(student, student_opt_grl, epoch=22, path="checkpoints/student_final.pth")
    stage3_time = time.time() - stage3_start
    print(f"  → Saved final student checkpoint (Val AUC={metrics['auc']:.4f} | Test AUC={test_metrics['auc']:.4f})")
    print(f"  → Stage 3 complete in {_fmt_time(stage3_time)}")

    if metrics["auc"] > best_student_auc:
        save_checkpoint(student, student_opt_grl, epoch=20, path="checkpoints/student_best.pth")
        print(f"  → New best student! AUC improved: {best_student_auc:.4f} → {metrics['auc']:.4f}")

    # ---------------------------
    # Final Summary
    # ---------------------------
    total_time = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Stage 1 (Teacher):     {_fmt_time(stage1_time)}")
    print(f"  Stage 2 (Student KD):  {_fmt_time(stage2_time)}")
    print(f"  Stage 3 (GRL):         {_fmt_time(stage3_time)}")
    print(f"  Total pipeline time:   {_fmt_time(total_time)}")
    print(f"  Best Teacher AUC:      {best_teacher_auc:.4f}")
    print(f"  Best Student Val AUC:  {max(best_student_auc, metrics['auc']):.4f}")
    print(f"  Final Student Test AUC:{test_metrics['auc']:.4f}")
    print(f"{'='*60}\n")


# ---------------------------
if __name__ == "__main__":
    main()
