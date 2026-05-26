import time
import yaml
import torch
from torch.utils.data import DataLoader
from collections import Counter
from data.dataset import DeepfakeDataset
from data.loader import load_samples

from models.teacher import TeacherModel
from models.student import StudentModel

from training.train import (
    train_teacher,
    train_student
)

from training.validate import evaluate

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

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    batch_size    = config["batch_size"]
    lr            = config["learning_rate"]
    seed          = config.get("seed", 42)
    deterministic = config.get("deterministic", False)
    seed_everything(seed, deterministic=deterministic)
    
    aug_grayscale_p = config.get("aug_grayscale_p", 0.2)
    aug_color_jitter_p = config.get("aug_color_jitter_p", 0.3)

    lambda_kd      = config["lambda_kd"]
    lambda_feat_kd = config.get("lambda_feat_kd", 0.5)
    temp_kd        = config.get("temperature_kd", 4.0)
    teacher_epochs = config.get("teacher_epochs", 5)
    student_epochs = config.get("student_epochs", 12)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Centralized Deepfake Detection Pipeline (Dual-Domain)")
    print(f"  Device: {device} | Batch: {batch_size} | LR: {lr}")
    print(f"  λ_kd: {lambda_kd} | λ_feat: {lambda_feat_kd}")
    print(f"{'='*60}")

    train_roots = config.get("train_roots", [config.get("train_root", "D:\\ff++_extracted")])
    test_data_dir  = config.get("test_root", "D:\\DFDC_EXTRACTED")
    val_split_ratio = config.get("val_split_ratio", 0.2)

    ffpp_samples = []
    load_start = time.time()
    for tr in train_roots:
        print(f"\n  Loading Train/Val Dataset from {tr}...")
        tr_samples = load_samples(tr)
        print(f"  Loaded {len(tr_samples)} samples from {tr}")
        ffpp_samples.extend(tr_samples)
    
    print(f"  Total Train/Val samples loaded: {len(ffpp_samples)} in {time.time() - load_start:.1f}s")

    print(f"  Loading Test Dataset from {test_data_dir}...")
    load_start = time.time()
    test_samples = load_samples(test_data_dir)
    print(f"  Loaded {len(test_samples)} test samples in {time.time() - load_start:.1f}s")

    from data.split import split_by_video_identity
    train_samples, val_samples = split_by_video_identity(ffpp_samples, test_size=val_split_ratio, random_state=seed)

    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    train_dataset = DeepfakeDataset(
        train_samples, augment=True, 
        aug_grayscale_p=aug_grayscale_p, 
        aug_color_jitter_p=aug_color_jitter_p
    )
    val_dataset   = DeepfakeDataset(val_samples,   augment=False)
    test_dataset  = DeepfakeDataset(test_samples,  augment=False)

    train_labels = [s[1] for s in train_samples]
    counts  = Counter(train_labels)
    class_weights = torch.tensor([1.0 / counts[0], 1.0 / counts[1]], dtype=torch.float32).to(device)

    def _build_train_loader(dataset, loader_seed):
        gen = make_generator(loader_seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=gen,
        )

    teacher_train_loader = _build_train_loader(train_dataset, seed)
    student_train_loader = _build_train_loader(train_dataset, seed + 100)

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

    print(f"  Train batches: {len(teacher_train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    print("\n  Initializing models...")
    init_start = time.time()
    teacher = TeacherModel().to(device)
    student = StudentModel().to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"  Teacher: {teacher_params:,} params | Student: {student_params:,} params")
    print(f"  Models initialized in {time.time() - init_start:.1f}s")

    teacher_opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    student_opt = torch.optim.Adam(student.parameters(), lr=lr)

    print(f"\n{'─'*60}")
    print(f"  STAGE 1: Training Teacher ({teacher_epochs} epochs)")
    print(f"{'─'*60}")

    stage1_start = time.time()
    train_teacher(
        model=teacher,
        dataloader=teacher_train_loader,
        optimizer=teacher_opt,
        device=device,
        epochs=teacher_epochs,
        class_weights=class_weights,
    )

    print("\n  Evaluating teacher...")
    metrics = evaluate(teacher, val_loader, device, calibrate_threshold=True)
    print_metrics("  [Teacher]", metrics)

    save_checkpoint(teacher, teacher_opt, epoch=teacher_epochs, path="checkpoints/teacher_stage1.pth")
    best_teacher_auc = metrics["auc"]
    stage1_time = time.time() - stage1_start
    print(f"  → Saved teacher checkpoint (AUC={best_teacher_auc:.4f})")
    print(f"  → Stage 1 complete in {_fmt_time(stage1_time)}")

    torch.cuda.empty_cache()

    print(f"\n{'─'*60}")
    print(f"  STAGE 2: Training Student with Multi-Level KD ({student_epochs} epochs)")
    print(f"{'─'*60}")

    stage2_start = time.time()
    train_student(
        student=student,
        teacher=teacher,
        dataloader=student_train_loader,
        optimizer=student_opt,
        device=device,
        epochs=student_epochs,
        lambda_kd=lambda_kd,
        lambda_feat_kd=lambda_feat_kd,
        val_loader=val_loader,
        val_every=3,
        patience=6,
        temperature_kd=temp_kd,
        class_weights=class_weights,
    )

    print("\n  Evaluating student...")
    metrics = evaluate(student, val_loader, device, calibrate_threshold=True)
    print_metrics("  [Student KD]", metrics)

    save_checkpoint(student, student_opt, epoch=student_epochs, path="checkpoints/student_stage2.pth")
    best_student_auc = metrics["auc"]
    stage2_time = time.time() - stage2_start
    print(f"  → Saved student checkpoint (AUC={best_student_auc:.4f})")
    print(f"  → Stage 2 complete in {_fmt_time(stage2_time)}")

    torch.cuda.empty_cache()

    print("\n  Final evaluation on validation split...")
    metrics = evaluate(student, val_loader, device, calibrate_threshold=True)
    print_metrics("  [Val]", metrics)

    print("\n  Final test evaluation using validation threshold...")
    test_metrics = evaluate(
        student,
        test_loader,
        device,
        threshold=metrics["threshold"],
        calibrate_threshold=False,
    )
    print_metrics("  [Test]", test_metrics)

    save_checkpoint(student, student_opt, epoch=teacher_epochs + student_epochs,
                    path="checkpoints/student_final.pth")

    total_time = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Stage 1 (Teacher):     {_fmt_time(stage1_time)}")
    print(f"  Stage 2 (Student KD):  {_fmt_time(stage2_time)}")

    print(f"  Total pipeline time:   {_fmt_time(total_time)}")
    print(f"  Best Teacher AUC:      {best_teacher_auc:.4f}")
    print(f"  Best Student Val AUC:  {max(best_student_auc, metrics['auc']):.4f}")
    print(f"  Final Student Test AUC:{test_metrics['auc']:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
