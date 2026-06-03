"""
Federated Deepfake Detection — Federated Training Entry Point

This script:
1. Trains the Teacher centrally (Phase 1)
2. Partitions data across federated clients by generator (non-IID)
3. Runs federated training of the Student via FedAvg/FedProx (Phase 2)

Usage:
    python main_federated.py
"""

import time
import yaml
import torch
from torch.utils.data import DataLoader
from data.dataset import DeepfakeDataset
from data.loader import load_samples
from data.partitioner import partition_by_generator, print_partition_stats

from models.teacher import TeacherModel
from models.student import StudentModel

from training.train import train_teacher
from training.validate import evaluate

from federation.client import FederatedClient
from federation.server import FederatedServer

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
    
    lambda_kd     = config["lambda_kd"]
    lambda_feat_kd = config.get("lambda_feat_kd", 0.5)
    temp_kd        = config.get("temperature_kd", 4.0)
    teacher_epochs = config.get("teacher_epochs", 5)

    # Federated params
    num_clients       = config.get("num_clients", 3)
    num_rounds        = config.get("num_rounds", 10)
    local_epochs      = config.get("local_epochs", 3)
    clients_per_round = config.get("clients_per_round", None)  # None = all
    mu                = config.get("fedprox_mu", 0.001)
    iid_partition     = config.get("iid_partition", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Federated Deepfake Detection Pipeline (Dual-Domain)")
    print(f"  Device: {device} | Batch: {batch_size} | LR: {lr}")
    print(f"  lambda_kd: {lambda_kd} | lambda_feat: {lambda_feat_kd}")
    print(f"  Clients: {num_clients} | Rounds: {num_rounds} | "
          f"Local epochs: {local_epochs}")
    print(f"  FedProx mu: {mu} | IID: {iid_partition}")
    print(f"{'='*60}")

    data_root = config.get("data_root", "./data")
    train_gens = config.get("train_generators", [])
    val_gens = config.get("val_generators", [])
    test_gens = config.get("test_generators", [])

    print(f"\n  Loading all samples from {data_root}...")
    load_start = time.time()
    all_samples = load_samples(data_root)
    print(f"  Loaded {len(all_samples)} total samples in {time.time() - load_start:.1f}s")

    from data.split import split_by_generator
    train_samples, val_samples, test_samples = split_by_generator(
        all_samples, 
        train_gens, 
        val_gens, 
        test_gens, 
        random_state=seed
    )

    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    val_dataset = DeepfakeDataset(val_samples, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=make_generator(seed + 1),
    )
    test_dataset = DeepfakeDataset(test_samples, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=make_generator(seed + 2),
    )

    print(f"\n{'-'*60}")
    print(f"  PHASE 1: Centralized Teacher Pretraining ({teacher_epochs} epochs)")
    print(f"{'-'*60}")

    phase1_start = time.time()

    teacher = TeacherModel().to(device)
    
    # Single LR for all teacher params — differential LR caused backbone_lr = lr*0.1 = 1e-5
    # which equalled CosineAnnealingLR's eta_min, freezing the backbone entirely.
    teacher_opt = torch.optim.Adam(teacher.parameters(), lr=lr)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher model: {teacher_params:,} parameters")

    train_dataset_full = DeepfakeDataset(
        train_samples, augment=True,
        aug_grayscale_p=aug_grayscale_p, 
        aug_color_jitter_p=aug_color_jitter_p
    )
    from collections import Counter
    train_labels = [s[1] for s in train_samples]
    counts = Counter(train_labels)
    # We are using a WeightedRandomSampler to balance the batches perfectly.
    # However, for the Teacher, applying class_weights on top of the sampler acts as an 
    # extreme "Anomaly Detection" bias. It forces the 35M param ResNet to over-index on 
    # pristine Reals (25x importance), which empirically yields much higher validation AUC 
    # on unseen deepfakes. We will use this for the Teacher, but disable it for the Student.
    n_real = counts[0]
    n_fake = counts[1]
    n_total = n_real + n_fake
    w_real = n_total / (2.0 * max(n_real, 1))
    w_fake = n_total / (2.0 * max(n_fake, 1))
    class_weights = torch.tensor([w_real, w_fake], dtype=torch.float32).to(device)
    print(f"  Teacher Class weights: real={w_real:.2f}, fake={w_fake:.2f} (Anomaly Detection Mode)")

    teacher_generator = make_generator(seed)
    # Create a WeightedRandomSampler to ensure 50/50 real/fake in every batch.
    # Calculate balanced weights for the sampler
    n_real = counts[0]
    n_fake = counts[1]
    n_total = n_real + n_fake
    w_real = n_total / (2.0 * max(n_real, 1))
    w_fake = n_total / (2.0 * max(n_fake, 1))
    
    sample_weights = [w_real if label == 0 else w_fake for label in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=teacher_generator
    )

    train_loader_full = DataLoader(
        train_dataset_full,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    print(f"  Train batches: {len(train_loader_full)}")

    train_teacher(
        model=teacher,
        dataloader=train_loader_full,
        optimizer=teacher_opt,
        device=device,
        epochs=teacher_epochs,
        class_weights=class_weights,
    )

    print("\n  Evaluating teacher...")
    metrics = evaluate(teacher, val_loader, device, calibrate_threshold=True)
    print_metrics("  [Teacher]", metrics)

    save_checkpoint(teacher, teacher_opt, epoch=teacher_epochs, path="checkpoints/teacher_federated.pth")
    phase1_time = time.time() - phase1_start
    print(f"  → Saved teacher (AUC={metrics['auc']:.4f})")
    print(f"  → Phase 1 complete in {_fmt_time(phase1_time)}")

    # Quality gate: abort if teacher is broken (AUC < 0.6 = near-random)
    min_teacher_auc = config.get("min_teacher_auc", 0.6)
    if metrics["auc"] < min_teacher_auc:
        print(f"\n  ✗ ABORTING: Teacher AUC ({metrics['auc']:.4f}) < {min_teacher_auc}")
        print(f"    A broken teacher produces useless KD targets — Phase 2 would be wasted compute.")
        print(f"    Diagnostics:")
        print(f"      - Check learning rate (should NOT be stuck at 1e-5)")
        print(f"      - Check loss is decreasing meaningfully")
        print(f"      - Try increasing teacher_epochs in config.yaml")
        print(f"      - Verify data labels (real=0, fake=1)")
        return
    elif metrics["auc"] < 0.7:
        print(f"\n  ⚠ WARNING: Teacher AUC ({metrics['auc']:.4f}) is low. "
              f"Student quality will be limited by teacher quality.")

    # Freeze teacher for distribution
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("\n  Initializing student model...")
    global_student = StudentModel().to(device)

    student_params = sum(p.numel() for p in global_student.parameters())
    print(f"  Student model: {student_params:,} parameters "
          f"({student_params/teacher_params*100:.1f}% of teacher)")

    print(f"\n{'-'*60}")
    print(f"  DATA PARTITIONING ({'IID' if iid_partition else 'Non-IID by generator'})")
    print(f"{'-'*60}")

    partition_start = time.time()
    client_data = partition_by_generator(
        train_samples,
        num_clients=num_clients,
        iid=iid_partition,
        seed=seed
    )
    print_partition_stats(client_data)
    print(f"  Partitioned in {time.time() - partition_start:.2f}s")

    print("\n  Creating federated clients...")
    client_start = time.time()
    clients = []
    for client_id in sorted(client_data.keys()):
        client = FederatedClient(
            client_id=client_id,
            samples=client_data[client_id],
            teacher=teacher,
            device=device,
            batch_size=batch_size,
            seed=seed,
            aug_grayscale_p=aug_grayscale_p,
            aug_color_jitter_p=aug_color_jitter_p,
        )
        # The Student ONLY uses the WeightedRandomSampler for balance.
        # It must NOT use class_weights, otherwise the double-correction causes
        # the small MobileNet model to collapse and guess "Real" for everything (AUC ~0.4).
        client.class_weights = None
        clients.append(client)
        gen_ids = sorted(client.local_generators)
        print(f"    Client {client_id}: {client.num_samples} samples | "
              f"{client.num_generators} generators {gen_ids} | "
              f"{len(client.dataloader)} batches")

    print(f"  {len(clients)} clients created in {time.time() - client_start:.1f}s")

    print(f"\n{'-'*60}")
    print(f"  PHASE 2: Federated Student Training")
    print(f"{'-'*60}")

    phase2_start = time.time()

    server = FederatedServer(
        global_student=global_student,
        clients=clients,
        val_loader=val_loader,
        device=device,
        clients_per_round=clients_per_round,
        seed=seed,
    )

    best_federated_path = "checkpoints/student_federated_best.pth"

    history = server.run(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        lr=lr,
        lambda_kd=lambda_kd,
        lambda_feat_kd=lambda_feat_kd,
        mu=mu,
        temperature_kd=temp_kd,
        save_path=best_federated_path,
        class_weights=None,   # Student uses WeightedRandomSampler for balance; no double-correction
    )

    phase2_time = time.time() - phase2_start

    print("\n  Loading best federated checkpoint for final test evaluation...")
    global_student.load_state_dict(torch.load(best_federated_path, map_location=device, weights_only=True))

    # Learn optimal temperature for probability calibration (Guo et al., 2017)
    print("\n  Calibrating temperature on validation set...")
    from training.validate import learn_temperature
    learned_temp = learn_temperature(global_student, val_loader, device)

    print("\n  Final evaluation on validation split (with TTA + temperature scaling)...")
    final_val_metrics = evaluate(
        global_student, val_loader, device,
        calibrate_threshold=True, use_tta=True, temperature=learned_temp,
    )
    print_metrics("  [Federated Val (Calibrated)]", final_val_metrics)

    print("\n  Final test evaluation using validation threshold (with TTA + temperature scaling)...")
    final_test_metrics = evaluate(
        global_student,
        test_loader,
        device,
        threshold=final_val_metrics["threshold"],
        calibrate_threshold=False,
        use_tta=True,
        temperature=learned_temp,
    )
    print_metrics("  [Federated Test]", final_test_metrics)

    total_time = time.time() - pipeline_start

    best_round = max(range(len(history)), key=lambda i: history[i]["auc"])
    best_metrics = history[best_round]

    print(f"\n{'='*75}")
    print(f"  FEDERATED PIPELINE COMPLETE")
    print(f"  Phase 1 (Teacher):    {_fmt_time(phase1_time)}")
    print(f"  Phase 2 (Federated):  {_fmt_time(phase2_time)}")
    print(f"  Total pipeline time:  {_fmt_time(total_time)}")
    print(f"\n  --- PERFORMANCE SUMMARY ---")
    print(f"  {'Model':<22} | {'Split':<5} | {'AUC':<6} | {'F1':<6} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6}")
    print(f"  {'-'*71}")
    print(f"  {'Teacher (Centralized)':<22} | {'Val':<5} | {metrics['auc']:.4f} | {metrics['f1']:.4f} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f}")
    print(f"  {f'Student (Round {best_round+1})':<22} | {'Val':<5} | {best_metrics['auc']:.4f} | {best_metrics['f1']:.4f} | {best_metrics['accuracy']:.4f} | {best_metrics['precision']:.4f} | {best_metrics['recall']:.4f}")
    print(f"  {'Student (Calibrated)':<22} | {'Val':<5} | {final_val_metrics['auc']:.4f} | {final_val_metrics['f1']:.4f} | {final_val_metrics['accuracy']:.4f} | {final_val_metrics['precision']:.4f} | {final_val_metrics['recall']:.4f}")
    print(f"  {'Student (Final)':<22} | {'Test':<5} | {final_test_metrics['auc']:.4f} | {final_test_metrics['f1']:.4f} | {final_test_metrics['accuracy']:.4f} | {final_test_metrics['precision']:.4f} | {final_test_metrics['recall']:.4f}")
    print(f"")
    print(f"  Run `python main.py` for centralized comparison.")
    print(f"{'='*75}\n")

if __name__ == "__main__":
    main()
