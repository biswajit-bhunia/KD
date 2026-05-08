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
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.model_selection import train_test_split

# Data
from data.dataset import DeepfakeDataset
from data.loader import load_samples
from data.split import split_samples
from data.partitioner import partition_by_generator, print_partition_stats

# Models
from models.teacher import TeacherModel
from models.student import StudentModel

# Training (centralized — for teacher)
from training.train import train_teacher
from training.validate import evaluate

# Federation
from federation.client import FederatedClient
from federation.server import FederatedServer

# Features
from features.gradient import GradientExtractor

# Utils
from utils.checkpoint import save_checkpoint


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
    lambda_kd     = config["lambda_kd"]
    lambda_supcon = config["lambda_supcon"]
    lambda_grl    = config["lambda_grl"]
    max_lambda_grl = config.get("max_lambda_grl", 0.3)
    temp_kd        = config.get("temperature_kd", 4.0)
    temp_supcon    = config.get("temperature_supcon", 0.07)

    # Federated params
    num_clients       = config.get("num_clients", 3)
    num_rounds        = config.get("num_rounds", 10)
    local_epochs      = config.get("local_epochs", 3)
    clients_per_round = config.get("clients_per_round", None)  # None = all
    mu                = config.get("fedprox_mu", 0.01)
    iid_partition     = config.get("iid_partition", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Federated Deepfake Detection Pipeline")
    print(f"  Device: {device} | Batch: {batch_size} | LR: {lr}")
    print(f"  λ_kd: {lambda_kd} | λ_sc: {lambda_supcon} | λ_grl: {lambda_grl}")
    print(f"  Clients: {num_clients} | Rounds: {num_rounds} | "
          f"Local epochs: {local_epochs}")
    print(f"  FedProx μ: {mu} | IID: {iid_partition}")
    print(f"{'='*60}")

    # ---------------------------
    # 2. Load & Split Dataset
    # ---------------------------
    print("\n  Loading dataset...")
    load_start = time.time()
    samples = load_samples("data/")

    n_real = sum(1 for s in samples if s[1] == 0)
    n_fake = sum(1 for s in samples if s[1] == 1)
    n_gens = len(set(s[2] for s in samples if s[1] == 1))
    print(f"  Loaded {len(samples)} samples ({n_real} real, {n_fake} fake, "
          f"{n_gens} generators) in {time.time() - load_start:.1f}s")

    # Previous label-stratified split allowed the same fake generators in train and validation.
    # train_samples, val_samples = train_test_split(
    #     samples,
    #     test_size=0.2,
    #     stratify=[s[1] for s in samples],
    #     random_state=42
    # )
    train_samples, val_samples = split_samples(
        samples,
        test_size=0.2,
        mode="generator_holdout",
        random_state=42
    )
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)}")

    val_dataset = DeepfakeDataset(val_samples, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ---------------------------
    # 3. Phase 1: Train Teacher Centrally
    # ---------------------------
    print(f"\n{'─'*60}")
    print(f"  PHASE 1: Centralized Teacher Training (5 epochs)")
    print(f"{'─'*60}")

    phase1_start = time.time()

    teacher = TeacherModel().to(device)
    teacher_opt = torch.optim.Adam(teacher.parameters(), lr=lr)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher model: {teacher_params:,} parameters")

    # Build centralized train loader for teacher
    train_dataset_full = DeepfakeDataset(train_samples, augment=True)
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler
    train_labels = [s[1] for s in train_samples]
    counts = Counter(train_labels)
    weights = [1.0 / counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader_full = DataLoader(
        train_dataset_full,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    print(f"  Train batches: {len(train_loader_full)}")

    train_teacher(
        model=teacher,
        dataloader=train_loader_full,
        optimizer=teacher_opt,
        device=device,
        epochs=5,
        lambda_supcon=lambda_supcon
    )

    print("\n  Evaluating teacher...")
    metrics = evaluate(teacher, val_loader, device)
    print_metrics("  [Teacher]", metrics)

    save_checkpoint(teacher, teacher_opt, epoch=5, path="checkpoints/teacher_federated.pth")
    phase1_time = time.time() - phase1_start
    print(f"  → Saved teacher (AUC={metrics['auc']:.4f})")
    print(f"  → Phase 1 complete in {_fmt_time(phase1_time)}")

    # Freeze teacher for distribution
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ---------------------------
    # 4. Initialize Student & Gradient Model
    # ---------------------------
    print("\n  Initializing student and gradient models...")
    global_student = StudentModel().to(device)
    grad_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(device)
    grad_extractor = GradientExtractor(grad_model).to(device)

    student_params = sum(p.numel() for p in global_student.parameters())
    print(f"  Student model: {student_params:,} parameters "
          f"({student_params/teacher_params*100:.1f}% of teacher)")

    # ---------------------------
    # 5. Partition Data Across Clients
    # ---------------------------
    print(f"\n{'─'*60}")
    print(f"  DATA PARTITIONING ({'IID' if iid_partition else 'Non-IID by generator'})")
    print(f"{'─'*60}")

    partition_start = time.time()
    client_data = partition_by_generator(
        train_samples,
        num_clients=num_clients,
        iid=iid_partition,
        seed=42
    )
    print_partition_stats(client_data)
    print(f"  Partitioned in {time.time() - partition_start:.2f}s")

    # ---------------------------
    # 6. Create Federated Clients
    # ---------------------------
    print("\n  Creating federated clients...")
    client_start = time.time()
    clients = []
    for client_id in sorted(client_data.keys()):
        client = FederatedClient(
            client_id=client_id,
            samples=client_data[client_id],
            teacher=teacher,
            grad_model=grad_model,
            device=device,
            batch_size=batch_size,
        )
        clients.append(client)
        gen_ids = sorted(client.local_generators)
        print(f"    Client {client_id}: {client.num_samples} samples | "
              f"{client.num_generators} generators {gen_ids} | "
              f"{len(client.dataloader)} batches")

    print(f"  {len(clients)} clients created in {time.time() - client_start:.1f}s")

    # ---------------------------
    # 7. Phase 2: Federated Student Training
    # ---------------------------
    print(f"\n{'─'*60}")
    print(f"  PHASE 2: Federated Student Training")
    print(f"{'─'*60}")

    phase2_start = time.time()

    server = FederatedServer(
        global_student=global_student,
        clients=clients,
        val_loader=val_loader,
        device=device,
        grad_extractor=grad_extractor,
        clients_per_round=clients_per_round,
    )

    history = server.run(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        lr=lr,
        lambda_kd=lambda_kd,
        lambda_supcon=lambda_supcon,
        lambda_grl=lambda_grl,
        max_lambda_grl=max_lambda_grl,
        mu=mu,
        temperature_kd=temp_kd,
        temperature_supcon=temp_supcon,
        save_path="checkpoints/student_federated_best.pth",
    )

    phase2_time = time.time() - phase2_start

    # ---------------------------
    # 8. Final Summary
    # ---------------------------
    total_time = time.time() - pipeline_start

    best_round = max(range(len(history)), key=lambda i: history[i]["auc"])
    best_metrics = history[best_round]

    print(f"\n{'='*60}")
    print(f"  FEDERATED PIPELINE COMPLETE")
    print(f"  Phase 1 (Teacher):    {_fmt_time(phase1_time)}")
    print(f"  Phase 2 (Federated):  {_fmt_time(phase2_time)}")
    print(f"  Total pipeline time:  {_fmt_time(total_time)}")
    print(f"")
    print(f"  Best round: {best_round + 1}/{num_rounds}")
    print(f"  Best AUC:   {best_metrics['auc']:.4f}")
    print(f"  Best F1:    {best_metrics['f1']:.4f}")
    print(f"  Best Acc:   {best_metrics['accuracy']:.4f}")
    print(f"")
    print(f"  Run `python main.py` for centralized comparison.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
