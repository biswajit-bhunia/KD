import copy
import math
import time
from collections import OrderedDict, Counter
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset import DeepfakeDataset
from features.forensic import build_forensic_stack
from models.kd import MultiLevelKD
from losses.losses import ClassificationLoss
from losses.fedprox import FedProxLoss
from utils.reproducibility import make_generator, seed_worker
from utils.debug_checks import (
    check_loss, check_kd_decomposition, check_forensic_stack,
    check_teacher_frozen, DEBUG as _DEBUG,
)

def _fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


def _get_round_lr(base_lr: float, round_idx: int, total_rounds: int) -> float:
    """
    Cosine decay of LR across federated rounds.
    Round 0  → base_lr
    Round N-1 → base_lr * 0.1  (never goes to zero)
    """
    min_lr = base_lr * 0.1
    cosine_decay = 0.5 * (1 + math.cos(math.pi * round_idx / total_rounds))
    return min_lr + (base_lr - min_lr) * cosine_decay


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        samples: List[Tuple[str, int, int]],
        teacher: nn.Module,
        device: torch.device,
        batch_size: int = 32,
        image_size: int = 256,
        seed: int = 42,
    ):
        self.client_id = client_id
        self.samples = samples
        self.teacher = teacher
        self.device = device
        self.batch_size = batch_size

        # Build dataloader
        dataset = DeepfakeDataset(samples, image_size=image_size, augment=True)

        labels = [s[1] for s in samples]
        counts = Counter(labels)
        weights = [1.0 / counts[l] for l in labels]
        data_generator = make_generator(seed + client_id)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=data_generator)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,  # Prevent excessive process spawning in federated simulation
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=data_generator,
        )

        self.num_samples = len(samples)
        self.local_generators = list(set(s[2] for s in samples if s[1] == 1))
        self.num_generators = len(self.local_generators)

    def train_local(
        self,
        student: nn.Module,
        global_state_dict: OrderedDict,
        lr: float = 3e-4,
        local_epochs: int = 3,
        lambda_kd: float = 0.1,
        lambda_feat_kd: float = 0.5,
        mu: float = 0.001,
        temperature_kd: float = 4.0,
        round_idx: int = 0,
        total_rounds: int = 12,
    ) -> OrderedDict:
        client_start = time.time()

        # Compute decayed LR for this round
        effective_lr = _get_round_lr(lr, round_idx, total_rounds)

        student = copy.deepcopy(student).to(self.device)
        student.train()

        # Freeze BN statistics entirely to prevent Non-IID corruption.
        # ImageNet pretrained layers preserve their high-quality statistics,
        # while randomly initialized layers will rely on their learnable affine parameters.
        for module in student.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        # Freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Loss functions
        cls_loss = ClassificationLoss(label_smoothing=0.05)
        multi_kd = MultiLevelKD(temperature=temperature_kd)
        fedprox_loss = FedProxLoss(mu=mu)

        # Fresh optimizer every round — never carry state across rounds.
        # Adam's moment tensors (exp_avg, exp_avg_sq) are computed relative
        # to a specific weight trajectory. After FedAvg those weights are
        # replaced by a global average, making stored moments invalid.
        optimizer = torch.optim.Adam(student.parameters(), lr=effective_lr)

        scaler = torch.amp.GradScaler(self.device.type, enabled=self.device.type == "cuda")

        # Move global params to device once (not on every batch inside FedProxLoss)
        global_params = {
            k: v.clone().detach().to(self.device)
            for k, v in global_state_dict.items()
        }

        num_batches = len(self.dataloader)

        print(f"      [Client {self.client_id}] LR this round: {effective_lr:.6f}")

        for epoch in range(local_epochs):
            epoch_start = time.time()
            total_loss = 0
            for batch_idx, batch in enumerate(self.dataloader):
                x_rgb = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                x_for = build_forensic_stack(x_rgb)

                with torch.no_grad():
                    teacher_out = self.teacher(x_rgb, x_for)

                with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
                    student_out = student(x_rgb, x_for)

                    loss_ce = cls_loss(student_out["logits"], labels)

                    # Multi-level KD
                    kd_losses = multi_kd(student_out, teacher_out)
                    loss_kd_val = kd_losses["logits"]
                    # Feature-only KD (excludes logits to avoid double-counting)
                    loss_feat = kd_losses["embedding"]

                    loss = (
                        loss_ce
                        + lambda_kd      * loss_kd_val
                        + lambda_feat_kd * loss_feat
                    )

                if mu > 0:
                    loss += fedprox_loss(student, global_params)

                if _DEBUG and batch_idx == 0 and epoch == 0:
                    check_forensic_stack(x_for, x_rgb.shape[0], x_rgb.shape[2], x_rgb.shape[3])
                    check_teacher_frozen(self.teacher)
                    check_kd_decomposition(kd_losses, loss_kd_val, loss_feat)
                    check_loss(loss, f"client{self.client_id}_loss")

                optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            epoch_time = time.time() - epoch_start
            avg_loss = total_loss / num_batches
            remaining = epoch_time * (local_epochs - epoch - 1)
            print(f"      [Client {self.client_id}] Epoch {epoch+1}/{local_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Time: {_fmt_time(epoch_time)} | "
                  f"ETA: {_fmt_time(remaining)}")

        client_time = time.time() - client_start
        print(f"      [Client {self.client_id}] Local training done in {_fmt_time(client_time)}")

        return student.state_dict()