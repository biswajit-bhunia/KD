"""
Federated Client for dual-domain deepfake detection.

Each client holds a local data partition and trains a local copy of the
student model using multi-level KD from a shared frozen teacher.

Changes from previous version:
  - Removed GradientExtractor — student now uses (x_rgb, x_forensic)
  - Added multi-level KD (semantic, forensic, embedding, logits)
  - GRL code preserved but inert when lambda_grl=0.0
"""

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
from models.grl import GradientReversal
from models.gen_classifier import GeneratorClassifier
from models.kd import MultiLevelKD
from losses.losses import ClassificationLoss, KDLoss, SupConLoss, GeneratorAdversarialLoss
from losses.fedprox import FedProxLoss
from utils.reproducibility import make_generator, seed_worker
from utils.debug_checks import (
    check_loss, check_kd_decomposition, check_forensic_stack,
    check_teacher_frozen, DEBUG as _DEBUG,
)


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


class FederatedClient:
    """
    A federated client that holds local data and trains a local copy
    of the student model using multi-level KD from a frozen teacher.
    """

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

        # Count local generators (fakes only)
        self.local_generators = set(s[2] for s in samples if s[1] == 1)
        self.num_generators = len(self.local_generators)

        # Build generator ID remapping (local gen_ids → 0-indexed for local gen_head)
        sorted_gens = sorted(self.local_generators)
        self.gen_id_remap = {g: i for i, g in enumerate(sorted_gens)}

        # Precompute vectorized mapping tensor to prevent CPU-GPU syncs during batch loops
        max_gen = max(self.local_generators) if self.local_generators else 0
        self.gen_map_tensor = torch.zeros(max_gen + 1, dtype=torch.long)
        for g, i in self.gen_id_remap.items():
            self.gen_map_tensor[g] = i

    def train_local(
        self,
        student: nn.Module,
        global_state_dict: OrderedDict,
        lr: float = 3e-4,
        local_epochs: int = 3,
        lambda_kd: float = 0.1,
        lambda_feat_kd: float = 0.5,
        lambda_supcon: float = 0.05,
        lambda_grl: float = 0.0,
        max_lambda_grl: float = 0.0,
        mu: float = 0.001,
        temperature_kd: float = 4.0,
        temperature_supcon: float = 0.07,
    ) -> OrderedDict:
        """
        Run local training on this client's data.

        Returns:
            trained local student state_dict
        """
        client_start = time.time()

        student = copy.deepcopy(student).to(self.device)
        student.train()

        # Freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Loss functions
        cls_loss = ClassificationLoss()
        multi_kd = MultiLevelKD(temperature=temperature_kd)
        supcon = SupConLoss(temperature=temperature_supcon)
        gen_loss_fn = GeneratorAdversarialLoss()
        fedprox_loss = FedProxLoss(mu=mu)

        # GRL + generator head (local only) — inert when lambda_grl=0.0
        grl = GradientReversal(lambda_=lambda_grl)

        if self.num_generators > 0 and max_lambda_grl > 0:
            if not hasattr(self, "gen_head") or self.gen_head is None:
                self.gen_head = GeneratorClassifier(
                    in_dim=student.embed_dim,
                    num_generators=self.num_generators
                ).to(self.device)

            self.gen_head.train()
            gen_head = self.gen_head
            gen_optimizer = torch.optim.Adam(gen_head.parameters(), lr=lr * 0.1)
        else:
            gen_head = None
            gen_optimizer = None

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        scaler = torch.amp.GradScaler(self.device.type, enabled=self.device.type == "cuda")

        # Move global params to device once (not on every batch inside FedProxLoss)
        global_params = {
            k: v.clone().detach().to(self.device)
            for k, v in global_state_dict.items()
        }

        num_batches = len(self.dataloader)

        for epoch in range(local_epochs):
            epoch_start = time.time()
            total_loss = 0

            # DANN sigmoid schedule — same as centralized GRL stage
            p = epoch / max(local_epochs, 1)
            lambda_grl_epoch = max_lambda_grl * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
            lambda_grl_epoch = max(lambda_grl * 0.1, lambda_grl_epoch) if lambda_grl > 0 else 0.0
            grl.lambda_ = lambda_grl_epoch

            for batch_idx, batch in enumerate(self.dataloader):
                x_rgb = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                gen_ids = batch["gen_id"].to(self.device)

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
                    loss_feat = kd_losses["semantic"] + kd_losses["forensic"] + kd_losses["embedding"]

                    loss_sup = supcon(student_out["embedding"], labels)

                    if gen_head is not None and lambda_grl_epoch > 0:
                        fake_mask = (gen_ids > 0)
                        if fake_mask.sum() > 0:
                            E_adv = grl(student_out["embedding"][fake_mask])
                            gen_logits = gen_head(E_adv)
                            # Vectorized mapping using precomputed tensor (no CPU-GPU syncs)
                            map_tensor = self.gen_map_tensor.to(self.device)
                            local_gen_ids = map_tensor[gen_ids[fake_mask]]
                            loss_gen = gen_loss_fn(gen_logits, local_gen_ids)
                        else:
                            loss_gen = torch.tensor(0.0, device=self.device)
                    else:
                        loss_gen = torch.tensor(0.0, device=self.device)

                # FedProx proximal term — computed outside autocast for fp32 precision
                loss_prox = fedprox_loss(student, global_params)

                loss = (
                    loss_ce
                    + lambda_kd      * loss_kd_val
                    + lambda_feat_kd * loss_feat
                    + lambda_supcon  * loss_sup
                    + lambda_grl_epoch * loss_gen
                    + loss_prox
                )

                # --- Debug checks (gated, zero overhead in production) ---
                if _DEBUG and batch_idx == 0 and epoch == 0:
                    check_forensic_stack(x_for, x_rgb.shape[0], x_rgb.shape[2], x_rgb.shape[3])
                    check_teacher_frozen(self.teacher)
                    check_kd_decomposition(kd_losses, loss_kd_val, loss_feat)
                    check_loss(loss, f"client{self.client_id}_loss")

                optimizer.zero_grad()
                if gen_optimizer is not None:
                    gen_optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

                if gen_optimizer is not None and gen_head is not None:
                    fake_mask = (gen_ids > 0)
                    if fake_mask.sum() > 0:
                        scaler.unscale_(gen_optimizer)
                        nn.utils.clip_grad_norm_(gen_head.parameters(), max_norm=1.0)
                        scaler.step(gen_optimizer)

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