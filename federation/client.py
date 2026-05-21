# import copy
# import math
# import time
# from collections import OrderedDict
# from typing import List, Tuple, Optional

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, WeightedRandomSampler
# from collections import Counter

# from data.dataset import DeepfakeDataset
# from features.forensic import build_forensic_stack
# from features.gradient import GradientExtractor
# from models.grl import GradientReversal
# from models.gen_classifier import GeneratorClassifier
# from losses.losses import ClassificationLoss, KDLoss, SupConLoss, GeneratorAdversarialLoss
# from losses.fedprox import FedProxLoss
# from utils.reproducibility import make_generator, seed_worker


# def _fmt_time(seconds):
#     """Format seconds into human-readable string."""
#     if seconds < 60:
#         return f"{seconds:.1f}s"
#     elif seconds < 3600:
#         m, s = divmod(seconds, 60)
#         return f"{int(m)}m {int(s)}s"
#     else:
#         h, rem = divmod(seconds, 3600)
#         m, s = divmod(rem, 60)
#         return f"{int(h)}h {int(m)}m {int(s)}s"


# class FederatedClient:
#     """
#     A federated client that holds local data and trains a local copy
#     of the student model using the existing KD + GRL pipeline.
#     """

#     def __init__(
#         self,
#         client_id: int,
#         samples: List[Tuple[str, int, int]],
#         teacher: nn.Module,
#         grad_model: nn.Module,
#         device: torch.device,
#         batch_size: int = 32,
#         image_size: int = 256,
#         seed: int = 42,
#     ):
#         self.client_id = client_id
#         self.samples = samples
#         self.teacher = teacher
#         self.grad_model = grad_model
#         self.device = device
#         self.batch_size = batch_size

#         # Build dataloader
#         dataset = DeepfakeDataset(samples, image_size=image_size, augment=True)

#         labels = [s[1] for s in samples]
#         counts = Counter(labels)
#         weights = [1.0 / counts[l] for l in labels]
#         data_generator = make_generator(seed + client_id)
#         sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=data_generator)

#         self.dataloader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             sampler=sampler,
#             num_workers=2,
#             pin_memory=True,
#             worker_init_fn=seed_worker,
#             generator=data_generator,
#         )

#         self.num_samples = len(samples)

#         # Count local generators (fakes only)
#         self.local_generators = set(s[2] for s in samples if s[1] == 1)
#         self.num_generators = len(self.local_generators)

#         # Build generator ID remapping (local gen_ids -> 0-indexed for local gen_head)
#         sorted_gens = sorted(self.local_generators)
#         self.gen_id_remap = {g: i for i, g in enumerate(sorted_gens)}

#     def train_local(
#         self,
#         student: nn.Module,
#         global_state_dict: OrderedDict,
#         lr: float = 3e-4,
#         local_epochs: int = 3,
#         lambda_kd: float = 0.1,
#         lambda_supcon: float = 0.05,
#         lambda_grl: float = 0.05,
#         max_lambda_grl: float = 0.3,    # ceiling for DANN sigmoid schedule
#         mu: float = 0.01,
#         temperature_kd: float = 4.0,
#         temperature_supcon: float = 0.07,
#     ) -> OrderedDict:
#         """
#         Run local training on this client's data.

#         Returns:
#             trained local student state_dict
#         """
#         client_start = time.time()

#         student = copy.deepcopy(student).to(self.device)
#         student.train()

#         # Freeze teacher
#         self.teacher.eval()
#         for p in self.teacher.parameters():
#             p.requires_grad = False

#         # Loss functions
#         cls_loss = ClassificationLoss()
#         kd_loss = KDLoss(temperature=temperature_kd)
#         supcon = SupConLoss(temperature=temperature_supcon)
#         gen_loss_fn = GeneratorAdversarialLoss()
#         fedprox_loss = FedProxLoss(mu=mu)

#         # Gradient extractor
#         grad_extractor = GradientExtractor(self.grad_model).to(self.device)

#         # GRL + generator head (local only)
#         grl = GradientReversal(lambda_=lambda_grl)

#         if self.num_generators > 0:
#             if not hasattr(self, "gen_head") or self.gen_head is None:
#                 self.gen_head = GeneratorClassifier(
#                     in_dim=student.mlp[0].out_features,
#                     num_generators=self.num_generators
#                 ).to(self.device)
#                 # Apply the same 0.1x LR slowdown fix from centralized training
#                 self.gen_optimizer = torch.optim.Adam(self.gen_head.parameters(), lr=lr * 0.1)
            
#             self.gen_head.train()
#             gen_head = self.gen_head
#             gen_optimizer = self.gen_optimizer
#         else:
#             gen_head = None
#             gen_optimizer = None

#         optimizer = torch.optim.Adam(student.parameters(), lr=lr)
#         scaler = torch.amp.GradScaler(self.device.type, enabled=self.device.type == "cuda")

#         # Snapshot of global weights for FedProx
#         global_params = {k: v.clone().detach() for k, v in global_state_dict.items()}

#         num_batches = len(self.dataloader)

#         for epoch in range(local_epochs):
#             epoch_start = time.time()
#             total_loss = 0

#             # DANN sigmoid schedule — same as centralized GRL stage
#             p = epoch / max(local_epochs, 1)
#             lambda_grl_epoch = max_lambda_grl * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
#             lambda_grl_epoch = max(lambda_grl * 0.1, lambda_grl_epoch)
#             grl.lambda_ = lambda_grl_epoch

#             for batch_idx, batch in enumerate(self.dataloader):
#                 x_rgb = batch["image"].to(self.device)
#                 labels = batch["label"].to(self.device)
#                 gen_ids = batch["gen_id"].to(self.device)

#                 x_for = build_forensic_stack(x_rgb)

#                 with torch.enable_grad():
#                     x_grad = grad_extractor(x_rgb)

#                 with torch.no_grad():
#                     teacher_out = self.teacher(x_rgb, x_for)

#                 with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
#                     student_out = student(x_for, x_grad)

#                     loss_ce = cls_loss(student_out["logits"], labels)
#                     loss_kd_val = kd_loss(student_out["logits"], teacher_out["logits"])
#                     loss_sup = supcon(student_out["embedding"], labels)

#                     if gen_head is not None:
#                         fake_mask = (gen_ids > 0)
#                         if fake_mask.sum() > 0:
#                             E_adv = grl(student_out["embedding"][fake_mask])
#                             gen_logits = gen_head(E_adv)
#                             # Raise on unknown gen_id instead of silently mapping to class 0
#                             local_gen_ids_list = []
#                             for g in gen_ids[fake_mask]:
#                                 g_val = g.item()
#                                 if g_val not in self.gen_id_remap:
#                                     raise ValueError(
#                                         f"Client {self.client_id}: unknown gen_id={g_val} "
#                                         f"in a fake-labeled sample. "
#                                         f"Known generators: {self.gen_id_remap}"
#                                     )
#                                 local_gen_ids_list.append(self.gen_id_remap[g_val])
#                             local_gen_ids = torch.tensor(
#                                 local_gen_ids_list, dtype=torch.long, device=self.device
#                             )
#                             loss_gen = gen_loss_fn(gen_logits, local_gen_ids)
#                         else:
#                             # Provide dummy loss connected to gen_head so optimizer receives gradients
#                             # This prevents the PyTorch AMP scaler from crashing when step() is called.
#                             dummy_in = torch.zeros(1, student.mlp[0].out_features, device=self.device)
#                             loss_gen = (gen_head(dummy_in) * 0.0).sum()
#                     else:
#                         loss_gen = torch.tensor(0.0, device=self.device)

#                 # FedProx proximal term — computed outside autocast for fp32 precision
#                 loss_prox = fedprox_loss(student, global_params)

#                 loss = (
#                     loss_ce
#                     + lambda_kd * loss_kd_val
#                     + lambda_supcon * loss_sup
#                     + lambda_grl_epoch * loss_gen  # use epoch-scheduled lambda
#                     + loss_prox
#                 )

#                 optimizer.zero_grad()
#                 if gen_optimizer is not None:
#                     gen_optimizer.zero_grad()

#                 scaler.scale(loss).backward()

#                 scaler.unscale_(optimizer)
#                 nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

#                 if gen_optimizer is not None:
#                     scaler.unscale_(gen_optimizer)
#                     nn.utils.clip_grad_norm_(gen_head.parameters(), max_norm=1.0)
#                     scaler.step(gen_optimizer)

#                 scaler.step(optimizer)
#                 scaler.update()

#                 total_loss += loss.item()

#             epoch_time = time.time() - epoch_start
#             avg_loss = total_loss / num_batches
#             remaining = epoch_time * (local_epochs - epoch - 1)
#             print(f"      [Client {self.client_id}] Epoch {epoch+1}/{local_epochs} | "
#                   f"Loss: {avg_loss:.4f} | "
#                   f"Time: {_fmt_time(epoch_time)} | "
#                   f"ETA: {_fmt_time(remaining)}")

#         client_time = time.time() - client_start
#         print(f"      [Client {self.client_id}] Local training done in {_fmt_time(client_time)}")

#         return student.state_dict()










import copy
import math
import time
from collections import OrderedDict
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

from data.dataset import DeepfakeDataset
from features.forensic import build_forensic_stack
from features.gradient import GradientExtractor
from models.grl import GradientReversal
from models.gen_classifier import GeneratorClassifier
from losses.losses import ClassificationLoss, KDLoss, SupConLoss, GeneratorAdversarialLoss
from losses.fedprox import FedProxLoss
from utils.reproducibility import make_generator, seed_worker


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
    of the student model using the existing KD + GRL pipeline.
    """

    def __init__(
        self,
        client_id: int,
        samples: List[Tuple[str, int, int]],
        teacher: nn.Module,
        grad_model: nn.Module,
        device: torch.device,
        batch_size: int = 32,
        image_size: int = 256,
        seed: int = 42,
    ):
        self.client_id = client_id
        self.samples = samples
        self.teacher = teacher
        self.grad_model = grad_model
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
            num_workers=0,  # FIX: Set to 0 to prevent excessive process spawning in federated simulation
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=data_generator,
        )

        self.num_samples = len(samples)

        # Count local generators (fakes only)
        self.local_generators = set(s[2] for s in samples if s[1] == 1)
        self.num_generators = len(self.local_generators)

        # Build generator ID remapping (local gen_ids -> 0-indexed for local gen_head)
        sorted_gens = sorted(self.local_generators)
        self.gen_id_remap = {g: i for i, g in enumerate(sorted_gens)}
        
        # FIX: Precompute vectorized mapping tensor to prevent CPU-GPU syncs during batch loops
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
        lambda_supcon: float = 0.05,
        lambda_grl: float = 0.05,
        max_lambda_grl: float = 0.3,
        mu: float = 0.01,
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
        kd_loss = KDLoss(temperature=temperature_kd)
        supcon = SupConLoss(temperature=temperature_supcon)
        gen_loss_fn = GeneratorAdversarialLoss()
        fedprox_loss = FedProxLoss(mu=mu)

        # Gradient extractor
        grad_extractor = GradientExtractor(self.grad_model).to(self.device)

        # GRL + generator head (local only)
        grl = GradientReversal(lambda_=lambda_grl)

        if self.num_generators > 0:
            if not hasattr(self, "gen_head") or self.gen_head is None:
                self.gen_head = GeneratorClassifier(
                    in_dim=student.mlp[0].out_features,
                    num_generators=self.num_generators
                ).to(self.device)

            self.gen_head.train()
            gen_head = self.gen_head
            
            # FIX: Recreate gen_optimizer every round to respect decaying LR and reset momentum
            gen_optimizer = torch.optim.Adam(gen_head.parameters(), lr=lr * 0.1)
        else:
            gen_head = None
            gen_optimizer = None

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        scaler = torch.amp.GradScaler(self.device.type, enabled=self.device.type == "cuda")

        # FIX: Move global params to device once here, not on every batch inside FedProxLoss.
        # global_state_dict arrives on CPU (isolated with .cpu().clone() in the server loop).
        # Without this, FedProxLoss.forward calls .to(param.device) on every parameter for
        # every batch — hundreds of redundant CPU→GPU transfers per round per client.
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
            lambda_grl_epoch = max(lambda_grl * 0.1, lambda_grl_epoch)
            grl.lambda_ = lambda_grl_epoch

            for batch_idx, batch in enumerate(self.dataloader):
                x_rgb = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                gen_ids = batch["gen_id"].to(self.device)

                x_for = build_forensic_stack(x_rgb)

                with torch.enable_grad():
                    x_grad = grad_extractor(x_rgb)

                with torch.no_grad():
                    teacher_out = self.teacher(x_rgb, x_for)

                with torch.amp.autocast(self.device.type, enabled=self.device.type == "cuda"):
                    student_out = student(x_for, x_grad)

                    loss_ce = cls_loss(student_out["logits"], labels)
                    loss_kd_val = kd_loss(student_out["logits"], teacher_out["logits"])
                    loss_sup = supcon(student_out["embedding"], labels)

                    if gen_head is not None:
                        fake_mask = (gen_ids > 0)
                        if fake_mask.sum() > 0:
                            E_adv = grl(student_out["embedding"][fake_mask])
                            gen_logits = gen_head(E_adv)
                            # FIX: Vectorized mapping using precomputed tensor (no CPU-GPU syncs)
                            map_tensor = self.gen_map_tensor.to(self.device)
                            local_gen_ids = map_tensor[gen_ids[fake_mask]]
                            loss_gen = gen_loss_fn(gen_logits, local_gen_ids)
                        else:
                            # No fake samples in this batch — skip gen head entirely.
                            # PyTorch's AMP scaler handles an optimizer with no gradients
                            # cleanly; no dummy forward pass needed.
                            loss_gen = torch.tensor(0.0, device=self.device)
                    else:
                        loss_gen = torch.tensor(0.0, device=self.device)

                # FedProx proximal term — computed outside autocast for fp32 precision
                loss_prox = fedprox_loss(student, global_params)

                loss = (
                    loss_ce
                    + lambda_kd * loss_kd_val
                    + lambda_supcon * loss_sup
                    + lambda_grl_epoch * loss_gen
                    + loss_prox
                )

                optimizer.zero_grad()
                if gen_optimizer is not None:
                    gen_optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

                if gen_optimizer is not None and fake_mask.sum() > 0:
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