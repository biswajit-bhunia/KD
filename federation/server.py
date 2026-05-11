import copy
import random
import time
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn

from models.student import StudentModel
from federation.client import FederatedClient
from federation.aggregation import fedavg_aggregate
from training.validate import evaluate


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


class FederatedServer:
    """
    Federated learning server that orchestrates training rounds.
    """

    def __init__(
        self,
        global_student: StudentModel,
        clients: List[FederatedClient],
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        grad_extractor: nn.Module,
        clients_per_round: Optional[int] = None,
        seed: int = 42,
    ):
        self.global_student = global_student
        self.clients = clients
        self.val_loader = val_loader
        self.device = device
        self.grad_extractor = grad_extractor
        self.clients_per_round = clients_per_round or len(clients)
        self.rng = random.Random(seed)

        self.best_auc = -1.0
        self.round_history = []

    def run(
        self,
        num_rounds: int = 10,
        local_epochs: int = 3,
        lr: float = 3e-4,
        lambda_kd: float = 0.1,
        lambda_supcon: float = 0.05,
        lambda_grl: float = 0.05,
        max_lambda_grl: float = 0.3,
        mu: float = 0.01,
        temperature_kd: float = 4.0,
        temperature_supcon: float = 0.07,
        save_path: str = "checkpoints/student_federated_best.pth",
    ):
        print(f"\n{'='*60}")
        print(f"  Federated Training Configuration")
        print(f"  Rounds: {num_rounds} | Clients/round: {self.clients_per_round}/{len(self.clients)}")
        print(f"  Local epochs: {local_epochs} | LR: {lr}")
        print(f"  FedProx μ: {mu} | λ_kd: {lambda_kd} | λ_sc: {lambda_supcon} | λ_grl: {lambda_grl}")
        print(f"{'='*60}\n")

        total_start = time.time()
        round_times = []

        for round_idx in range(num_rounds):
            round_start = time.time()
            print(f"\n{'─'*50}")
            print(f"  ROUND {round_idx + 1}/{num_rounds}")
            print(f"{'─'*50}")

            # 1. Select clients
            selected = self._select_clients()
            client_ids = [c.client_id for c in selected]
            total_samples = sum(c.num_samples for c in selected)
            print(f"  Selected clients: {client_ids} ({total_samples} total samples)")

            # 2. Local training
            global_state = copy.deepcopy(self.global_student.state_dict())
            client_updates = []

            for i, client in enumerate(selected):
                print(f"\n    ┌─ Client {client.client_id} ({i+1}/{len(selected)}) "
                      f"| {client.num_samples} samples | "
                      f"{client.num_generators} generators")

                local_state = client.train_local(
                    student=self.global_student,
                    global_state_dict=global_state,
                    lr=lr,
                    local_epochs=local_epochs,
                    lambda_kd=lambda_kd,
                    lambda_supcon=lambda_supcon,
                    lambda_grl=lambda_grl,
                    max_lambda_grl=max_lambda_grl,
                    mu=mu,
                    temperature_kd=temperature_kd,
                    temperature_supcon=temperature_supcon,
                )

                client_updates.append((local_state, client.num_samples))
                print(f"    └─ Client {client.client_id} complete")

            # 3. Aggregate
            agg_start = time.time()
            print(f"\n  Aggregating {len(client_updates)} client updates...")
            aggregated_state = fedavg_aggregate(global_state, client_updates)
            self.global_student.load_state_dict(aggregated_state)
            agg_time = time.time() - agg_start
            print(f"  Aggregation done in {agg_time:.2f}s")

            # 4. Evaluate
            print(f"\n  Evaluating global model...")
            metrics = evaluate(
                self.global_student, self.val_loader,
                self.device, self.grad_extractor
            )

            round_time = time.time() - round_start
            round_times.append(round_time)
            elapsed_total = time.time() - total_start
            avg_round_time = sum(round_times) / len(round_times)
            eta = avg_round_time * (num_rounds - round_idx - 1)

            print(f"\n  ╔══ Round {round_idx+1}/{num_rounds} Results ══")
            print(f"  ║ Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  ║ Precision: {metrics['precision']:.4f}")
            print(f"  ║ Recall:    {metrics['recall']:.4f}")
            print(f"  ║ F1:        {metrics['f1']:.4f}")
            print(f"  ║ AUC:       {metrics['auc']:.4f}")
            print(f"  ╟──")
            print(f"  ║ Round time: {_fmt_time(round_time)} | "
                  f"Total elapsed: {_fmt_time(elapsed_total)} | "
                  f"ETA: {_fmt_time(eta)}")
            print(f"  ╚══")

            self.round_history.append(metrics)

            # 5. Save best
            if metrics["auc"] > self.best_auc:
                self.best_auc = metrics["auc"]
                torch.save(self.global_student.state_dict(), save_path)
                print(f"  ★ New best model! AUC: {self.best_auc:.4f} → saved to {save_path}")

        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"  FEDERATED TRAINING COMPLETE")
        print(f"  Total time:  {_fmt_time(total_time)}")
        print(f"  Best AUC:    {self.best_auc:.4f}")
        print(f"  Total rounds: {num_rounds}")
        print(f"  Avg round:   {_fmt_time(sum(round_times)/len(round_times))}")
        print(f"{'='*60}\n")

        return self.round_history

    def _select_clients(self) -> List[FederatedClient]:
        """Randomly select clients for this round."""
        if self.clients_per_round >= len(self.clients):
            return list(self.clients)
        return self.rng.sample(self.clients, self.clients_per_round)
