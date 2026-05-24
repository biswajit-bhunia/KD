import random

import numpy as np
import torch

def seed_everything(seed: int = 42, deterministic: bool = False):
    """Seed Python, NumPy, PyTorch, CUDA, and DataLoader workers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_generator(seed: int = 42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
