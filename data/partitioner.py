import os
import random
from collections import defaultdict
from typing import List, Tuple, Dict

def partition_by_generator(
    samples: List[Tuple[str, int, int]],
    num_clients: int,
    iid: bool = False,
    seed: int = 42
) -> Dict[int, List[Tuple[str, int, int]]]:
    """
    Partition samples across federated clients.

    Args:
        samples: list of (image_path, label, generator_id)
        num_clients: number of federated clients
        iid: if True, shuffle and split evenly (each client gets all generators).
             if False, split by generator (each client gets dominant + mixed generators).
        seed: random seed for reproducibility

    Returns:
        dict mapping client_id -> list of (image_path, label, generator_id)
    """
    rng = random.Random(seed)

    if iid:
        return _partition_iid(samples, num_clients, rng)
    else:
        return _partition_by_generator(samples, num_clients, rng)

def _partition_iid(samples, num_clients, rng):
    """Shuffle all samples and split evenly across clients."""
    samples_copy = list(samples)
    rng.shuffle(samples_copy)

    client_data = {i: [] for i in range(num_clients)}
    chunk_size = len(samples_copy) // num_clients

    for i in range(num_clients):
        start = i * chunk_size
        end = start + chunk_size if i < num_clients - 1 else len(samples_copy)
        client_data[i].extend(samples_copy[start:end])

    return client_data

def _partition_by_generator(samples, num_clients, rng):
    """
    Non-IID partitioning: Assign real images evenly across all clients. 
    Assign fake images from specific generators to assigned clients.
    """
    # Group real and fake samples
    real_samples = [s for s in samples if s[1] == 0]
    fake_by_gen = defaultdict(list)
    for s in samples:
        if s[1] == 1:
            fake_by_gen[s[2]].append(s)

    generator_ids = sorted(fake_by_gen.keys())
    
    # Assign ALL generators to clients via round-robin.
    gen_to_clients = defaultdict(list)
    for i, gen_id in enumerate(generator_ids):
        client_id = i % num_clients
        gen_to_clients[gen_id].append(client_id)

    client_data = defaultdict(list)
    
    # Distribute real images evenly across all clients
    rng.shuffle(real_samples)
    reals_per_client = len(real_samples) // num_clients
    
    for client_id in range(num_clients):
        start = client_id * reals_per_client
        end = start + reals_per_client if client_id < num_clients - 1 else len(real_samples)
        client_data[client_id].extend(real_samples[start:end])

    # Distribute fake images by splitting generator fakes among assigned clients
    for gen_id, clients in gen_to_clients.items():
        gen_samples = fake_by_gen[gen_id]
        rng.shuffle(gen_samples)
        samples_per_chunk = len(gen_samples) // len(clients)
        
        for idx, client_id in enumerate(clients):
            start = idx * samples_per_chunk
            end = start + samples_per_chunk if idx < len(clients) - 1 else len(gen_samples)
            client_data[client_id].extend(gen_samples[start:end])
                
            if len(gen_samples[start:end]) == 0:
                print(f"  \u26a0  Client {client_id} has 0 fake images from generator {gen_id}.")

    return dict(client_data)

def print_partition_stats(client_data: Dict[int, List[Tuple[str, int, int]]]):
    """Print summary of how data was distributed across clients."""
    print("\n=== Data Partition Summary ===")
    for client_id in sorted(client_data.keys()):
        samples = client_data[client_id]
        n_real = sum(1 for s in samples if s[1] == 0)
        n_fake = sum(1 for s in samples if s[1] == 1)
        gens = set(s[2] for s in samples if s[1] == 1)
        gen_str = ", ".join(str(g) for g in sorted(gens)) if gens else "none"

        # Show per-generator counts for transparency
        from collections import Counter
        gen_counts = Counter(s[2] for s in samples if s[1] == 1)
        gen_detail = " | ".join(f"g{g}:{c}" for g, c in sorted(gen_counts.items()))

        print(f"  Client {client_id}: {len(samples)} samples "
              f"({n_real} real, {n_fake} fake) | generators: [{gen_str}]")
        if gen_detail:
            print(f"    breakdown: {gen_detail}")
    print()
