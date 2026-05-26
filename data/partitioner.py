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
    """Group by video, shuffle videos, and split evenly across clients."""
    from data.split import get_video_id
    video_groups = defaultdict(list)
    for s in samples:
        video_groups[get_video_id(s[0], gen_id=s[2])].append(s)
        
    vids = list(video_groups.keys())
    rng.shuffle(vids)

    client_data = {i: [] for i in range(num_clients)}
    chunk_size = len(vids) // num_clients

    for i in range(num_clients):
        start = i * chunk_size
        end = start + chunk_size if i < num_clients - 1 else len(vids)
        for vid in vids[start:end]:
            client_data[i].extend(video_groups[vid])

    return client_data

def _partition_by_generator(samples, num_clients, rng):
    """
    Non-IID partitioning: group frames by video identity. Assign real videos evenly 
    across all clients. Assign fake videos from specific generators to assigned clients.
    """
    from data.split import get_video_id
    
    # Group real and fake samples by video identity
    real_videos = defaultdict(list)
    fake_by_gen_videos = defaultdict(lambda: defaultdict(list))
    
    for s in samples:
        vid = get_video_id(s[0], gen_id=s[2])
        if s[1] == 0:
            real_videos[vid].append(s)
        else:
            fake_by_gen_videos[s[2]][vid].append(s)

    generator_ids = sorted(fake_by_gen_videos.keys())
    
    # Assign ALL generators to clients via round-robin.
    # Old code iterated over num_clients, silently dropping generators when
    # len(generators) > num_clients (e.g. 4 generators, 2 clients → 2 dropped).
    gen_to_clients = defaultdict(list)
    for i, gen_id in enumerate(generator_ids):
        client_id = i % num_clients
        gen_to_clients[gen_id].append(client_id)

    # Distribute real videos evenly across all clients
    real_vids = list(real_videos.keys())
    rng.shuffle(real_vids)
    real_vids_per_client = len(real_vids) // num_clients

    client_data = defaultdict(list)
    
    # Allocate real frames for all clients
    for client_id in range(num_clients):
        start = client_id * real_vids_per_client
        end = start + real_vids_per_client if client_id < num_clients - 1 else len(real_vids)
        for vid in real_vids[start:end]:
            client_data[client_id].extend(real_videos[vid])

    # Distribute fake videos by splitting generator fakes among assigned clients
    for gen_id, clients in gen_to_clients.items():
        gen_vids = list(fake_by_gen_videos[gen_id].keys())
        rng.shuffle(gen_vids)
        vids_per_chunk = len(gen_vids) // len(clients)
        
        for idx, client_id in enumerate(clients):
            start = idx * vids_per_chunk
            end = start + vids_per_chunk if idx < len(clients) - 1 else len(gen_vids)
            for vid in gen_vids[start:end]:
                client_data[client_id].extend(fake_by_gen_videos[gen_id][vid])
                
            if len(gen_vids[start:end]) == 0:
                print(f"  \u26a0  Client {client_id} has 0 fake videos from generator {gen_id}.")

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
