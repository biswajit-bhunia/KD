import copy
from collections import OrderedDict
from typing import List, Tuple

import torch




def fedavg_aggregate(
    global_state_dict: OrderedDict,
    client_updates: List[Tuple[OrderedDict, int]]
) -> OrderedDict:
    if not client_updates:
        return copy.deepcopy(global_state_dict)

    total_samples = sum(n for _, n in client_updates)
    aggregated = OrderedDict()

    for key in global_state_dict.keys():
        global_device = global_state_dict[key].device

        # BatchNorm step counter: sum across clients, don't average
        if key.endswith('num_batches_tracked'):
            aggregated[key] = sum(
                client_state[key] for client_state, _ in client_updates
            ).to(global_device)
            continue

        # Accumulate on CPU (client states are already there after isolation)
        acc = torch.zeros_like(global_state_dict[key], dtype=torch.float32, device='cpu')
        for client_state, num_samples in client_updates:
            weight = num_samples / total_samples
            acc += weight * client_state[key].float()

        aggregated[key] = acc.to(dtype=global_state_dict[key].dtype, device=global_device)

    return aggregated