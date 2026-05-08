import copy
from collections import OrderedDict
from typing import List, Tuple

import torch


def fedavg_aggregate(
    global_state_dict: OrderedDict,
    client_updates: List[Tuple[OrderedDict, int]]
) -> OrderedDict:
    """
    Federated Averaging (FedAvg) — McMahan et al. 2017.

    Computes a weighted average of client model weights, where each client's
    contribution is proportional to its dataset size.

    Args:
        global_state_dict: current global model state_dict (used as fallback
                           for keys not present in updates)
        client_updates: list of (client_state_dict, num_samples) tuples

    Returns:
        aggregated state_dict
    """
    if not client_updates:
        return copy.deepcopy(global_state_dict)

    # Total samples across all clients
    total_samples = sum(n for _, n in client_updates)

    # Initialize aggregated dict
    aggregated = OrderedDict()

    for key in global_state_dict.keys():
        # Weighted sum of client parameters
        aggregated[key] = torch.zeros_like(global_state_dict[key], dtype=torch.float32)

        for client_state, num_samples in client_updates:
            weight = num_samples / total_samples
            aggregated[key] += weight * client_state[key].float()

        # Preserve original dtype (e.g., BatchNorm running stats are float)
        aggregated[key] = aggregated[key].to(global_state_dict[key].dtype)

    return aggregated
