"""Global determinism control.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> dict:
    """Seed python/numpy/torch and (optionally) force deterministic cuDNN.

    Returns the seed state as a dict.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

    return {
        "seed": seed,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
    }


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn so that augmentation RNG is reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
