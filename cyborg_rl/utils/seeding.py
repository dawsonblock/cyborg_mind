"""Seeding utilities for reproducibility."""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
