"""
Synthetic Minecraft‑style dataset for Cyborg Mind v2.0

This dataset generates random pixel frames and game state vectors to
simulate inputs to the brain.  It is not meant to resemble real
Minecraft frames but serves as a quick stand‑in to test the training
pipeline before integrating a true environment or dataset such as
OpenAI's VPT dataset.  Each sample consists of:

* **pixels**: [3, 128, 128] tensor of random uint8 values in [0,255].
* **scalars**: [20] tensor where the first 5 entries are continuous
  values (health, hunger, x, y, z) scaled to [0,100] and the
  remaining 15 entries are binary inventory flags.
* **goals**: [4] tensor with random floats in [0,1] representing
  high‑level objectives (explore, build, fight, mine).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader


class MinecraftSyntheticDataset(Dataset):
    """
    Generate synthetic Minecraft‑like data for training.

    Parameters
    ----------
    num_samples : int, optional
        Number of samples in the dataset.  Defaults to 10 000.
    image_size : int, optional
        Spatial resolution of the square images.  Defaults to 128.
    """

    def __init__(self, num_samples: int = 10_000, image_size: int = 128):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # Random RGB image
        pixels = torch.randint(
            0,
            256,
            size=(3, self.image_size, self.image_size),
            dtype=torch.float32,
        )
        # Continuous scalars: health, hunger, x, y, z in [0,100]
        continuous = torch.rand(5) * 100.0
        # Binary inventory: 15 bits
        inventory = torch.randint(0, 2, (15,), dtype=torch.float32)
        scalars = torch.cat([continuous, inventory], dim=0)  # [20]
        # Goals: random 4‑dim vector in [0,1]
        goals = torch.rand(4)
        return pixels, scalars, goals


def create_dataloader(
    num_samples: int = 10_000,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    Helper to construct a DataLoader for the synthetic dataset.
    """
    dataset = MinecraftSyntheticDataset(num_samples=num_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )