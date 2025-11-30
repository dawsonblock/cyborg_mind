"""Device management utilities for GPU-correct operations."""

from typing import Union, TypeVar
import torch
import torch.nn as nn

T = TypeVar("T", torch.Tensor, nn.Module)


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate torch device.

    Args:
        device: Device specification. "auto" selects CUDA if available.

    Returns:
        torch.device: The selected device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def to_device(x: T, device: torch.device) -> T:
    """
    Move tensor or module to device.

    Args:
        x: Tensor or module to move.
        device: Target device.

    Returns:
        The input moved to the specified device.
    """
    return x.to(device)


def ensure_tensor(
    x: Union[torch.Tensor, list, tuple, float, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Ensure input is a tensor on the correct device.

    Args:
        x: Input data (tensor, list, tuple, or scalar).
        device: Target device.
        dtype: Target dtype.

    Returns:
        torch.Tensor: Tensor on the specified device.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)
