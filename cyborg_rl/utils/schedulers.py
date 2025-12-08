#!/usr/bin/env python3
"""
Learning Rate Schedulers for V7 Training

Includes:
- WarmupCosineScheduler: Linear warmup + cosine decay
- WarmupLinearScheduler: Linear warmup + linear decay
- AdaptiveKLScheduler: Adjust LR based on KL divergence
"""

from typing import Optional
import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.
    
    LR Schedule:
        - Warmup: linearly increase from 0 to base_lr
        - Decay: cosine decay from base_lr to min_lr
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class WarmupLinearScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and linear decay.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            decay_factor = 1 - progress
            return [
                self.min_lr + (base_lr - self.min_lr) * decay_factor
                for base_lr in self.base_lrs
            ]


class AdaptiveKLScheduler:
    """
    Adaptive learning rate based on KL divergence.
    
    If KL is too high, decrease LR.
    If KL is too low, increase LR.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        target_kl: float = 0.02,
        kl_range: tuple = (0.5, 2.0),
        lr_range: tuple = (0.5, 2.0),
        initial_lr: Optional[float] = None,
    ) -> None:
        """
        Args:
            optimizer: PyTorch optimizer
            target_kl: Target KL divergence
            kl_range: (low_mult, high_mult) - thresholds relative to target
            lr_range: (min_mult, max_mult) - LR adjustment multipliers
            initial_lr: Initial learning rate (or use optimizer's current)
        """
        self.optimizer = optimizer
        self.target_kl = target_kl
        self.kl_low = target_kl * kl_range[0]
        self.kl_high = target_kl * kl_range[1]
        self.lr_decrease = lr_range[0]
        self.lr_increase = lr_range[1]
        
        self.initial_lr = initial_lr or optimizer.param_groups[0]["lr"]
        self.current_lr = self.initial_lr
    
    def step(self, kl: float) -> float:
        """
        Update LR based on current KL.
        
        Returns:
            New learning rate
        """
        if kl > self.kl_high:
            # KL too high -> decrease LR
            self.current_lr *= self.lr_decrease
        elif kl < self.kl_low:
            # KL too low -> increase LR (up to initial)
            self.current_lr = min(self.current_lr * self.lr_increase, self.initial_lr)
        
        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr
        
        return self.current_lr
    
    def get_lr(self) -> float:
        return self.current_lr


def create_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
    min_lr: float = 0.0,
    **kwargs,
) -> _LRScheduler:
    """
    Factory function to create schedulers.
    
    Args:
        scheduler_type: "cosine", "linear", or "constant"
        optimizer: PyTorch optimizer
        total_steps: Total training steps
        warmup_ratio: Fraction of steps for warmup
        min_lr: Minimum learning rate
    """
    warmup_steps = int(total_steps * warmup_ratio)
    
    if scheduler_type == "cosine":
        return WarmupCosineScheduler(
            optimizer, warmup_steps, total_steps, min_lr
        )
    elif scheduler_type == "linear":
        return WarmupLinearScheduler(
            optimizer, warmup_steps, total_steps, min_lr
        )
    elif scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
