#!/usr/bin/env python3
"""
PopArt Value Normalization

Implements Preserving Outputs Precisely, while Adaptively Rescaling Targets (PopArt).
Normalizes value targets to improve learning stability in environments with
varying reward magnitudes.

Reference: https://arxiv.org/abs/1602.07714
"""

from typing import Tuple
import torch
import torch.nn as nn


class PopArt(nn.Module):
    """
    PopArt: Adaptive value normalization for PPO.
    
    Maintains running mean/std of value targets and normalizes
    the value head output accordingly.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        beta: float = 0.0001,
        epsilon: float = 1e-5,
    ) -> None:
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (usually 1 for value)
            beta: Update rate for running statistics
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        
        self.beta = beta
        self.epsilon = epsilon
        
        # Linear layer for value prediction
        self.weight = nn.Parameter(torch.zeros(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Running statistics (not trainable)
        self.register_buffer("mean", torch.zeros(output_dim))
        self.register_buffer("std", torch.ones(output_dim))
        self.register_buffer("count", torch.zeros(1))
        
        # Initialize weights
        nn.init.orthogonal_(self.weight, gain=1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning denormalized value prediction.
        
        Args:
            x: Input features (B, D)
            
        Returns:
            Denormalized value (B, 1)
        """
        normalized_value = torch.addmm(self.bias, x, self.weight.t())
        return normalized_value * self.std + self.mean
    
    def forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized value (for training)."""
        return torch.addmm(self.bias, x, self.weight.t())
    
    def normalize(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize targets using current statistics."""
        return (targets - self.mean) / (self.std + self.epsilon)
    
    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize values."""
        return normalized * self.std + self.mean
    
    @torch.no_grad()
    def update_stats(self, targets: torch.Tensor) -> None:
        """
        Update running mean/std from new targets.
        Uses Welford's online algorithm.
        """
        batch_mean = targets.mean(dim=0)
        batch_var = targets.var(dim=0)
        batch_count = targets.shape[0]
        
        # Update count
        new_count = self.count + batch_count
        
        # Update mean
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / new_count
        
        # Update variance (Welford's algorithm)
        m_a = self.std.pow(2) * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / new_count
        new_var = m2 / new_count
        new_std = (new_var + self.epsilon).sqrt()
        
        # PopArt: preserve output magnitudes
        self._preserve_outputs(new_mean, new_std)
        
        # Update buffers
        self.mean.copy_(new_mean)
        self.std.copy_(new_std)
        self.count.copy_(new_count)
    
    def _preserve_outputs(
        self, new_mean: torch.Tensor, new_std: torch.Tensor
    ) -> None:
        """Adjust weights/biases to preserve output distribution."""
        # Scale weights
        self.weight.data.mul_(self.std / new_std)
        
        # Adjust bias
        self.bias.data.mul_(self.std / new_std)
        self.bias.data.add_((self.mean - new_mean) / new_std)


class PopArtValueHead(nn.Module):
    """
    Value head with PopArt normalization.
    
    Usage:
        value_head = PopArtValueHead(input_dim=256)
        value = value_head(features)  # Denormalized value
        
        # During training:
        value_head.update(returns)
        normalized_value = value_head.forward_normalized(features)
        loss = (normalized_value - value_head.normalize(returns)).pow(2)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        beta: float = 0.0001,
    ) -> None:
        super().__init__()
        
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.popart = PopArt(hidden_dim, output_dim=1, beta=beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return denormalized value."""
        features = self.base(x)
        return self.popart(features)
    
    def forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized value for training."""
        features = self.base(x)
        return self.popart.forward_normalized(features)
    
    def normalize(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize targets."""
        return self.popart.normalize(targets)
    
    def update(self, targets: torch.Tensor) -> None:
        """Update statistics from new returns."""
        self.popart.update_stats(targets)
