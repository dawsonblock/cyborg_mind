#!/usr/bin/env python3
"""
Action Distribution Logger

Logs action distribution statistics, entropy, and concentration metrics
for debugging policy behavior.
"""

from typing import Any, Dict, List, Optional
from collections import deque
import numpy as np
import torch


class ActionLogger:
    """
    Logs action distribution statistics over training.
    
    Tracks:
    - Action frequency histogram
    - Entropy over time
    - Top-K action concentration
    - Action switching rate
    """
    
    def __init__(
        self,
        num_actions: int,
        history_size: int = 10000,
    ) -> None:
        """
        Args:
            num_actions: Number of discrete actions
            history_size: Number of steps to retain
        """
        self.num_actions = num_actions
        self.history_size = history_size
        
        # Counts
        self.action_counts = np.zeros(num_actions, dtype=np.int64)
        self.total_count = 0
        
        # Time series
        self.entropy_history: deque = deque(maxlen=history_size)
        self.top1_prob_history: deque = deque(maxlen=history_size)
        
        # Recent actions for switching rate
        self.recent_actions: deque = deque(maxlen=100)
        
    def log_action(
        self,
        action: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Log a batch of actions.
        
        Args:
            action: Action indices (B,) or (B, N)
            probs: Optional action probabilities (B, A) or (B, N, A)
            
        Returns:
            Current metrics
        """
        actions = action.detach().cpu().numpy().flatten()
        
        # Update counts
        for a in actions:
            self.action_counts[a] += 1
            self.total_count += 1
        
        # Recent actions
        self.recent_actions.extend(actions.tolist())
        
        metrics = {}
        
        if probs is not None:
            probs_np = probs.detach().cpu().numpy()
            if probs_np.ndim == 3:
                probs_np = probs_np.reshape(-1, probs_np.shape[-1])
            
            # Entropy
            entropy = -np.sum(probs_np * np.log(probs_np + 1e-8), axis=-1)
            avg_entropy = entropy.mean()
            self.entropy_history.append(avg_entropy)
            metrics["entropy"] = float(avg_entropy)
            
            # Top-1 probability
            top1_prob = probs_np.max(axis=-1).mean()
            self.top1_prob_history.append(top1_prob)
            metrics["top1_prob"] = float(top1_prob)
            
            # Concentration (Gini coefficient)
            metrics["concentration"] = self._gini(probs_np.mean(axis=0))
        
        return metrics
    
    def _gini(self, probs: np.ndarray) -> float:
        """Calculate Gini coefficient for action concentration."""
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        index = np.arange(1, n + 1)
        return float((np.sum((2 * index - n - 1) * sorted_probs)) / (n * np.sum(sorted_probs)))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        freq = self.action_counts / max(1, self.total_count)
        
        # Switching rate
        if len(self.recent_actions) > 1:
            recent = list(self.recent_actions)
            switches = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
            switch_rate = switches / (len(recent) - 1)
        else:
            switch_rate = 0.0
        
        stats = {
            "action_frequency": freq.tolist(),
            "most_common_action": int(np.argmax(self.action_counts)),
            "least_common_action": int(np.argmin(self.action_counts)),
            "total_actions": self.total_count,
            "switch_rate": switch_rate,
        }
        
        if self.entropy_history:
            stats["avg_entropy"] = float(np.mean(list(self.entropy_history)))
            stats["entropy_trend"] = float(
                np.mean(list(self.entropy_history)[-100:]) - 
                np.mean(list(self.entropy_history)[:100])
            ) if len(self.entropy_history) > 100 else 0.0
        
        if self.top1_prob_history:
            stats["avg_top1_prob"] = float(np.mean(list(self.top1_prob_history)))
        
        return stats
    
    def get_histogram(self) -> Dict[int, float]:
        """Get action frequency histogram."""
        return {i: float(c / max(1, self.total_count)) for i, c in enumerate(self.action_counts)}
    
    def reset(self) -> None:
        """Reset all counts."""
        self.action_counts.fill(0)
        self.total_count = 0
        self.entropy_history.clear()
        self.top1_prob_history.clear()
        self.recent_actions.clear()
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ActionLogger(total={stats['total_actions']}, "
            f"most_common={stats['most_common_action']}, "
            f"switch_rate={stats['switch_rate']:.3f})"
        )
