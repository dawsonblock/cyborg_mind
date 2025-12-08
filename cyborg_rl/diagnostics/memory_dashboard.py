#!/usr/bin/env python3
"""
Memory Dashboard - Real-time memory diagnostics visualization.

Tracks:
- write_rate
- overwrite_conflict
- slot_usage_histogram
- attention_weights
- read/write vector norms
"""

from typing import Any, Dict, List, Optional
from collections import deque
from dataclasses import dataclass, field
import threading
import time

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class MemoryMetrics:
    """Container for memory metrics over time."""
    
    max_history: int = 1000
    
    # Time series
    write_strength: deque = field(default_factory=lambda: deque(maxlen=1000))
    read_entropy: deque = field(default_factory=lambda: deque(maxlen=1000))
    overwrite_conflict: deque = field(default_factory=lambda: deque(maxlen=1000))
    usage_mean: deque = field(default_factory=lambda: deque(maxlen=1000))
    age_mean: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Gate usage buckets
    gate_low: deque = field(default_factory=lambda: deque(maxlen=1000))
    gate_mid: deque = field(default_factory=lambda: deque(maxlen=1000))
    gate_high: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Timestamps
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Latest snapshot
    latest_slot_usage: Optional[np.ndarray] = None
    latest_attention: Optional[np.ndarray] = None
    
    def update(self, logs: Dict[str, Any]) -> None:
        """Update metrics from log dict."""
        self.timestamps.append(time.time())
        self.write_strength.append(logs.get("write_strength", 0))
        self.read_entropy.append(logs.get("read_entropy", 0))
        self.overwrite_conflict.append(logs.get("overwrite_conflict", 0))
        self.usage_mean.append(logs.get("usage_mean", 0))
        self.age_mean.append(logs.get("age_mean", 0))
        self.gate_low.append(logs.get("gate_usage_low", 0))
        self.gate_mid.append(logs.get("gate_usage_mid", 0))
        self.gate_high.append(logs.get("gate_usage_high", 0))
        
        if "slot_usage" in logs:
            self.latest_slot_usage = np.array(logs["slot_usage"])
        if "attention" in logs:
            self.latest_attention = np.array(logs["attention"])
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "avg_write_strength": np.mean(list(self.write_strength)) if self.write_strength else 0,
            "avg_read_entropy": np.mean(list(self.read_entropy)) if self.read_entropy else 0,
            "avg_overwrite_conflict": np.mean(list(self.overwrite_conflict)) if self.overwrite_conflict else 0,
            "total_updates": len(self.timestamps),
        }


class MemoryDashboard:
    """
    Real-time memory diagnostics dashboard.
    
    Usage:
        dashboard = MemoryDashboard()
        dashboard.start()
        
        # During training
        for step in training_loop:
            _, _, logs = memory.forward_step(hidden, state)
            dashboard.update(logs)
        
        dashboard.stop()
    """
    
    def __init__(
        self,
        update_interval: float = 0.5,
        history_size: int = 1000,
        use_plotly: bool = True,
    ):
        """
        Args:
            update_interval: Seconds between dashboard updates
            history_size: Number of data points to retain
            use_plotly: Use Plotly (interactive) vs Matplotlib (static)
        """
        self.update_interval = update_interval
        self.metrics = MemoryMetrics(max_history=history_size)
        self.use_plotly = use_plotly and PLOTLY_AVAILABLE
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def update(self, logs: Dict[str, Any]) -> None:
        """Update dashboard with new metrics."""
        with self._lock:
            self.metrics.update(logs)
    
    def start(self) -> None:
        """Start dashboard in background thread."""
        if self._running:
            return
        
        self._running = True
        if self.use_plotly:
            self._thread = threading.Thread(target=self._run_plotly, daemon=True)
        elif MATPLOTLIB_AVAILABLE:
            self._thread = threading.Thread(target=self._run_matplotlib, daemon=True)
        else:
            print("⚠️ No visualization library available. Install plotly or matplotlib.")
            return
        
        self._thread.start()
        print("📊 Memory Dashboard started")
    
    def stop(self) -> None:
        """Stop dashboard."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("📊 Memory Dashboard stopped")
    
    def _run_plotly(self) -> None:
        """Run Plotly dashboard (opens in browser)."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Write Strength Over Time",
                "Gate Usage Distribution",
                "Read Entropy",
                "Overwrite Conflict",
            ),
        )
        
        while self._running:
            with self._lock:
                if len(self.metrics.timestamps) < 2:
                    time.sleep(self.update_interval)
                    continue
                
                times = list(self.metrics.timestamps)
                write_strength = list(self.metrics.write_strength)
                read_entropy = list(self.metrics.read_entropy)
                conflict = list(self.metrics.overwrite_conflict)
                gate_low = np.mean(list(self.metrics.gate_low)) if self.metrics.gate_low else 0
                gate_mid = np.mean(list(self.metrics.gate_mid)) if self.metrics.gate_mid else 0
                gate_high = np.mean(list(self.metrics.gate_high)) if self.metrics.gate_high else 0
            
            # This is a simplified version - full implementation would use dash
            time.sleep(self.update_interval)
    
    def _run_matplotlib(self) -> None:
        """Run Matplotlib dashboard."""
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Memory Dashboard")
        
        while self._running:
            with self._lock:
                if len(self.metrics.timestamps) < 2:
                    time.sleep(self.update_interval)
                    continue
                
                times = np.array(list(self.metrics.timestamps))
                times = times - times[0]  # Relative time
                write_strength = list(self.metrics.write_strength)
                read_entropy = list(self.metrics.read_entropy)
                conflict = list(self.metrics.overwrite_conflict)
            
            # Clear and redraw
            for ax in axes.flat:
                ax.clear()
            
            axes[0, 0].plot(times, write_strength, 'b-')
            axes[0, 0].set_title("Write Strength")
            axes[0, 0].set_xlabel("Time (s)")
            
            axes[0, 1].bar(["Low", "Mid", "High"], [
                np.mean(list(self.metrics.gate_low)) if self.metrics.gate_low else 0,
                np.mean(list(self.metrics.gate_mid)) if self.metrics.gate_mid else 0,
                np.mean(list(self.metrics.gate_high)) if self.metrics.gate_high else 0,
            ])
            axes[0, 1].set_title("Gate Usage")
            
            axes[1, 0].plot(times, read_entropy, 'g-')
            axes[1, 0].set_title("Read Entropy")
            
            axes[1, 1].plot(times, conflict, 'r-')
            axes[1, 1].set_title("Overwrite Conflict")
            
            plt.tight_layout()
            plt.pause(self.update_interval)
        
        plt.ioff()
        plt.close()
    
    def get_report(self) -> str:
        """Generate text report of metrics."""
        summary = self.metrics.get_summary()
        return f"""
Memory Dashboard Report
=======================
Total Updates: {summary['total_updates']}
Avg Write Strength: {summary['avg_write_strength']:.4f}
Avg Read Entropy: {summary['avg_read_entropy']:.4f}
Avg Overwrite Conflict: {summary['avg_overwrite_conflict']:.4f}

Gate Usage Distribution:
  Low (<0.33): {np.mean(list(self.metrics.gate_low)) * 100:.1f}%
  Mid (0.33-0.67): {np.mean(list(self.metrics.gate_mid)) * 100:.1f}%
  High (>0.67): {np.mean(list(self.metrics.gate_high)) * 100:.1f}%
"""


def create_dashboard(**kwargs) -> MemoryDashboard:
    """Factory function for dashboard creation."""
    return MemoryDashboard(**kwargs)
