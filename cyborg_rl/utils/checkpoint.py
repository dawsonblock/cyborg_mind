#!/usr/bin/env python3
"""
Checkpoint Manager for V7 Training

Features:
- Automatic checkpoint saving (periodic, best, latest)
- Resume from checkpoint
- Model export (ONNX, TorchScript)
- Checkpoint metrics tracking
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import time
import shutil

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages training checkpoints with automatic saving and resumption.
    
    Usage:
        ckpt = CheckpointManager("checkpoints/", max_to_keep=5)
        
        # Save
        ckpt.save(step=1000, model=model, optimizer=optimizer, metrics={"reward": 100})
        
        # Resume
        state = ckpt.load_latest()
        model.load_state_dict(state["model"])
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_to_keep: int = 5,
        keep_best: int = 3,
        metric_name: str = "reward",
        metric_mode: str = "max",
    ) -> None:
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_to_keep: Maximum number of periodic checkpoints to keep
            keep_best: Number of best checkpoints to keep
            metric_name: Metric to use for best checkpoint selection
            metric_mode: "max" or "min" for best metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        self.checkpoints: List[Dict[str, Any]] = []
        self.best_checkpoints: List[Dict[str, Any]] = []
        
        # Load existing checkpoint history
        self._load_history()
    
    def _load_history(self) -> None:
        """Load checkpoint history from manifest."""
        manifest_path = self.checkpoint_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)
                self.checkpoints = data.get("checkpoints", [])
                self.best_checkpoints = data.get("best", [])
    
    def _save_history(self) -> None:
        """Save checkpoint history to manifest."""
        manifest_path = self.checkpoint_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "checkpoints": self.checkpoints,
                "best": self.best_checkpoints,
            }, f, indent=2)
    
    def save(
        self,
        step: int,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        **state_dicts,
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            step: Current training step
            model: Model to save
            optimizer: Optimizer to save
            scheduler: LR scheduler to save
            metrics: Training metrics
            extra: Extra data to save
            is_best: Force save as best
            **state_dicts: Additional state dicts (e.g., encoder=encoder.state_dict())
        
        Returns:
            Path to saved checkpoint
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_step{step}_{timestamp}.pt"
        filepath = self.checkpoint_dir / filename
        
        # Build checkpoint
        checkpoint = {
            "step": step,
            "timestamp": timestamp,
            "metrics": metrics or {},
        }
        
        if model is not None:
            checkpoint["model"] = model.state_dict()
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()
        if extra is not None:
            checkpoint["extra"] = extra
        
        checkpoint.update(state_dicts)
        
        # Save
        torch.save(checkpoint, filepath)
        
        # Update history
        ckpt_info = {
            "path": str(filepath),
            "step": step,
            "timestamp": timestamp,
            "metrics": metrics or {},
        }
        self.checkpoints.append(ckpt_info)
        
        # Check if best
        if metrics and self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            if self._is_best(metric_value) or is_best:
                self.best_checkpoints.append(ckpt_info)
                self._prune_best()
        
        # Prune old checkpoints
        self._prune_checkpoints()
        
        # Save latest symlink
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists():
            latest_path.unlink()
        shutil.copy(filepath, latest_path)
        
        self._save_history()
        
        return str(filepath)
    
    def _is_best(self, value: float) -> bool:
        """Check if value is the best so far."""
        if not self.best_checkpoints:
            return True
        
        best_values = [
            c["metrics"].get(self.metric_name, float("-inf") if self.metric_mode == "max" else float("inf"))
            for c in self.best_checkpoints
        ]
        
        if self.metric_mode == "max":
            return value > min(best_values)
        else:
            return value < max(best_values)
    
    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_to_keep."""
        # Keep only non-best checkpoints for pruning
        best_paths = {c["path"] for c in self.best_checkpoints}
        
        while len(self.checkpoints) > self.max_to_keep:
            oldest = self.checkpoints[0]
            if oldest["path"] not in best_paths:
                path = Path(oldest["path"])
                if path.exists():
                    path.unlink()
                self.checkpoints.pop(0)
            else:
                # Don't delete best checkpoints
                self.checkpoints.pop(0)
    
    def _prune_best(self) -> None:
        """Keep only top-N best checkpoints."""
        if len(self.best_checkpoints) <= self.keep_best:
            return
        
        # Sort by metric
        reverse = self.metric_mode == "max"
        self.best_checkpoints.sort(
            key=lambda x: x["metrics"].get(self.metric_name, 0),
            reverse=reverse,
        )
        
        # Remove worst
        while len(self.best_checkpoints) > self.keep_best:
            worst = self.best_checkpoints.pop()
            # Don't delete file, it might still be in regular checkpoints
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load a specific checkpoint."""
        return torch.load(path, map_location="cpu", weights_only=False)
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint."""
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists():
            return self.load(latest_path)
        
        if self.checkpoints:
            return self.load(self.checkpoints[-1]["path"])
        
        return None
    
    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint."""
        if not self.best_checkpoints:
            return None
        
        # Sort by metric
        reverse = self.metric_mode == "max"
        sorted_best = sorted(
            self.best_checkpoints,
            key=lambda x: x["metrics"].get(self.metric_name, 0),
            reverse=reverse,
        )
        
        return self.load(sorted_best[0]["path"])
    
    def get_resume_step(self) -> int:
        """Get step to resume from."""
        if self.checkpoints:
            return self.checkpoints[-1]["step"]
        return 0
    
    def export_onnx(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        filename: str = "model.onnx",
    ) -> str:
        """Export model to ONNX format."""
        export_path = self.checkpoint_dir / filename
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        return str(export_path)
    
    def export_torchscript(
        self,
        model: nn.Module,
        filename: str = "model.pt",
    ) -> str:
        """Export model to TorchScript."""
        export_path = self.checkpoint_dir / filename
        scripted = torch.jit.script(model)
        scripted.save(export_path)
        return str(export_path)
