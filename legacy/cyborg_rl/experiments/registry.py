import os
import json
import time
import uuid
import yaml
import torch
import git
import platform
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class ExperimentRegistry:
    """
    Registry for tracking experiment metadata, configs, and artifacts.
    
    Features:
    - Auto-generates unique run IDs
    - Saves full config YAML
    - Tracks git commit hash and diff
    - Logs system info (OS, Python, PyTorch, GPU)
    - Manages checkpoints and metrics
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        run_name: Optional[str] = None,
        base_dir: Optional[str] = None,
    ):
        self.config = config
        self.run_id = str(uuid.uuid4())[:8]
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

        if run_name:
            self.run_name = f"{run_name}_{self.timestamp}_{self.run_id}"
        else:
            self.run_name = f"run_{self.timestamp}_{self.run_id}"

        # Setup paths
        if base_dir:
            self.base_dir = Path(base_dir) / self.run_name
        else:
            self.base_dir = Path("experiments") / "runs" / self.run_name
        self.ckpt_dir = self.base_dir / "checkpoints"
        self.logs_dir = self.base_dir / "logs"
        
        self._create_dirs()
        self._save_manifest()
        
        logger.info(f"Initialized ExperimentRegistry: {self.run_name}")
        logger.info(f"Artifacts will be saved to: {self.base_dir}")

    def _create_dirs(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

    def _get_git_info(self) -> Dict[str, str]:
        try:
            repo = git.Repo(search_parent_directories=True)
            return {
                "commit_hash": repo.head.object.hexsha,
                "branch": repo.active_branch.name,
                "is_dirty": repo.is_dirty(),
            }
        except Exception as e:
            logger.warning(f"Could not get git info: {e}")
            return {"error": str(e)}

    def _get_system_info(self) -> Dict[str, str]:
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": os.cpu_count(),
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
        return info

    def _save_manifest(self):
        manifest = {
            "run_name": self.run_name,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "git_info": self._get_git_info(),
            "system_info": self._get_system_info(),
            "config": self.config,
        }
        
        # Save manifest JSON
        with open(self.base_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)
            
        # Save Config YAML
        with open(self.base_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """Log metrics to a local CSV file (in addition to wandb/tb)."""
        metrics["step"] = step
        metrics["timestamp"] = time.time()
        
        csv_path = self.logs_dir / "metrics.csv"
        file_exists = csv_path.exists()
        
        with open(csv_path, "a") as f:
            if not file_exists:
                header = ",".join(metrics.keys())
                f.write(f"{header}\n")
            
            row = ",".join(str(v) for v in metrics.values())
            f.write(f"{row}\n")

    def save_checkpoint(self, state_dict: Dict[str, Any], step: int, is_best: bool = False):
        filename = f"checkpoint_{step}.pt"
        path = self.ckpt_dir / filename
        torch.save(state_dict, path)
        
        # Update 'latest' symlink or copy
        latest_path = self.ckpt_dir / "latest.pt"
        torch.save(state_dict, latest_path)
        
        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            torch.save(state_dict, best_path)
            logger.info(f"Saved new best checkpoint to {best_path}")

    def get_checkpoint_path(self, tag: str = "latest") -> Optional[Path]:
        path = self.ckpt_dir / f"{tag}.pt"
        if path.exists():
            return path
        return None
