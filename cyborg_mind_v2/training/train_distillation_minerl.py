"""
Teacher-Student distillation using RealTeacher on actual MineRL data.

This script extends the synthetic distillation trainer to work with real
MineRL demonstration data. It loads trajectories from the MineRL dataset,
uses RealTeacher to generate action and value predictions, and trains
BrainCyborgMind to match the teacher's behavior.

The script handles the MineRL data pipeline, observation preprocessing,
and provides options for training on different tasks (TreeChop, Navigate, etc.).
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Optional, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

try:
    import minerl
    MINERL_AVAILABLE = True
except ImportError:
    MINERL_AVAILABLE = False
    print("Warning: MineRL not available. Install with: pip install gym==0.21.0 minerl==0.4.4")

from ..capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind
from .real_teacher import RealTeacher
from ..envs.minerl_obs_adapter import obs_to_brain
from ..data.synthetic_dataset import create_dataloader


@dataclass
class MineRLDistillationConfig:
    """Configuration for MineRL distillation training."""

    # Environment settings
    env_name: str = "MineRLTreechop-v0"
    data_dir: str = "data/minerl"
    use_synthetic: bool = False  # Fallback to synthetic if MineRL unavailable
    synthetic_samples: int = 10_000

    # Training settings
    device: str = "cuda"
    num_steps: int = 100_000
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Distillation settings
    temperature: float = 2.0
    value_weight: float = 0.5
    mem_reg_weight: float = 0.01

    # Logging and checkpointing
    log_interval: int = 100
    ckpt_interval: int = 5_000
    output_dir: str = "checkpoints"
    tensorboard_dir: str = "runs/distill_minerl"

    # Teacher settings
    teacher_ckpt: Optional[str] = None  # Path to pretrained teacher


class MineRLDataIterator:
    """Iterator that yields batches from MineRL demonstration data."""

    def __init__(
        self,
        env_name: str,
        data_dir: str,
        batch_size: int,
        max_trajectory_length: int = 1000,
    ):
        self.env_name = env_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_trajectory_length = max_trajectory_length

        # Load MineRL data
        self.data = minerl.data.make(environment=env_name, data_dir=data_dir)
        self.trajectory_names = self.data.get_trajectory_names()
        print(f"[MineRLDataIterator] Loaded {len(self.trajectory_names)} trajectories from {env_name}")

        # Buffers for batching
        self.buffer_pixels = []
        self.buffer_scalars = []
        self.buffer_goals = []

    def __iter__(self):
        """Iterate over trajectories and yield batches."""
        while True:
            # Sample random trajectory
            traj_name = np.random.choice(self.trajectory_names)

            try:
                for obs, action, reward, next_obs, done in self.data.load_data(traj_name, skip_interval=0):
                    # Convert observation to brain format
                    pixels, scalars, goal = obs_to_brain(obs)

                    # Add to buffer
                    self.buffer_pixels.append(pixels)
                    self.buffer_scalars.append(scalars)
                    self.buffer_goals.append(goal)

                    # Yield batch when buffer is full
                    if len(self.buffer_pixels) >= self.batch_size:
                        batch_pixels = torch.from_numpy(np.stack(self.buffer_pixels[:self.batch_size]))
                        batch_scalars = torch.from_numpy(np.stack(self.buffer_scalars[:self.batch_size]))
                        batch_goals = torch.from_numpy(np.stack(self.buffer_goals[:self.batch_size]))

                        # Clear buffer
                        self.buffer_pixels = self.buffer_pixels[self.batch_size:]
                        self.buffer_scalars = self.buffer_scalars[self.batch_size:]
                        self.buffer_goals = self.buffer_goals[self.batch_size:]

                        yield batch_pixels, batch_scalars, batch_goals

            except Exception as e:
                print(f"Warning: Error loading trajectory {traj_name}: {e}")
                continue


def distillation_loss(
    student_out,
    teacher_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    temperature: float = 2.0,
    value_weight: float = 0.5,
    mem_reg_weight: float = 0.01,
) -> torch.Tensor:
    """
    Compute the distillation loss.

    Uses KL divergence for action policies and MSE for values,
    plus a small regularization on memory writes.
    """
    # KL divergence on softened action logits
    student_log_probs = nn.functional.log_softmax(
        student_out["action_logits"] / temperature, dim=-1
    )
    teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    kl = nn.functional.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean"
    ) * (temperature ** 2)

    # Value MSE
    mse = nn.functional.mse_loss(student_out["value"], teacher_values)

    # Regularise memory writes (encourage sparse writes)
    mem_norm = torch.mean(torch.norm(student_out["mem_write"], dim=-1))

    return kl + value_weight * mse + mem_reg_weight * mem_norm


def train_distillation_minerl(cfg: MineRLDistillationConfig) -> None:
    """
    Train BrainCyborgMind with RealTeacher on MineRL data.

    If MineRL is not available or use_synthetic is True, falls back
    to synthetic data for testing the pipeline.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Using device: {device}")

    # Create student and teacher
    student = BrainCyborgMind().to(device)
    student.train()
    print(f"[Trainer] Student parameters: {sum(p.numel() for p in student.parameters()):,}")

    teacher = RealTeacher(
        ckpt_path=cfg.teacher_ckpt,
        device=str(device),
        num_actions=student.num_actions
    ).to(device)
    teacher.eval()
    print(f"[Trainer] Teacher loaded (frozen)")

    # Data loader
    if MINERL_AVAILABLE and not cfg.use_synthetic:
        print(f"[Trainer] Using MineRL data from {cfg.env_name}")
        data_iter = iter(MineRLDataIterator(
            env_name=cfg.env_name,
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
        ))
    else:
        print(f"[Trainer] Using synthetic data ({cfg.synthetic_samples} samples)")
        loader = create_dataloader(
            num_samples=cfg.synthetic_samples,
            batch_size=cfg.batch_size,
            num_workers=0,
            shuffle=True
        )
        data_iter = iter(loader)

    # Optimizer
    optimizer = optim.AdamW(
        student.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=cfg.tensorboard_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Metrics
    losses, accs, vmses, entropies, pressures = [], [], [], [], []

    print(f"[Trainer] Starting training for {cfg.num_steps} steps...")

    for step in range(1, cfg.num_steps + 1):
        # Get batch
        try:
            pixels, scalars, goals = next(data_iter)
        except StopIteration:
            # Restart iterator (for synthetic data)
            if not MINERL_AVAILABLE or cfg.use_synthetic:
                loader = create_dataloader(
                    num_samples=cfg.synthetic_samples,
                    batch_size=cfg.batch_size,
                    num_workers=0,
                    shuffle=True
                )
                data_iter = iter(loader)
                pixels, scalars, goals = next(data_iter)
            else:
                raise

        pixels = pixels.to(device)
        scalars = scalars.to(device)
        goals = goals.to(device)
        B = pixels.size(0)

        # Initialize recurrent state
        thoughts = torch.zeros(B, student.thought_dim, device=device)
        emotions = torch.zeros(B, student.emotion_dim, device=device)
        workspaces = torch.zeros(B, student.workspace_dim, device=device)

        # Teacher predictions (frozen)
        with torch.no_grad():
            t_logits, t_values = teacher.predict(pixels, scalars)

        # Student forward
        out = student(
            pixels,
            scalars,
            goals,
            thoughts,
            emotion=emotions,
            workspace=workspaces,
        )

        # Compute loss
        loss = distillation_loss(
            out, t_logits, t_values,
            cfg.temperature, cfg.value_weight, cfg.mem_reg_weight
        )

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            losses.append(loss.item())

            # Action accuracy
            s_actions = out["action_logits"].argmax(dim=-1)
            t_actions = t_logits.argmax(dim=-1)
            accs.append((s_actions == t_actions).float().mean().item())

            # Value MSE
            vmses.append(nn.functional.mse_loss(out["value"], t_values).item())

            # Policy entropy
            probs = nn.functional.softmax(out["action_logits"], dim=-1)
            log_probs = nn.functional.log_softmax(out["action_logits"], dim=-1)
            entropies.append(-(probs * log_probs).sum(dim=-1).mean().item())

            # Memory pressure
            pressure = out["pressure"].item()
            pressures.append(pressure)

        # Memory expansion
        if pressure > 0.85:
            if student.pmm.expand():
                print(f"[step {step}] Memory expanded to {student.pmm.mem_slots} slots (pressure={pressure:.3f})")
                # Reinitialize optimizer for new parameters
                optimizer = optim.AdamW(
                    student.parameters(),
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay
                )

        # Logging
        if step % cfg.log_interval == 0:
            avg_loss = np.mean(losses[-cfg.log_interval:])
            avg_acc = np.mean(accs[-cfg.log_interval:])
            avg_vmse = np.mean(vmses[-cfg.log_interval:])
            avg_ent = np.mean(entropies[-cfg.log_interval:])
            avg_pressure = np.mean(pressures[-cfg.log_interval:])

            print(
                f"[step {step:6d}] "
                f"loss={avg_loss:.4f} "
                f"acc={avg_acc*100:.1f}% "
                f"v_mse={avg_vmse:.4f} "
                f"H={avg_ent:.4f} "
                f"pressure={avg_pressure:.3f}"
            )

            # TensorBoard
            writer.add_scalar("train/loss", avg_loss, step)
            writer.add_scalar("train/accuracy", avg_acc, step)
            writer.add_scalar("train/value_mse", avg_vmse, step)
            writer.add_scalar("train/entropy", avg_ent, step)
            writer.add_scalar("train/pressure", avg_pressure, step)
            writer.add_scalar("train/memory_slots", student.pmm.mem_slots, step)

        # Checkpoint
        if step % cfg.ckpt_interval == 0:
            ckpt_path = os.path.join(cfg.output_dir, f"student_mind_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"[checkpoint] saved {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(cfg.output_dir, "student_mind_final.pt")
    torch.save({
        'step': cfg.num_steps,
        'model_state_dict': student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"[Trainer] Training complete. Final model saved to {final_path}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BrainCyborgMind via distillation from RealTeacher")
    parser.add_argument("--env", type=str, default="MineRLTreechop-v0", help="MineRL environment name")
    parser.add_argument("--data-dir", type=str, default="data/minerl", help="MineRL data directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of MineRL")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--steps", type=int, default=100_000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="Path to pretrained teacher checkpoint")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for checkpoints")

    args = parser.parse_args()

    cfg = MineRLDistillationConfig(
        env_name=args.env,
        data_dir=args.data_dir,
        use_synthetic=args.synthetic,
        device=args.device,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        teacher_ckpt=args.teacher_ckpt,
        output_dir=args.output_dir,
    )

    train_distillation_minerl(cfg)
