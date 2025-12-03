"""
Teacher–Student distillation using a real teacher for Cyborg Mind v2.0.

This script trains a ``BrainCyborgMind`` by distilling behaviour from
``RealTeacher``, a CLIP‑based vision model with trainable heads for
action and value prediction.  Training minimises a composite loss
combining Kullback–Leibler divergence for the action policies,
mean squared error for value estimates and a small regularisation on
the student's memory write vectors.  Memory pressure is monitored
and expansion is triggered automatically.  Optionally, the synthetic
Minecraft dataset from ``data/synthetic_dataset.py`` is used to
generate random observations.

The resulting model can be loaded by ``CyborgMindController`` for
inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..capsule_brain.policy.brain_cyborg_mind import BrainCyborgMind
from .real_teacher import RealTeacher
from ..data.synthetic_dataset import MinecraftSyntheticDataset, create_dataloader


@dataclass
class TrainerConfig:
    """Configuration for distillation training."""

    device: str = "cuda"
    num_steps: int = 100_000
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    temperature: float = 2.0
    value_weight: float = 0.5
    mem_reg_weight: float = 0.01
    grad_clip: float = 1.0
    synthetic_samples: int = 10_000
    ckpt_interval: int = 5_000
    log_interval: int = 100
    output_dir: str = "checkpoints"


def distillation_loss(
    student_out: any,
    teacher_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    temperature: float = 2.0,
    value_weight: float = 0.5,
    mem_reg_weight: float = 0.01,
) -> torch.Tensor:
    """
    Compute the distillation loss given student outputs and teacher targets.

    Parameters
    ----------
    student_out : any
        Output of ``BrainCyborgMind.forward``.  Must have attributes
        ``action_logits``, ``value`` and ``mem_write``.
    teacher_logits : torch.Tensor
        Teacher action logits of shape [B, A].
    teacher_values : torch.Tensor
        Teacher value estimates of shape [B, 1].
    temperature : float
        Softmax temperature for KL divergence.
    value_weight : float
        Weight for the value loss relative to the policy loss.
    mem_reg_weight : float
        Weight for the memory write regularisation term.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    # KL divergence on softened action logits
    student_log_probs = nn.functional.log_softmax(student_out.action_logits / temperature, dim=-1)
    teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    kl = nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    # Value MSE
    mse = nn.functional.mse_loss(student_out.value, teacher_values)
    # Regularise memory writes
    mem_norm = torch.mean(torch.norm(student_out.mem_write, dim=-1))
    return kl + value_weight * mse + mem_reg_weight * mem_norm


def train_student_real(cfg: TrainerConfig) -> None:
    """
    Train ``BrainCyborgMind`` with ``RealTeacher`` for a specified number of steps.

    Observations are drawn from ``MinecraftSyntheticDataset`` unless
    replaced with real data by modifying the dataloader.  Checkpoints
    and metrics are saved periodically.  Memory expansion is handled
    automatically based on pressure reported by the student's PMM.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    # Create student and teacher
    student = BrainCyborgMind().to(device)
    student.train()
    teacher = RealTeacher(device=str(device), num_actions=student.num_actions).to(device)
    teacher.eval()

    # Data loader
    loader = create_dataloader(num_samples=cfg.synthetic_samples, batch_size=cfg.batch_size, num_workers=0, shuffle=True)
    data_iter = iter(loader)

    # Optimiser
    optimiser = optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.output_dir, exist_ok=True)
    losses, accs, vmses, entropies = [], [], [], []

    for step in range(1, cfg.num_steps + 1):
        try:
            pixels, scalars, goals = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            pixels, scalars, goals = next(data_iter)
        pixels = pixels.to(device)
        scalars = scalars.to(device)
        goals = goals.to(device)
        B = pixels.size(0)
        # Prepare initial recurrent state
        thoughts = torch.zeros(B, student.thought_dim, device=device)
        emotions = torch.zeros(B, student.emotion_dim, device=device)
        workspaces = torch.zeros(B, student.workspace_dim, device=device)

        # Teacher predictions
        with torch.no_grad():
            t_logits, t_values = teacher.predict(pixels, scalars)
            t_logits = t_logits.to(device)
            t_values = t_values.to(device)

        # Student forward
        out = student(
            pixels,
            scalars,
            goals,
            thoughts,
            emotion=emotions,
            workspace=workspaces,
        )
        loss = distillation_loss(out, t_logits, t_values, cfg.temperature, cfg.value_weight, cfg.mem_reg_weight)

        # Backprop
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        optimiser.step()

        # Metrics
        with torch.no_grad():
            losses.append(loss.item())
            # Argmax accuracy against teacher
            s_actions = out.action_logits.argmax(dim=-1)
            t_actions = t_logits.argmax(dim=-1)
            accs.append((s_actions == t_actions).float().mean().item())
            vmses.append(nn.functional.mse_loss(out.value, t_values).item())
            probs = nn.functional.softmax(out.action_logits, dim=-1)
            log_probs = nn.functional.log_softmax(out.action_logits, dim=-1)
            entropies.append(-(probs * log_probs).sum(dim=-1).mean().item())

        # Memory expansion
        pressure = out.pressure
        if pressure > 0.85:
            if student.pmm.expand():
                # Reinitialise optimiser for new params
                optimiser = optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                print(f"[Trainer] Memory expanded to {student.pmm.mem_slots} slots at step {step}.")

        # Logging
        if step % cfg.log_interval == 0:
            avg_loss = sum(losses[-cfg.log_interval:]) / cfg.log_interval
            avg_acc = sum(accs[-cfg.log_interval:]) / cfg.log_interval
            avg_vmse = sum(vmses[-cfg.log_interval:]) / cfg.log_interval
            avg_ent = sum(entropies[-cfg.log_interval:]) / cfg.log_interval
            print(
                f"[step {step:6d}] loss={avg_loss:.4f} acc={avg_acc*100:.1f}% v_mse={avg_vmse:.4f} H={avg_ent:.4f} pressure={pressure:.3f}"
            )
        # Checkpoint
        if step % cfg.ckpt_interval == 0:
            ckpt_path = os.path.join(cfg.output_dir, f"student_mind_step_{step}.pt")
            torch.save(student.state_dict(), ckpt_path)
            print(f"[checkpoint] saved {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(cfg.output_dir, "student_mind_final.pt")
    torch.save(student.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    cfg = TrainerConfig()
    train_student_real(cfg)