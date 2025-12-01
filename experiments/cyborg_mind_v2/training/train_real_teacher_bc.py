# cyborg_mind_v2/training/train_real_teacher_bc.py

import os
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

import minerl  # MineRL dataset

from torch.utils.tensorboard import SummaryWriter

from .real_teacher import RealTeacher
from ..envs.action_mapping import (
    NUM_ACTIONS,
    minerl_action_to_index,
)


# ---------------------------
# 1. Obs -> pixels, scalars
# ---------------------------

def obs_to_teacher_inputs(obs: Dict[str, Any], image_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert MineRL observation dict -> (pixels, scalars) for RealTeacher.

    pixels: [3, H, W], float32 in [0,1], resized to image_size.
    scalars: [20] float32, currently zeros (extend later).
    """
    pov = obs["pov"]  # [H, W, 3], uint8
    pov_resized = cv2.resize(pov, (image_size, image_size), interpolation=cv2.INTER_AREA)
    pixels = pov_resized.astype(np.float32) / 255.0
    pixels = np.transpose(pixels, (2, 0, 1))  # [3, H, W]

    scalars = np.zeros(20, dtype=np.float32)

    return pixels, scalars


# ---------------------------
# 3. Training loop
# ---------------------------

def train_real_teacher_bc(
    env_name: str,
    data_dir: str,
    output_ckpt: str,
    device: str = "cuda",
    epochs: int = 1,
    max_seq_len: int = 64,
    batch_size: int = 64,
    lr: float = 3e-4,
    max_grad_norm: float = 1.0,
    num_actions: int = NUM_ACTIONS,
    log_dir: str = "runs/real_teacher_bc",
    lr_schedule: bool = True,
) -> None:
    """
    Enhanced behavioral cloning trainer for RealTeacher.
    
    Improvements:
    - Gradient clipping for stability
    - Optional learning rate scheduling
    - Better buffer management
    - Accuracy logging
    """
    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"

    print(f"[BC] Using device: {device}")
    print(f"[BC] Loading dataset {env_name} from {data_dir} ...")

    data = minerl.data.make(environment=env_name, data_dir=data_dir)

    teacher = RealTeacher(
        ckpt_path=None,  # we are training from scratch
        device=device,
        num_actions=num_actions,
        scalar_dim=20,
    ).to(device)

    params = [p for p in teacher.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    
    # Learning rate scheduler
    scheduler = None
    if lr_schedule:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * 1000, eta_min=lr * 0.1
        )
    
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=log_dir)

    teacher.train()

    # Pre-allocate buffers for efficiency
    pixel_buffer: List[np.ndarray] = []
    scalar_buffer: List[np.ndarray] = []
    action_buffer: List[int] = []

    global_step = 0
    best_loss = float('inf')

    def flush_batch():
        nonlocal global_step, best_loss
        if not pixel_buffer:
            return 0.0

        pixels_np = np.stack(pixel_buffer, axis=0)  # [B, 3, H, W]
        scalars_np = np.stack(scalar_buffer, axis=0)  # [B, 20]
        actions_np = np.array(action_buffer, dtype=np.int64)

        # Clear buffers efficiently
        pixel_buffer.clear()
        scalar_buffer.clear()
        action_buffer.clear()

        pixels_t = torch.from_numpy(pixels_np).to(device)
        scalars_t = torch.from_numpy(scalars_np).to(device)
        actions_t = torch.from_numpy(actions_np).to(device)

        logits, _ = teacher.predict(pixels_t, scalars_t)
        loss = criterion(logits, actions_t)

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        global_step += 1
        loss_val = float(loss.item())
        
        # Track best loss
        if loss_val < best_loss:
            best_loss = loss_val
        
        writer.add_scalar("loss/train_ce", loss_val, global_step)
        writer.add_scalar("loss/best", best_loss, global_step)
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("train/lr", current_lr, global_step)

        # Compute accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == actions_t).float().mean().item()
            writer.add_scalar("train/accuracy", acc, global_step)
            
            probs = torch.softmax(logits, dim=-1).mean(0)
            for idx in range(min(num_actions, 20)):  # Log first 20 actions
                writer.add_scalar(
                    f"action_prob/idx_{idx}", 
                    float(probs[idx].item()), 
                    global_step
                )

        return loss_val

    for epoch in range(epochs):
        print(f"[BC] Epoch {epoch+1}/{epochs}")
        seq_iter = data.sarsd_iter(num_epochs=1, max_sequence_len=max_seq_len)

        for obs_seq, act_seq, rew_seq, next_obs_seq, done_seq in seq_iter:
            T = len(obs_seq["pov"])
            for t in range(T):
                obs_t = {k: v[t] for k, v in obs_seq.items()}
                act_t = {k: v[t] for k, v in act_seq.items()}

                pixels, scalars = obs_to_teacher_inputs(obs_t)
                action_idx = minerl_action_to_index(act_t)

                pixel_buffer.append(pixels)
                scalar_buffer.append(scalars)
                action_buffer.append(action_idx)

                if len(pixel_buffer) >= batch_size:
                    loss_value = flush_batch()
                    if global_step % 50 == 0:
                        print(f"[BC] step={global_step} loss={loss_value:.4f}")

        if pixel_buffer:
            loss_value = flush_batch()
            print(f"[BC] final epoch flush step={global_step} loss={loss_value:.4f}")

        os.makedirs(os.path.dirname(output_ckpt), exist_ok=True)
        torch.save(
            {
                "state_dict": teacher.state_dict(),
                "env_name": env_name,
                "num_actions": num_actions,
            },
            output_ckpt,
        )
        print(f"[BC] Saved teacher checkpoint to: {output_ckpt}")

    writer.close()
    print("[BC] Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Behavioral cloning for RealTeacher on MineRL dataset.")
    parser.add_argument("--env-name", type=str, default="MineRLTreechop-v0")
    parser.add_argument("--data-dir", type=str, default=os.path.expanduser("~/.minerl"))
    parser.add_argument("--output", type=str, default="checkpoints/real_teacher_bc.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-dir", type=str, default="runs/real_teacher_bc")

    args = parser.parse_args()

    train_real_teacher_bc(
        env_name=args.env_name,
        data_dir=args.data_dir,
        output_ckpt=args.output,
        device=args.device,
        epochs=args.epochs,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
