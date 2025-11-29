#!/bin/bash
set -e

echo "==================================================================="
echo "  CyborgMind V2 - TreeChop PPO Training"
echo "==================================================================="
echo ""

# Configuration
EXPERIMENT_NAME="treechop_ppo"
ENV_NAME="MineRLTreechop-v0"
OUTPUT_DIR="checkpoints"
LOGS_DIR="logs/treechop_ppo"
RESULTS_DIR="docs/results"
NUM_EPISODES=1000
SAVE_INTERVAL=100
DEVICE="cuda"

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOGS_DIR}
mkdir -p ${RESULTS_DIR}

# Print configuration
echo "Configuration:"
echo "  Environment: ${ENV_NAME}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Device: ${DEVICE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Logs: ${LOGS_DIR}"
echo ""

# Run training
python -u cyborg_mind_v2/training/train_cyborg_mind_ppo.py \
    --env ${ENV_NAME} \
    --adapter minerl \
    --num-episodes ${NUM_EPISODES} \
    --save-interval ${SAVE_INTERVAL} \
    --output-dir ${OUTPUT_DIR} \
    --logs-dir ${LOGS_DIR} \
    --device ${DEVICE} \
    --experiment-name ${EXPERIMENT_NAME} \
    --learning-rate 3e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2 \
    --entropy-coef 0.01 \
    --value-loss-coef 0.5 \
    --max-grad-norm 0.5 \
    --batch-size 64 \
    --num-epochs 10

# Generate plots
echo ""
echo "Generating result plots..."
python << 'PLOT_EOF'
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_results():
    """Generate plots from training logs."""
    logs_dir = Path("logs/treechop_ppo")
    results_dir = Path("docs/results")
    results_dir.mkdir(exist_ok=True, parents=True)

    # Load metrics
    metrics_file = logs_dir / "metrics.json"
    if not metrics_file.exists():
        print("No metrics file found, skipping plots")
        return

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    if not metrics:
        print("No metrics data, skipping plots")
        return

    episodes = [m["episode"] for m in metrics]
    rewards = [m["reward"] for m in metrics]
    losses = [m.get("loss", 0) for m in metrics]
    values = [m.get("value", 0) for m in metrics]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("CyborgMind V2 - TreeChop PPO Training Results", fontsize=16)

    # Plot 1: Reward curve
    axes[0, 0].plot(episodes, rewards, label="Episode Reward", alpha=0.6)
    if len(rewards) > 10:
        # Moving average
        window = min(50, len(rewards) // 10)
        ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ma_episodes = episodes[window-1:]
        axes[0, 0].plot(ma_episodes, ma_rewards, label=f"MA({window})", linewidth=2)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Loss curve
    axes[0, 1].plot(episodes, losses, label="Policy Loss", color="orange")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Training Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Value estimates
    axes[1, 0].plot(episodes, values, label="Value Estimate", color="green")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_title("Value Estimates")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Reward distribution
    axes[1, 1].hist(rewards, bins=30, alpha=0.7, color="purple")
    axes[1, 1].axvline(np.mean(rewards), color="red", linestyle="--", label=f"Mean: {np.mean(rewards):.2f}")
    axes[1, 1].set_xlabel("Reward")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Reward Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = results_dir / "treechop_ppo.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    # Print summary statistics
    print("\nTraining Summary:")
    print(f"  Total Episodes: {len(episodes)}")
    print(f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Max Reward: {np.max(rewards):.2f}")
    print(f"  Final 100 Episode Mean: {np.mean(rewards[-100:]):.2f}")

if __name__ == "__main__":
    plot_training_results()
PLOT_EOF

# Print completion message
echo ""
echo "==================================================================="
echo "  Training Complete!"
echo "==================================================================="
echo ""
echo "Checkpoint saved to: ${OUTPUT_DIR}/treechop_brain.pt"
echo "TensorBoard logs: ${LOGS_DIR}"
echo "Results plot: ${RESULTS_DIR}/treechop_ppo.png"
echo ""
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir=${LOGS_DIR}"
echo ""
echo "==================================================================="
