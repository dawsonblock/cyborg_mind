#!/bin/bash
set -e

echo "==================================================================="
echo "  CyborgMind V2 - TreeChop Behavior Cloning (RealTeacher)"
echo "==================================================================="
echo ""

# Configuration
EXPERIMENT_NAME="real_teacher_treechop"
ENV_NAME="MineRLTreechop-v0"
OUTPUT_DIR="checkpoints"
LOGS_DIR="logs/teacher_bc"
RESULTS_DIR="docs/results"
NUM_EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=1e-4
DEVICE="cuda"

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOGS_DIR}
mkdir -p ${RESULTS_DIR}

# Print configuration
echo "Configuration:"
echo "  Environment: ${ENV_NAME}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Device: ${DEVICE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Logs: ${LOGS_DIR}"
echo ""

# Download MineRL dataset if not present
echo "Checking for MineRL dataset..."
python << 'DATASET_EOF'
import minerl
import os

data_dir = os.path.join(os.path.expanduser("~"), ".minerl", "data")
print(f"Dataset directory: {data_dir}")

if not os.path.exists(data_dir):
    print("Downloading MineRL dataset... (this may take a while)")
    data = minerl.data.make('MineRLTreechop-v0')
    print("Dataset download complete")
else:
    print("Dataset already exists")
DATASET_EOF

# Run BC training
echo ""
echo "Starting Behavior Cloning training..."
python -u cyborg_mind_v2/training/train_real_teacher_bc.py \
    --env ${ENV_NAME} \
    --adapter minerl \
    --num-epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --output-dir ${OUTPUT_DIR} \
    --logs-dir ${LOGS_DIR} \
    --device ${DEVICE} \
    --experiment-name ${EXPERIMENT_NAME} \
    --max-samples 100000 \
    --validation-split 0.1 \
    --save-best

# Generate plots
echo ""
echo "Generating result plots..."
python << 'PLOT_EOF'
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_bc_results():
    """Generate plots from BC training logs."""
    logs_dir = Path("logs/teacher_bc")
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

    epochs = [m["epoch"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    train_acc = [m.get("train_accuracy", 0) for m in metrics]
    val_loss = [m.get("val_loss", 0) for m in metrics]
    val_acc = [m.get("val_accuracy", 0) for m in metrics]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("CyborgMind V2 - TreeChop Behavior Cloning Results", fontsize=16)

    # Plot 1: Training loss
    axes[0, 0].plot(epochs, train_loss, label="Train Loss", color="blue")
    if val_loss and any(val_loss):
        axes[0, 0].plot(epochs, val_loss, label="Val Loss", color="orange")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs, train_acc, label="Train Accuracy", color="green")
    if val_acc and any(val_acc):
        axes[0, 1].plot(epochs, val_acc, label="Val Accuracy", color="red")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Training and Validation Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # Plot 3: Loss improvement
    if len(train_loss) > 1:
        loss_delta = [train_loss[i] - train_loss[i-1] for i in range(1, len(train_loss))]
        axes[1, 0].plot(epochs[1:], loss_delta, label="Loss Delta", color="purple")
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss Delta")
        axes[1, 0].set_title("Loss Improvement per Epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    Training Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Total Epochs: {len(epochs)}

    Final Train Loss: {train_loss[-1]:.4f}
    Best Train Loss: {min(train_loss):.4f}

    Final Train Acc: {train_acc[-1]:.2%}
    Best Train Acc: {max(train_acc):.2%}

    """
    if val_loss and any(val_loss):
        summary_text += f"""
    Final Val Loss: {val_loss[-1]:.4f}
    Best Val Loss: {min([v for v in val_loss if v > 0]):.4f}

    Final Val Acc: {val_acc[-1]:.2%}
    Best Val Acc: {max(val_acc):.2%}
    """

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()

    # Save plot
    output_path = results_dir / "treechop_bc.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    # Print summary statistics
    print("\nBehavior Cloning Summary:")
    print(f"  Total Epochs: {len(epochs)}")
    print(f"  Final Train Loss: {train_loss[-1]:.4f}")
    print(f"  Final Train Accuracy: {train_acc[-1]:.2%}")
    if val_loss and any(val_loss):
        print(f"  Final Val Loss: {val_loss[-1]:.4f}")
        print(f"  Final Val Accuracy: {val_acc[-1]:.2%}")

if __name__ == "__main__":
    plot_bc_results()
PLOT_EOF

# Print completion message
echo ""
echo "==================================================================="
echo "  Behavior Cloning Training Complete!"
echo "==================================================================="
echo ""
echo "Checkpoint saved to: ${OUTPUT_DIR}/real_teacher_treechop.pt"
echo "TensorBoard logs: ${LOGS_DIR}"
echo "Results plot: ${RESULTS_DIR}/treechop_bc.png"
echo ""
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir=${LOGS_DIR}"
echo ""
echo "==================================================================="
