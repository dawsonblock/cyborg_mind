#!/bin/bash
# scripts/train_treechop.sh
# Launcher for Cyborg MineRL Treechop Training (v3.0)

# Settings
ENV_ID="MineRLTreechop-v0"
EXP_NAME="cyborg-v3-treechop-gru-pmm"
DEVICE="auto" # or "cuda"/"mps"

# Hyperparameters
NUM_ENVS=4          # Adjust based on CPU cores
TOTAL_STEPS=1000000 # 1M steps
ENCODER="gru"       # "gru", "mamba", "mamba_gru"
MEMORY="pmm"        # "pmm", "none"

echo "🚀 Launching Cyborg MineRL Training: $EXP_NAME"
echo "Env: $ENV_ID | Encoder: $ENCODER | Memory: $MEMORY"

python3 mine_rl_train.py \
    --env $ENV_ID \
    --encoder $ENCODER \
    --memory $MEMORY \
    --num-envs $NUM_ENVS \
    --wandb \
    --config configs/unified_config.yaml

echo "✅ Training Job Finished"
