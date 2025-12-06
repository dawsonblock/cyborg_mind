#!/bin/bash
# scripts/verify_treechop.sh
# Verification Run (Short Duration)

ENV_ID="MineRLTreechop-v0"
EXP_NAME="verify-treechop"

# Production settings but short duration
python3 mine_rl_train.py \
    --env $ENV_ID \
    --encoder gru \
    --memory pmm \
    --num-envs 2 \
    --steps 2000 \
    --smoke-test \
    --config configs/unified_config.yaml

echo "✅ Verification Run Completed"
