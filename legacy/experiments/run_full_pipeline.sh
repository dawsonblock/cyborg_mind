#!/bin/bash
set -e

echo "==================================================================="
echo "  CyborgMind V2 - Complete Training Pipeline"
echo "  Teacher BC → Student Distillation → PPO Fine-tuning"
echo "==================================================================="
echo ""

# Configuration
ENV_NAME="MineRLTreechop-v0"
DEVICE="cuda"
OUTPUT_DIR="checkpoints"
LOGS_DIR="logs"

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOGS_DIR}

# ========== PHASE 1: Train Teacher with BC ==========
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 1: Training RealTeacher with Behavior Cloning"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bash experiments/run_treechop_teacher_bc.sh

if [ ! -f "${OUTPUT_DIR}/real_teacher_treechop.pt" ]; then
    echo "ERROR: Teacher checkpoint not found"
    exit 1
fi

echo "✓ Phase 1 complete: Teacher trained"

# ========== PHASE 2: Distill to Student ==========
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 2: Distilling Knowledge to Student Brain"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python -u cyborg_mind_v2/training/train_distillation_minerl.py \
    --teacher-checkpoint ${OUTPUT_DIR}/real_teacher_treechop.pt \
    --env ${ENV_NAME} \
    --num-epochs 30 \
    --batch-size 64 \
    --learning-rate 5e-4 \
    --output-dir ${OUTPUT_DIR} \
    --logs-dir ${LOGS_DIR}/distillation \
    --device ${DEVICE} \
    --kd-temperature 2.0 \
    --kd-alpha 0.7

if [ ! -f "${OUTPUT_DIR}/distilled_student.pt" ]; then
    echo "ERROR: Distilled student checkpoint not found"
    exit 1
fi

echo "✓ Phase 2 complete: Student distilled"

# ========== PHASE 3: PPO Fine-tuning ==========
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 3: Fine-tuning with PPO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python -u cyborg_mind_v2/training/train_cyborg_mind_ppo.py \
    --env ${ENV_NAME} \
    --adapter minerl \
    --load-checkpoint ${OUTPUT_DIR}/distilled_student.pt \
    --num-episodes 500 \
    --save-interval 50 \
    --output-dir ${OUTPUT_DIR} \
    --logs-dir ${LOGS_DIR}/ppo_finetune \
    --device ${DEVICE} \
    --learning-rate 1e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95

if [ ! -f "${OUTPUT_DIR}/treechop_brain.pt" ]; then
    echo "ERROR: Final PPO checkpoint not found"
    exit 1
fi

echo "✓ Phase 3 complete: PPO fine-tuned"

# ========== PHASE 4: Evaluation ==========
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 4: Evaluating Final Agent"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python << 'EVAL_EOF'
import torch
import gym
import minerl
import numpy as np
from cyborg_mind_v2.integration.cyborg_mind_controller import CyborgMindController

# Load trained agent
controller = CyborgMindController(
    ckpt_path="checkpoints/treechop_brain.pt",
    device="cuda"
)

# Evaluate
env = gym.make("MineRLTreechop-v0")
num_eval_episodes = 10
rewards = []

for ep in range(num_eval_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Convert obs to brain format (simplified)
        pixels = torch.from_numpy(obs["pov"]).permute(2, 0, 1).float() / 255.0
        pixels = pixels.unsqueeze(0)
        scalars = torch.zeros(1, 20)
        goal = torch.tensor([[1, 0, 0, 0]])

        # Get action
        actions = controller.step(["eval_agent"], pixels, scalars, goal)
        action_idx = actions[0]

        # Map to MineRL action (simplified)
        minerl_action = {
            "camera": [0, 0],
            "forward": 1 if action_idx == 1 else 0,
            "attack": 1 if action_idx == 6 else 0,
        }

        obs, reward, done, _ = env.step(minerl_action)
        episode_reward += reward

    rewards.append(episode_reward)
    print(f"Episode {ep+1}: Reward = {episode_reward:.2f}")

print(f"\nMean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
env.close()
EVAL_EOF

# ========== Summary ==========
echo ""
echo "==================================================================="
echo "  🎉 Pipeline Complete!"
echo "==================================================================="
echo ""
echo "Checkpoints:"
echo "  - Teacher (BC):       ${OUTPUT_DIR}/real_teacher_treechop.pt"
echo "  - Student (Distill):  ${OUTPUT_DIR}/distilled_student.pt"
echo "  - Final (PPO):        ${OUTPUT_DIR}/treechop_brain.pt"
echo ""
echo "Logs:"
echo "  - Teacher:            ${LOGS_DIR}/teacher_bc"
echo "  - Distillation:       ${LOGS_DIR}/distillation"
echo "  - PPO:                ${LOGS_DIR}/ppo_finetune"
echo ""
echo "Next steps:"
echo "  - Deploy API: uvicorn cyborg_mind_v2.deployment.api_server:app"
echo "  - Visualize: Open frontend/demo/index.html"
echo "  - Fine-tune more: bash experiments/run_treechop_ppo.sh"
echo ""
echo "==================================================================="
