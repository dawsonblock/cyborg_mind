# CyborgMind V2 Architecture Overview

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Brain Components](#brain-components)
3. [Environment Adapter System](#environment-adapter-system)
4. [Training Pipeline](#training-pipeline)
5. [Deployment Architecture](#deployment-architecture)

---

## System Architecture

CyborgMind V2 is a unified emotion-consciousness brain for RL agents. It combines:
- **Vision processing** via frozen adapters
- **Dynamic GPU PMM memory** (auto-expands)
- **LSTM temporal processing**
- **FRNN workspace** (global consciousness)
- **Emotion system** (8-channel affect model)
- **Thought vectors** (32D persistent cognition)

```
┌─────────────────────────────────────────────────────────────┐
│                    CyborgMind V2 System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   MineRL     │    │     Gym      │    │    CC3D      │  │
│  │   Adapter    │    │   Adapter    │    │   Adapter    │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │           │
│         └───────────────────┴───────────────────┘           │
│                            │                                │
│         ┌──────────────────▼─────────────────────┐          │
│         │  Universal Environment Interface       │          │
│         │  (pixels, scalars, goal) → action_idx  │          │
│         └──────────────────┬─────────────────────┘          │
│                            │                                │
│  ╔═════════════════════════▼═════════════════════════════╗  │
│  ║              BrainCyborgMind                          ║  │
│  ║                                                       ║  │
│  ║  ┌──────────┐  ┌──────────┐  ┌──────────┐           ║  │
│  ║  │  Vision  │  │   PMM    │  │   LSTM   │           ║  │
│  ║  │ Adapter  │  │ Memory   │  │  Core    │           ║  │
│  ║  └─────┬────┘  └────┬─────┘  └─────┬────┘           ║  │
│  ║        └────────────┴──────────────┬┘                ║  │
│  ║                                    │                 ║  │
│  ║  ┌───────────┬────────────────────┴──────────────┐  ║  │
│  ║  │  Emotion  │  Thought  │  Workspace  │  FRNN  │  ║  │
│  ║  └───────────┴───────────┴─────────────┴────────┘  ║  │
│  ║                                                       ║  │
│  ║  Output: action, value, emotion, thought, workspace  ║  │
│  ╚═══════════════════════════════════════════════════════╝  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Brain Components

### 1. Vision Adapter
- **Input**: RGB images [B, 3, H, W]
- **Architecture**: CNN (3 conv layers + FC)
- **Output**: vision_dim=512 features
- **Status**: Can be frozen or fine-tuned

### 2. Dynamic GPU PMM
- **Purpose**: Content-addressable episodic memory
- **Initial slots**: 256
- **Max slots**: 2048
- **Expansion**: Auto-expands when pressure > 0.85
- **Key features**:
  - Cosine similarity retrieval
  - Least-used eviction
  - Garbage collection
  - Pressure metrics

### 3. LSTM Core
- **Input**: [embedding + memory_readout]
- **Hidden dim**: 512
- **Purpose**: Temporal reasoning
- **Output**: hidden states for heads

### 4. FRNN Workspace
- **Dim**: 64
- **Purpose**: Global conscious workspace
- **Dynamics**: Fully recurrent (all-to-all connections)
- **Theory**: Based on Global Workspace Theory

### 5. Emotion System
- **Channels**: 8 (valence, arousal, dominance, joy, fear, anger, sadness, surprise)
- **Range**: [-1, 1]
- **Update**: Every forward pass via emotion head
- **Integration**: Fed back to encoder on next step

### 6. Thought Vector
- **Dim**: 32
- **Purpose**: Persistent cognitive state
- **Stability**: Anchored every 10 steps to prevent runaway
- **Clipping**: [-3, 3] bounds

---

## Environment Adapter System

### Design Principles
- **Universal interface**: All envs → (pixels, scalars, goal)
- **Action abstraction**: Brain outputs discrete indices
- **Type-safe**: Uses Python Protocols
- **Extensible**: Add new envs without changing brain

### Adapter Structure

```python
class BrainEnvAdapter(Protocol):
    def reset(self) -> BrainInputs:
        """Return (pixels, scalars, goal)"""
        ...

    def step(self, action_idx: int) -> Tuple[BrainInputs, float, bool, Dict]:
        """Execute action, return (obs, reward, done, info)"""
        ...
```

### Available Adapters

**MineRLAdapter**
- Env: MineRLTreechop-v0, Navigate, etc.
- Actions: 19 discrete (movement + camera + attack)
- Scalars: inventory + compass + step ratio
- Goal: one-hot task encoding

**GymAdapter**
- Env: CartPole, Atari, etc.
- Actions: Env-specific or discretized
- Scalars: state + step ratio
- Goal: one-hot task encoding

**CC3DAdapter (stub)**
- Env: CompuCell3D biological sims
- Ready for future integration

---

## Training Pipeline

### 1. PPO Training
```bash
# Configure
vim configs/treechop_ppo.yaml

# Train
bash experiments/run_treechop_ppo.sh
```

Flow:
1. Create adapter
2. Collect rollout (4096 steps)
3. Compute GAE returns
4. PPO update (4 epochs, 256 minibatch)
5. Memory expansion if pressure high
6. Log to TensorBoard

### 2. Behavior Cloning
```bash
bash experiments/run_treechop_teacher_bc.sh
```

Flow:
1. Load MineRL demonstrations
2. Sample (pixels, scalars, goal, action) tuples
3. Supervised learning on action prediction
4. Evaluate on validation set
5. Save best checkpoint

### 3. Distillation Pipeline
```bash
# 1. Train teacher with BC
bash experiments/run_treechop_teacher_bc.sh

# 2. Distill to student
python cyborg_mind_v2/training/train_distillation_minerl.py

# 3. Fine-tune with PPO
bash experiments/run_treechop_ppo.sh --load checkpoints/real_teacher_treechop.pt
```

---

## Deployment Architecture

### API Server
```bash
uvicorn cyborg_mind_v2.deployment.api_server:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /reset`: Initialize agent
- `POST /step`: Get action from observation
- `GET /state/{agent_id}`: Query brain state
- `GET /metrics`: Performance metrics

### Web Visualizer
```bash
# Serve frontend
cd frontend/demo
python -m http.server 8080

# Open http://localhost:8080
```

Features:
- Real-time emotion visualization
- Thought vector heatmap
- Memory pressure tracking
- Action/value display

### Production Deployment

**Docker Container**:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

COPY . /app
RUN pip install -e /app

EXPOSE 8000
CMD ["uvicorn", "cyborg_mind_v2.deployment.api_server:app", "--host", "0.0.0.0"]
```

**Scaling**:
- Single brain instance can handle multiple agents
- Per-agent state stored in AgentStateManager
- GPU batching for parallel inference

---

## Data Flow Example

```
Environment Step:
  1. Env observation → Adapter.obs_to_brain_inputs()
  2. BrainInputs (pixels, scalars, goal)
  3. Brain.forward() with previous (thought, emotion, workspace, hidden)
  4. Output (action_logits, value, new_thought, new_emotion, new_workspace, new_hidden)
  5. Sample action from logits
  6. Adapter.step(action_idx) → env_action
  7. Env.step(env_action) → next_obs, reward, done

Brain Internal Flow:
  1. Vision(pixels) → vision_features
  2. Encoder([vision, scalars, goal, thought, emotion, workspace]) → embedding
  3. Align(embedding) → query
  4. PMM(query) → memory_readout
  5. LSTM([embedding, memory]) → hidden
  6. ActionHead(hidden) → action_logits
  7. ValueHead(hidden) → value
  8. EmotionHead(hidden) → new_emotion
  9. ThoughtHead(hidden) → new_thought
  10. WorkspaceHead(hidden) → workspace_update
  11. FRNN(workspace_update, prev_workspace) → new_workspace
  12. SelfWriter(new_workspace) → write_modulation
  13. PMM.write(mem_head(hidden) * write_modulation)
```

---

## Configuration System

**YAML Configs** (`configs/`):
- `treechop_ppo.yaml`: MineRL PPO training
- `treechop_bc.yaml`: MineRL BC training
- `gym_cartpole.yaml`: Gym CartPole demo
- `synthetic.yaml`: Synthetic data training

**Usage**:
```python
import yaml

with open("configs/treechop_ppo.yaml") as f:
    config = yaml.safe_load(f)

# Access nested values
lr = config["ppo"]["learning_rate"]
```

---

## Performance Characteristics

**Inference Speed**:
- Single agent: ~100 FPS (GPU)
- Batch 32 agents: ~1500 FPS (GPU)

**Memory Usage**:
- Brain weights: ~10 MB
- PMM (256 slots): ~1 MB
- Per-agent state: ~50 KB

**Training**:
- PPO (200k steps): ~2-4 hours (single GPU)
- BC (50 epochs): ~1-2 hours

---

## Extension Points

1. **New Environment**: Implement `BrainEnvAdapter`
2. **Custom Reward**: Modify adapter's reward shaping
3. **Larger Memory**: Increase `max_slots` in PMM
4. **Multimodal Input**: Add audio/text encoders to fusion
5. **Hierarchical Control**: Add option critics
6. **Multi-Agent**: Implement communication protocol

---

For detailed API docs, see `docs/API.md`
For deployment guide, see `docs/DEPLOYMENT.md`
