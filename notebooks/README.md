# 📓 CyborgMind Notebooks

This directory contains Jupyter notebooks for training and experimenting with CyborgMind.

---

## 🚀 Quick Start: Google Colab

### Step 1: Open in Colab

Click this link to open the training notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dawsonblock/cyborg_mind/blob/main/notebooks/train_colab_cuberite.ipynb)

Or manually:
1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Open Notebook**
3. Select the **GitHub** tab
4. Enter: `dawsonblock/cyborg_mind`
5. Select `notebooks/train_colab_cuberite.ipynb`

---

### Step 2: Enable GPU

1. Click **Runtime → Change runtime type**
2. Select **GPU** (T4 is free, A100 is faster but requires Pro)
3. Click **Save**

---

### Step 3: Run the Notebook

Execute cells in order:

| Section | What it Does |
|---------|--------------|
| **1. Setup** | Installs dependencies, clones repo |
| **2. Cuberite** | Downloads Minecraft server for visualization |
| **3. VNC** | Sets up virtual display (optional) |
| **4. Config** | Customize training parameters |
| **5. Train** | Runs PPO training loop |
| **6. Checkpoint** | Saves and downloads trained model |
| **7. Evaluate** | Tests agent performance |

---

## ⚙️ Configuration Options

Edit the `CONFIG` cell to customize training:

```python
CONFIG = {
    # Environment
    'env_name': 'MineRLTreechop-v0',  # MineRL environment
    'num_envs': 2,                     # Parallel environments
    'frame_stack': 4,                  # Frames to stack
    
    # Model
    'encoder': 'gru',     # 'gru', 'mamba', or 'mamba_gru'
    'hidden_dim': 384,    # Hidden layer size
    
    # Memory (PMM)
    'pmm_enabled': True,  # Enable memory module
    'pmm_slots': 16,      # Memory slots
    
    # Training
    'total_steps': 100000,  # Increase for better results
    'learning_rate': 3e-4,
    
    # Logging
    'use_wandb': False,   # Enable for experiment tracking
}
```

---

## 📊 Recommended Settings

| Colab Tier | GPU | Recommended Steps | Training Time |
|------------|-----|-------------------|---------------|
| Free | T4 | 100,000 | ~30 min |
| Pro | A100 | 1,000,000 | ~2 hours |
| Pro+ | A100 | 8,000,000 | ~12 hours |

---

## 🎮 Viewing the Agent (Cuberite)

The notebook sets up a Cuberite Minecraft server. To connect:

1. **Install ngrok** (in a new cell):
   ```python
   !pip install pyngrok
   from pyngrok import ngrok
   tunnel = ngrok.connect(25565, "tcp")
   print(f"Connect Minecraft to: {tunnel.public_url}")
   ```

2. **Connect with Minecraft Java Edition** to the displayed address

> **Note**: Cuberite is a separate Minecraft server for visualization. The actual training uses MineRL's built-in environment.

---

## 🔧 Troubleshooting

### MineRL Installation Fails
```python
# Try installing without dependencies first
!pip install minerl --no-deps

# Then install required deps
!pip install gym numpy pillow
```

### GPU Out of Memory
Reduce these values in CONFIG:
- `num_envs`: 1
- `batch_size`: 1024
- `hidden_dim`: 256

### Session Disconnects
- Enable **Colab Pro** for longer sessions
- Use `--resume checkpoints/latest.pt` to continue training

---

## 📁 Available Notebooks

| Notebook | Description |
|----------|-------------|
| `train_colab_cuberite.ipynb` | Full training with Cuberite visualization |

---

## 💾 Saving Your Work

Checkpoints are saved to `/content/cyborg_mind/checkpoints/`. Download before session ends:

```python
from google.colab import files
files.download('/content/cyborg_mind/checkpoints/best.pt')
```

Or mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/cyborg_mind/checkpoints/*.pt /content/drive/MyDrive/
```
