# Installing Missing Dependencies

Quick guide to install missing dependencies identified by `verify_setup.py`.

## Core Dependencies

### Install minigrid (Required for Gym environments)

```bash
pip install minigrid==2.3.1
```

### Install slowapi (Required for API rate limiting)

```bash
pip install slowapi==0.1.9
```

## Optional Dependencies

### Install MineRL (For Treechop training)

```bash
# Option 1: Use setup script
./setup_minerl.sh

# Option 2: Manual installation
pip install minerl shimmy gymnasium[all]
```

### Install Mamba-SSM (For Mamba encoder)

```bash
# Option 1: Use setup script (requires CUDA)
./setup_mamba_gpu.sh

# Option 2: Manual installation
pip install mamba-ssm>=1.1.1 causal-conv1d>=1.1.0
```

## Install Package in Development Mode

To make `cyborg_rl` importable without path manipulation:

```bash
# Install in editable mode
pip install -e .
```

## Verify Installation

After installing dependencies:

```bash
python scripts/verify_setup.py
```

## Quick Fix for All Core Dependencies

```bash
# Install all core dependencies
pip install minigrid==2.3.1 slowapi==0.1.9

# Install package in development mode
pip install -e .

# Verify
python scripts/verify_setup.py
```

## Docker Alternative

If you prefer not to install locally, use Docker:

```bash
# Build image (includes all dependencies)
docker build -t cyborg-mind:latest .

# Run verification in container
docker run --rm cyborg-mind:latest python scripts/verify_setup.py
```

## Troubleshooting

### "No module named 'cyborg_rl'"

**Solution 1:** Install in development mode
```bash
pip install -e .
```

**Solution 2:** Add to PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### "No module named 'minigrid'"

```bash
pip install minigrid==2.3.1
```

### "No module named 'slowapi'"

```bash
pip install slowapi==0.1.9
```

### Mamba-SSM installation fails

Mamba requires CUDA. If you don't have GPU:
- Use `--backbone gru` or `--backbone pseudo_mamba` instead
- The system will automatically fall back to GRU if Mamba is unavailable

### MineRL installation issues

MineRL requires Java. Install OpenJDK:

```bash
# Ubuntu/Debian
sudo apt-get install openjdk-8-jdk

# macOS
brew install openjdk@8

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64  # Linux
export JAVA_HOME=/usr/local/opt/openjdk@8           # macOS
```

## Minimal Working Setup

For basic functionality without MineRL or Mamba:

```bash
# Install core dependencies
pip install -r requirements.txt
pip install minigrid==2.3.1 slowapi==0.1.9

# Install package
pip install -e .

# Test with Gym environments
python scripts/train_gym_cartpole.py
```

## Full Setup (All Features)

```bash
# Install all dependencies
pip install -r requirements.txt
pip install minigrid==2.3.1 slowapi==0.1.9

# Install optional dependencies
./setup_minerl.sh
./setup_mamba_gpu.sh  # Only if you have CUDA

# Install package
pip install -e .

# Verify
python scripts/verify_setup.py
```
