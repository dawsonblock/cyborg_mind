# CyborgMind V2.6 Production Dockerfile
# Multi-stage build for optimized production deployment

# ============================================================================
# Stage 1: Base image with CUDA support
# ============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    openjdk-8-jdk \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# ============================================================================
# Stage 2: Dependencies
# ============================================================================
FROM base AS dependencies

WORKDIR /tmp

# Install PyTorch with CUDA support
RUN pip3 install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
RUN pip3 install \
    numpy \
    gymnasium \
    stable-baselines3 \
    tensorboard \
    wandb \
    prometheus-client \
    fastapi \
    uvicorn[standard] \
    pydantic \
    python-multipart \
    opencv-python-headless \
    pillow \
    requests

# Install MineRL (optional)
RUN pip3 install minerl || echo "MineRL installation skipped"

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies AS application

WORKDIR /app

# Copy application code
COPY cyborg_mind_v2/ /app/cyborg_mind_v2/
COPY setup.py /app/
COPY README.md /app/

# Install package
RUN pip3 install -e .

# Create runtime directories
RUN mkdir -p /app/checkpoints /app/logs /app/data /app/models

# ============================================================================
# Stage 4: Production
# ============================================================================
FROM application AS production

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["python3", "-m", "uvicorn", "cyborg_mind_v2.deployment.api_server:app", \
     "--host", "0.0.0.0", "--port", "8000"]
