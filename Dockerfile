# CyborgMind RL - Production-grade Multi-stage Dockerfile
# Supports CPU and GPU builds with optimized layers

# ==============================================================================
# Base stage - Common dependencies
# ==============================================================================
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONFAULTHANDLER=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Builder stage - Install Python dependencies
# ==============================================================================
FROM base AS builder

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ==============================================================================
# Development stage - Full dev environment
# ==============================================================================
FROM builder AS development

# Install dev dependencies
RUN pip install pytest pytest-cov black flake8 mypy ipython jupyter

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install -e ".[dev]"

# Default command for development
CMD ["bash"]

# ==============================================================================
# Production stage - Optimized runtime
# ==============================================================================
FROM base AS production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary files
COPY cyborg_rl/ ./cyborg_rl/
COPY scripts/ ./scripts/
COPY pyproject.toml .
COPY requirements.txt .

# Install package
RUN pip install -e .

# Create directories for runtime
RUN mkdir -p /app/checkpoints /app/logs /app/data && \
    chmod -R 755 /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash cyborg && \
    chown -R cyborg:cyborg /app
USER cyborg

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/metrics || exit 1

# Expose ports
EXPOSE 8000 8001

# Default entrypoint
# Default entrypoint
ENTRYPOINT ["python"]
CMD ["launcher.py", "train", "--env", "CartPole-v1", "--steps", "100000"]

# ==============================================================================
# GPU stage - CUDA-enabled runtime
# ==============================================================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS gpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121

# Copy source code
COPY cyborg_rl/ ./cyborg_rl/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p /app/checkpoints /app/logs /app/data

# Expose ports
EXPOSE 8000 8001

ENTRYPOINT ["python"]
CMD ["launcher.py", "train", "--env", "CartPole-v1", "--steps", "100000", "--device", "cuda"]
