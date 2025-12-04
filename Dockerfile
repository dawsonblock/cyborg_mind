FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# MineRL requires OpenJDK 8
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    openjdk-17-jdk-headless \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MineRL specifically (if not in requirements)
RUN pip install --no-cache-dir minerl shimmy gymnasium[all]

# Copy source code
COPY . .

# Install package
RUN pip install -e .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MINERL_DATA_ROOT=/app/data/minerl

# Expose API port
EXPOSE 8000

# Default command: Run API server
# Expects CONFIG_PATH and CHECKPOINT_PATH env vars or args
CMD ["python", "scripts/run_api_server.py", "--config", "configs/treechop_ppo.yaml", "--checkpoint", "artifacts/minerl_treechop/latest/best_model.pt"]
