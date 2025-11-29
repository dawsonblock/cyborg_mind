FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Upgrade pip and install build tools
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install the package in editable mode with all dependencies
RUN pip3 install --no-cache-dir -e .

# Install additional deployment-specific dependencies
RUN pip3 install --no-cache-dir -r cyborg_mind_v2/deployment/requirements.txt

# Create directories for checkpoints and logs
RUN mkdir -p /app/checkpoints /app/runs /app/logs

# Expose port for monitoring (Prometheus)
EXPOSE 9090

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command: run the real teacher-student distillation trainer
CMD ["python3", "-m", "cyborg_mind_v2.training.teacher_student_trainer_real"]