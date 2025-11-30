# CyborgMind RL - Build Automation
# Usage: make [target]

.PHONY: help install dev test lint format train train-gpu inference docker-build docker-up docker-down clean

# Default target
help:
	@echo "CyborgMind RL - Available targets:"
	@echo ""
	@echo "  install      Install production dependencies"
	@echo "  dev          Install development dependencies"
	@echo "  test         Run unit tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black"
	@echo ""
	@echo "  train        Train CartPole agent (CPU)"
	@echo "  train-gpu    Train CartPole agent (GPU)"
	@echo "  inference    Run inference on trained model"
	@echo ""
	@echo "  docker-build Build Docker images"
	@echo "  docker-up    Start training with Docker"
	@echo "  docker-down  Stop Docker services"
	@echo "  docker-logs  View Docker logs"
	@echo ""
	@echo "  monitor      Start monitoring stack"
	@echo "  clean        Clean build artifacts"

# =============================================================================
# Installation
# =============================================================================
install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pip install black flake8 mypy

# =============================================================================
# Testing
# =============================================================================
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=cyborg_rl --cov-report=html --cov-report=term

# =============================================================================
# Code Quality
# =============================================================================
lint:
	flake8 cyborg_rl/ scripts/ tests/ --max-line-length=100
	mypy cyborg_rl/ --ignore-missing-imports

format:
	black cyborg_rl/ scripts/ tests/ --line-length=100

# =============================================================================
# Training
# =============================================================================
train:
	python scripts/train_gym_cartpole.py \
		--total-timesteps 100000 \
		--checkpoint-dir checkpoints/cartpole \
		--device cpu \
		--no-metrics

train-gpu:
	python scripts/train_gym_cartpole.py \
		--total-timesteps 500000 \
		--checkpoint-dir checkpoints/cartpole \
		--device cuda

train-minerl:
	python scripts/train_minerl_navigate.py \
		--total-timesteps 1000000 \
		--checkpoint-dir checkpoints/minerl \
		--device cuda

inference:
	python scripts/inference.py \
		--checkpoint checkpoints/cartpole/final_policy.pt \
		--env CartPole-v1 \
		--episodes 10 \
		--deterministic

# =============================================================================
# Docker
# =============================================================================
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d trainer

docker-up-gpu:
	docker-compose --profile gpu up -d trainer-gpu

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

monitor:
	docker-compose --profile monitoring up -d prometheus grafana
	@echo "Grafana: http://localhost:3000 (admin/cyborgmind)"
	@echo "Prometheus: http://localhost:9090"

# =============================================================================
# Cleanup
# =============================================================================
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
