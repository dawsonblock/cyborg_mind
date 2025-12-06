# CyborgMind Documentation Index

Complete documentation for CyborgMind v4.0 - Production-Grade MineRL RL Agent System

---

## Getting Started

### 🚀 [Quick Start Guide](QUICK_START_GUIDE.md)
Get running in 5 minutes. Installation, first training run, and basic commands.

**Start here if you're new!**

### 📦 [Installation Guide](INSTALL.md)
Detailed installation instructions for different environments (Gym, MineRL, Mamba).

### ✅ [Setup Verification](../scripts/verify_setup.py)
Script to verify all dependencies are installed correctly.

---

## Core Documentation

### 🧠 [How It Works](HOW_IT_WORKS.md)
**Complete system overview**:
- Delayed-cue memory task explanation
- System architecture (environment, agent, trainer)
- Training process step-by-step
- Performance metrics and learning phases
- Code flow diagrams

**Read this to understand the system.**

### 🏗️ [Memory Architecture](MEMORY_ARCHITECTURE.md)
**Deep technical dive**:
- Neural network components (GRU, PMM, Policy/Value heads)
- Information flow and gradient propagation
- Memory mechanisms (short-term vs long-term)
- Training dynamics and loss functions
- Hyperparameter impact analysis
- Advanced techniques and optimizations

**Read this for technical details.**

### 🎯 [Training Guide](HOW_TO_TRAIN.md)
**Comprehensive training workflows**:
- Memory benchmark training
- MineRL Treechop training
- Hyperparameter tuning
- Curriculum learning strategies
- Multi-stage training pipelines

---

## API & Deployment

### 🌐 [API Reference](API.md)
**FastAPI server documentation**:
- Authentication (JWT and static tokens)
- Endpoints (`/step`, `/reset`, `/health`, `/metrics`, `/stream`)
- Request/response schemas
- Rate limiting
- WebSocket streaming
- Python client examples

### 🚢 [Deployment Guide](DEPLOYMENT.md)
**Production deployment**:
- Docker deployment (CPU and GPU)
- Kubernetes with HPA
- Production checklist
- Monitoring setup (Prometheus + Grafana)
- Scaling strategies
- Security best practices
- Load balancing with Nginx

---

## Troubleshooting & Support

### 🔧 [Troubleshooting Guide](TROUBLESHOOTING.md)
**Common issues and solutions**:
- Installation problems
- Training issues (low accuracy, crashes, NaN loss)
- Runtime errors
- Performance problems
- API server issues
- Debugging tips

### 🐛 [Bug Fixes & Enhancements](../BUGFIXES_AND_ENHANCEMENTS.md)
Recent fixes and improvements to the system.

### 📋 [Fixes Summary](../FIXES_SUMMARY.md)
Summary of all fixes applied to v4.0.

---

## Architecture & Design

### 🏛️ [Architecture v3](ARCHITECTURE_V3.md)
System architecture overview for v3.0+.

### 📐 [Architecture](ARCHITECTURE.md)
Original architecture documentation.

### 🔄 [Adapter System](ADAPTER_SYSTEM.md)
Environment adapter design and implementation.

### 🧪 [Experiments](EXPERIMENTS.md)
Experiment tracking and registry system.

---

## Advanced Topics

### 🔁 [Recurrent PPO](RECURRENT_PPO.md)
Recurrent PPO implementation details for memory tasks.

### 📊 [Monitoring](MONITORING.md)
Monitoring setup with Prometheus and Grafana.

### 🔒 [Security](SECURITY.md)
Security best practices and guidelines.

### 📈 [MineRL Results](MINERL_RESULTS.md)
Results and benchmarks on MineRL tasks.

---

## Quick Reference

### Common Commands

**Train memory benchmark**:
```bash
python3 -m cyborg_rl.memory_benchmarks.pseudo_mamba_memory_suite \
    --task delayed_cue --backbone gru --horizon 20 \
    --num-envs 8 --total-timesteps 50000 --run-name my_run
```

**Train MineRL agent**:
```bash
python scripts/run_treechop_pipeline.py \
    --config configs/treechop_ppo.yaml \
    --run-name treechop_v1 --steps 200000
```

**Start API server**:
```bash
python scripts/run_api_server.py \
    --config configs/treechop_ppo.yaml \
    --checkpoint artifacts/best_model.pt
```

**Verify setup**:
```bash
python scripts/verify_setup.py
```

---

## Documentation by Use Case

### I want to...

**...understand how the system works**
→ Read [How It Works](HOW_IT_WORKS.md)

**...train my first agent**
→ Follow [Quick Start Guide](QUICK_START_GUIDE.md)

**...improve training accuracy**
→ Check [Memory Architecture](MEMORY_ARCHITECTURE.md) and [Training Guide](HOW_TO_TRAIN.md)

**...deploy to production**
→ Follow [Deployment Guide](DEPLOYMENT.md)

**...fix an error**
→ Check [Troubleshooting Guide](TROUBLESHOOTING.md)

**...understand the API**
→ Read [API Reference](API.md)

**...tune hyperparameters**
→ See [Memory Architecture](MEMORY_ARCHITECTURE.md) hyperparameter section

**...scale training**
→ Check [Deployment Guide](DEPLOYMENT.md) scaling section

---

## File Structure

```
docs/
├── INDEX.md                    # This file
├── QUICK_START_GUIDE.md        # 5-minute getting started
├── HOW_IT_WORKS.md             # Complete system overview
├── MEMORY_ARCHITECTURE.md      # Technical deep dive
├── HOW_TO_TRAIN.md             # Training workflows
├── API.md                      # API reference
├── DEPLOYMENT.md               # Production deployment
├── TROUBLESHOOTING.md          # Common issues
├── ARCHITECTURE_V3.md          # System architecture
├── ADAPTER_SYSTEM.md           # Environment adapters
├── RECURRENT_PPO.md            # Recurrent PPO details
├── EXPERIMENTS.md              # Experiment tracking
├── MONITORING.md               # Monitoring setup
├── SECURITY.md                 # Security guidelines
└── MINERL_RESULTS.md           # Benchmark results
```

---

## External Resources

### Papers
- [PPO](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization
- [GAE](https://arxiv.org/abs/1506.02438) - Generalized Advantage Estimation
- [Mamba](https://arxiv.org/abs/2312.00752) - Linear-Time Sequence Modeling
- [GRU](https://arxiv.org/abs/1406.1078) - Gated Recurrent Units
- [NTM](https://arxiv.org/abs/1410.5401) - Neural Turing Machines

### Tools
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Gymnasium](https://gymnasium.farama.org/) - RL environment standard
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [Prometheus](https://prometheus.io/) - Monitoring
- [Grafana](https://grafana.com/) - Visualization

---

## Version History

- **v4.0** (Current) - Production-ready with memory benchmarks
- **v3.0** - FastAPI server and deployment
- **v2.6** - MineRL integration
- **v2.0** - Adapter system
- **v1.0** - Initial release

---

## Contributing

See main [README.md](../README.md) for contribution guidelines.

---

## Support

1. **Check documentation** - Start with relevant guide above
2. **Run verification** - `python scripts/verify_setup.py`
3. **Check troubleshooting** - [Troubleshooting Guide](TROUBLESHOOTING.md)
4. **Review fixes** - [Fixes Summary](../FIXES_SUMMARY.md)

---

**Last updated**: 2025-12-05  
**Version**: 4.0  
**Status**: Production Ready ✅
