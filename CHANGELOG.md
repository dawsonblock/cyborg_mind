# Changelog

All notable changes to CyborgMind are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] - 2025-12-03

### Major Release: Production-Ready v3

Complete architectural overhaul to unified `cyborg_rl` core with production training and API systems.

### Added

#### Core Engine
- **Unified PPO Implementation:** `cyborg_rl.trainers.PPOTrainer` with Mixed-Precision (AMP), gradient clipping, vectorized environments
- **Mamba/GRU Encoder:** `cyborg_rl.models.MambaGRUEncoder` with configurable Mamba SSM or GRU backends
- **Predictive Memory Module (PMM):** `cyborg_rl.memory.PredictiveMemoryModule` with content-based addressing and intrinsic rewards
- **PPOAgent:** `cyborg_rl.agents.PPOAgent` with discrete/continuous action support, save/load, state management

#### Configuration System
- **Config Dataclasses:** Full configuration via `cyborg_rl.config.Config` with env, model, memory, PPO, training, and API sections
- **YAML Support:** `Config.from_yaml()` and `Config.from_dict()` for flexible config loading
- **Production Configs:** Ready-to-use configs for CartPole, Pendulum in `configs/envs/`

#### Production Training
- **train_production.py:** Main entry point with ExperimentRegistry, vectorized envs, checkpointing
- **ExperimentRegistry:** Automatic experiment tracking with git hash, system info, metrics logging, checkpoint management
- **Gym Adapter:** `cyborg_rl.envs.GymAdapter` for Gymnasium environments

#### API Server
- **FastAPI Server:** `cyborg_rl.server.CyborgServer` with async inference
- **Bearer Token Auth:** Simple auth system (configurable token)
- **Rate Limiting:** SlowAPI integration for request throttling
- **Batch Inference:** `/step_batch` endpoint for efficient multi-agent inference
- **Prometheus Metrics:** `/metrics` endpoint with request count, latency histograms

#### Testing
- **Comprehensive Test Suite:** 9 test files covering Config, PPOAgent, RolloutBuffer, PMM, API, ExperimentRegistry
- **pytest Configuration:** `pytest.ini` with strict markers, legacy exclusion
- **CI Integration:** Tests run on all PRs and main branch pushes

#### Documentation
- **README.md:** Honest feature descriptions, quick start, architecture diagram
- **docs/HOW_TO_TRAIN.md:** Complete training guide with config reference, examples
- **docs/API.md:** Full API reference with auth, endpoints, examples, security best practices
- **tests/README.md:** Test suite guide with coverage goals, how to run, troubleshooting

### Changed

#### Architecture
- **Unified Namespace:** All production code under `cyborg_rl` (vs. scattered `experiments.*`)
- **v2 Legacy:** Moved `cyborg_mind_v2` to `legacy/cyborg_mind_v2/` (preserved for reference)
- **Config Schema:** Changed from `agent:` → `model:`, `training:` → `train:` in YAML files
- **Checkpoint Paths:** Now use `experiments/runs/<run_name>/checkpoints/` instead of flat `checkpoints/`

#### Training Pipeline
- **RolloutBuffer:** Moved to `cyborg_rl.trainers.rollout_buffer` (was in memory module)
- **TrainConfig:** Added 13 PPO-specific parameters (lr, n_steps, batch_size, etc.)
- **Vectorized Envs:** Default to 4 parallel envs via `AsyncVectorEnv`

#### API
- **Auth System:** Clarified as "Bearer Token Auth" (not JWT) in all docs
- **Endpoint Naming:** Consistent `/step`, `/step_batch`, `/reset` patterns
- **Error Handling:** Structured error responses with proper status codes

### Fixed

#### Critical Blockers
- **PPOTrainer Import:** Fixed `from cyborg_rl.memory.replay_buffer` → `from cyborg_rl.trainers.rollout_buffer`
- **Config.from_dict:** Implemented missing method (required by `train_production.py`)
- **TrainConfig Fields:** Added missing PPO parameters (lr, weight_decay, use_amp, n_steps, etc.)
- **Version Sync:** Updated `pyproject.toml` from 2.8.0 → 3.0.0 to match README

#### Test Suite
- **v2 Test Cleanup:** Moved 8 broken v2-dependent tests to `tests/legacy_v2/`
- **Import Paths:** All active tests now import from `cyborg_rl` (no `experiments.*`)
- **pytest Config:** Excluded legacy tests via `norecursedirs`

#### Documentation
- **Honest Claims:** Removed "JWT Auth" claims (actual: static bearer token)
- **Domain Adapters:** Clarified Trading/EEG/Lab as "stubs" (not production-ready)
- **K8s Manifests:** Removed claims about K8s until actually implemented

### Removed

- Removed placeholder AI-generated status docs (BUILD_STATUS.md, FINAL_VERIFICATION.md, etc.)
- Removed broken v2 imports from active test suite
- Removed misleading "full Prometheus/Grafana stack" claims (metrics available, dashboards optional)

### Security

- **Auth Token Configuration:** Now clearly documented as static token (warn about production use)
- **Security Best Practices:** Added section to API.md with HTTPS, network isolation, logging recommendations

---

## [2.8.0] - 2025-11-XX

### Legacy Release (Pre-v3)

Final release of the v2.x architecture (CapsuleBrain-based). See `legacy/cyborg_mind_v2/` for code.

**Note:** v2.x is no longer actively maintained. Migrate to v3.0.

---

## [2.6.0] - 2025-XX-XX

### v2.6 Features

- CapsuleBrain architecture with thought/emotion modules
- MineRL TreeChop adapter
- Basic experiment registry
- Hydra configuration system

**Migration:** See `docs/V2.6_MIGRATION_GUIDE.md` (archived)

---

## Development Guidelines

### Semantic Versioning

- **MAJOR (X.0.0):** Breaking changes, architecture rewrites
- **MINOR (3.X.0):** New features, backward-compatible additions
- **PATCH (3.0.X):** Bug fixes, documentation updates

### Changelog Format

When adding entries:
- Use present tense ("Add feature" not "Added feature")
- Group by Added/Changed/Fixed/Removed/Security
- Link to issues/PRs when relevant

---

## Links

- [Repository](https://github.com/dawsonblock/cyborg_mind)
- [Issues](https://github.com/dawsonblock/cyborg_mind/issues)
- [Documentation](docs/)
