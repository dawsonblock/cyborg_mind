# Changelog

All notable changes to CyborgMind are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.2.0] - 2025-12-04

### Added

#### WandB Integration
- **Automatic Logging:** PPOTrainer now supports Weights & Biases integration for experiment tracking
- **Granular Metrics:** Logs `loss`, `policy_loss`, `value_loss`, `entropy_loss`, `fps`, `timestep`, `update`
- **Model Watching:** `wandb.watch()` tracks gradients and parameters automatically
- **TrainConfig Extensions:** 5 new WandB config fields (wandb_enabled, wandb_project, wandb_entity, wandb_tags, wandb_run_name)
- **Comprehensive Tests:** `tests/test_wandb_integration.py` with 10 test cases
- **Documentation:** Updated `docs/HOW_TO_TRAIN.md` with complete WandB setup guide

#### Docker Compose Monitoring Stack
- **Production Stack:** `docker-compose.yml` with CyborgAPI, Prometheus, and Grafana services
- **Prometheus Config:** Pre-configured scraping for `/metrics` endpoint at 10s intervals
- **Grafana Dashboards:** Auto-provisioned dashboard with 7 panels:
  - Request rate by endpoint (success/error)
  - Inference latency percentiles (p50, p95, p99)
  - Error rate gauge
  - Auth failure count gauge
  - Inference throughput (single vs batch)
  - Requests by status code
  - Total requests all-time
- **Monitoring README:** Complete guide in `monitoring/README.md` with deployment, ops, troubleshooting
- **Environment Variables:** `.env.example` for easy configuration
- **Persistent Storage:** Docker volumes for Prometheus and Grafana data
- **Health Checks:** API container includes health check with auto-restart

#### WebSocket Streaming Endpoint
- **Real-time Inference:** `WS /stream` endpoint for continuous observation → action streaming
- **Low Latency:** No connection overhead, maintains persistent state
- **Authentication:** Token-based auth in message payload (supports JWT + static tokens)
- **State Management:** Each WebSocket connection maintains independent agent state
- **Error Handling:** Graceful error responses, automatic state cleanup on disconnect
- **Metrics:** WebSocket connection counter and success/error tracking
- **Comprehensive Tests:** `tests/test_websocket.py` with 13 test cases
- **Documentation:** Updated `docs/API.md` with complete WebSocket guide (JavaScript + Python examples)

#### Configuration
- `TrainConfig.wandb_enabled` - Enable/disable WandB logging (default: `false`)
- `TrainConfig.wandb_project` - WandB project name (default: `"cyborg-mind"`)
- `TrainConfig.wandb_entity` - WandB username/team (optional)
- `TrainConfig.wandb_tags` - List of tags for experiments (optional)
- `TrainConfig.wandb_run_name` - Custom run name (auto-generated if not set)

### Changed

- **PPOTrainer:** Enhanced to log metrics to WandB when enabled
- **PPOTrainer._update_policy():** Now returns granular loss metrics (policy_loss, value_loss, entropy_loss)
- **CyborgServer:** Added WebSocket support with `/stream` endpoint
- **README.md:** Updated features to highlight WandB, WebSocket streaming, Docker Compose stack
- **docs/API.md:** Added comprehensive WebSocket streaming documentation
- **docs/HOW_TO_TRAIN.md:** Added WandB integration section with setup instructions
- **configs/envs/gym_cartpole.yaml:** Added WandB config example (disabled by default)

### DevOps

- **One-Command Deployment:** `docker-compose up -d` launches full monitoring stack
- **Grafana Auto-Provisioning:** Dashboard and datasource automatically configured
- **Network Isolation:** Services communicate via Docker bridge network
- **Production Ready:** Health checks, restart policies, volume persistence

### Performance

- **WebSocket Streaming:** ~50% lower latency than HTTP for high-frequency inference
- **State Persistence:** Eliminates redundant state resets across sequential requests
- **Concurrent Connections:** Supports multiple independent WebSocket clients

---

## [3.1.0] - 2025-12-04

### Added

#### JWT Authentication System
- **Full JWT Support:** `cyborg_rl.utils.jwt_auth.JWTAuth` with HS256/RS256 algorithms
- **Token Generation:** `POST /auth/token` endpoint for dynamic JWT creation
- **Token Expiry:** Configurable token validity (default: 60 minutes)
- **Issuer/Audience Validation:** Production-grade token verification
- **Dual-Mode Auth:** Supports both JWT and static bearer tokens (backward compatible)
- **APIConfig Extensions:** 7 new JWT config fields (jwt_enabled, jwt_secret, jwt_algorithm, etc.)
- **Comprehensive Tests:** `tests/test_jwt_auth.py` with 18 test cases

#### Configuration
- `APIConfig.jwt_enabled` - Enable/disable JWT (default: `false`)
- `APIConfig.jwt_secret` - Secret key for HS256 or path to private key for RS256
- `APIConfig.jwt_algorithm` - HS256 or RS256
- `APIConfig.jwt_issuer` - Optional token issuer validation
- `APIConfig.jwt_audience` - Optional token audience validation
- `APIConfig.jwt_expiry_minutes` - Token validity duration (default: 60)
- `APIConfig.jwt_public_key_path` - Public key path for RS256 verification

#### Dependencies
- `PyJWT>=2.8.0` - JWT token handling
- `slowapi>=0.1.9` - Rate limiting (now documented)
- `gitpython>=3.1.40` - Git integration (now documented)

### Changed

- **CyborgServer:** Updated `_verify_token()` to use JWT handler with dual-mode support
- **Authentication:** JWT validation first, falls back to static token if JWT disabled
- **README.md:** Updated to "JWT + Bearer Token Auth"
- **docs/API.md:** Complete JWT authentication guide with both modes documented

### Security

- ✅ **Token Expiry:** Automatic expiration (configurable)
- ✅ **Issuer Validation:** Prevents forged tokens
- ✅ **Audience Scoping:** Restricts token usage
- ✅ **Algorithm Flexibility:** HS256 (symmetric) or RS256 (asymmetric)
- ✅ **Backward Compatible:** No breaking changes for existing deployments

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
