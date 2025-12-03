# Experiments & Registry

## Experiment Registry

Every training run is automatically tracked by `cyborg_rl.experiments.registry.ExperimentRegistry`.

### Artifacts
For each run, the following are saved in `experiments/runs/<run_name>/`:
- `manifest.json`: Metadata (Git hash, System info, Timestamp).
- `config.yaml`: Exact config used.
- `logs/metrics.csv`: Training metrics.
- `checkpoints/`: Model checkpoints (`latest.pt`, `best.pt`).

## Reproducibility

To reproduce a run:
1. Check `manifest.json` for the Git commit hash.
2. Checkout that commit.
3. Run training with the saved `config.yaml`.

## Legacy Experiments
Old experiments from v2 have been archived in `legacy/cyborg_mind_v2`.
