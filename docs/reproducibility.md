# Reproducibility

## Controls

- All entrypoints require an explicit config file.
- Seeds are expanded into `python`, `numpy`, and any future ML backend.
- Results are written into run-scoped directories.
- Analysis scripts consume logged artifacts instead of notebooks.

## Run contract

Each run directory contains:

- `config_resolved.json`
- `manifest.json`
- `metrics.jsonl`
- `episode_summary.json`
- `summary.json`
