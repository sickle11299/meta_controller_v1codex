# Meta-Controller

This repository scaffolds a paper-grade DRL meta-controller around a deterministic UTAA scheduler.

## Design constraints

- DRL tunes parameters only.
- UTAA remains the scheduling authority.
- The meta-controller runs as a sidecar and never blocks scheduling.
- Experiments must be reproducible from versioned configs, seeds, and logged artifacts.

## Layout

- `src/meta_controller/`: package source.
- `configs/`: versioned experiment configurations.
- `tests/`: regression and reproducibility checks.
- `scripts/`: one-command experiment entrypoints.
- `docs/`: architecture and experiment protocol.

## Quick start

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m meta_controller.experiments.train --config configs/training.yaml
python -m meta_controller.experiments.evaluate --config configs/evaluation.yaml
python -m meta_controller.analysis.make_figures --results-dir results --output-dir artifacts/paper
```

## Reproduction

The intended paper reproduction flow is:

```bash
bash scripts/run_all.sh
```

The shell scripts call the Python modules in `src/meta_controller/` and generate summarized artifacts in `artifacts/paper/`.
