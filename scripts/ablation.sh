#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

python -m meta_controller.experiments.ablation --config configs/ablation.yaml
