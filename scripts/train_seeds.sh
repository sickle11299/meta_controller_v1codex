#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

for seed in 7 11 19; do
  python -m meta_controller.experiments.train --config configs/training.yaml --set seed="${seed}"
done
