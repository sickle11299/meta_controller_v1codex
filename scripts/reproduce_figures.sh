#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

python -m meta_controller.analysis.make_figures --results-dir results --output-dir artifacts/paper
python -m meta_controller.analysis.make_tables --results-dir results --output-dir artifacts/paper
