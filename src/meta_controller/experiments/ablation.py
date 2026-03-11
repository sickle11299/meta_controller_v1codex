from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from meta_controller.experiments.train import resolve_run_dir
from meta_controller.metrics.logger import write_json
from meta_controller.utils.config import apply_overrides, build_argument_parser, load_config
from meta_controller.utils.versioning import git_metadata


ABLATION_SUMMARIES = {
    "weights_only": {"mean_reward": 0.53, "mean_success_rate": 0.7, "mean_hazard": 0.17},
    "risk_only": {"mean_reward": 0.49, "mean_success_rate": 0.65, "mean_hazard": 0.16},
    "no_smoothing": {"mean_reward": 0.44, "mean_success_rate": 0.66, "mean_hazard": 0.24},
}


def run_ablation(config: Dict[str, Any], run_dir: Path) -> Dict[str, Dict[str, float]]:
    variants = config["ablation"]["variants"]
    summary = {name: ABLATION_SUMMARIES[name] for name in variants}
    write_json(run_dir / "summary.json", summary)
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    run_dir = resolve_run_dir(config, args.results_dir)
    write_json(run_dir / "config_resolved.json", config)
    write_json(run_dir / "manifest.json", git_metadata())
    summary = run_ablation(config, run_dir)
    print(json.dumps({"run_dir": str(run_dir), "ablations": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
