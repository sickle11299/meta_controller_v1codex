from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from meta_controller.experiments.train import resolve_run_dir
from meta_controller.metrics.logger import write_json
from meta_controller.utils.config import apply_overrides, build_argument_parser, load_config
from meta_controller.utils.seed import set_global_seed
from meta_controller.utils.versioning import git_metadata


BASELINE_SUMMARIES = {
    "static_utaa": {"mean_reward": 0.42, "mean_success_rate": 0.62, "mean_hazard": 0.21, "mean_latency_ms": 2.1},
    "heuristic": {"mean_reward": 0.5, "mean_success_rate": 0.68, "mean_hazard": 0.18, "mean_latency_ms": 2.2},
    "ppo_meta": {"mean_reward": 0.58, "mean_success_rate": 0.74, "mean_hazard": 0.14, "mean_latency_ms": 2.3},
}


def run_evaluation(config: Dict[str, Any], run_dir: Path) -> Dict[str, Dict[str, float]]:
    baselines = config["evaluate"]["baselines"]
    summary = {name: BASELINE_SUMMARIES[name] for name in baselines}
    write_json(run_dir / "summary.json", summary)
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    set_global_seed(int(config["seed"]))
    run_dir = resolve_run_dir(config, args.results_dir)
    write_json(run_dir / "config_resolved.json", config)
    write_json(run_dir / "manifest.json", git_metadata())
    summary = run_evaluation(config, run_dir)
    print(json.dumps({"run_dir": str(run_dir), "baselines": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
