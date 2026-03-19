from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from meta_controller.controller.action_mapping import SafeActionMapper
from meta_controller.controller.policy import MetaPolicy
from meta_controller.controller.ppo import PPOTrainer
from meta_controller.controller.value import ValueEstimator
from meta_controller.env.edge_env import EdgeEnv
from meta_controller.metrics.collector import MetricsCollector
from meta_controller.metrics.logger import write_json
from meta_controller.runtime.sidecar import SidecarRuntime
from meta_controller.utils.checkpoint import save_checkpoint
from meta_controller.utils.config import apply_overrides, build_argument_parser, load_config
from meta_controller.utils.seed import set_global_seed
from meta_controller.utils.versioning import git_metadata


def resolve_run_dir(config: Dict[str, Any], override_results_dir: str | None = None) -> Path:
    root = Path(override_results_dir or config.get("results_root", "results"))
    run_dir = root / config["experiment_name"] / f"seed_{config['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _prepare_runtime_config(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    runtime_config = copy.deepcopy(config)
    calibration_cfg = runtime_config.setdefault("env", {}).setdefault("calibration", {})
    report_path = calibration_cfg.get("report_path")
    if not report_path or report_path == "auto":
        calibration_cfg["report_path"] = str(run_dir / "calibration_report.json")
    return runtime_config


def _build_mapper(config: Dict[str, Any]) -> SafeActionMapper:
    return SafeActionMapper(
        beta=float(config["controller"]["beta"]),
        risk_base=float(config["env"]["risk_base"]),
        risk_min=float(config["env"]["risk_min"]),
        risk_max=float(config["env"]["risk_max"]),
        weight_min=float(config["env"]["weight_min"]),
        weight_max=float(config["env"]["weight_max"]),
        weight_sum_target=float(config["env"]["weight_sum_target"]),
        smoothing=float(config["controller"]["smoothing"]),
    )


def _average_stats(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "explained_variance": 0.0,
        }
    keys = items[0].keys()
    return {key: sum(item[key] for item in items) / len(items) for key in keys}


def run_training(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    runtime_config = _prepare_runtime_config(config, run_dir)
    mapper = _build_mapper(runtime_config)
    collector = MetricsCollector()
    policy = MetaPolicy()
    value_estimator = ValueEstimator()
    env = EdgeEnv(runtime_config)
    runtime = SidecarRuntime(env=env, collector=collector, mapper=mapper, policy=policy)
    trainer = PPOTrainer(policy=policy, value_estimator=value_estimator, config=runtime_config)

    episodes = int(runtime_config["train"]["episodes"])
    train_stats: List[Dict[str, float]] = []
    for episode in range(episodes):
        rollout = runtime.run_episode(
            episode_index=episode,
            max_steps=int(runtime_config["env"]["episode_horizon"]),
            collect_rollout=True,
            value_estimator=value_estimator,
        )
        train_stats.append(trainer.train_epoch(rollout.steps))

    summary = collector.summarize()
    summary.update(_average_stats(train_stats))
    summary["calibration_report_path"] = runtime_config["env"]["calibration"]["report_path"]

    collector.write_jsonl(run_dir / "metrics.jsonl")
    write_json(run_dir / "episode_summary.json", {"episodes": episodes, **summary})
    save_checkpoint(run_dir / "checkpoints" / "last.json", summary)
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    seed_bundle = set_global_seed(int(config["seed"]))
    run_dir = resolve_run_dir(config, args.results_dir)
    runtime_config = _prepare_runtime_config(config, run_dir)
    write_json(run_dir / "config_resolved.json", runtime_config)
    write_json(run_dir / "manifest.json", {"seed_bundle": seed_bundle, **git_metadata()})
    summary = run_training(runtime_config, run_dir)
    write_json(run_dir / "summary.json", summary)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
