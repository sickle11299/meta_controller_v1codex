from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from meta_controller.controller.action_mapping import SafeActionMapper
from meta_controller.controller.ppo import PPOTrainer
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


def run_training(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    mapper = SafeActionMapper(
        beta=float(config["controller"]["beta"]),
        risk_base=float(config["env"]["risk_base"]),
        risk_min=float(config["env"]["risk_min"]),
        risk_max=float(config["env"]["risk_max"]),
        weight_min=float(config["env"]["weight_min"]),
        weight_max=float(config["env"]["weight_max"]),
        weight_sum_target=float(config["env"]["weight_sum_target"]),
        smoothing=float(config["controller"]["smoothing"]),
    )
    env = EdgeEnv(config)
    collector = MetricsCollector()
    runtime = SidecarRuntime(env=env, collector=collector, mapper=mapper)
    trainer = PPOTrainer()
    episodes = int(config["train"]["episodes"])
    for episode in range(episodes):
        runtime.run_episode(episode_index=episode, max_steps=int(config["env"]["episode_horizon"]))
    for record in collector.records:
        trainer.record_step(record.reward, record.success_rate, record.hazard_integral)
    summary = collector.summarize()
    summary.update(trainer.train_epoch())
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
    write_json(run_dir / "config_resolved.json", config)
    write_json(run_dir / "manifest.json", {"seed_bundle": seed_bundle, **git_metadata()})
    summary = run_training(config, run_dir)
    write_json(run_dir / "summary.json", summary)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
