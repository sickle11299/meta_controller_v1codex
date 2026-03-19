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
    # 4. 执行训练循环
    episodes = int(config["train"]["episodes"])  # 训练局数（来自配置）
    for episode in range(episodes):
        # 跑完一整局，返回一系列奖励
        runtime.run_episode(episode_index=episode, max_steps=int(config["env"]["episode_horizon"]))

    # 5. 将收集到的数据喂给 PPO 训练器
    for record in collector.records:     # 遍历所有步骤记录
        trainer.record_step(record.reward, record.success_rate, record.hazard_integral)

    # 6. 生成训练摘要并更新模型
    summary = collector.summarize()          # 统计整体表现
    summary.update(trainer.train_epoch())    # 执行一次 PPO 训练迭代

    # 7. 保存结果到磁盘
    collector.write_jsonl(run_dir / "metrics.jsonl")                        # 详细指标（每步一条）
    write_json(run_dir / "episode_summary.json", {"episodes": episodes, **summary})  # 汇总信息
    save_checkpoint(run_dir / "checkpoints" / "last.json", summary)         # 检查点（用于恢复训练）

    return summary  # 返回训练摘要


def main(argv: List[str] | None = None) -> int:     #解析参数             
    parser = build_argument_parser()
    args = parser.parse_args(argv)            #argv = Argument Vector（参数向量）这是从 C/C++ 继承来的传统命名：arg = argument（参数）v = vector（向量/数组）
    config = apply_overrides(load_config(args.config), args.set)
    seed_bundle = set_global_seed(int(config["seed"]))                  #固定随机性
    run_dir = resolve_run_dir(config, args.results_dir)       #创建实验目录
    write_json(run_dir / "config_resolved.json", config)      
    write_json(run_dir / "manifest.json", {"seed_bundle": seed_bundle, **git_metadata()})  ## ← 写入 Git 信息   
    summary = run_training(config, run_dir)           #执行
    write_json(run_dir / "summary.json", summary)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
