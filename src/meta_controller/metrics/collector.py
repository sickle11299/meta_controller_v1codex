from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from meta_controller.controller.action_mapping import ParameterSnapshot
from meta_controller.env.edge_env import Transition
from meta_controller.metrics.schema import StepRecord


class MetricsCollector:
    def __init__(self) -> None:
        self.records: List[StepRecord] = []

    def log_step(self, episode_index: int, step: int, transition: Transition, snapshot: ParameterSnapshot) -> None:   # episode_index: 回合编号  step: 当前回合内的步数   环境交互的转移数据（s, a, r, s', info）
        #将单步奖励封装为StepRecord对象，并将其添加到MetricsCollector的记录列表中。每个StepRecord包含了当前回合编号、步数、奖励、成功率、风险积分、动作变化惩罚、调度器延迟以及当前的风险预算和权重信息。这些数据将被用于后续的分析和评估，以了解控制策略在不同回合和步骤中的表现。
        self.records.append(    # 构造单步记录并添加到日志列表 
            StepRecord(
                episode_index=episode_index,
                step=step,
                reward=transition.reward,
                success_rate=float(transition.info["success_rate"]),
                hazard_integral=float(transition.info["hazard_integral"]),
                action_delta_penalty=float(transition.info["action_delta_penalty"]),
                scheduler_latency_ms=float(transition.info["latency_ms"]),
                risk_budget=snapshot.risk_budget,
                weights=list(snapshot.weights),
            )
        )

    def write_jsonl(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in self.records:     #序列化写入
                handle.write(json.dumps(record.to_dict()) + "\n")

    def summarize(self) -> Dict[str, float]:
        if not self.records:
            return {"mean_reward": 0.0, "mean_success_rate": 0.0, "mean_hazard": 0.0, "mean_latency_ms": 0.0}
        count = float(len(self.records))
        return {
            "mean_reward": sum(item.reward for item in self.records) / count,       #这计算的还是平均奖励 并非  GAE 回报
            "mean_success_rate": sum(item.success_rate for item in self.records) / count,
            "mean_hazard": sum(item.hazard_integral for item in self.records) / count,
            "mean_latency_ms": sum(item.scheduler_latency_ms for item in self.records) / count,
        }
