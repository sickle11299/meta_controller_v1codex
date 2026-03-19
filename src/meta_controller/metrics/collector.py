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

    def log_step(self, episode_index: int, step: int, transition: Transition, snapshot: ParameterSnapshot) -> None:
        self.records.append(
            StepRecord(
                episode_index=episode_index,
                step=step,
                reward=transition.reward,
                success_rate=float(transition.info["success_rate"]),
                load_balance=float(transition.info["load_balance"]),
                hazard_integral=float(transition.info["hazard_integral"]),
                action_delta_penalty=float(transition.info["action_delta_penalty"]),
                latency_penalty=float(transition.info["latency_penalty"]),
                scheduler_latency_ms=float(transition.info["latency_ms"]),
                cpu_temp_c=float(transition.info["cpu_temp_c"]),
                power_total_w=float(transition.info["power_total_w"]),
                power_pred_error_w=float(transition.info["power_pred_error_w"]),
                temp_pred_error_c=float(transition.info["temp_pred_error_c"]),
                risk_budget=snapshot.risk_budget,
                model_validation_passed=bool(transition.info["model_validation_passed"]),
                weights=list(snapshot.weights),
                reward_components=dict(transition.info["reward_components"]),
            )
        )

    def write_jsonl(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def summarize(self) -> Dict[str, float | bool]:
        if not self.records:
            return {
                "mean_reward": 0.0,
                "mean_success_rate": 0.0,
                "mean_hazard": 0.0,
                "mean_latency_ms": 0.0,
                "mean_temp_c": 0.0,
                "mean_power_w": 0.0,
                "model_validation_passed": False,
            }
        count = float(len(self.records))
        return {
            "mean_reward": sum(item.reward for item in self.records) / count,
            "mean_success_rate": sum(item.success_rate for item in self.records) / count,
            "mean_hazard": sum(item.hazard_integral for item in self.records) / count,
            "mean_latency_ms": sum(item.scheduler_latency_ms for item in self.records) / count,
            "mean_temp_c": sum(item.cpu_temp_c for item in self.records) / count,
            "mean_power_w": sum(item.power_total_w for item in self.records) / count,
            "model_validation_passed": all(item.model_validation_passed for item in self.records),
        }
