from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from meta_controller.controller.action_mapping import ParameterSnapshot


@dataclass
class SchedulerSummary:
    scheduled: int
    succeeded: int
    latency_ms: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "scheduled": self.scheduled,
            "succeeded": self.succeeded,
            "latency_ms": self.latency_ms,
        }


class UTAAClient:
    """调度器代理。

    当前主线状态空间切到 7 维后，这里也同步只依赖：
    `soc / load / temp / rssi / rtt / temp-load 梯度 / rssi 梯度`
    这些与调度稳定性直接相关的量。
    """

    def schedule(self, observation: list[float], parameters: ParameterSnapshot) -> SchedulerSummary:
        soc, load, temp_norm, rssi, latency_norm, temp_load_gradient, rssi_gradient = observation

        quality = 0.35 * soc + 0.25 * (1.0 - load) + 0.20 * rssi + 0.20 * (1.0 - latency_norm)
        thermal_penalty = max(0.0, temp_norm - 0.75)
        thermal_trend_penalty = max(0.0, temp_load_gradient)
        link_trend_penalty = max(0.0, -rssi_gradient)
        risk_penalty = max(0.0, parameters.risk_budget - 0.10)

        success_ratio = quality
        success_ratio -= 0.20 * risk_penalty
        success_ratio -= 0.25 * thermal_penalty
        success_ratio -= 0.10 * thermal_trend_penalty
        success_ratio -= 0.05 * link_trend_penalty
        success_ratio = min(1.0, max(0.0, success_ratio))

        scheduled = max(1, int(round(4 + parameters.weights[0])))
        succeeded = int(round(scheduled * success_ratio))

        latency_ms = 6.0 + 18.0 * latency_norm + 3.0 * max(0.0, load - rssi) + 2.0 * parameters.risk_budget
        latency_ms += 2.0 * max(0.0, -rssi_gradient)
        return SchedulerSummary(scheduled=scheduled, succeeded=succeeded, latency_ms=latency_ms)
