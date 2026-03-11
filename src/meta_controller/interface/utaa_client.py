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
    def schedule(self, observation: list[float], parameters: ParameterSnapshot) -> SchedulerSummary:
        soc, load, cpu_temp, rssi, rtt, _, _ = observation
        quality = max(0.0, min(1.0, (soc + (1.0 - load) + rssi + (1.0 - rtt)) / 4.0))
        risk_penalty = max(0.0, parameters.risk_budget - 0.1)
        success_ratio = max(0.0, min(1.0, quality - risk_penalty * 0.15 - max(0.0, cpu_temp - 0.7)))
        scheduled = max(1, int(round(4 + parameters.weights[0])))
        succeeded = int(round(scheduled * success_ratio))
        latency_ms = 2.0 + parameters.weights[1] * 0.2
        return SchedulerSummary(scheduled=scheduled, succeeded=succeeded, latency_ms=latency_ms)
