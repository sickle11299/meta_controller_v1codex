from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HazardSummary:
    integral: float
    instantaneous: float


def estimate_hazard(temp_penalty: float, power_penalty: float, latency_penalty: float, risk_budget: float) -> HazardSummary:
    """把多目标惩罚折叠成诊断性 hazard 指标。"""

    instantaneous = 0.5 * max(0.0, temp_penalty)
    instantaneous += 0.3 * max(0.0, power_penalty)
    instantaneous += 0.2 * max(0.0, latency_penalty)
    instantaneous += 0.1 * max(0.0, risk_budget - 0.3)
    return HazardSummary(integral=instantaneous, instantaneous=instantaneous)
