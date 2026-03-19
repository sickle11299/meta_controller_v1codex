from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from meta_controller.env.hazard_model import HazardSummary, estimate_hazard


@dataclass
class RewardTerms:
    success_rate: float
    load_balance: float
    temp_penalty: float
    power_penalty: float
    latency_penalty: float
    hazard_integral: float
    action_delta_penalty: float
    reward: float
    reward_components: Dict[str, float] = field(default_factory=dict)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def compute_load_balance(load_frac: float, weights: Iterable[float]) -> float:
    """根据当前总负载和调度权重，估算负载是否均匀分配。"""

    values = list(weights)
    total = sum(values)
    if total <= 1e-9:
        return 1.0
    allocation_loads = [load_frac * weight / total for weight in values]
    mean_load = sum(allocation_loads) / len(allocation_loads) if allocation_loads else 0.0
    if mean_load <= 1e-9:
        return 1.0
    variance = sum((item - mean_load) ** 2 for item in allocation_loads) / len(allocation_loads)
    cv = (variance ** 0.5) / mean_load
    return _clip01(1.0 - cv)


def compute_reward(
    success_rate: float,
    load_balance: float,
    temp_penalty: float,
    power_penalty: float,
    latency_penalty: float,
    action_delta_penalty: float,
    risk_budget: float,
    weights: Dict[str, float],
) -> RewardTerms:
    """按分项加权方式计算奖励。"""

    hazard: HazardSummary = estimate_hazard(temp_penalty, power_penalty, latency_penalty, risk_budget)
    reward = 0.0
    reward += float(weights["success"]) * success_rate
    reward += float(weights["balance"]) * load_balance
    reward -= float(weights["temp"]) * temp_penalty
    reward -= float(weights["power"]) * power_penalty
    reward -= float(weights["latency"]) * latency_penalty
    reward -= float(weights["action_delta"]) * action_delta_penalty
    reward_components = {
        "success": success_rate,
        "balance": load_balance,
        "temp_penalty": temp_penalty,
        "power_penalty": power_penalty,
        "latency_penalty": latency_penalty,
        "action_delta_penalty": action_delta_penalty,
        "hazard_integral": hazard.integral,
    }
    return RewardTerms(
        success_rate=success_rate,
        load_balance=load_balance,
        temp_penalty=temp_penalty,
        power_penalty=power_penalty,
        latency_penalty=latency_penalty,
        hazard_integral=hazard.integral,
        action_delta_penalty=action_delta_penalty,
        reward=reward,
        reward_components=reward_components,
    )
