from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class StepRecord:
    episode_index: int
    step: int
    reward: float
    success_rate: float
    load_balance: float
    hazard_integral: float
    action_delta_penalty: float
    latency_penalty: float
    scheduler_latency_ms: float
    cpu_temp_c: float
    power_total_w: float
    power_pred_error_w: float
    temp_pred_error_c: float
    risk_budget: float
    model_validation_passed: bool
    weights: List[float]
    reward_components: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
