from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class StepRecord:
    episode_index: int
    step: int
    reward: float       #奖励字段定义  StepRecord中存储实时奖励的
    success_rate: float
    hazard_integral: float
    action_delta_penalty: float
    scheduler_latency_ms: float
    risk_budget: float
    weights: List[float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
