from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PPOTrainer:
    trajectories: List[Dict[str, float]] = field(default_factory=list)

    def record_step(self, reward: float, success_rate: float, hazard: float) -> None:
        self.trajectories.append(
            {
                "reward": reward,
                "success_rate": success_rate,
                "hazard": hazard,
            }
        )

    def train_epoch(self) -> Dict[str, float]:
        if not self.trajectories:
            return {"mean_reward": 0.0, "mean_success_rate": 0.0, "mean_hazard": 0.0}
        count = float(len(self.trajectories))
        return {
            "mean_reward": sum(item["reward"] for item in self.trajectories) / count,   # PPO内计算的也是平均奖励并非GAE 回报
            "mean_success_rate": sum(item["success_rate"] for item in self.trajectories) / count,
            "mean_hazard": sum(item["hazard"] for item in self.trajectories) / count,
        }
