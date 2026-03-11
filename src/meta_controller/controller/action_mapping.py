from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class ParameterSnapshot:
    weights: List[float]
    risk_budget: float
    version: int = 0
    valid_for_seconds: float = 15.0

    def to_dict(self) -> dict:
        return asdict(self)


class ActionMappingError(ValueError):
    pass


class SafeActionMapper:
    def __init__(
        self,
        beta: float,
        risk_base: float,
        risk_min: float,
        risk_max: float,
        weight_min: float,
        weight_max: float,
        weight_sum_target: float,
        smoothing: float = 0.25,
    ) -> None:
        self.beta = beta
        self.risk_base = risk_base
        self.risk_min = risk_min
        self.risk_max = risk_max
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.weight_sum_target = weight_sum_target
        self.smoothing = smoothing

    def map_action(self, action: Iterable[float], previous: ParameterSnapshot | None, version: int) -> ParameterSnapshot:
        action_values = list(action)
        if len(action_values) != 5:
            raise ActionMappingError("action must contain 5 values")
        if any(not math.isfinite(value) or value < -1.0 or value > 1.0 for value in action_values):
            raise ActionMappingError("action values must be finite and in [-1, 1]")

        raw_weights = [self._softplus(value * self.beta) for value in action_values[1:]]
        clipped = [min(self.weight_max, max(self.weight_min, weight)) for weight in raw_weights]
        total = sum(clipped)
        normalized = [weight * self.weight_sum_target / total for weight in clipped]

        risk_budget = min(self.risk_max, max(self.risk_min, self.risk_base + action_values[0] * (self.risk_max - self.risk_base)))

        if previous is not None:
            normalized = [self._blend(old, new) for old, new in zip(previous.weights, normalized)]
            risk_budget = self._blend(previous.risk_budget, risk_budget)

        bounded = [min(self.weight_max, max(self.weight_min, weight)) for weight in normalized]
        scale = self.weight_sum_target / sum(bounded)
        bounded = [weight * scale for weight in bounded]
        return ParameterSnapshot(weights=bounded, risk_budget=risk_budget, version=version)

    def _blend(self, current: float, proposed: float) -> float:
        return (1.0 - self.smoothing) * current + self.smoothing * proposed

    @staticmethod
    def _softplus(value: float) -> float:
        return math.log1p(math.exp(value))
