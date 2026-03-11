from __future__ import annotations

from typing import Iterable


class ValueEstimator:
    def estimate(self, observation: Iterable[float]) -> float:
        values = list(observation)
        return sum(values) / len(values)
