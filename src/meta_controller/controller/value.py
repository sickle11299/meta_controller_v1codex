from __future__ import annotations

from typing import Iterable


class ValueEstimator:
    def estimate(self, observation: Iterable[float]) -> float:  #param observation: 可迭代的观测状态向量
        values = list(observation)   # 对观测状态做价值估计（这里用均值作为估计值）
        return sum(values) / len(values)         #return: 观测值的平均值
