from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class PowerModelParams:
    """功耗模型参数。

    这些值直接来自上学期实验报告，不在代码里二次“拍脑袋”修改。
    """

    p_idle: float = 3.6474844481882505
    amplitude: float = 2.844673382018813
    exponent_b: float = 34.45251041743898
    p_idle_total: float = 7.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class SaturatingExpPowerModel:
    """基于饱和指数曲线的处理器功耗模型。"""

    def __init__(self, params: PowerModelParams | None = None) -> None:
        self.params = params or PowerModelParams()

    @staticmethod
    def _clamp_utilization(load_frac: float) -> float:
        return min(1.0, max(0.0, float(load_frac)))

    def predict_mpu_power(self, load_frac: float) -> float:
        """对应实验报告中的公式：P = P_idle + A * (1 - exp(-b*u))。"""

        u = self._clamp_utilization(load_frac)
        return self.params.p_idle + self.params.amplitude * (1.0 - math.exp(-self.params.exponent_b * u))

    def dynamic_mpu_power(self, load_frac: float) -> float:
        """动态功耗项，供总功耗校准使用。"""

        return max(0.0, self.predict_mpu_power(load_frac) - self.params.p_idle)

    def predict_total_power(self, load_frac: float, alpha_total: float) -> float:
        """把处理器动态功耗映射成系统总功耗。

        alpha_total 由离线校准得到，不允许在状态转移里写死来源不明的常数。
        """

        dynamic = self.dynamic_mpu_power(load_frac)
        return self.params.p_idle_total + max(0.0, float(alpha_total)) * dynamic
