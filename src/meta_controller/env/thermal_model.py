from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class ThermalModelParams:
    """热模型参数，直接复用实验报告结果。"""

    t_idle_c: float = 34.43080813439079
    p_idle_total_w: float = 7.0
    c_th: float = 33.076555038714595
    r_off: float = 3.794006935090086
    r_on: float = 2.717063804611588
    t_change_s: float = 520.0
    k_on_suggested: float = 1.3963628416272873
    active_load_threshold: float = 0.70

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ThermalState:
    """保存当前热状态，便于测试和日志追踪。"""

    temperature_c: float
    active_duration_s: float = 0.0
    r_th: float = 0.0
    k_dynamic: float = 1.0
    branch: str = "off"

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "temperature_c": self.temperature_c,
            "active_duration_s": self.active_duration_s,
            "r_th": self.r_th,
            "k_dynamic": self.k_dynamic,
            "branch": self.branch,
        }


class PiecewiseThermalModel:
    """分段热阻 RC 热模型。

    触发逻辑本轮写死成：
    1. 负载率连续高于阈值 `active_load_threshold`；
    2. 累积时长达到 `t_change_s`；
    3. 从 `R_off` 切到 `R_on`；
    4. 只要高负载中断，累计时长清零并回到 `R_off`。
    """

    def __init__(self, params: ThermalModelParams | None = None) -> None:
        self.params = params or ThermalModelParams()

    def reset(self, initial_temp_c: float | None = None) -> ThermalState:
        start_temp = self.params.t_idle_c if initial_temp_c is None else float(initial_temp_c)
        return ThermalState(
            temperature_c=start_temp,
            active_duration_s=0.0,
            r_th=self.params.r_off,
            k_dynamic=1.0,
            branch="off",
        )

    def step(self, state: ThermalState, power_total_w: float, load_frac: float, dt_s: float) -> ThermalState:
        """推进一步温度状态。"""

        active_duration_s = state.active_duration_s + dt_s if load_frac >= self.params.active_load_threshold else 0.0
        branch = "on" if active_duration_s >= self.params.t_change_s else "off"
        r_th = self.params.r_on if branch == "on" else self.params.r_off
        k_dynamic = self.params.k_on_suggested if branch == "on" else 1.0

        p_dyn = max(0.0, float(power_total_w) - self.params.p_idle_total_w)
        tau = max(1e-6, r_th * self.params.c_th)
        decay = math.exp(-max(1e-6, dt_s) / tau)
        next_temp = self.params.t_idle_c + (state.temperature_c - self.params.t_idle_c) * decay
        next_temp += r_th * (k_dynamic * p_dyn) * (1.0 - decay)

        return ThermalState(
            temperature_c=next_temp,
            active_duration_s=active_duration_s,
            r_th=r_th,
            k_dynamic=k_dynamic,
            branch=branch,
        )
