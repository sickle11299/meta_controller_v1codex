from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from meta_controller.interface.telemetry_bus import TelemetryFrame


@dataclass
class Observation:
    values: List[float]
    raw: Dict[str, float]

    def to_dict(self) -> Dict[str, float]:
        return dict(self.raw)


@dataclass(frozen=True)
class ObservationNormalizer:
    """观测归一化参数。

    这轮我们把主线状态空间收敛到 7 维，优先保证：
    1. 状态语义足够清晰；
    2. 和项目目标相关；
    3. 不过度依赖当前还在继续校准的功耗拟合误差。
    """

    temp_idle_c: float = 34.43080813439079
    temp_max_c: float = 60.0


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def build_observation(
    current: TelemetryFrame,
    previous: TelemetryFrame | None,
    normalizer: ObservationNormalizer,
) -> Observation:
    """构造 7 维观测向量。

    最终顺序固定为：
    [soc, load_frac, temp_norm, rssi_norm, latency_norm, temp_load_gradient, rssi_gradient]

    这 7 维的取舍思路是：
    - 保留对调度最关键的电量、负载、温度、链路和时延；
    - 用两个梯度项补“趋势信息”，帮助 PPO 判断系统正在变好还是变坏；
    - 暂时不把功耗显式塞进状态，避免把当前物理模型误差直接灌进策略输入。
    """

    temp_span = max(1.0, normalizer.temp_max_c - normalizer.temp_idle_c)
    temp_norm = _clip01((current.cpu_temp - normalizer.temp_idle_c) / temp_span)

    if previous is None:
        temp_load_gradient = 0.0
        rssi_gradient = 0.0
    else:
        load_delta = current.load - previous.load
        temp_delta = current.cpu_temp - previous.cpu_temp
        # 这里保留“温度相对负载变化率”，而不是单纯温度差，
        # 是为了让状态能表达“负载一变，温度响应有多敏感”。
        temp_load_gradient = temp_delta / load_delta if abs(load_delta) > 1e-6 else 0.0
        temp_load_gradient /= temp_span
        rssi_gradient = current.rssi - previous.rssi

    values = [
        _clip01(current.soc),
        _clip01(current.load),
        temp_norm,
        _clip01(current.rssi),
        _clip01(current.rtt),
        max(-1.0, min(1.0, temp_load_gradient)),
        max(-1.0, min(1.0, rssi_gradient)),
    ]
    raw = {
        "soc": current.soc,
        "load_frac": current.load,
        "cpu_temp_c": current.cpu_temp,
        "rssi_norm": current.rssi,
        "latency_norm": current.rtt,
        "temp_load_gradient": values[5],
        "rssi_gradient": values[6],
        # 原始功耗仍然保留在 raw 里，方便日志、奖励和后续调试，不必彻底丢掉。
        "power_total_w": current.power_total_w,
    }
    return Observation(values=values, raw=raw)
