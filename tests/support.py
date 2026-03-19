from __future__ import annotations

import csv
from pathlib import Path

from meta_controller.env.power_model import SaturatingExpPowerModel
from meta_controller.env.thermal_model import PiecewiseThermalModel


def write_calibration_csv(path: str | Path, alpha_total: float = 2.1, dt_s: float = 5.0) -> Path:
    """写入一份自包含的校准 CSV。

    测试数据刻意用和实现相同的物理公式生成，
    目的是验证“代码链路是否正确”，而不是验证实验数据本身。
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    power_model = SaturatingExpPowerModel()
    thermal_model = PiecewiseThermalModel()
    thermal_state = thermal_model.reset()

    loads = []
    loads.extend([0.08] * 20)
    loads.extend([0.82] * 120)
    loads.extend([0.18] * 20)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t_s", "cpu_load_pct", "soc_temp_c", "power_sum_w"])
        writer.writeheader()
        for index, load_frac in enumerate(loads):
            power_total_w = power_model.predict_total_power(load_frac, alpha_total)
            thermal_state = thermal_model.step(thermal_state, power_total_w, load_frac, dt_s)
            writer.writerow(
                {
                    "t_s": round(index * dt_s, 4),
                    "cpu_load_pct": round(load_frac * 100.0, 4),
                    "soc_temp_c": round(thermal_state.temperature_c, 4),
                    "power_sum_w": round(power_total_w, 4),
                }
            )
    return output
