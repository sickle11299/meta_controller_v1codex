from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List

from meta_controller.env.calibration import CalibrationRow, load_calibration_rows
from meta_controller.env.power_model import SaturatingExpPowerModel
from meta_controller.env.thermal_model import PiecewiseThermalModel, ThermalState


@dataclass
class TelemetryFrame:
    """环境一步推进后暴露给上层的原始物理状态。"""

    soc: float
    load: float
    cpu_temp: float
    rssi: float
    rtt: float
    power_total_w: float
    power_pred_error_w: float = 0.0
    temp_pred_error_c: float = 0.0
    r_th: float = 0.0
    active_duration_s: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "soc": self.soc,
            "load": self.load,
            "cpu_temp": self.cpu_temp,
            "rssi": self.rssi,
            "rtt": self.rtt,
            "power_total_w": self.power_total_w,
            "power_pred_error_w": self.power_pred_error_w,
            "temp_pred_error_c": self.temp_pred_error_c,
            "r_th": self.r_th,
            "active_duration_s": self.active_duration_s,
        }


class TelemetryBus:
    """基于实验拟合参数和校准报告的遥测总线。"""

    def __init__(self, config: Dict[str, Any], calibration_report: Dict[str, Any], seed: int = 0) -> None:
        self._random = random.Random(seed)
        self._config = config
        self._env_config = config["env"]
        self._control_interval_seconds = float(config.get("control_interval_seconds", 5.0))
        self._calibration_report = calibration_report
        calibration_cfg = self._env_config.get("calibration", {})
        self._rows: List[CalibrationRow] = load_calibration_rows(calibration_cfg.get("csv_path", ""))
        self._power_model = SaturatingExpPowerModel()
        self._thermal_model = PiecewiseThermalModel()
        self._alpha_total = float(calibration_report.get("alpha_total", 1.0))
        self._cursor = 0
        self._soc = 0.95
        self._thermal_state: ThermalState = self._thermal_model.reset()
        self._load_noise = float(self._env_config.get("load_noise_scale", 0.03))
        self._rssi_noise = float(self._env_config.get("rssi_noise_scale", 0.03))
        self._latency_noise = float(self._env_config.get("latency_noise_scale", 0.02))
        self._battery_capacity_ws = float(self._env_config.get("battery_capacity_ws", 25000.0))

    @property
    def validation_summary(self) -> Dict[str, Any]:
        return self._calibration_report

    def reset(self) -> None:
        self._cursor = 0
        self._soc = 0.95
        initial_temp = self._rows[0].soc_temp_c if self._rows else None
        self._thermal_state = self._thermal_model.reset(initial_temp_c=initial_temp)

    def _reference_row(self) -> CalibrationRow | None:
        if not self._rows:
            return None
        row = self._rows[self._cursor % len(self._rows)]
        self._cursor += 1
        return row

    def _sample_load(self, row: CalibrationRow | None) -> float:
        base_load = row.load_frac if row is not None else 0.4 + 0.1 * self._random.random()
        mode = str(self._env_config.get("mode", "sim"))
        if mode == "trace":
            return min(1.0, max(0.0, base_load))
        return min(1.0, max(0.0, base_load + self._random.uniform(-self._load_noise, self._load_noise)))

    def _sample_rssi(self, load_frac: float) -> float:
        # 链路强度目前没有真实实验拟合，这里先保留轻量解析代理模型，并限制在合理噪声范围内。
        proposal = 0.88 - 0.45 * load_frac + self._random.uniform(-self._rssi_noise, self._rssi_noise)
        return min(0.99, max(0.05, proposal))

    def _sample_rtt_proxy(self, load_frac: float, rssi: float) -> float:
        # 时延仍是代理量，后续拿到真实时延实验后可以在这一层替换。
        proposal = 0.10 + 0.55 * load_frac + 0.25 * (1.0 - rssi)
        proposal += self._random.uniform(-self._latency_noise, self._latency_noise)
        return min(1.0, max(0.0, proposal))

    def _update_soc(self, power_total_w: float) -> None:
        soc_drop = self._control_interval_seconds * power_total_w / max(1.0, self._battery_capacity_ws)
        self._soc = max(0.05, self._soc - soc_drop)

    def next_frame(self) -> TelemetryFrame:
        row = self._reference_row()
        load_frac = self._sample_load(row)
        power_total_w = self._power_model.predict_total_power(load_frac, self._alpha_total)
        self._thermal_state = self._thermal_model.step(
            self._thermal_state,
            power_total_w=power_total_w,
            load_frac=load_frac,
            dt_s=self._control_interval_seconds,
        )
        rssi = self._sample_rssi(load_frac)
        rtt = self._sample_rtt_proxy(load_frac, rssi)
        self._update_soc(power_total_w)

        reference_power = row.power_sum_w if row is not None else power_total_w
        reference_temp = row.soc_temp_c if row is not None else self._thermal_state.temperature_c
        return TelemetryFrame(
            soc=self._soc,
            load=load_frac,
            cpu_temp=self._thermal_state.temperature_c,
            rssi=rssi,
            rtt=rtt,
            power_total_w=power_total_w,
            power_pred_error_w=power_total_w - reference_power,
            temp_pred_error_c=self._thermal_state.temperature_c - reference_temp,
            r_th=self._thermal_state.r_th,
            active_duration_s=self._thermal_state.active_duration_s,
        )
