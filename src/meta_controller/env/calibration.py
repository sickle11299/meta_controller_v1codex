from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from meta_controller.env.power_model import PowerModelParams, SaturatingExpPowerModel
from meta_controller.env.thermal_model import PiecewiseThermalModel, ThermalModelParams


@dataclass(frozen=True)
class CalibrationRow:
    """从真实 CSV 提取出的最小必要字段。"""

    t_s: float
    load_frac: float
    soc_temp_c: float
    power_sum_w: float


def _safe_float(raw: str | None, default: float = 0.0) -> float:
    if raw in (None, ""):
        return default
    return float(raw)


def load_calibration_rows(csv_path: str | Path) -> List[CalibrationRow]:
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            CalibrationRow(
                t_s=_safe_float(record.get("t_s")),
                load_frac=min(1.0, max(0.0, _safe_float(record.get("cpu_load_pct")) / 100.0)),
                soc_temp_c=_safe_float(record.get("soc_temp_c")),
                power_sum_w=_safe_float(record.get("power_sum_w")),
            )
            for record in reader
        ]


def _percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * ratio))
    return ordered[index]


def _regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    actual = list(y_true)
    pred = list(y_pred)
    if not actual:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "max_abs_error": 0.0}
    errors = [p - y for y, p in zip(actual, pred)]
    mae = sum(abs(err) for err in errors) / len(errors)
    rmse = math.sqrt(sum(err * err for err in errors) / len(errors))
    max_abs_error = max(abs(err) for err in errors)
    mean_y = sum(actual) / len(actual)
    ss_tot = sum((y - mean_y) ** 2 for y in actual)
    ss_res = sum((y - p) ** 2 for y, p in zip(actual, pred))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2, "max_abs_error": max_abs_error}


def _fit_zero_intercept_scale(xs: List[float], ys: List[float]) -> float:
    denominator = sum(value * value for value in xs)
    if denominator <= 1e-12:
        return 1.0
    numerator = sum(x * y for x, y in zip(xs, ys))
    return max(1e-6, numerator / denominator)


def _median_dt(rows: List[CalibrationRow], fallback_dt_s: float) -> float:
    if len(rows) < 2:
        return fallback_dt_s
    deltas = [max(1e-6, rows[index].t_s - rows[index - 1].t_s) for index in range(1, len(rows))]
    return statistics.median(deltas) if deltas else fallback_dt_s


def _build_summary(rows: List[CalibrationRow]) -> Dict[str, float]:
    powers = [row.power_sum_w for row in rows]
    temps = [row.soc_temp_c for row in rows]
    loads = [row.load_frac for row in rows]
    return {
        "row_count": float(len(rows)),
        "power_p95_w": _percentile(powers, 0.95),
        "power_mean_w": sum(powers) / len(powers) if powers else 0.0,
        "temp_p95_c": _percentile(temps, 0.95),
        "temp_max_c": max(temps) if temps else 60.0,
        "load_p95": _percentile(loads, 0.95),
    }


def build_calibration_report(
    csv_path: str | Path,
    report_path: str | Path | None = None,
    power_params: PowerModelParams | None = None,
    thermal_params: ThermalModelParams | None = None,
    fallback_dt_s: float = 5.0,
    power_r2_threshold: float = 0.60,
    power_rmse_threshold: float = 2.0,
    temp_mae_threshold: float = 4.0,
) -> Dict[str, Any]:
    """构建校准与回放验证报告。"""

    power_model = SaturatingExpPowerModel(power_params)
    thermal_model = PiecewiseThermalModel(thermal_params)
    rows = load_calibration_rows(csv_path)
    summary = _build_summary(rows)

    if not rows:
        report = {
            "csv_path": str(csv_path),
            "available": False,
            "message": "calibration csv not found",
            "alpha_total": 1.0,
            "sample_count": 0,
            "power_metrics": {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "max_abs_error": 0.0},
            "temperature_metrics": {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "max_abs_error": 0.0},
            "summary": summary,
            "model_validation_passed": False,
            "power_model_params": power_model.params.to_dict(),
            "thermal_model_params": thermal_model.params.to_dict(),
        }
        if report_path is not None:
            output = Path(report_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    dynamic_terms = [power_model.dynamic_mpu_power(row.load_frac) for row in rows]
    target_dynamic = [max(0.0, row.power_sum_w - power_model.params.p_idle_total) for row in rows]
    alpha_total = _fit_zero_intercept_scale(dynamic_terms, target_dynamic)
    power_predictions = [power_model.predict_total_power(row.load_frac, alpha_total) for row in rows]
    power_metrics = _regression_metrics([row.power_sum_w for row in rows], power_predictions)

    dt_s = _median_dt(rows, fallback_dt_s)
    thermal_state = thermal_model.reset(initial_temp_c=rows[0].soc_temp_c)
    temp_predictions = [thermal_state.temperature_c]
    for row in rows[1:]:
        predicted_power = power_model.predict_total_power(row.load_frac, alpha_total)
        thermal_state = thermal_model.step(thermal_state, predicted_power, row.load_frac, dt_s)
        temp_predictions.append(thermal_state.temperature_c)
    temp_metrics = _regression_metrics([row.soc_temp_c for row in rows], temp_predictions)

    model_validation_passed = (
        (power_metrics["r2"] >= power_r2_threshold or power_metrics["rmse"] <= power_rmse_threshold)
        and temp_metrics["mae"] <= temp_mae_threshold
    )
    report = {
        "csv_path": str(csv_path),
        "available": True,
        "alpha_total": alpha_total,
        "sample_count": len(rows),
        "dt_s": dt_s,
        "power_metrics": power_metrics,
        "temperature_metrics": temp_metrics,
        "summary": summary,
        "model_validation_passed": model_validation_passed,
        "power_model_params": power_model.params.to_dict(),
        "thermal_model_params": thermal_model.params.to_dict(),
    }
    if report_path is not None:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def ensure_calibration_report(
    env_config: Dict[str, Any],
    control_interval_seconds: float,
    report_path_override: str | Path | None = None,
) -> Dict[str, Any]:
    calibration_config = env_config.get("calibration", {})
    return build_calibration_report(
        csv_path=calibration_config.get("csv_path", ""),
        report_path=report_path_override or calibration_config.get("report_path"),
        fallback_dt_s=float(calibration_config.get("fallback_dt_s", control_interval_seconds)),
        power_r2_threshold=float(calibration_config.get("power_r2_threshold", 0.60)),
        power_rmse_threshold=float(calibration_config.get("power_rmse_threshold", 2.0)),
        temp_mae_threshold=float(calibration_config.get("temp_mae_threshold", 4.0)),
    )
