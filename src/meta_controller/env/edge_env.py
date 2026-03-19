from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from meta_controller.controller.action_mapping import ParameterSnapshot
from meta_controller.env.calibration import ensure_calibration_report
from meta_controller.env.observation import ObservationNormalizer, build_observation
from meta_controller.env.reward import RewardTerms, compute_load_balance, compute_reward
from meta_controller.env.termination import should_terminate
from meta_controller.interface.telemetry_bus import TelemetryBus, TelemetryFrame
from meta_controller.interface.utaa_client import SchedulerSummary, UTAAClient


@dataclass
class Transition:
    observation: List[float]
    reward: float
    done: bool
    info: Dict[str, Any]


class EdgeEnv:
    """强化学习环境主类。

    这一层负责把三个世界连起来：
    1. 物理仿真世界：`TelemetryBus` 提供温度、功耗、链路等原始状态；
    2. 调度动作世界：`ParameterSnapshot` 表示 RL 侧给调度器的参数建议；
    3. 强化学习世界：输出 `observation / reward / done / info`。
    """

    def __init__(self, config: Dict[str, object], telemetry_bus: TelemetryBus | None = None, scheduler: UTAAClient | None = None) -> None:
        self.config = config
        env_config = config["env"]
        reward_config = config["reward"]
        self.risk_base = float(env_config["risk_base"])
        self.horizon = int(env_config["episode_horizon"])
        self.reward_weights = {
            "success": float(reward_config["weights"]["success"]),
            "balance": float(reward_config["weights"]["balance"]),
            "temp": float(reward_config["weights"]["temp"]),
            "power": float(reward_config["weights"]["power"]),
            "latency": float(reward_config["weights"]["latency"]),
            "action_delta": float(reward_config["weights"]["action_delta"]),
        }
        self.calibration_report = ensure_calibration_report(
            env_config=env_config,
            control_interval_seconds=float(config.get("control_interval_seconds", 5.0)),
            report_path_override=env_config.get("calibration", {}).get("report_path"),
        )
        observation_cfg = env_config.get("observation", {})
        self.normalizer = ObservationNormalizer(
            temp_idle_c=float(observation_cfg.get("temp_idle_c", 34.43080813439079)),
            temp_max_c=float(observation_cfg.get("temp_max_c", 60.0)),
        )
        self.telemetry_bus = telemetry_bus or TelemetryBus(config=config, calibration_report=self.calibration_report, seed=int(config["seed"]))
        self.scheduler = scheduler or UTAAClient()
        self._previous_frame: Optional[TelemetryFrame] = None
        self._previous_action: Optional[ParameterSnapshot] = None
        self._step = 0

    def reset(self) -> List[float]:
        self.telemetry_bus.reset()
        self._previous_frame = None
        self._previous_action = None
        self._step = 0
        frame = self.telemetry_bus.next_frame()
        observation = build_observation(frame, self._previous_frame, self.normalizer)
        self._previous_frame = frame
        return observation.values

    def step(self, snapshot: ParameterSnapshot) -> Transition:
        frame = self.telemetry_bus.next_frame()
        observation = build_observation(frame, self._previous_frame, self.normalizer)
        scheduler_summary = self.scheduler.schedule(observation.values, snapshot)
        reward_terms = self._reward(snapshot, scheduler_summary, frame)
        self._step += 1
        done = should_terminate(self._step, self.horizon, frame.soc, frame.cpu_temp, frame.rssi)
        info = {
            "success_rate": reward_terms.success_rate,
            "load_balance": reward_terms.load_balance,
            "temp_penalty": reward_terms.temp_penalty,
            "power_penalty": reward_terms.power_penalty,
            "latency_penalty": reward_terms.latency_penalty,
            "hazard_integral": reward_terms.hazard_integral,
            "action_delta_penalty": reward_terms.action_delta_penalty,
            "scheduled": float(scheduler_summary.scheduled),
            "succeeded": float(scheduler_summary.succeeded),
            "latency_ms": scheduler_summary.latency_ms,
            "cpu_temp_c": frame.cpu_temp,
            "power_total_w": frame.power_total_w,
            "power_pred_error_w": frame.power_pred_error_w,
            "temp_pred_error_c": frame.temp_pred_error_c,
            "r_th": frame.r_th,
            "active_duration_s": frame.active_duration_s,
            "reward_components": reward_terms.reward_components,
            "model_validation_passed": bool(self.calibration_report.get("model_validation_passed", False)),
        }
        self._previous_frame = frame
        self._previous_action = snapshot
        return Transition(observation=observation.values, reward=reward_terms.reward, done=done, info=info)

    def _reward(self, snapshot: ParameterSnapshot, summary: SchedulerSummary, frame: TelemetryFrame) -> RewardTerms:
        success_rate = summary.succeeded / max(1, summary.scheduled)
        action_delta_penalty = 0.0
        if self._previous_action is not None:
            action_delta_penalty = sum((a - b) ** 2 for a, b in zip(snapshot.weights, self._previous_action.weights))
            action_delta_penalty += (snapshot.risk_budget - self._previous_action.risk_budget) ** 2

        load_balance = compute_load_balance(frame.load, snapshot.weights)
        temp_penalty = max(0.0, (frame.cpu_temp - 55.0) / 5.0)
        power_penalty = max(0.0, (frame.power_total_w - 14.0) / 2.0)
        latency_penalty = max(0.0, (summary.latency_ms - 12.0) / 12.0)
        return compute_reward(
            success_rate=success_rate,
            load_balance=load_balance,
            temp_penalty=temp_penalty,
            power_penalty=power_penalty,
            latency_penalty=latency_penalty,
            action_delta_penalty=action_delta_penalty,
            risk_budget=snapshot.risk_budget,
            weights=self.reward_weights,
        )
