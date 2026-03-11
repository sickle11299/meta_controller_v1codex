from __future__ import annotations

from typing import List

from meta_controller.controller.action_mapping import ParameterSnapshot, SafeActionMapper
from meta_controller.controller.inference_service import InferenceService
from meta_controller.controller.policy import MetaPolicy
from meta_controller.env.edge_env import EdgeEnv
from meta_controller.interface.parameter_store import ParameterStore
from meta_controller.metrics.collector import MetricsCollector
from meta_controller.runtime.control_loop import run_control_step


class SidecarRuntime:
    def __init__(self, env: EdgeEnv, collector: MetricsCollector, mapper: SafeActionMapper) -> None:
        self.env = env
        self.collector = collector
        self.inference = InferenceService(policy=MetaPolicy(), mapper=mapper)
        self.store = ParameterStore(default_snapshot=ParameterSnapshot(weights=[1.0, 1.0, 1.0, 1.0], risk_budget=env.risk_base))

    def run_episode(self, episode_index: int, max_steps: int) -> List[float]:
        observation = self.env.reset()
        rewards: List[float] = []
        for version in range(1, max_steps + 1):
            snapshot = run_control_step(self.inference, self.store, observation, version)
            transition = self.env.step(snapshot)
            self.collector.log_step(episode_index=episode_index, step=version, transition=transition, snapshot=snapshot)
            rewards.append(transition.reward)
            observation = transition.observation
            if transition.done:
                break
        return rewards
