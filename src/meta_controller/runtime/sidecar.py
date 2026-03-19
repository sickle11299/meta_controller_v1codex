from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from meta_controller.controller.action_mapping import ParameterSnapshot, SafeActionMapper
from meta_controller.controller.inference_service import InferenceService
from meta_controller.controller.policy import MetaPolicy
from meta_controller.controller.ppo import RolloutStep
from meta_controller.controller.value import ValueEstimator
from meta_controller.env.edge_env import EdgeEnv
from meta_controller.interface.parameter_store import ParameterStore
from meta_controller.metrics.collector import MetricsCollector
from meta_controller.runtime.control_loop import run_control_step


@dataclass
class RolloutEpisode:
    rewards: List[float]
    steps: List[RolloutStep]


class SidecarRuntime:
    def __init__(self, env: EdgeEnv, collector: MetricsCollector, mapper: SafeActionMapper, policy: MetaPolicy | None = None) -> None:
        self.env = env
        self.collector = collector
        self.policy = policy or MetaPolicy()
        self.inference = InferenceService(policy=self.policy, mapper=mapper)
        self.mapper = mapper
        self.store = ParameterStore(default_snapshot=ParameterSnapshot(weights=[1.0, 1.0, 1.0, 1.0], risk_budget=env.risk_base))

    def run_episode(
        self,
        episode_index: int,
        max_steps: int,
        collect_rollout: bool = False,
        value_estimator: ValueEstimator | None = None,
    ) -> RolloutEpisode | List[float]:
        observation = self.env.reset()
        rewards: List[float] = []
        rollout_steps: List[RolloutStep] = []
        for version in range(1, max_steps + 1):
            if collect_rollout:
                if value_estimator is None:
                    raise ValueError("value_estimator is required when collect_rollout=True")
                with torch.no_grad():
                    action_tensor, log_prob_tensor, _ = self.policy.sample_action(observation)
                    action = action_tensor.squeeze(0).detach().cpu().tolist()
                    value = float(value_estimator.forward(observation).item())
                previous = self.store.get_latest()
                snapshot = self.mapper.map_action(action, previous=previous, version=version)
                self.store.publish(snapshot)
            else:
                snapshot = run_control_step(self.inference, self.store, observation, version)
                action = None
                log_prob_tensor = None
                value = 0.0

            transition = self.env.step(snapshot)
            self.collector.log_step(episode_index=episode_index, step=version, transition=transition, snapshot=snapshot)
            rewards.append(transition.reward)

            if collect_rollout:
                with torch.no_grad():
                    next_value = 0.0 if transition.done else float(value_estimator.forward(transition.observation).item())
                rollout_steps.append(
                    RolloutStep(
                        observation=list(observation),
                        action=list(action),
                        log_prob=float(log_prob_tensor.item()),
                        reward=transition.reward,
                        done=transition.done,
                        value=value,
                        next_value=next_value,
                    )
                )

            observation = transition.observation
            if transition.done:
                break

        if collect_rollout:
            return RolloutEpisode(rewards=rewards, steps=rollout_steps)
        return rewards
