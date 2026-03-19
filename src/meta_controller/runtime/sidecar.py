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
        rewards: List[float] = []       # 记录当前回合所有奖励
        for version in range(1, max_steps + 1):
            snapshot = run_control_step(self.inference, self.store, observation, version)    # 执行推理，生成动作快照
            transition = self.env.step(snapshot)   # 环境步进，获取转移信息
            #在每个控制步骤后调用 MetricsCollector 的 log_step 方法，记录当前步的数据，包括回合编号、步数、环境交互的转移数据以及动作快照。这些数据将被存储在 MetricsCollector 中，以便后续分析和评估。
            self.collector.log_step(episode_index=episode_index, step=version, transition=transition, snapshot=snapshot)  # 记录当前步数据   
            rewards.append(transition.reward)    # 累计奖励
            observation = transition.observation   # 更新下一时刻观测
            if transition.done:    # 回合结束则退出
                break
        return rewards    # 返回本回合所有奖励
