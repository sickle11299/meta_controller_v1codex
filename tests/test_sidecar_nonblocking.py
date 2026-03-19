import unittest

from meta_controller.controller.action_mapping import SafeActionMapper
from meta_controller.env.edge_env import EdgeEnv
from meta_controller.metrics.collector import MetricsCollector
from meta_controller.runtime.sidecar import SidecarRuntime
from meta_controller.utils.config import load_config


class SidecarTests(unittest.TestCase):
    def test_sidecar_runs_without_blocking_scheduler_path(self) -> None:
        config = load_config('configs/default.yaml')
        # 2. 创建安全动作映射器
        mapper = SafeActionMapper(
            beta=float(config['controller']['beta']),
            risk_base=float(config['env']['risk_base']),
            risk_min=float(config['env']['risk_min']),
            risk_max=float(config['env']['risk_max']),
            weight_min=float(config['env']['weight_min']),
            weight_max=float(config['env']['weight_max']),
            weight_sum_target=float(config['env']['weight_sum_target']),
            smoothing=float(config['controller']['smoothing']),
        )
        runtime = SidecarRuntime(env=EdgeEnv(config), collector=MetricsCollector(), mapper=mapper)
        rewards = runtime.run_episode(episode_index=0, max_steps=3)
        # 5. 验证两个核心契约
        self.assertGreater(len(rewards), 0)  # 必须有奖励值返回
        self.assertEqual(len(runtime.collector.records), len(rewards))  # 记录数与奖励数一致



if __name__ == '__main__':
    unittest.main()
