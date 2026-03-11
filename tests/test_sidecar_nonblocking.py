import unittest

from meta_controller.controller.action_mapping import SafeActionMapper
from meta_controller.env.edge_env import EdgeEnv
from meta_controller.metrics.collector import MetricsCollector
from meta_controller.runtime.sidecar import SidecarRuntime
from meta_controller.utils.config import load_config


class SidecarTests(unittest.TestCase):
    def test_sidecar_runs_without_blocking_scheduler_path(self) -> None:
        config = load_config('configs/default.yaml')
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
        self.assertGreater(len(rewards), 0)
        self.assertEqual(len(runtime.collector.records), len(rewards))


if __name__ == '__main__':
    unittest.main()
