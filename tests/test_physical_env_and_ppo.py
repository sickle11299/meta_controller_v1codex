import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from meta_controller.controller.action_mapping import ParameterSnapshot, SafeActionMapper
from meta_controller.controller.policy import MetaPolicy
from meta_controller.controller.ppo import PPOTrainer
from meta_controller.controller.value import ValueEstimator, compute_gae
from meta_controller.env.calibration import build_calibration_report
from meta_controller.env.edge_env import EdgeEnv
from meta_controller.metrics.collector import MetricsCollector
from meta_controller.runtime.sidecar import SidecarRuntime
from meta_controller.utils.config import load_config

from tests.support import write_calibration_csv


class PhysicalEnvAndPPOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.calibration_csv = write_calibration_csv(self.temp_dir / "calibration.csv")
        self.report_path = self.temp_dir / "calibration_report.json"
        self.config = load_config("configs/default.yaml")
        self.config["env"]["calibration"]["csv_path"] = str(self.calibration_csv)
        self.config["env"]["calibration"]["report_path"] = str(self.report_path)
        self.config["env"]["mode"] = "trace"
        self.config["env"]["episode_horizon"] = 160

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_calibration_report_contains_metrics(self) -> None:
        report = build_calibration_report(self.calibration_csv, self.report_path)
        self.assertGreater(report["alpha_total"], 0.0)
        self.assertIn("r2", report["power_metrics"])
        self.assertIn("mae", report["temperature_metrics"])
        self.assertTrue(self.report_path.exists())
        parsed = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["sample_count"], report["sample_count"])

    def test_env_switches_thermal_branch_after_sustained_high_load(self) -> None:
        env = EdgeEnv(self.config)
        observation = env.reset()
        self.assertEqual(len(observation), 7)
        snapshot = ParameterSnapshot(weights=[1.0, 1.0, 1.0, 1.0], risk_budget=0.2, version=1)
        seen_on = False
        last_r_th = 0.0
        for _ in range(150):
            transition = env.step(snapshot)
            last_r_th = float(transition.info["r_th"])
            if last_r_th < 3.0:
                seen_on = True
        self.assertTrue(seen_on)
        self.assertGreater(last_r_th, 3.0)

    def test_compute_gae_shapes(self) -> None:
        advantages, returns = compute_gae(
            rewards=[1.0, 0.5, -0.2],
            values=[0.1, 0.2, 0.3],
            dones=[False, False, True],
            next_values=[0.2, 0.3, 0.0],
            gamma=0.99,
            gae_lambda=0.95,
        )
        self.assertEqual(tuple(advantages.shape), (3,))
        self.assertEqual(tuple(returns.shape), (3,))

    def test_ppo_update_changes_parameters(self) -> None:
        mapper = SafeActionMapper(
            beta=float(self.config["controller"]["beta"]),
            risk_base=float(self.config["env"]["risk_base"]),
            risk_min=float(self.config["env"]["risk_min"]),
            risk_max=float(self.config["env"]["risk_max"]),
            weight_min=float(self.config["env"]["weight_min"]),
            weight_max=float(self.config["env"]["weight_max"]),
            weight_sum_target=float(self.config["env"]["weight_sum_target"]),
            smoothing=float(self.config["controller"]["smoothing"]),
        )
        policy = MetaPolicy()
        value_estimator = ValueEstimator()
        trainer = PPOTrainer(policy=policy, value_estimator=value_estimator, config=self.config)
        runtime = SidecarRuntime(env=EdgeEnv(self.config), collector=MetricsCollector(), mapper=mapper, policy=policy)

        before = [parameter.detach().clone() for parameter in policy.parameters()]
        rollout = runtime.run_episode(
            episode_index=0,
            max_steps=32,
            collect_rollout=True,
            value_estimator=value_estimator,
        )
        stats = trainer.train_epoch(rollout.steps)
        after = list(policy.parameters())
        changed = any(not torch.allclose(old, new.detach(), atol=1e-7) for old, new in zip(before, after))
        self.assertTrue(changed)
        self.assertIn("approx_kl", stats)


if __name__ == "__main__":
    unittest.main()
