import unittest

from meta_controller.env.reward import compute_reward


class RewardTests(unittest.TestCase):
    def test_reward_contains_all_components(self) -> None:
        terms = compute_reward(
            success_rate=0.8,
            load_balance=0.9,
            temp_penalty=0.1,
            power_penalty=0.2,
            latency_penalty=0.3,
            action_delta_penalty=0.4,
            risk_budget=0.2,
            weights={
                "success": 1.0,
                "balance": 0.25,
                "temp": 0.35,
                "power": 0.30,
                "latency": 0.10,
                "action_delta": 0.05,
            },
        )
        expected = 0.8 + 0.25 * 0.9 - 0.35 * 0.1 - 0.30 * 0.2 - 0.10 * 0.3 - 0.05 * 0.4
        self.assertAlmostEqual(terms.reward, expected)
        self.assertAlmostEqual(terms.reward_components["success"], 0.8)
        self.assertIn("hazard_integral", terms.reward_components)


if __name__ == "__main__":
    unittest.main()
