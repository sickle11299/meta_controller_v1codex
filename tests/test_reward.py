import unittest

from meta_controller.env.reward import compute_reward


class RewardTests(unittest.TestCase):
    def test_reward_penalizes_hazard_and_action_drift(self) -> None:
        terms = compute_reward(
            success_rate=0.8,
            hazard_integral=0.2,
            action_delta_penalty=0.1,
            psi=1.0,
            xi=0.5,
            phi=0.1,
        )
        self.assertAlmostEqual(terms.reward, 0.69)
        self.assertAlmostEqual(terms.success_rate, 0.8)


if __name__ == '__main__':
    unittest.main()
