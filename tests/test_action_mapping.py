import unittest

from meta_controller.controller.action_mapping import SafeActionMapper


class ActionMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = SafeActionMapper(
            beta=1.5,
            risk_base=0.2,
            risk_min=0.05,
            risk_max=0.5,
            weight_min=0.1,
            weight_max=4.0,
            weight_sum_target=4.0,
            smoothing=0.25,
        )

    def test_mapping_stays_within_constraints(self) -> None:
        snapshot = self.mapper.map_action([0.4, -0.5, 0.1, 0.8, -0.2], previous=None, version=1)
        self.assertEqual(len(snapshot.weights), 4)
        self.assertAlmostEqual(sum(snapshot.weights), 4.0, places=6)
        self.assertGreaterEqual(snapshot.risk_budget, 0.05)
        self.assertLessEqual(snapshot.risk_budget, 0.5)


if __name__ == "__main__":
    unittest.main()
