import unittest

from meta_controller.controller.policy import MetaPolicy
from meta_controller.utils.seed import set_global_seed


class SeedDeterminismTests(unittest.TestCase):
    def test_policy_output_is_repeatable(self) -> None:
        set_global_seed(7)
        policy = MetaPolicy()
        first = policy.act([0.9, 0.4, 0.5, 0.7, 0.2, 0.0, 0.0])
        set_global_seed(7)
        second = policy.act([0.9, 0.4, 0.5, 0.7, 0.2, 0.0, 0.0])
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
