import unittest

from meta_controller.controller.action_mapping import ParameterSnapshot
from meta_controller.interface.parameter_store import ParameterStore


class ConstraintTests(unittest.TestCase):
    def test_store_falls_back_when_snapshot_is_stale(self) -> None:
        default = ParameterSnapshot(weights=[1.0, 1.0, 1.0, 1.0], risk_budget=0.2, version=0, valid_for_seconds=0.1)
        store = ParameterStore(default_snapshot=default)
        stale = store.get_latest(now=9999999999.0)
        self.assertEqual(stale, default)


if __name__ == "__main__":
    unittest.main()
