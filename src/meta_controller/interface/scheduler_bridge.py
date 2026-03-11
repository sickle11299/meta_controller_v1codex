from __future__ import annotations

from meta_controller.controller.action_mapping import ParameterSnapshot
from meta_controller.interface.parameter_store import ParameterStore


class SchedulerBridge:
    def __init__(self, store: ParameterStore) -> None:
        self.store = store

    def get_scheduler_parameters(self) -> ParameterSnapshot:
        return self.store.get_latest()
