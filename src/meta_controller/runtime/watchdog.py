from __future__ import annotations

import time

from meta_controller.interface.parameter_store import ParameterStore


class Watchdog:
    def __init__(self, store: ParameterStore, max_staleness_seconds: float) -> None:
        self.store = store
        self.max_staleness_seconds = max_staleness_seconds

    def healthy(self) -> bool:
        snapshot = self.store.get_latest(now=time.time())
        return snapshot.valid_for_seconds >= self.max_staleness_seconds or snapshot == self.store.default_snapshot()
