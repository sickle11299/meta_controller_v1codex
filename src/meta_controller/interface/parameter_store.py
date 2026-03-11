from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from meta_controller.controller.action_mapping import ParameterSnapshot


@dataclass(frozen=True)
class StoredParameters:
    snapshot: ParameterSnapshot
    published_at: float


class ParameterStore:
    def __init__(self, default_snapshot: ParameterSnapshot) -> None:
        self._default = default_snapshot
        self._current = StoredParameters(snapshot=default_snapshot, published_at=time.time())
        self._lock = threading.Lock()

    def publish(self, snapshot: ParameterSnapshot) -> None:
        with self._lock:
            self._current = StoredParameters(snapshot=snapshot, published_at=time.time())

    def get_latest(self, now: Optional[float] = None) -> ParameterSnapshot:
        with self._lock:
            current = self._current
        now = time.time() if now is None else now
        age = now - current.published_at
        if age > current.snapshot.valid_for_seconds:
            return self._default
        return current.snapshot

    def default_snapshot(self) -> ParameterSnapshot:
        return self._default
