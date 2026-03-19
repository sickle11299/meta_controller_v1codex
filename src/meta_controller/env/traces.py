from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from meta_controller.interface.telemetry_bus import TelemetryFrame


def load_trace(path: str | Path) -> List[TelemetryFrame]:
    records = json.loads(Path(path).read_text())
    return [TelemetryFrame(**record) for record in records]


def dump_trace(path: str | Path, frames: Iterable[TelemetryFrame]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([frame.to_dict() for frame in frames], indent=2))
