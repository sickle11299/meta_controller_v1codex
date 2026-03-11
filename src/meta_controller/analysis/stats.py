from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def load_summary(path: str | Path) -> Dict[str, float]:
    return json.loads(Path(path).read_text())


def collect_metric(summary: Dict[str, object], key: str) -> float:
    value = summary.get(key, 0.0)
    return float(value) if not isinstance(value, dict) else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
