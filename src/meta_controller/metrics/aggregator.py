from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable


def aggregate_run_summaries(paths: Iterable[str | Path]) -> Dict[str, float]:
    summaries = []
    for path in paths:
        summaries.append(Path(path).read_text())
    return {"num_runs": float(len(summaries))}
