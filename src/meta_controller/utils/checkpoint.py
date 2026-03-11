from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_checkpoint(path: str | Path, state: Dict[str, Any]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(state, indent=2))


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())
