from __future__ import annotations

import json
from pathlib import Path
from typing import List

from meta_controller.experiments.train import main as train_main


def main(argv: List[str] | None = None) -> int:
    seeds = [7, 11, 19]
    runs = []
    for seed in seeds:
        train_main(["--config", "configs/training.yaml", "--set", f"seed={seed}"])
        runs.append(seed)
    print(json.dumps({"completed_seeds": runs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
