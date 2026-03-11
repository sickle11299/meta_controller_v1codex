from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for path in sorted(results_dir.glob("*/seed_*/summary.json")):
        summary = json.loads(path.read_text())
        if "mean_reward" in summary:
            rows.append({
                "experiment": path.parent.parent.name,
                "seed": path.parent.name,
                "mean_reward": summary.get("mean_reward", 0.0),
                "mean_success_rate": summary.get("mean_success_rate", 0.0),
                "mean_hazard": summary.get("mean_hazard", 0.0),
            })
    with (output_dir / "Table1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "seed", "mean_reward", "mean_success_rate", "mean_hazard"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_dir": str(output_dir), "table": "Table1.csv"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
