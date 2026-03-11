from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from meta_controller.analysis.stats import collect_metric


def _iter_summary_files(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob("*/seed_*/summary.json"))


def _load_pairs(results_dir: Path) -> List[Tuple[str, float]]:
    pairs: List[Tuple[str, float]] = []
    for path in _iter_summary_files(results_dir):
        summary = json.loads(path.read_text())
        if "mean_reward" in summary:
            pairs.append((path.parent.parent.name, collect_metric(summary, "mean_reward")))
    return pairs


def _write_svg(path: Path, title: str, pairs: List[Tuple[str, float]]) -> None:
    width = 480
    height = 240
    bar_width = 80
    spacing = 30
    baseline = 200
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="20" y="24" font-size="18">{title}</text>',
    ]
    for index, (label, value) in enumerate(pairs):
        x = 20 + index * (bar_width + spacing)
        bar_height = max(1, int(value * 120))
        y = baseline - bar_height
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#24557a" />')
        parts.append(f'<text x="{x}" y="{baseline + 16}" font-size="12">{label}</text>')
        parts.append(f'<text x="{x}" y="{y - 4}" font-size="12">{value:.3f}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = _load_pairs(results_dir)
    _write_svg(output_dir / "Fig1.svg", "Mean Reward by Run", pairs)
    _write_svg(output_dir / "Fig2.svg", "Mean Reward by Run", pairs)
    print(json.dumps({"output_dir": str(output_dir), "figures": ["Fig1.svg", "Fig2.svg"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
