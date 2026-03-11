from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


Config = Dict[str, Any]


def _deep_merge(base: Config, override: Config) -> Config:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    data = json.loads(config_path.read_text())
    parent = data.get("extends")
    if parent:
        parent_config = load_config(config_path.parent.parent / Path(parent).name) if not Path(parent).exists() else load_config(parent)
        data = _deep_merge(parent_config, data)
    return data


def parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def apply_overrides(config: Config, overrides: list[str]) -> Config:
    updated = dict(config)
    for item in overrides:
        key, raw_value = item.split("=", 1)
        parts = key.split(".")
        cursor = updated
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = parse_value(raw_value)
    return updated


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--results-dir")
    return parser
