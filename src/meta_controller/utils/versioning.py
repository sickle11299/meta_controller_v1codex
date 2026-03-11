from __future__ import annotations

import subprocess
from typing import Dict


def git_metadata() -> Dict[str, str]:
    def _run(*args: str) -> str:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    commit = _run("git", "rev-parse", "HEAD")
    branch = _run("git", "rev-parse", "--abbrev-ref", "HEAD")
    dirty = "true" if _run("git", "status", "--porcelain") not in {"", "unknown"} else "false"
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }
