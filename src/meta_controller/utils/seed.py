from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SeedBundle:
    python: int
    numpy: int


def expand_seed(seed: int) -> SeedBundle:
    return SeedBundle(python=seed, numpy=seed + 1)


def set_global_seed(seed: int) -> Dict[str, int]:
    bundle = expand_seed(seed)
    random.seed(bundle.python)
    try:
        import numpy as np  # type: ignore

        np.random.seed(bundle.numpy)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(bundle.python)
    except Exception:
        pass
    return {"python": bundle.python, "numpy": bundle.numpy}
