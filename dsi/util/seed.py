"""Determinism helpers. Set seeds at every entry point and log them."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RNGState:
    python: tuple[Any, ...]
    numpy: dict[str, Any]
    torch_cpu: Any
    torch_cuda: list[Any] | None


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def capture_rng() -> RNGState:
    import torch

    return RNGState(
        python=random.getstate(),
        numpy={"state": np.random.get_state()},
        torch_cpu=torch.get_rng_state(),
        torch_cuda=(
            [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else None
        ),
    )


def restore_rng(s: RNGState) -> None:
    import torch

    random.setstate(s.python)
    np.random.set_state(s.numpy["state"])
    torch.set_rng_state(s.torch_cpu)
    if s.torch_cuda is not None and torch.cuda.is_available():
        for i, st in enumerate(s.torch_cuda):
            torch.cuda.set_rng_state(st, device=i)
