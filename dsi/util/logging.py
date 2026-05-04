"""Plain stdout/file logger per CLAUDE.md §6 (in addition to WandB)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_path: Path | str | None = None, level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    log.propagate = False
    return log
