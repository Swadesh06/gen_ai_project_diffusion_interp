"""Checkpoint helpers per CLAUDE.md §7.

Rolling-deque retention (latest N), atomic last.pt symlink, full state save
(model + optimizer + scheduler + RNG + step + epoch + best_metric + config).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from dsi.util.seed import capture_rng


def _serialize_cfg(c: Any) -> dict:
    if is_dataclass(c):
        return asdict(c)
    if isinstance(c, dict):
        return c
    return {"value": str(c)}


class RollingCheckpointer:
    """Saves checkpoints, keeps only the latest `keep_n`, plus a `last.pt` symlink and `best.pt`."""

    def __init__(self, ckpt_dir: Path | str, keep_n: int = 4):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_n = keep_n
        self._files: list[Path] = sorted(self.dir.glob("step_*.pt"))
        self.best_metric: float | None = None
        self._prune()

    def _prune(self) -> None:
        while len(self._files) > self.keep_n:
            old = self._files.pop(0)
            if old.exists():
                old.unlink()

    def save(
        self,
        step: int,
        epoch: int,
        model_state: dict,
        optimizer_state: dict | None = None,
        scheduler_state: dict | None = None,
        config: Any = None,
        metric: float | None = None,
        better: str = "lower",
        extra: dict | None = None,
    ) -> Path:
        import torch

        payload = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "rng_state": capture_rng(),
            "config": _serialize_cfg(config) if config is not None else None,
            "metric": metric,
            "extra": extra or {},
        }
        path = self.dir / f"step_{step:09d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

        last = self.dir / "last.pt"
        if last.exists() or last.is_symlink():
            last.unlink()
        try:
            last.symlink_to(path.name)
        except OSError:
            shutil.copy2(path, last)

        self._files.append(path)
        self._prune()

        if metric is not None:
            improved = (
                self.best_metric is None
                or (better == "lower" and metric < self.best_metric)
                or (better == "higher" and metric > self.best_metric)
            )
            if improved:
                self.best_metric = metric
                shutil.copy2(path, self.dir / "best.pt")
                with (self.dir / "best.json").open("w") as f:
                    json.dump({"step": step, "epoch": epoch, "metric": metric}, f)

        return path

    @staticmethod
    def load(path: Path | str) -> dict:
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
