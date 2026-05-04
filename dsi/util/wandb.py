"""WandB init helper. Standard tags, config dict, log dir per CLAUDE.md §6."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, is_dataclass
from typing import Any

from dsi.config import cfg


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=str(cfg.paths.repo),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _to_serializable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    return x


def init_wandb(
    exp_id: str,
    config: dict | Any,
    tags: list[str],
    *,
    project: str | None = None,
    notes: str = "",
    group: str | None = None,
    job_type: str | None = None,
):
    """Init a WandB run. Wrapper that fills in standard fields per CLAUDE.md §6.

    Returns the wandb run object, or None if WANDB_MODE=disabled.
    """
    import wandb

    cfg_dict = _to_serializable(config)
    cfg_dict["git_commit"] = _git_commit()

    log_dir = cfg.paths.log_root / "wandb"
    log_dir.mkdir(parents=True, exist_ok=True)

    return wandb.init(
        project=project or cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=exp_id,
        config=cfg_dict,
        tags=tags,
        notes=notes,
        group=group,
        job_type=job_type,
        dir=str(log_dir),
        mode=cfg.wandb.mode,
        reinit=False,
    )
