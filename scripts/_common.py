"""Shared script scaffolding.

Common argparse parser, dry-run reporter, exp_id minting, log-path resolution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import cfg  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--exp-id", type=str, default="", help="explicit experiment id; auto-generated if empty")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--wandb", action="store_true", help="log to WandB")
    p.add_argument("--no-wandb", action="store_true", help="explicitly disable WandB even if env asks")
    p.add_argument("--dry-run", action="store_true", help="print what would happen, do not launch GPU work")
    p.add_argument("--dry-smoke", action="store_true", help="run a tiny version (a few iters) to sanity-check")
    return p


def auto_exp_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time())}"


def serialize(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, (list, tuple)):
        return [serialize(v) for v in x]
    if isinstance(x, dict):
        return {k: serialize(v) for k, v in x.items()}
    return x


def write_dry_report(exp_id: str, plan: dict, out_dir: Path | None = None) -> Path:
    out = out_dir or (cfg.paths.report_root / "dry")
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{exp_id}.dry.json"
    p.write_text(json.dumps(serialize(plan), indent=2, default=str))
    return p


def echo_dry(plan: dict) -> None:
    print("== DRY RUN ==")
    print(json.dumps(serialize(plan), indent=2, default=str))
    print("== END DRY RUN ==")


def resolve_exp_id(args, prefix: str) -> str:
    return args.exp_id or auto_exp_id(prefix)
