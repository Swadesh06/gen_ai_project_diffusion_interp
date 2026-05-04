"""Adversarial prompt loaders: MMA-Diffusion, UnlearnDiffAtk, Ring-A-Bell.

These are the standard adversarial benchmarks for T2I safety.
Cloned source repos live under `cfg.paths.data_root/<name>/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from dsi.config import cfg
from dsi.data.types import Prompt

AdvSource = Literal["mma_text", "mma_image", "unlearndiff", "ringabell"]


def _root() -> Path:
    return cfg.paths.data_root


def load_mma_text(limit: int | None = None) -> list[Prompt]:
    """MMA-Diffusion text-modality adversarial prompts.

    Looks under `cfg.paths.data_root/MMA-Diffusion/` for any of the standard prompt CSV / JSON files.
    """
    base = _root() / "MMA-Diffusion"
    if not base.exists():
        return []
    out: list[Prompt] = []
    for p in sorted(base.rglob("*.csv")):
        if "text" not in p.name.lower() and "prompt" not in p.name.lower():
            continue
        try:
            import csv

            with p.open() as f:
                for row in csv.DictReader(f):
                    text = row.get("adv_prompt") or row.get("prompt") or row.get("text")
                    if text:
                        out.append(
                            Prompt(
                                text=str(text),
                                source=f"mma_text/{p.stem}",
                                label="nsfw",
                                category="mma_adversarial",
                            )
                        )
        except (FileNotFoundError, UnicodeDecodeError):
            continue
        if limit and len(out) >= limit:
            return out[:limit]
    for p in sorted(base.rglob("*.json")):
        if "prompt" not in p.name.lower():
            continue
        try:
            with p.open() as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        if isinstance(data, list):
            for d in data:
                txt = d if isinstance(d, str) else (d.get("prompt") if isinstance(d, dict) else "")
                if txt:
                    out.append(Prompt(text=str(txt), source=f"mma_text/{p.stem}",
                                      label="nsfw", category="mma_adversarial"))
        if limit and len(out) >= limit:
            return out[:limit]
    return out


def load_unlearndiff(limit: int | None = None) -> list[Prompt]:
    """UnlearnDiffAtk crafted prompts."""
    base = _root() / "Diffusion-MU-Attack"
    if not base.exists():
        return []
    out: list[Prompt] = []
    for p in sorted(base.rglob("*.csv")):
        try:
            import csv

            with p.open() as f:
                for row in csv.DictReader(f):
                    text = row.get("prompt") or row.get("adv_prompt") or row.get("text")
                    if text:
                        out.append(
                            Prompt(
                                text=str(text),
                                source=f"unlearndiff/{p.stem}",
                                label="nsfw",
                                category=row.get("concept", "unlearndiff"),
                            )
                        )
        except (FileNotFoundError, UnicodeDecodeError):
            continue
        if limit and len(out) >= limit:
            return out[:limit]
    return out


def load_ringabell(limit: int | None = None) -> list[Prompt]:
    """Ring-A-Bell adversarial concept-embedding prompts."""
    base = _root() / "Ring-A-Bell"
    if not base.exists():
        return []
    out: list[Prompt] = []
    for p in sorted(base.rglob("*.csv")):
        try:
            import csv

            with p.open() as f:
                for row in csv.DictReader(f):
                    text = row.get("prompt") or row.get("text")
                    if text:
                        out.append(
                            Prompt(
                                text=str(text),
                                source=f"ringabell/{p.stem}",
                                label="nsfw",
                                category=row.get("concept", "ringabell"),
                            )
                        )
        except (FileNotFoundError, UnicodeDecodeError):
            continue
        if limit and len(out) >= limit:
            return out[:limit]
    return out


def load_adversarial(source: AdvSource, limit: int | None = None) -> list[Prompt]:
    if source == "mma_text":
        return load_mma_text(limit)
    if source == "mma_image":
        return []
    if source == "unlearndiff":
        return load_unlearndiff(limit)
    if source == "ringabell":
        return load_ringabell(limit)
    raise ValueError(f"Unknown adversarial source: {source}")
