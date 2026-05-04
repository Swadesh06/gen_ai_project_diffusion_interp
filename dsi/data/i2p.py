"""I2P prompt loader.

Two splits:
  - I2P (`AIML-TUDA/i2p`): 4703 prompts with category labels (nudity, violence, hate, ...).
  - I2P-adversarial (`AIML-TUDA/i2p-adversarial-split`): the harder adversarial subset.

`load_i2p()` returns a list of `Prompt` records. Loading uses HF `datasets` if available;
otherwise falls back to a local cached parquet/CSV at `cfg.paths.data_root/i2p/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

from dsi.config import cfg
from dsi.data.types import Prompt

I2PSplit = Literal["full", "adversarial"]
NSFW_CATEGORIES = {
    "sexual", "nudity", "violence", "self-harm", "hate", "harassment",
    "shocking", "illegal activity",
}


def _category_label(cat: str) -> str:
    cat_l = cat.lower().strip()
    return "nsfw" if any(k in cat_l for k in NSFW_CATEGORIES) else "benign"


def load_i2p(split: I2PSplit = "full", limit: int | None = None) -> list[Prompt]:
    """Load I2P prompts. Returns up to `limit` prompts (None = all)."""
    repo = "AIML-TUDA/i2p" if split == "full" else "AIML-TUDA/i2p-adversarial-split"

    try:
        from datasets import load_dataset

        ds = load_dataset(repo, split="train", cache_dir=str(cfg.paths.model_root))
    except Exception:
        return _load_local_csv(split, limit)

    out: list[Prompt] = []
    for row in ds:
        text = row.get("prompt", row.get("text", ""))
        cat = row.get("categories", row.get("category", ""))
        if isinstance(cat, list):
            cat = ",".join(cat)
        out.append(
            Prompt(
                text=str(text),
                source=f"i2p/{split}",
                label=_category_label(str(cat)),
                category=str(cat),
                seed=int(row.get("sd_seed", row.get("seed", 0)) or 0),
                extra={"hard_nsfw": bool(row.get("hard", False))},
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def _load_local_csv(split: I2PSplit, limit: int | None) -> list[Prompt]:
    path = cfg.paths.data_root / "i2p" / f"{split}.csv"
    if not path.exists():
        return []
    import csv

    out: list[Prompt] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row.get("categories", row.get("category", ""))
            out.append(
                Prompt(
                    text=row.get("prompt", row.get("text", "")),
                    source=f"i2p/{split}",
                    label=_category_label(cat),
                    category=cat,
                    seed=int(row.get("sd_seed", row.get("seed", 0)) or 0),
                )
            )
            if limit is not None and len(out) >= limit:
                break
    return out


def i2p_nsfw_subset(limit: int | None = None) -> list[Prompt]:
    return [p for p in load_i2p("full", limit=None) if p.label == "nsfw"][:limit]


def iter_i2p(split: I2PSplit = "full") -> Iterable[Prompt]:
    yield from load_i2p(split, limit=None)
