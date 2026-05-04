"""LAION-COCO subset loader.

For SAE activation collection we need ~50K benign prompts; LAION-COCO captions are
diverse and well-aligned with the SDXL distribution. We only need text (no images).
"""

from __future__ import annotations

from pathlib import Path

from dsi.config import cfg
from dsi.data.types import Prompt

DEFAULT_REPO = "laion/laion-coco"


def load_laion_coco(limit: int = 50_000) -> list[Prompt]:
    """Returns up to `limit` LAION-COCO captions.

    Streams from HF if available; falls back to a local parquet under
    `cfg.paths.data_root/laion_coco/captions.parquet`.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(DEFAULT_REPO, split="train", streaming=True, cache_dir=str(cfg.paths.model_root))
    except Exception:
        return _load_local(limit)

    out: list[Prompt] = []
    for row in ds:
        text = row.get("caption") or row.get("TEXT") or row.get("top_caption") or ""
        if not text:
            continue
        out.append(Prompt(text=str(text), source="laion-coco", label="benign", category="laion"))
        if len(out) >= limit:
            break
    return out


def _load_local(limit: int) -> list[Prompt]:
    parquet = cfg.paths.data_root / "laion_coco" / "captions.parquet"
    if not parquet.exists():
        return []
    import pandas as pd

    df = pd.read_parquet(parquet)
    text_col = "caption" if "caption" in df.columns else df.columns[0]
    out: list[Prompt] = []
    for txt in df[text_col].head(limit).tolist():
        out.append(Prompt(text=str(txt), source="laion-coco", label="benign", category="laion"))
    return out
