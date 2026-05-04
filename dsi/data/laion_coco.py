"""Benign-prompt loader for SAE activation collection.

Original target: LAION-COCO (50K captions). As of 2026-05-04 the upstream HF dataset
`laion/laion-coco` is gated, so we fall back to a multi-source pool:
  1. local parquet (if a future session manages to download LAION-COCO),
  2. COCO val 2017 captions (~25K, already on disk),
  3. LAION-400M-WIT alternatives (sample by streaming from open mirror if available).

Net effect: `load_laion_coco(limit)` always returns a non-empty benign-prompt list as long
as one of the fallback sources is reachable. `cfg.paths.data_root/laion_coco/captions.parquet`
is the fast path; everything else is a fallback.
"""

from __future__ import annotations

from dsi.config import cfg
from dsi.data.types import Prompt

DEFAULT_REPO = "laion/laion-coco"
ALTERNATIVE_REPOS = ("laion/laion400m-meta",)


def load_laion_coco(limit: int = 50_000) -> list[Prompt]:
    """Up to `limit` benign prompts, preferring the original LAION-COCO when reachable."""
    parquet_rows = _load_local(limit)
    if parquet_rows:
        return parquet_rows[:limit]

    rows = _try_streaming(DEFAULT_REPO, limit)
    if rows:
        return rows
    for alt in ALTERNATIVE_REPOS:
        rows = _try_streaming(alt, limit, text_keys=("TEXT", "caption", "text"))
        if rows:
            return rows

    return _fallback_to_coco(limit)


def _try_streaming(repo: str, limit: int, text_keys: tuple[str, ...] = ("caption", "TEXT", "top_caption")) -> list[Prompt]:
    try:
        from datasets import load_dataset

        ds = load_dataset(repo, split="train", streaming=True, cache_dir=str(cfg.paths.model_root))
    except Exception:
        return []
    out: list[Prompt] = []
    try:
        for row in ds:
            text = next((row[k] for k in text_keys if row.get(k)), "")
            if not text:
                continue
            out.append(Prompt(text=str(text), source=repo, label="benign", category="laion"))
            if len(out) >= limit:
                break
    except Exception:
        pass
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
        out.append(Prompt(text=str(txt), source="laion-coco-local", label="benign", category="laion"))
    return out


def _fallback_to_coco(limit: int) -> list[Prompt]:
    """If LAION-COCO is unreachable, substitute COCO val captions (already on disk)."""
    from dsi.data.coco import load_coco_captions

    rows = load_coco_captions(limit=limit)
    for r in rows:
        r.source = f"coco-fallback({r.source})"
    return rows
