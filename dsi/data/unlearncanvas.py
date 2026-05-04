"""UnlearnCanvas loader: 60 styles x 20 objects = 1200 image groups; concept-erasure benchmark.

Supports two on-disk layouts:
  - directory tree `<root>/<style>/<object>/*.{png,jpg}` (original v1 layout)
  - HF parquet files `<root>/data/train-*.parquet` with columns {image, text}
    where `text` is e.g. "An Architectures image in Abstractionism style"
"""

from __future__ import annotations

import re
from pathlib import Path

from dsi.config import cfg
from dsi.data.types import ImageRecord

_TEXT_RE = re.compile(r"An?\s+(?P<obj>[A-Za-z]+?)s?\s+image\s+in\s+(?P<style>.+?)\s+style", re.I)


def _root() -> Path:
    return cfg.paths.data_root / "unlearncanvas"


def _is_parquet_layout() -> bool:
    return (_root() / "data").exists() and any((_root() / "data").glob("*.parquet"))


def _parse_caption(text: str) -> tuple[str, str]:
    m = _TEXT_RE.match(text or "")
    if not m:
        return "", ""
    return m.group("style").strip(), m.group("obj").strip()


def _list_parquets() -> list[Path]:
    return sorted((_root() / "data").glob("*.parquet"))


def list_styles() -> list[str]:
    root = _root()
    if not root.exists():
        return []
    if _is_parquet_layout():
        import pandas as pd

        styles: set[str] = set()
        # Sample up to 30 shards (the full dataset spans 153 shards; 30 is enough to
        # cover all 60 styles assuming roughly balanced shuffling). Early-out at 60.
        for p in _list_parquets()[:30]:
            try:
                df = pd.read_parquet(p, columns=["text"])
            except Exception:
                continue
            for txt in df["text"].astype(str).tolist():
                s, _ = _parse_caption(txt)
                if s:
                    styles.add(s)
            if len(styles) >= 60:
                break
        return sorted(styles)
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_objects(style: str) -> list[str]:
    if _is_parquet_layout():
        import pandas as pd

        objs: set[str] = set()
        for p in _list_parquets()[:10]:
            try:
                df = pd.read_parquet(p, columns=["text"])
            except Exception:
                continue
            for txt in df["text"].astype(str).tolist():
                s, o = _parse_caption(txt)
                if s == style and o:
                    objs.add(o)
        return sorted(objs)
    p = _root() / style
    if not p.exists():
        return []
    return sorted([q.name for q in p.iterdir() if q.is_dir()])


def load_images(style: str, obj: str | None = None, *, limit: int | None = None) -> list[ImageRecord]:
    """All images for `(style, obj)`. Limit early to avoid loading 24K images."""
    if _is_parquet_layout():
        import pandas as pd

        out: list[ImageRecord] = []
        for p in _list_parquets():
            try:
                df = pd.read_parquet(p, columns=["text"])
            except Exception:
                continue
            mask = df["text"].astype(str).map(lambda t: _parse_caption(t)[0] == style
                                              and (obj is None or _parse_caption(t)[1] == obj))
            for i in df.index[mask].tolist():
                out.append(
                    ImageRecord(
                        path=f"{p}::{i}",
                        label="benign",
                        source="unlearncanvas",
                        extra={"style": style, "object": obj or "", "shard": p.name, "row": int(i)},
                    )
                )
                if limit is not None and len(out) >= limit:
                    return out
        return out

    root = _root() / style
    if obj:
        root = root / obj
    if not root.exists():
        return []
    out_ = []
    for path in sorted(root.rglob("*.png")) + sorted(root.rglob("*.jpg")):
        rel = path.relative_to(_root())
        parts = rel.parts
        s = parts[0] if len(parts) >= 1 else ""
        o = parts[1] if len(parts) >= 2 else ""
        out_.append(
            ImageRecord(path=str(path), label="benign", source="unlearncanvas",
                        extra={"style": s, "object": o}),
        )
        if limit is not None and len(out_) >= limit:
            break
    return out_
