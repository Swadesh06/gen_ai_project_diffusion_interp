"""UnlearnCanvas loader: 60 styles x 20 objects = 1200 image groups; concept-erasure benchmark."""

from __future__ import annotations

from pathlib import Path

from dsi.config import cfg
from dsi.data.types import ImageRecord


def _root() -> Path:
    return cfg.paths.data_root / "unlearncanvas"


def list_styles() -> list[str]:
    root = _root()
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_objects(style: str) -> list[str]:
    p = _root() / style
    if not p.exists():
        return []
    return sorted([q.name for q in p.iterdir() if q.is_dir()])


def load_images(style: str, obj: str | None = None) -> list[ImageRecord]:
    """All images under `style/` (and optionally narrowed to a single `object`)."""
    root = _root() / style
    if obj:
        root = root / obj
    if not root.exists():
        return []
    out: list[ImageRecord] = []
    for path in sorted(root.rglob("*.png")) + sorted(root.rglob("*.jpg")):
        rel = path.relative_to(_root())
        parts = rel.parts
        s = parts[0] if len(parts) >= 1 else ""
        o = parts[1] if len(parts) >= 2 else ""
        out.append(
            ImageRecord(
                path=str(path),
                label="benign",
                source="unlearncanvas",
                extra={"style": s, "object": o},
            )
        )
    return out
