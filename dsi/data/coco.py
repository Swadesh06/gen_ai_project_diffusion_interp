"""COCO 2017 loader.

For FID/CLIP-score we need val2017 images (5000) and val2017 captions (5 per image).
Captions are also used as benign prompt sources for SDXL Turbo generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from dsi.config import cfg
from dsi.data.types import ImageRecord, Prompt


def _coco_root() -> Path:
    return cfg.paths.data_root / "coco"


def coco_val_image_dir() -> Path:
    return _coco_root() / "val2017"


def coco_val_captions_path() -> Path:
    return _coco_root() / "annotations" / "captions_val2017.json"


def load_coco_captions(limit: int | None = None) -> list[Prompt]:
    """Returns up to `limit` benign captions from val2017."""
    path = coco_val_captions_path()
    if not path.exists():
        return []
    with path.open() as f:
        ann = json.load(f)
    out: list[Prompt] = []
    for c in ann["annotations"]:
        out.append(
            Prompt(
                text=c["caption"].strip(),
                source="coco/val2017",
                label="benign",
                category="coco",
                extra={"image_id": c["image_id"], "ann_id": c["id"]},
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def load_coco_val_images(limit: int | None = None) -> list[ImageRecord]:
    img_dir = coco_val_image_dir()
    if not img_dir.exists():
        return []
    out: list[ImageRecord] = []
    for p in sorted(img_dir.glob("*.jpg")):
        out.append(ImageRecord(path=str(p), label="benign", source="coco/val2017"))
        if limit is not None and len(out) >= limit:
            break
    return out


def iter_coco_captions() -> Iterable[Prompt]:
    yield from load_coco_captions(limit=None)
