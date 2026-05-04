"""Image-saving discipline (CLAUDE.md §7 / Item 1c-2 binding).

Every attack / intervention / detector experiment going forward saves:
  - every bypass case as PNG
  - every corrected case (interventions)
  - every false-positive case (detectors)
  - 50 random non-bypass / non-corrected as negative-control sample
  - perturbation viz `<seed>.perturb.png` = (post - pre) * 10 clipped [0, 255]
  - per-feature activation heatmap `<seed>.heatmap.png` (intervention only)

Plus a 4×4 `figure.png` per exp_id of the most informative cases.

Usage:
    from dsi.util.img_saving import CaseRecorder
    rec = CaseRecorder(out_dir, exp_id="A01_pixel_eps4_n200")
    rec.save_attack_case(seed, kind="bypass", pre=pre_img, post=post_img,
                         meta={"prompt": prompt, "perturb_norm": 0.13})
    ...
    rec.save_figure_grid()  # 4x4 of best cases by score
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
from PIL import Image

CaseKind = Literal["bypass", "corrected", "false_positive", "true_negative", "true_positive", "negative_control"]


def _to_pil(x):
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, np.ndarray):
        a = x
        if a.dtype != np.uint8:
            a = (a.clip(0, 1) * 255).astype("uint8") if a.max() <= 1.0 else a.clip(0, 255).astype("uint8")
        if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[0] != a.shape[-1]:
            a = a.transpose(1, 2, 0)
        if a.ndim == 3 and a.shape[2] == 1:
            a = a[..., 0]
        return Image.fromarray(a)
    raise TypeError(f"_to_pil unsupported type {type(x)}")


def _perturb_viz(pre, post, *, multiplier: float = 10.0) -> Image.Image:
    """`(post - pre) * multiplier` clipped to [0, 255], centered at 128."""
    p = np.asarray(_to_pil(pre).convert("RGB"), dtype="float32")
    q = np.asarray(_to_pil(post).convert("RGB"), dtype="float32")
    if p.shape != q.shape:
        q = np.asarray(_to_pil(post).convert("RGB").resize(_to_pil(pre).size), dtype="float32")
    diff = (q - p) * multiplier + 128.0
    return Image.fromarray(diff.clip(0, 255).astype("uint8"))


def _heatmap_2d(arr2d: np.ndarray, size: tuple[int, int] = (256, 256)) -> Image.Image:
    """`arr2d` (H, W) → grayscale PIL. Min-max normalised to [0, 255]."""
    a = arr2d.astype("float32")
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        a = np.zeros_like(a)
    else:
        a = (a - lo) / (hi - lo)
    img = Image.fromarray((a * 255).astype("uint8"), mode="L")
    return img.resize(size, resample=Image.NEAREST)


class CaseRecorder:
    """Per-experiment image-case recorder.

    Lays out:
        <out_dir>/cases/<kind>/<seed>.png          # pre image
        <out_dir>/cases/<kind>/<seed>.post.png     # post image (if differs)
        <out_dir>/cases/<kind>/<seed>.perturb.png  # 10x diff
        <out_dir>/cases/<kind>/<seed>.heatmap.png  # feature activation heatmap (if provided)
        <out_dir>/cases/<kind>/<seed>.meta.json    # prompt + score + tags
    """

    def __init__(self, out_dir: Path | str, exp_id: str,
                 negative_control_n: int = 50, seed: int = 0):
        self.out_dir = Path(out_dir)
        self.exp_id = exp_id
        self.cases_dir = self.out_dir / "cases"
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        self._neg_pool: list[tuple] = []
        self._neg_n = negative_control_n
        self._rng = random.Random(seed)

    def save_attack_case(self, seed: int | str, *, kind: CaseKind,
                         pre, post=None, perturb=None, heatmap=None,
                         meta: Optional[dict] = None) -> None:
        stem = f"{int(seed):08d}" if isinstance(seed, int) else str(seed)
        kind_dir = self.cases_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        if pre is not None:
            _to_pil(pre).save(kind_dir / f"{stem}.png")
        if post is not None:
            _to_pil(post).save(kind_dir / f"{stem}.post.png")
        if perturb is not None:
            _to_pil(perturb).save(kind_dir / f"{stem}.perturb.png")
        elif pre is not None and post is not None:
            try:
                _perturb_viz(pre, post).save(kind_dir / f"{stem}.perturb.png")
            except Exception:
                pass
        if heatmap is not None:
            if isinstance(heatmap, np.ndarray) and heatmap.ndim == 2:
                _heatmap_2d(heatmap).save(kind_dir / f"{stem}.heatmap.png")
            elif isinstance(heatmap, Image.Image):
                heatmap.save(kind_dir / f"{stem}.heatmap.png")
        if meta is not None:
            (kind_dir / f"{stem}.meta.json").write_text(json.dumps(meta, default=str))
        self.records.append({"seed": str(seed), "kind": kind, "meta": meta or {}})

    def offer_negative_control(self, seed, pre, *, meta: Optional[dict] = None) -> None:
        """Reservoir-sample non-event cases. Call on every benign / no-flag image;
        keeps `negative_control_n` random samples saved to disk.
        """
        if len(self._neg_pool) < self._neg_n:
            self._neg_pool.append((seed, pre, meta))
        else:
            j = self._rng.randint(0, len(self._neg_pool))
            if j < self._neg_n:
                self._neg_pool[j] = (seed, pre, meta)

    def flush_negative_control(self) -> None:
        for seed, pre, meta in self._neg_pool:
            self.save_attack_case(seed, kind="negative_control", pre=pre, meta=meta)

    def save_figure_grid(self, *, n: int = 16, kind_priority: Iterable[CaseKind] = (
            "bypass", "corrected", "false_positive", "negative_control"),
                         out_name: str = "figure.png") -> Path:
        """4x4 grid of the most informative cases. Picks first N from priority kinds in order."""
        chosen: list[Path] = []
        for kind in kind_priority:
            kind_dir = self.cases_dir / kind
            if not kind_dir.exists():
                continue
            for p in sorted(kind_dir.glob("*.png")):
                if p.name.endswith(".post.png") or p.name.endswith(".perturb.png") or p.name.endswith(".heatmap.png"):
                    continue
                chosen.append(p)
                if len(chosen) >= n:
                    break
            if len(chosen) >= n:
                break

        side = int(np.ceil(np.sqrt(max(1, len(chosen)))))
        side = min(side, 4)
        rows = (len(chosen) + side - 1) // side if chosen else 1
        rows = max(1, min(rows, 4))
        cell_size = 256
        grid = Image.new("RGB", (side * cell_size, rows * cell_size), color=(20, 20, 20))
        for i, p in enumerate(chosen[: side * rows]):
            r, c = i // side, i % side
            try:
                im = Image.open(p).convert("RGB").resize((cell_size, cell_size))
                grid.paste(im, (c * cell_size, r * cell_size))
            except Exception:
                pass
        out = self.out_dir / out_name
        grid.save(out)
        return out

    def write_index(self) -> Path:
        out = self.out_dir / "cases.index.json"
        out.write_text(json.dumps({"exp_id": self.exp_id, "n": len(self.records),
                                   "records": self.records}, indent=2))
        return out
