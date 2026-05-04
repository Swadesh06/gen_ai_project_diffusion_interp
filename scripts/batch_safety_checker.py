#!/usr/bin/env python
"""Batched CompVis safety-checker labelling for any directory of PNGs.

GPU-resident (~3 GB), so it runs alongside any large GPU job. Writes
`<image>.png.safety.json` sidecars with `{flagged, score}` per image.
Skips images that already have a sidecar.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import cfg  # noqa: E402

SUFFIX = ".safety.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    args = ap.parse_args()

    import torch
    from PIL import Image

    from dsi.detectors.baselines.safety_checker import SafetyCheckerWrapper

    sc = SafetyCheckerWrapper(model_id=cfg.model.safety_checker, device=args.device).load()
    if args.device == "cuda" and args.dtype == "fp16":
        sc.model.half()

    root = Path(args.root)
    paths = sorted(root.rglob("*.png"))
    todo = [p for p in paths if not p.with_suffix(p.suffix + SUFFIX).exists()]
    print(f"{root}: {len(paths)} PNGs, {len(todo)} unscored", flush=True)

    t0 = time.time()
    n_flagged = 0
    for i in range(0, len(todo), args.batch_size):
        batch_paths = todo[i : i + args.batch_size]
        imgs = []
        valid = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                valid.append(p)
            except Exception:
                continue
        if not imgs:
            continue
        with torch.no_grad():
            results = sc.score(imgs)
        for p, r in zip(valid, results):
            r["score"] = float(r["score"])
            p.with_suffix(p.suffix + SUFFIX).write_text(json.dumps(r))
            if r["flagged"]:
                n_flagged += 1
        if (i // args.batch_size) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+len(batch_paths)}/{len(todo)}] {elapsed:.1f}s flagged_so_far={n_flagged}",
                  flush=True)
    elapsed = time.time() - t0
    print(f"DONE root={root} n={len(todo)} flagged={n_flagged} elapsed={elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
