#!/usr/bin/env python
"""CPU worker — always-on watcher for un-evaluated bypass images.

Scans `cfg.paths.output_root/**/*.png` and `*.jpg` recursively. For each image
without a sidecar `<image>.labels.json`, runs NudeNet + Q16 + SD safety-checker
and writes the JSON. Idempotent. Sleeps `--interval` between scans.

Per CLAUDE.md §4 — keeps the CPU busy whenever GPU is busy.
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

LABEL_SUFFIX = ".labels.json"


def list_unlabelled(root: Path, shard: tuple[int, int] | None = None) -> list[Path]:
    """List unlabelled images. If `shard=(i, n)`, only return paths whose hash mod n == i."""
    import hashlib
    out = []
    for p in sorted(root.rglob("*.png")):
        if (p.with_suffix(p.suffix + LABEL_SUFFIX)).exists():
            continue
        if shard is not None:
            i, n = shard
            h = int(hashlib.md5(str(p).encode()).hexdigest(), 16) % n
            if h != i:
                continue
        out.append(p)
    for p in sorted(root.rglob("*.jpg")):
        if (p.with_suffix(p.suffix + LABEL_SUFFIX)).exists():
            continue
        if shard is not None:
            i, n = shard
            h = int(hashlib.md5(str(p).encode()).hexdigest(), 16) % n
            if h != i:
                continue
        out.append(p)
    return out


def label_image(path: Path, scorers: dict) -> dict:
    rec = {"path": str(path), "ts": time.time()}
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
    except Exception as e:
        rec["error_load"] = repr(e)
        return rec

    if "nudenet" in scorers:
        try:
            rec["nudenet"] = scorers["nudenet"].score_path(str(path))
        except Exception as e:
            rec["nudenet_err"] = repr(e)
    if "q16" in scorers:
        try:
            rec["q16"] = scorers["q16"].score_image(img)
        except Exception as e:
            rec["q16_err"] = repr(e)
    if "safety" in scorers:
        try:
            rec["safety_checker"] = scorers["safety"].score([img])[0]
        except Exception as e:
            rec["safety_err"] = repr(e)
    rec["flagged_any"] = any(
        rec.get(k, {}).get("flagged", False) for k in ("nudenet", "q16", "safety_checker")
    )
    return rec


def load_scorers(use_safety: bool) -> dict:
    out = {}
    print("loading NudeNet ...")
    from dsi.detectors.baselines.nudenet import NudeNetWrapper

    out["nudenet"] = NudeNetWrapper().load()
    print("loading Q16 ...")
    from dsi.detectors.baselines.q16 import Q16Wrapper

    out["q16"] = Q16Wrapper(device="cpu").load()
    if use_safety:
        print("loading SD safety checker ...")
        from dsi.detectors.baselines.safety_checker import SafetyCheckerWrapper

        out["safety"] = SafetyCheckerWrapper(device="cpu").load()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(cfg.paths.output_root))
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--no-safety-checker", action="store_true",
                    help="skip the diffusion safety_checker (heavier; needs GPU for speed)")
    ap.add_argument("--once", action="store_true", help="single pass, exit")
    ap.add_argument("--shard", default=None, help="e.g. 3/8 — only label paths whose md5 mod 8 == 3")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    shard = None
    if args.shard:
        i, n = args.shard.split("/")
        shard = (int(i), int(n))
        print(f"shard {shard[0]}/{shard[1]}")
    scorers = load_scorers(use_safety=not args.no_safety_checker)
    print(f"watching {root}")

    while True:
        try:
            todo = list_unlabelled(root, shard=shard)
        except Exception as e:
            print(f"scan error: {e}", flush=True)
            todo = []
        for p in todo:
            try:
                if not p.exists():
                    continue                       # disappeared between scan and label
                t0 = time.time()
                rec = label_image(p, scorers)
                sidecar = p.with_suffix(p.suffix + LABEL_SUFFIX)
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(json.dumps(rec, indent=2))
                if rec.get("flagged_any") or (time.time() - t0) > 1.0:
                    print(f"  {p.name}: flagged={rec.get('flagged_any')} ({time.time()-t0:.2f}s)", flush=True)
            except Exception as e:
                print(f"  {p.name}: label error {type(e).__name__}: {e}", flush=True)
                continue
        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
