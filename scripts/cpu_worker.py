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


def list_unlabelled(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*.png")):
        if not (p.with_suffix(p.suffix + LABEL_SUFFIX)).exists():
            out.append(p)
    for p in sorted(root.rglob("*.jpg")):
        if not (p.with_suffix(p.suffix + LABEL_SUFFIX)).exists():
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
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    scorers = load_scorers(use_safety=not args.no_safety_checker)
    print(f"watching {root}")

    while True:
        try:
            todo = list_unlabelled(root)
        except Exception as e:
            print(f"scan error: {e}")
            todo = []
        for p in todo:
            t0 = time.time()
            rec = label_image(p, scorers)
            sidecar = p.with_suffix(p.suffix + LABEL_SUFFIX)
            sidecar.write_text(json.dumps(rec, indent=2))
            print(f"  {p.name}: flagged={rec.get('flagged_any')} ({time.time()-t0:.1f}s)")
        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
