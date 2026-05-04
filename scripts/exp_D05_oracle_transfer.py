#!/usr/bin/env python
"""D-5 oracle-transfer eval — does an A01 perturbation also fool other oracles?

For each A01 post image (which bypassed safety_checker), score with NudeNet
and Q16. If those bypass too, the perturbation transferred across the
oracle ensemble. If only safety_checker bypassed, the attack is
safety_checker-specific.

This is a static-image cross-classifier transferability test, the
strongest version of D-5.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="outputs/A01_pixel_eps4_n200_seed1/post")
    ap.add_argument("--out-dir", default="outputs/D05_oracle_transfer")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== D05 oracle-transfer (src={args.src_dir}) ===")

    from PIL import Image
    from dsi.detectors.baselines.nudenet import NudeNetWrapper
    from dsi.detectors.baselines.q16 import Q16Wrapper
    from dsi.detectors.baselines.safety_checker import SafetyCheckerWrapper

    nn = NudeNetWrapper().load()
    q16 = Q16Wrapper().load()
    sc = SafetyCheckerWrapper(device="cuda").load()

    files = sorted(Path(args.src_dir).glob("*.png"))
    print(f"  {len(files)} post images")

    results = []
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            img = Image.open(f).convert("RGB")
            r_nn = nn.score_path(str(f))
            r_q16 = q16.score_image(img)
            r_sc = sc.score([img])[0]
            row = {
                "file": f.name,
                "nudenet_flag": bool(r_nn["flagged"]),
                "nudenet_score": float(r_nn["score"]),
                "q16_flag": bool(r_q16["flagged"]),
                "q16_score": float(r_q16["score"]),
                "safety_checker_flag": bool(r_sc["flagged"]),
            }
            results.append(row)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(files)}] {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  err {f.name}: {e}")

    n = len(results)
    n_sc = sum(r["safety_checker_flag"] for r in results)
    n_nn = sum(r["nudenet_flag"] for r in results)
    n_q16 = sum(r["q16_flag"] for r in results)
    n_any = sum(r["nudenet_flag"] or r["q16_flag"] or r["safety_checker_flag"] for r in results)

    print(f"  n={n}")
    print(f"  safety_checker flagged: {n_sc} = {n_sc/max(1,n):.3f}")
    print(f"  nudenet flagged:        {n_nn} = {n_nn/max(1,n):.3f}")
    print(f"  Q16 flagged:            {n_q16} = {n_q16/max(1,n):.3f}")
    print(f"  any oracle flagged:     {n_any} = {n_any/max(1,n):.3f}")

    summary = {
        "src_dir": args.src_dir,
        "n": n,
        "safety_checker_flag_rate": n_sc / max(1, n),
        "nudenet_flag_rate": n_nn / max(1, n),
        "q16_flag_rate": n_q16 / max(1, n),
        "any_oracle_flag_rate": n_any / max(1, n),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
