#!/usr/bin/env python
"""Score D09_sd3_i2p_n100 outputs with NudeNet + Q16 oracles.

Already has safety_checker scores in .safety.json sidecars (4/100 flagged).
This adds oracle-ensemble cross-classifier rates for SD3 outputs.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="outputs/D09_sd3_i2p_n100/post")
    ap.add_argument("--out-dir", default="outputs/D09_sd3_i2p_n100_oracle")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== score_sd3_oracles (src={args.src_dir}) ===")

    from PIL import Image
    from dsi.detectors.baselines.nudenet import NudeNetWrapper
    from dsi.detectors.baselines.q16 import Q16Wrapper
    from dsi.detectors.baselines.safety_checker import SafetyCheckerWrapper

    nn = NudeNetWrapper().load()
    q16 = Q16Wrapper().load()
    sc = SafetyCheckerWrapper(device="cuda").load()

    files = sorted(Path(args.src_dir).glob("*.png"))
    print(f"  {len(files)} images")

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
                "q16_flag": bool(r_q16["flagged"]),
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
    print(f"  safety_checker flagged: {n_sc} = {n_sc/n:.3f}")
    print(f"  nudenet flagged:        {n_nn} = {n_nn/n:.3f}")
    print(f"  Q16 flagged:            {n_q16} = {n_q16/n:.3f}")
    print(f"  any oracle flagged:     {n_any} = {n_any/n:.3f}")

    summary = {
        "src_dir": args.src_dir,
        "n": n,
        "safety_checker_flag_rate": n_sc / n,
        "nudenet_flag_rate": n_nn / n,
        "q16_flag_rate": n_q16 / n,
        "any_oracle_flag_rate": n_any / n,
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
