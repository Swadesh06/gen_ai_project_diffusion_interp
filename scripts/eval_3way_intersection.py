#!/usr/bin/env python
"""3-way intersection ensemble test on B02-v3 Square BB bypass images.

Take 8 successful Square BB bypasses against B02-v3. Score with:
  - safety_checker (the original target)
  - B02-adv detector
Already done above for B02-adv (8/8 still flagged).
This script adds safety_checker scoring."""
from __future__ import annotations

import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    print("=== eval_3way_intersection ===")
    from PIL import Image
    from dsi.detectors.baselines.safety_checker import SafetyCheckerWrapper

    sc = SafetyCheckerWrapper(device="cuda").load()

    bypass_dir = REPO / "outputs/C01_square_vs_B02v3_bb_n200_q500/post"
    pre_dir = REPO / "outputs/C01_square_vs_B02v3_bb_n200_q500/pre"
    rows = []
    for png in sorted(bypass_dir.glob("*.png")):
        seed = int(png.stem)
        post_img = Image.open(png).convert("RGB")
        pre_png = pre_dir / png.name
        if not pre_png.exists():
            continue
        pre_img = Image.open(pre_png).convert("RGB")
        flag_pre = bool(sc.score([pre_img])[0]["flagged"])
        flag_post = bool(sc.score([post_img])[0]["flagged"])
        rows.append({"seed": seed, "SC_pre_flag": flag_pre, "SC_post_flag": flag_post})
        print(f"  seed={seed}: SC pre={'F' if flag_pre else 'S'}, post={'F' if flag_post else 'S'}")

    n_post_sc = sum(1 for r in rows if r["SC_post_flag"])
    n_pre_sc = sum(1 for r in rows if r["SC_pre_flag"])
    out = REPO / "outputs/C01_xtarget_3way_b02v3_bypasses.json"
    out.write_text(json.dumps({"rows": rows, "n_pre_sc_flag": n_pre_sc,
                                 "n_post_sc_flag": n_post_sc}, indent=2))
    print(f"\n  pre  SC flag: {n_pre_sc}/{len(rows)}")
    print(f"  post SC flag: {n_post_sc}/{len(rows)}")
    print(f"DONE -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
