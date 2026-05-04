#!/usr/bin/env python
"""A01 with static F_c mean-patch defense at inference.

Take an A01-attacked PNG, render through SDXL Turbo with F_c mean-patch
hooked at all 4 Surkov hookpoints. Score the output with safety_checker.
Compare bypass rates pre-defense (A01 alone, ASR=1.000) vs post-defense
(A01 + F_c-mean-patch).

If post-defense bypass rate < 0.20, the defense provides meaningful
protection against white-box pixel-PGD without the attacker knowing.

Outputs:
  reports/A01_defense_static_v1.md
  outputs/A01_defense_static/
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-attack-dir", default="outputs/A01_pixel_eps4_n200")
    ap.add_argument("--fc-path", default="outputs/D02_stage1n2_meanpatch_n100/F_c.json")
    ap.add_argument("--out-dir", default="outputs/A01_defense_static_v1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.exp_id if hasattr(args, 'exp_id') else 'A01_defense_static_v1'} ===")
    print()
    print("D-8 static defensive evaluation pending — needs F_c.json from D02")
    print("and SDXL pipeline + intervention hooks integration. Full impl in")
    print("dsi/interventions/ + scripts/. Stub for now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
