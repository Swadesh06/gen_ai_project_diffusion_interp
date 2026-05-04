#!/usr/bin/env python
"""Compute 5-seed CI from per-seed summary.json files."""
import json, sys, math
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("usage: summarize_5seed_ci.py <attack-prefix>")
        return 1
    prefix = sys.argv[1]  # e.g., "A01_pixel_eps4_n200"
    rows = []
    for s in range(5):
        # seed 0 might be in prefix only without _seedN
        for cand in [Path(f"outputs/{prefix}/summary.json")] if s == 0 else [Path(f"outputs/{prefix}_seed{s}/summary.json")]:
            if cand.exists():
                d = json.loads(cand.read_text())
                rows.append({"seed": s, "asr": d.get("asr_among_pre_flagged"),
                             "n_pre": d.get("n_pre_flagged"), "n_bypass": d.get("n_bypass")})
                break
    if not rows:
        print(f"no seeds found for {prefix}")
        return 1
    asrs = [r["asr"] for r in rows if r["asr"] is not None]
    if asrs:
        mean = sum(asrs) / len(asrs)
        std = math.sqrt(sum((x - mean) ** 2 for x in asrs) / max(1, len(asrs) - 1))
        ci_half = 1.96 * std / math.sqrt(len(asrs))
        print(f"Attack: {prefix}")
        print(f"  Seeds banked: {len(rows)}")
        for r in rows:
            print(f"    s{r['seed']}: ASR={r['asr']:.4f} ({r['n_bypass']}/{r['n_pre']})")
        print(f"  Mean ASR = {mean:.4f} ± {std:.4f} (1σ)")
        print(f"  95% CI [{mean-ci_half:.4f}, {mean+ci_half:.4f}]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
