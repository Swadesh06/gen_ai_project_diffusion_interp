#!/usr/bin/env python
"""Build the canonical closure prompt list — 500 I2P-NSFW prompts cached for v3 matched-budget grid.

Same prompts used across every Gate 3 cell so they are paired across (target, attack, budget).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="outputs/closure_prompt_list_v1.json")
    args = ap.parse_args()

    from dsi.data.i2p import i2p_nsfw_subset
    prompts = i2p_nsfw_subset(limit=None)
    print(f"loaded {len(prompts)} I2P-NSFW prompts")
    sub = prompts[: args.n]
    out = {
        "n": len(sub),
        "source": "i2p_nsfw_subset (full split, deterministic order)",
        "prompts": [{"idx": i, "prompt": p.text, "category": p.category, "seed": int(p.seed or i)} for i, p in enumerate(sub)],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} ({len(sub)} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
