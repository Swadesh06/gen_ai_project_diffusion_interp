#!/usr/bin/env python
"""Cross-space SAE-feature overlap analysis (Item 2.5 / Contribution 1 pass criterion).

Reads `attribution.json` from N attack runs (one per attack space), takes the
top-k features per hookpoint per space, and reports Jaccard overlap matrices.

Pass criterion (per task_description_v1.md §3 Contribution 1):
  ≥ 60% feature-set overlap between pixel and CLIP-embedding bypasses on the same prompt.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attack-dirs", nargs="+", required=True,
                    help="paths to outputs/<exp_id> dirs that have attribution.json")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out", default="reports/cross_space_overlap.md")
    args = ap.parse_args()

    parts = {}
    for d in args.attack_dirs:
        path = Path(d) / "attribution.json"
        if not path.exists():
            print(f"skip {d}: no attribution.json")
            continue
        info = json.loads(path.read_text())
        parts[Path(d).name] = info["blocks"]

    if len(parts) < 2:
        print(f"need ≥ 2 attack dirs with attribution; got {len(parts)}")
        return 1

    rows = ["# Cross-space SAE feature overlap", "",
            f"Top-k features per block per attack run (k={args.top_k}); Jaccard between every pair.",
            ""]
    overlap = {}
    for hp in sorted(set().union(*[set(b.keys()) for b in parts.values()])):
        rows.append(f"## hookpoint `{hp}`")
        rows.append("")
        names = sorted(parts.keys())
        # Header
        rows.append("| run \\ run | " + " | ".join(names) + " |")
        rows.append("|" + "|".join(["---"] * (len(names) + 1)) + "|")
        for n1 in names:
            cells = [n1]
            for n2 in names:
                a = parts[n1].get(hp, {}).get("top_indices", [])[: args.top_k]
                b = parts[n2].get(hp, {}).get("top_indices", [])[: args.top_k]
                ov = jaccard(a, b)
                cells.append(f"{ov:.3f}")
                overlap.setdefault(hp, {}).setdefault(n1, {})[n2] = ov
            rows.append("| " + " | ".join(cells) + " |")
        rows.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows))
    print("\n".join(rows[: 30]))
    print(f"\nwrote {out_path}")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(overlap, indent=2))
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
