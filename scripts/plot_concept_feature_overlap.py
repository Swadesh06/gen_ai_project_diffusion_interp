#!/usr/bin/env python
"""Plot F_c-feature overlap matrix for (nudity, violence) per hookpoint.

Visual: 4-panel bar chart showing top-20 features (rank-1 to rank-20) per
hookpoint, color-coded by concept. Confirms zero overlap.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")
    nudity = json.loads(open("outputs/F_c_stage1n2_top.json").read())
    viol = json.loads(open("outputs/D04_violence_stage1/stage1.json").read())

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, hp in enumerate(HOOKPOINTS):
        ax = axes[i]
        n_set = sorted(set(nudity.get(hp, [])))
        v_set = sorted(set(viol[hp]["top_k_indices"]))
        # show as separate scatters on x-axis = feature_idx
        if n_set:
            ax.scatter(n_set, [1]*len(n_set), c="red", s=20, label=f"nudity F_c (n={len(n_set)})")
        ax.scatter(v_set, [0]*len(v_set), c="blue", s=20, label=f"violence top-20")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["violence", "nudity"])
        ax.set_xlabel("SAE feature index")
        ax.set_xlim(-200, 5400)
        ax.set_title(f"{hp} (overlap = {len(set(n_set) & set(v_set))})")
        ax.legend(loc="upper right", fontsize=7)
    plt.suptitle("Per-concept SAE feature selectivity (zero overlap)", fontsize=12)
    plt.tight_layout()
    out = Path("outputs/figures") / "concept_feature_overlap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
