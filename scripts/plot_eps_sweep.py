"""Plot ASR vs ε for A01 pixel-PGD and A02 latent-PGD."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def wilson_ci(k, n, z=1.96):
    if n == 0: return 0, 1
    p = k / n
    center = (k + z*z/2) / (n + z*z)
    margin = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / (1 + z*z/n)
    return max(0, center - margin), min(1, center + margin)


def main():
    a01 = [
        ("ε=1/255", 1/255, 11, 11),
        ("ε=2/255", 2/255, 9, 9),
        ("ε=4/255", 4/255, 88, 88),
    ]
    a02 = [
        ("ε=0.025", 0.025, 9, 9),
        ("ε=0.05", 0.05, 11, 11),
        ("ε=0.1", 0.1, 100, 100),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, data, title in [(axes[0], a01, "A01 pixel-PGD"), (axes[1], a02, "A02 latent-PGD")]:
        eps = [e for _, e, _, _ in data]
        rates = [k/n for _, _, k, n in data]
        cis = [wilson_ci(k, n) for _, _, k, n in data]
        err_lo = [max(0.0, r - lo) for r, (lo, _) in zip(rates, cis)]
        err_hi = [max(0.0, hi - r) for r, (_, hi) in zip(rates, cis)]
        ax.errorbar(eps, rates, yerr=[err_lo, err_hi], marker="o", linestyle="-",
                    color="#4287f5", capsize=6, ecolor="black")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("ε (perturbation budget)")
        ax.set_ylabel("ASR (Wilson 95% CI)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        for i, ((label, _, k, n), r) in enumerate(zip(data, rates)):
            ax.text(eps[i], r + 0.04, f"{k}/{n}", ha="center", fontsize=8)
    plt.suptitle("Attack ASR vs ε on safety_checker (saturation persists at quarter ε)", fontsize=12)
    plt.tight_layout()
    out = Path("outputs/figures/eps_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
