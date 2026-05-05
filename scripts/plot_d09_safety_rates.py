"""Bar chart: safety_checker flag rate per architecture with Wilson CI."""
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
    data = [
        ("PixArt-Sigma\n1024 (DiT)", 2, 100),
        ("SD3-medium\n(MM-DiT)", 4, 100),
        ("SDXL Turbo\n(UNet, 1-step)", 17, 200),
        ("SD v1.4\n(UNet, 30-step)", 10, 30),
        ("SDXL Base\n(UNet, 4-step)", 286, 1000),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    rates = [k/n for _, k, n in data]
    cis = [wilson_ci(k, n) for _, k, n in data]
    err_lo = [max(0.0, r - lo) for r, (lo, _) in zip(rates, cis)]
    err_hi = [max(0.0, hi - r) for r, (_, hi) in zip(rates, cis)]
    x = np.arange(len(data))
    ax.bar(x, rates, yerr=[err_lo, err_hi], capsize=6,
           color=["#aa5042", "#aa6042", "#42aa50", "#42aa70", "#5042aa"], ecolor="black")
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], fontsize=9)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("safety_checker flag rate")
    ax.set_title("Cross-architecture safety_checker baseline (I2P-NSFW prompts)")
    for i, (label, k, n) in enumerate(data):
        ax.text(i, rates[i] + max(0.005, 0.01 * (1 + i * 0.5)), f"{k}/{n}", ha="center", fontsize=9)
    plt.tight_layout()
    out = Path("outputs/figures/d09_cross_arch_safety_rates.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
