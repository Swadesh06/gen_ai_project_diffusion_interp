#!/usr/bin/env python
"""Phase D-1 — causal feature graphs across the denoising trajectory.

A static feature attribution map (Phase 1 contribution 1) tells us "feature 879
fires on bypass". The story we want for the paper is the *trajectory* — which
features at hookpoint h' at step t-1 causally drive activations at hookpoint h
at step t.

Without per-step trajectories cached at scale, we approximate the static
cross-hookpoint causal graph using attribution-by-correlation (Marks et al.
2024 "Sparse Feature Circuits"): treat the four Surkov hookpoints as a
pseudo-sequence (down.2.1 -> mid.0 -> up.0.0 -> up.0.1) and compute the
linear regression coefficient `Beta_{f,g}` predicting feature f at hookpoint
h+1 from feature g at hookpoint h, evaluated on the cached unsafe-condition
SAE z's. Edges with `|Beta| > tau` and `g, f` both Stage-2 survivors define
the causal subgraph.

Output: outputs/D01_causal_graph/ with per-edge tables, top-feature lists,
and a Sankey-style PDF.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")
HOOKPOINT_PAIRS = (("down.2.1", "mid.0"), ("mid.0", "up.0.0"), ("up.0.0", "up.0.1"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="outputs/dataset_axbench_v1")
    ap.add_argument("--out-dir", default="outputs/D01_causal_graph")
    ap.add_argument("--top-k-per-hp", type=int, default=20,
                    help="number of top features per hookpoint to include in the graph")
    ap.add_argument("--tau", type=float, default=0.10,
                    help="absolute regression coefficient threshold for an edge")
    ap.add_argument("--feature-source", choices=["sae", "raw"], default="sae")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import numpy as np
    import torch
    from sklearn.linear_model import LinearRegression

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    y = np.load(data_dir / "y.npy")
    X_per_hp = {}
    for hp in HOOKPOINTS:
        hp_safe = hp.replace(".", "_")
        if args.feature_source == "sae":
            X_per_hp[hp] = np.load(data_dir / f"X_sae_{hp_safe}.npy").astype("float32")
        else:
            X_per_hp[hp] = np.load(data_dir / f"X_raw_{hp_safe}.npy").astype("float32")
    print(f"y: {(y == 1).sum()} pos / {(y == 0).sum()} neg")
    for hp, X in X_per_hp.items():
        print(f"  {hp}: {X.shape}")

    unsafe_mask = (y == 1)
    benign_mask = (y == 0)

    # 1. Per-hookpoint top-K features by Stage-1 Fisher ratio
    #    s_forget(f) = E_unsafe[z_f^2], s_retain(f) = E_benign[z_f^2]
    top_per_hp: dict = {}
    for hp in HOOKPOINTS:
        Xf = X_per_hp[hp]
        s_for = (Xf[unsafe_mask] ** 2).mean(axis=0)
        s_ret = (Xf[benign_mask] ** 2).mean(axis=0).clip(min=1e-12)
        ratio = s_for / s_ret
        top_idx = np.argsort(-ratio)[: args.top_k_per_hp]
        top_per_hp[hp] = top_idx.tolist()
        print(f"  {hp} top-{args.top_k_per_hp} fisher-ratio features: {top_idx[:5].tolist()}...")

    # 2. Per (h, h+1) hookpoint pair: linear regression of TOP features at h+1
    #    on TOP features at h, using unsafe samples.
    edges: list[dict] = []
    for src_hp, dst_hp in HOOKPOINT_PAIRS:
        src_idx = top_per_hp[src_hp]
        dst_idx = top_per_hp[dst_hp]
        Xs = X_per_hp[src_hp][unsafe_mask][:, src_idx]
        Xd = X_per_hp[dst_hp][unsafe_mask][:, dst_idx]
        # standardize
        Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
        Xd = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-8)
        for j, fdst in enumerate(dst_idx):
            reg = LinearRegression().fit(Xs, Xd[:, j])
            for i, gsrc in enumerate(src_idx):
                beta = float(reg.coef_[i])
                if abs(beta) >= args.tau:
                    edges.append({
                        "src_hp": src_hp, "src_feature": int(gsrc),
                        "dst_hp": dst_hp, "dst_feature": int(fdst),
                        "beta": beta,
                    })
    edges.sort(key=lambda e: -abs(e["beta"]))
    print(f"  {len(edges)} edges (|beta| >= {args.tau})")

    # 3. Identify root features (no incoming edge, has outgoing) and sinks (no outgoing).
    sources = {(e["src_hp"], e["src_feature"]) for e in edges}
    targets = {(e["dst_hp"], e["dst_feature"]) for e in edges}
    roots = sources - targets
    sinks = targets - sources
    print(f"  {len(roots)} root features, {len(sinks)} sink features")

    # 4. Save
    out = {
        "hookpoint_order": list(HOOKPOINTS),
        "top_per_hp": top_per_hp,
        "edges": edges,
        "n_edges": len(edges),
        "tau": args.tau,
        "top_k_per_hp": args.top_k_per_hp,
        "feature_source": args.feature_source,
        "roots": [{"hp": hp, "feature": f} for hp, f in roots],
        "sinks": [{"hp": hp, "feature": f} for hp, f in sinks],
    }
    (out_dir / "graph.json").write_text(json.dumps(out, indent=2, default=int))

    # 5. Render Sankey-style plot via matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 8))
        # Layout: 4 columns for the 4 hookpoints, vertically stack top-K features
        col_x = {hp: i for i, hp in enumerate(HOOKPOINTS)}
        rows_per_hp = args.top_k_per_hp
        for hp, feats in top_per_hp.items():
            for j, f in enumerate(feats):
                x = col_x[hp]
                y = rows_per_hp - j
                color = "tab:red" if (hp, int(f)) in roots else (
                    "tab:green" if (hp, int(f)) in sinks else "tab:blue")
                ax.scatter(x, y, s=60, color=color, zorder=3)
                ax.text(x, y + 0.15, f"f{f}", fontsize=6, ha="center", va="bottom",
                        color="black", zorder=4)
        for e in edges:
            x0 = col_x[e["src_hp"]]
            x1 = col_x[e["dst_hp"]]
            y0 = rows_per_hp - top_per_hp[e["src_hp"]].index(e["src_feature"])
            y1 = rows_per_hp - top_per_hp[e["dst_hp"]].index(e["dst_feature"])
            color = "tab:red" if e["beta"] > 0 else "tab:purple"
            ax.plot([x0, x1], [y0, y1], color=color, alpha=min(1.0, abs(e["beta"])),
                    linewidth=0.5, zorder=2)
        ax.set_xticks(list(col_x.values()))
        ax.set_xticklabels(list(col_x.keys()))
        ax.set_yticks([])
        ax.set_title(f"Causal feature graph ({args.feature_source} features, top-{args.top_k_per_hp}, tau={args.tau})")
        fig.tight_layout()
        fig.savefig(out_dir / "graph.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "graph.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"plot failed: {e}")

    print(f"\nwrote {out_dir}/graph.json, {out_dir}/graph.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
