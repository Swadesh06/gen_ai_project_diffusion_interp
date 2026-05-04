#!/usr/bin/env python
"""Phase C analysis — internal structure of F_c (the 69 Stage1∩Stage2 features).

For the existing F_c bundle, compute on the dataset_axbench_v1 raw activations:
  1. Per-feature mean activation in NSFW vs benign.
  2. Pairwise correlation matrix of F_c feature activations.
  3. Sparsity statistics.
  4. Effective rank of the F_c subspace (how much actual variance is in the
     F_c subspace, not the full 5120-d hidden).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="outputs/dataset_axbench_v1")
    ap.add_argument("--fc-file", default="outputs/F_c_stage1n2_top.json")
    ap.add_argument("--out-dir", default="outputs/F_c_analysis")
    args = ap.parse_args()

    import numpy as np

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    F_c = json.loads(Path(args.fc_file).read_text())
    print(f"F_c bundle: {sum(len(v) for v in F_c.values())} features across {len(F_c)} hookpoints")
    for hp, idx in F_c.items():
        print(f"  {hp}: |F_c|={len(idx)} indices")

    data_dir = Path(args.data_dir)
    y = np.load(data_dir / "y.npy")
    benign = (y == 0); nsfw = (y == 1)

    z_fc_concat = []
    per_hp_stats = {}
    for hp, idx in F_c.items():
        if not idx: continue
        hp_us = hp.replace(".", "_")
        Xsae = np.load(data_dir / f"X_sae_{hp_us}.npy")  # (N, 5120)
        Xsae_fc = Xsae[:, np.asarray(idx, dtype=int)]    # (N, |F_c|)
        # mean activation per class
        mu_b = Xsae_fc[benign].mean(axis=0)
        mu_n = Xsae_fc[nsfw].mean(axis=0)
        sep = (mu_n - mu_b)
        # sparsity: fraction of zeros in F_c slice
        sparsity = float((Xsae_fc == 0).mean())
        # rank
        U, S, _ = np.linalg.svd(Xsae_fc - Xsae_fc.mean(axis=0), full_matrices=False)
        eff_rank = float((S.sum() ** 2) / (S * S).sum())
        per_hp_stats[hp] = {
            "n_features": len(idx),
            "mean_benign": mu_b.tolist(),
            "mean_nsfw": mu_n.tolist(),
            "separation_per_feature": sep.tolist(),
            "sparsity": sparsity,
            "effective_rank": eff_rank,
            "singular_values_top_10": S[:10].tolist(),
        }
        z_fc_concat.append(Xsae_fc)

    # Pairwise correlation
    Z = np.concatenate(z_fc_concat, axis=1)  # (N, 69)
    print(f"\nF_c concat shape: {Z.shape}")
    Z_centered = Z - Z.mean(axis=0)
    Z_std = Z.std(axis=0) + 1e-8
    corr = (Z_centered.T @ Z_centered) / (Z.shape[0] * Z_std[:, None] * Z_std[None, :])
    # Off-diagonal stats
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    print(f"Pairwise corr off-diag: |r| mean={np.abs(off_diag).mean():.3f}, max={np.abs(off_diag).max():.3f}")
    # Effective rank of full F_c
    _, S_full, _ = np.linalg.svd(Z_centered, full_matrices=False)
    eff_rank_full = float((S_full.sum() ** 2) / (S_full * S_full).sum())
    print(f"Effective rank of F_c (69 features): {eff_rank_full:.2f}")

    out = {
        "F_c": {hp: list(map(int, idx)) for hp, idx in F_c.items()},
        "n_features_total": int(Z.shape[1]),
        "per_hp": per_hp_stats,
        "pairwise_corr_offdiag_abs_mean": float(np.abs(off_diag).mean()),
        "pairwise_corr_offdiag_abs_max": float(np.abs(off_diag).max()),
        "effective_rank_full": eff_rank_full,
        "singular_values_top_20": S_full[:20].tolist(),
    }
    (out_dir / "fc_structure.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {out_dir / 'fc_structure.json'}")

    # Save corr matrix as npy
    np.save(out_dir / "fc_pairwise_corr.npy", corr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
