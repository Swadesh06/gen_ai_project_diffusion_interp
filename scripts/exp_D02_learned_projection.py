#!/usr/bin/env python
"""Phase D-2 — learned-projection intervention.

For each Surkov hookpoint, train a single linear projection
`Pi: R^d -> R^d` (d = SAE d_hidden) that:
  - preserves benign-distribution SAE z (identity-on-benign).
  - projects unsafe-distribution SAE z toward the benign-mean feature
    set (collapse-to-benign).

Loss:
    L = || Pi(z_benign) - z_benign ||^2  +  lambda * || Pi(z_unsafe) - z_benign_mean ||^2

z values are mean-pooled per sample at each hookpoint (consistent with
the detector training inputs). Result is a per-hookpoint matrix
Pi_<hp> ∈ R^{d_hidden × d_hidden}.

At intervention time, for any z (unsafe-flagged) the patched z_patched
= Pi_<hp>(z) is decoded back to feature space and added to the
residual, replacing the per-feature scalar mean-patch.

This is the v2 §6 D-2 idea: the patch primitive ablation (D02 mean ≈
D03 zero ≈ D04 resample) suggests the patch primitive doesn't matter
much under good Stage-1 ∩ Stage-2 selection. A learned projection can
capture distributional structure scalar mean-patching cannot.

Output: outputs/D02_learned_projection/projection_<hp>.pt + summary.json.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="outputs/dataset_axbench_v1")
    ap.add_argument("--out-dir", default="outputs/D02_learned_projection")
    ap.add_argument("--feature-source", choices=["sae", "raw"], default="sae",
                    help="train projection on SAE-encoded features (Stage-1∩Stage-2 territory) or raw hookpoint activations")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="weight on unsafe→benign-mean term")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    y = np.load(data_dir / "y.npy")
    print(f"y: {(y == 1).sum()} pos / {(y == 0).sum()} neg")

    summary = {"hookpoints": {}, "feature_source": args.feature_source,
               "lam": args.lam, "epochs": args.epochs}

    for hp in HOOKPOINTS:
        hp_safe = hp.replace(".", "_")
        if args.feature_source == "raw":
            X = np.load(data_dir / f"X_raw_{hp_safe}.npy").astype("float32")
        else:
            X = np.load(data_dir / f"X_sae_{hp_safe}.npy").astype("float32")
        d = X.shape[1]
        print(f"\n=== {hp}  d={d}  ({args.feature_source}) ===", flush=True)
        Xb = X[y == 0]   # benign
        Xu = X[y == 1]   # unsafe
        mu_benign = torch.from_numpy(Xb.mean(axis=0)).to(args.device)

        Xb_t = torch.from_numpy(Xb).to(args.device)
        Xu_t = torch.from_numpy(Xu).to(args.device)

        torch.manual_seed(args.seed)
        Pi = nn.Linear(d, d, bias=True).to(args.device)
        # init: identity (so initial Pi is no-op on benign)
        with torch.no_grad():
            Pi.weight.copy_(torch.eye(d, device=args.device))
            Pi.bias.zero_()
        opt = torch.optim.Adam(Pi.parameters(), lr=args.lr, weight_decay=0.0)

        t0 = time.time()
        for ep in range(args.epochs):
            opt.zero_grad()
            # benign preservation
            yb = Pi(Xb_t)
            l_pres = ((yb - Xb_t) ** 2).mean()
            # unsafe → benign-mean
            yu = Pi(Xu_t)
            l_proj = ((yu - mu_benign[None, :]) ** 2).mean()
            loss = l_pres + args.lam * l_proj
            loss.backward()
            opt.step()
            if (ep + 1) % 50 == 0:
                print(f"  ep {ep+1:4d}  l_pres={l_pres.item():.6f}  l_proj={l_proj.item():.6f}  total={loss.item():.6f}", flush=True)

        elapsed = time.time() - t0
        # Save projection
        ckpt_path = out_dir / f"projection_{hp_safe}.pt"
        torch.save({
            "weight": Pi.weight.detach().cpu(),
            "bias": Pi.bias.detach().cpu(),
            "mu_benign": mu_benign.detach().cpu(),
            "feature_source": args.feature_source,
            "hookpoint": hp,
        }, ckpt_path)
        # Sanity AUC: distance from Pi(z) to mu_benign should rank benign closer
        with torch.no_grad():
            yb_dist = ((Pi(Xb_t) - mu_benign[None, :]) ** 2).sum(dim=-1).cpu().numpy()
            yu_dist = ((Pi(Xu_t) - mu_benign[None, :]) ** 2).sum(dim=-1).cpu().numpy()
        scores = np.concatenate([yb_dist, yu_dist])
        labels = np.concatenate([np.zeros(len(yb_dist)), np.ones(len(yu_dist))])
        try:
            auc = float(roc_auc_score(labels, scores))
        except Exception:
            auc = float("nan")
        post_pres = float(((Pi(Xb_t) - Xb_t) ** 2).mean().item())
        post_proj = float(((Pi(Xu_t) - mu_benign[None, :]) ** 2).mean().item())
        print(f"  done in {elapsed:.1f}s  benign_pres_mse={post_pres:.6f}  unsafe_proj_mse={post_proj:.6f}  ranking_auc={auc:.4f}")
        summary["hookpoints"][hp] = {
            "d": d,
            "n_benign": int(len(Xb)),
            "n_unsafe": int(len(Xu)),
            "elapsed_s": elapsed,
            "post_benign_preservation_mse": post_pres,
            "post_unsafe_projection_mse": post_proj,
            "ranking_auc": auc,
            "ckpt_path": str(ckpt_path),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
