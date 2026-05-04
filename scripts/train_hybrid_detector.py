#!/usr/bin/env python
"""Phase C-6 — hybrid SAE + raw-residual detector.

Concatenates [SAE features || raw mean-pooled residuals] as detector input.
Compares against:
  - SAE only (re-uses B01/B02 numbers)
  - raw only (Phase C-2 axbench probe)

Pass criterion (per appendix §G C-6):
  hybrid AUC > max(SAE-only, raw-only) AUC by >= 1pp on I2P-adversarial.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _split(X, y, val_frac, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_val = max(1, int(len(y) * val_frac))
    return X[perm[n_val:]], y[perm[n_val:]], X[perm[:n_val]], y[perm[:n_val]]


def train_one(name, X, y, args):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, roc_auc_score

    Xtr, ytr, Xva, yva = _split(X, y, args.val_frac, args.seed)
    Xtr_t = torch.from_numpy(Xtr).to(args.device).float()
    ytr_t = torch.from_numpy(ytr).float().to(args.device)
    Xva_t = torch.from_numpy(Xva).to(args.device).float()

    head = nn.Linear(X.shape[1], 1).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pw = None
    if args.auto_pos_weight:
        n_pos = int(ytr.sum())
        n_neg = int((1 - ytr).sum())
        pw = torch.tensor([n_neg / max(1, n_pos)], device=args.device)

    n_tr = len(Xtr_t)
    best_va_auc = 0.0
    best_va_ap = 0.0
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n_tr, device=args.device)
        for i in range(0, n_tr, args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = head(Xtr_t[idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, ytr_t[idx], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            va_logits = head(Xva_t).squeeze(-1).float().cpu().numpy()
        va_auc = roc_auc_score(yva, va_logits)
        va_ap = average_precision_score(yva, va_logits)
        best_va_auc = max(best_va_auc, va_auc)
        best_va_ap = max(best_va_ap, va_ap)
    return {"name": name, "in_dim": int(X.shape[1]), "best_va_auc": float(best_va_auc),
            "best_va_ap": float(best_va_ap), "n_tr": int(n_tr), "n_val": int(len(yva))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="dir with X_raw_<hp>.npy + X_sae_<hp>.npy + y.npy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auto-pos-weight", action="store_true")
    args = ap.parse_args()

    import numpy as np

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y = np.load(data_dir / "y.npy")
    print(f"y: {len(y)} samples ({int(y.sum())} pos / {int((1-y).sum())} neg)")

    # Load all per-block features for both raw and sae
    Xs = {"raw": {}, "sae": {}}
    for kind in ("raw", "sae"):
        for X_path in sorted(data_dir.glob(f"X_{kind}_*.npy")):
            hp = X_path.stem.removeprefix(f"X_{kind}_").replace("_", ".")
            Xs[kind][hp] = np.load(X_path).astype("float32")
            print(f"  {kind}_{hp}: {Xs[kind][hp].shape}")

    if not Xs["raw"] or not Xs["sae"]:
        print("missing raw or sae data")
        return 2

    raw_cat = np.concatenate([v for v in Xs["raw"].values()], axis=1)
    sae_cat = np.concatenate([v for v in Xs["sae"].values()], axis=1)
    hybrid_cat = np.concatenate([raw_cat, sae_cat], axis=1)

    print(f"\n=== raw_only AUC ===")
    r_raw = train_one("raw_only", raw_cat, y, args)
    print(f"  best_va_auc={r_raw['best_va_auc']:.4f} best_va_ap={r_raw['best_va_ap']:.4f}")

    print(f"\n=== sae_only AUC ===")
    r_sae = train_one("sae_only", sae_cat, y, args)
    print(f"  best_va_auc={r_sae['best_va_auc']:.4f} best_va_ap={r_sae['best_va_ap']:.4f}")

    print(f"\n=== hybrid AUC (raw || sae) ===")
    r_hybrid = train_one("hybrid_raw_plus_sae", hybrid_cat, y, args)
    print(f"  best_va_auc={r_hybrid['best_va_auc']:.4f} best_va_ap={r_hybrid['best_va_ap']:.4f}")

    delta_vs_sae = r_hybrid["best_va_auc"] - r_sae["best_va_auc"]
    delta_vs_raw = r_hybrid["best_va_auc"] - r_raw["best_va_auc"]
    print(f"\n=== Pass criterion (Phase C-6) ===")
    print(f"hybrid - SAE = {delta_vs_sae:+.4f} ({delta_vs_sae*100:+.2f} pp)")
    print(f"hybrid - raw = {delta_vs_raw:+.4f} ({delta_vs_raw*100:+.2f} pp)")
    print(f"PASS criterion (hybrid > max(SAE,raw) by ≥ 1pp): {'YES' if min(delta_vs_sae, delta_vs_raw) >= 0.01 else 'NO'}")

    out_path = out_dir / "hybrid_results.json"
    out_path.write_text(json.dumps({"raw_only": r_raw, "sae_only": r_sae, "hybrid": r_hybrid,
                                    "delta_vs_sae": delta_vs_sae, "delta_vs_raw": delta_vs_raw}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
