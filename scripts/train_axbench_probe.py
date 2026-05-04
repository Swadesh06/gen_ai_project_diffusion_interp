#!/usr/bin/env python
"""Phase C-2 — AxBench-style raw-activation probe.

Loads X_raw_<hookpoint>.npy + X_sae_<hookpoint>.npy + y.npy from the Phase C-2
dataset dir and trains:
  - linear probe on RAW activations (per-block + all-cat)
  - linear probe on SAE activations (per-block + all-cat)

Reports head-to-head AUC. Pass criterion (per appendix §C.1):
  SAE-detector AUC >= raw-activation-detector AUC + 2pp on the same data.
  If not met → reframe Contribution 2 as interpretability + Stage-2 enablement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import cfg  # noqa: E402


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
    print(f"  {name}: best_va_auc={best_va_auc:.4f} best_va_ap={best_va_ap:.4f} (n_tr={n_tr}, n_val={len(yva)})", flush=True)
    return {"name": name, "in_dim": int(X.shape[1]), "best_va_auc": float(best_va_auc),
            "best_va_ap": float(best_va_ap), "n_tr": int(n_tr), "n_val": int(len(yva))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
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

    results = []
    # Per-block + all-concat for both raw and sae
    for prefix in ("raw", "sae"):
        Xs = {}
        for X_path in sorted(data_dir.glob(f"X_{prefix}_*.npy")):
            hp = X_path.stem.removeprefix(f"X_{prefix}_").replace("_", ".")
            Xs[hp] = np.load(X_path).astype("float32")
        if not Xs:
            print(f"no {prefix} data")
            continue
        for hp, X in Xs.items():
            print(f"=== {prefix} per-block {hp} ===")
            results.append(train_one(f"{prefix}_per_{hp}", X, y, args))
        print(f"=== {prefix} all-blocks-cat ===")
        X_cat = np.concatenate([v for v in Xs.values()], axis=1)
        results.append(train_one(f"{prefix}_all_cat", X_cat, y, args))

    out_path = out_dir / "axbench_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    # Pass criterion
    sae_all = next((r for r in results if r["name"] == "sae_all_cat"), None)
    raw_all = next((r for r in results if r["name"] == "raw_all_cat"), None)
    print()
    print("=== Pass criterion (Phase C-2) ===")
    if sae_all and raw_all:
        delta = sae_all["best_va_auc"] - raw_all["best_va_auc"]
        print(f"sae_all AUC = {sae_all['best_va_auc']:.4f}")
        print(f"raw_all AUC = {raw_all['best_va_auc']:.4f}")
        print(f"Δ = SAE − raw = {delta:+.4f} ({delta*100:.2f} pp)")
        print(f"PASS criterion (Δ ≥ +0.02): {'YES' if delta >= 0.02 else 'NO — SAE story collapses to interpretability + Stage-2 enablement'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
