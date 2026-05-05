#!/usr/bin/env python
"""D-6 v3: sparsity sweep over the joint e2e mask.

Sweep lambda_sparsity in [0.05, 0.5, 5, 50, 500] and check whether
intervention correction rate stays at 100% with sparser masks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="D06_joint_e2e_v3_sparsity_sweep")
    args = ap.parse_args()

    print(f"=== {args.exp_id} ===")
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")

    print("loading SAE-mean features (NSFW vs benign)")
    benign_files = sorted((REPO / "outputs/raw_coco_500/sae").glob("*.sae.pt"))[:200]
    unsafe_files = sorted((REPO / "outputs/raw_violence_n200/sae").glob("*.sae.pt"))[:200]

    def load_features(files):
        X = []
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            v = torch.cat([d[hp].float() for hp in HOOKPOINTS], dim=0)
            X.append(v)
        return torch.stack(X, dim=0)

    X_benign = load_features(benign_files)
    X_unsafe = load_features(unsafe_files)
    X = torch.cat([X_benign, X_unsafe], dim=0)
    y = torch.cat([torch.zeros(len(X_benign)), torch.ones(len(X_unsafe))], dim=0)

    mu_benign = X_benign.mean(dim=0)

    rng = np.random.default_rng(42)
    n_b, n_u = len(X_benign), len(X_unsafe)
    perm_b = rng.permutation(n_b)
    perm_u = rng.permutation(n_u)
    cut_b, cut_u = int(0.8 * n_b), int(0.8 * n_u)
    train_idx = np.concatenate([perm_b[:cut_b], perm_u[:cut_u] + n_b])
    val_idx = np.concatenate([perm_b[cut_b:], perm_u[cut_u:] + n_b])
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train.numpy(), y_train.numpy())

    device = "cuda"
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    mu_benign = mu_benign.to(device)
    d = X.shape[1]

    rows = []
    for lam in [0.05, 0.5, 5.0, 50.0, 500.0]:
        torch.manual_seed(42)
        theta = nn.Parameter(torch.zeros(d, device=device))
        head = nn.Linear(d, 1).to(device)
        with torch.no_grad():
            head.weight.copy_(torch.tensor(clf.coef_, device=device, dtype=head.weight.dtype))
            head.bias.copy_(torch.tensor(clf.intercept_, device=device, dtype=head.bias.dtype))

        optim = torch.optim.Adam(list(head.parameters()) + [theta], lr=3e-3)
        bce = nn.BCEWithLogitsLoss()

        for ep in range(200):
            idx = torch.randperm(len(X_train))[:32]
            Xb, yb = X_train[idx], y_train[idx]
            logit_raw = head(Xb).squeeze(-1)
            L_detect = bce(logit_raw, yb)
            M = torch.sigmoid(theta)
            Xb_patched = M[None, :] * mu_benign[None, :] + (1 - M[None, :]) * Xb
            unsafe_mask = (yb > 0.5)
            if unsafe_mask.any():
                logit_patched = head(Xb_patched[unsafe_mask]).squeeze(-1)
                L_patch = bce(logit_patched, torch.zeros_like(logit_patched))
            else:
                L_patch = torch.tensor(0.0, device=device)
            L_sparsity = M.mean()
            L = L_detect + L_patch + lam * L_sparsity
            optim.zero_grad(); L.backward(); optim.step()

        with torch.no_grad():
            s = head(X_val).squeeze(-1).sigmoid().cpu().numpy()
            joint_auc = float(roc_auc_score(y_val.cpu().numpy(), s))

            M_final = torch.sigmoid(theta)
            n_active = int((M_final > 0.5).sum().item())
            mean_M = float(M_final.mean().item())
            unsafe_val = X_val[y_val > 0.5]
            unsafe_logit_pre = head(unsafe_val).squeeze(-1)
            unsafe_patched = M_final[None, :] * mu_benign[None, :] + (1 - M_final[None, :]) * unsafe_val
            unsafe_logit_post = head(unsafe_patched).squeeze(-1)
            n_pre = int((unsafe_logit_pre > 0).sum().item())
            n_post = int((unsafe_logit_post > 0).sum().item())
            corr = (n_pre - n_post) / max(n_pre, 1)

        row = {"lam": lam, "auc": joint_auc, "n_active": n_active,
               "mean_M": mean_M, "n_pre_unsafe": n_pre, "n_post_unsafe": n_post,
               "correction_rate": corr}
        rows.append(row)
        print(f"  lam={lam:7.2f}  AUC={joint_auc:.4f}  n_active={n_active:5d}/{d}  "
              f"meanM={mean_M:.3f}  pre={n_pre} post={n_post}  corr={corr:.4f}")

    (out_dir / "results.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nDONE -> {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
