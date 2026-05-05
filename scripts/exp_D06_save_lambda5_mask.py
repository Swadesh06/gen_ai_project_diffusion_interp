#!/usr/bin/env python
"""Save the lambda=5.0 joint state from D-6 v3 sparsity sweep."""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    print("=== D06_save_lambda5_mask ===")
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression

    out_dir = REPO / "outputs/D06_lambda5_state"
    out_dir.mkdir(parents=True, exist_ok=True)

    HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")

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
    X_train, y_train = X[train_idx], y[train_idx]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train.numpy(), y_train.numpy())

    device = "cuda"
    X_train, y_train = X_train.to(device), y_train.to(device)
    mu_benign = mu_benign.to(device)
    d = X.shape[1]

    torch.manual_seed(42)
    theta = nn.Parameter(torch.zeros(d, device=device))
    head = nn.Linear(d, 1).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(clf.coef_, device=device, dtype=head.weight.dtype))
        head.bias.copy_(torch.tensor(clf.intercept_, device=device, dtype=head.bias.dtype))

    optim = torch.optim.Adam(list(head.parameters()) + [theta], lr=3e-3)
    bce = nn.BCEWithLogitsLoss()
    lam_sparsity = 5.0

    for ep in range(200):
        idx = torch.randperm(len(X_train))[:32]
        Xb, yb = X_train[idx], y_train[idx]
        logit_raw = head(Xb).squeeze(-1)
        L_detect = bce(logit_raw, yb)
        M = torch.sigmoid(theta)
        Xb_patched = M[None, :] * mu_benign[None, :] + (1 - M[None, :]) * Xb
        unsafe_mask_b = (yb > 0.5)
        if unsafe_mask_b.any():
            logit_patched = head(Xb_patched[unsafe_mask_b]).squeeze(-1)
            L_patch = bce(logit_patched, torch.zeros_like(logit_patched))
        else:
            L_patch = torch.tensor(0.0, device=device)
        L_sparsity = M.mean()
        L = L_detect + L_patch + lam_sparsity * L_sparsity
        optim.zero_grad(); L.backward(); optim.step()

    M_final = torch.sigmoid(theta)
    n_active = int((M_final > 0.5).sum().item())
    print(f"  final n_active = {n_active}")

    # Save
    torch.save({"theta": theta.detach().cpu(),
                "head_state_dict": {"weight": head.weight.detach().cpu(),
                                     "bias": head.bias.detach().cpu()},
                "mu_benign": mu_benign.cpu(),
                "lam_sparsity": lam_sparsity},
                out_dir / "joint_state.pt")
    print(f"  saved {out_dir / 'joint_state.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
