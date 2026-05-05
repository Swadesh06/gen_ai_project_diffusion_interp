#!/usr/bin/env python
"""D-8 v1: round-based adversarial training of the joint mask + head.

Per task_description_v2.md §6 D-8:
> round r: (1) compute F_c^(r) via Stage 1 ∩ Stage 2 on D^(r).
> (2) Run pixel + latent + embedding attacks against the deployed
> pipeline using F_c^(r). (3) Augment D^(r+1) with successful bypasses.
> (4) Repeat for 5 rounds.

This v1 runs **2 rounds** with the joint-trained mask from D-6 v2.

Round 0: train on (200 benign, 200 unsafe) with joint mask + head.
         Compute F_c^(0) = mask features at M > 0.5.
Round 1: simulate adversarial bypasses by perturbing benign features
         toward unsafe along the F_c^(0) direction (synthetic
         "white-box-like" augmentations). Re-train head on
         (benign, unsafe, synthetic-bypasses).
         Compute F_c^(1) = updated mask.

Pass: |F_c^(1)| within 2× |F_c^(0)|; correction rate stays high on
held-out test.

Outputs:
  reports/D08_trades_v1.md
  outputs/D08_trades_v1/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def train_joint_one_round(X_train, y_train, X_val, y_val, mu_benign, d, 
                          init_head_w, init_head_b, lam_sparsity=5.0, 
                          epochs=200, lr=3e-3, seed=42):
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    
    torch.manual_seed(seed)
    theta = nn.Parameter(torch.zeros(d, device=X_train.device))
    head = nn.Linear(d, 1).to(X_train.device)
    with torch.no_grad():
        head.weight.copy_(init_head_w.to(head.weight.dtype))
        head.bias.copy_(init_head_b.to(head.bias.dtype))

    optim = torch.optim.Adam(list(head.parameters()) + [theta], lr=lr)
    bce = nn.BCEWithLogitsLoss()
    
    for ep in range(epochs):
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
            L_patch = torch.tensor(0.0, device=X_train.device)
        L_sparsity = M.mean()
        L = L_detect + L_patch + lam_sparsity * L_sparsity
        optim.zero_grad(); L.backward(); optim.step()

    with torch.no_grad():
        s_val = head(X_val).squeeze(-1).sigmoid().cpu().numpy()
        auc = float(roc_auc_score(y_val.cpu().numpy(), s_val))
        M_final = torch.sigmoid(theta)
        n_active = int((M_final > 0.5).sum().item())
        unsafe_val = X_val[y_val > 0.5]
        unsafe_logit_pre = head(unsafe_val).squeeze(-1)
        unsafe_patched = M_final[None, :] * mu_benign[None, :] + (1 - M_final[None, :]) * unsafe_val
        unsafe_logit_post = head(unsafe_patched).squeeze(-1)
        n_pre = int((unsafe_logit_pre > 0).sum().item())
        n_post = int((unsafe_logit_post > 0).sum().item())
        corr = (n_pre - n_post) / max(n_pre, 1)
    return {"theta": theta.detach(), "head_w": head.weight.detach(),
            "head_b": head.bias.detach(), "M_final": M_final.detach(),
            "auc": auc, "n_active": n_active, "corr": corr,
            "n_pre": n_pre, "n_post": n_post}


def attack_features(X_unsafe, theta, head_w, head_b, mu_benign, eps=2.0):
    """Synthetic 'gradient-like' attack: perturb features along the
    detection-head's negative gradient (in feature space) to drive
    unsafe -> safe in head's prediction."""
    import torch
    
    # Detection direction in feature space = head_w
    # To drive unsafe -> safe, subtract head_w * eps along the unsafe samples.
    direction = head_w.squeeze().detach()
    direction_norm = direction / (direction.norm() + 1e-8)
    X_attacked = X_unsafe - eps * direction_norm[None, :]
    return X_attacked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="D08_trades_v1")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--lam-sparsity", type=float, default=5.0)
    ap.add_argument("--attack-eps", type=float, default=3.0)
    args = ap.parse_args()

    print(f"=== {args.exp_id} ===")
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression

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

    init_head_w = torch.tensor(clf.coef_, device=device)
    init_head_b = torch.tensor(clf.intercept_, device=device)

    # ROUND 0: standard joint training
    print(f"\n--- Round 0: clean training ---")
    r0 = train_joint_one_round(X_train, y_train, X_val, y_val, mu_benign, d,
                                init_head_w, init_head_b,
                                lam_sparsity=args.lam_sparsity)
    print(f"  AUC={r0['auc']:.4f}, n_active={r0['n_active']}, corr={r0['corr']:.4f}")

    rounds = [{"r": 0, **{k: r0[k] for k in ["auc", "n_active", "corr", "n_pre", "n_post"]},
               "n_train": int(len(X_train)), "n_unsafe_synthetic": 0}]

    # ROUNDS 1..N: synthesise adversarial bypasses, augment data, re-train
    X_aug = X_train.clone()
    y_aug = y_train.clone()
    for r in range(1, args.rounds + 1):
        print(f"\n--- Round {r}: attack + augment ---")
        # Attack: take train-set unsafe samples, perturb toward "safe" direction
        # along current head_w, but label them unsafe to teach head to handle them.
        unsafe_train = X_train[y_train > 0.5]
        attacked = attack_features(unsafe_train, theta=None,
                                    head_w=r0["head_w"] if r == 1 else last_round["head_w"],
                                    head_b=None, mu_benign=mu_benign,
                                    eps=args.attack_eps)
        # Augment training set: add attacked features as positive labels
        X_aug = torch.cat([X_aug, attacked], dim=0)
        y_aug = torch.cat([y_aug, torch.ones(len(attacked), device=device)], dim=0)

        # Re-train
        rN = train_joint_one_round(X_aug, y_aug, X_val, y_val, mu_benign, d,
                                    init_head_w, init_head_b,
                                    lam_sparsity=args.lam_sparsity, seed=42 + r)
        # Test correction rate on the synthetic attacked val features
        unsafe_val_attacked = attack_features(X_val[y_val > 0.5],
                                               theta=None,
                                               head_w=rN["head_w"],
                                               head_b=None, mu_benign=mu_benign,
                                               eps=args.attack_eps)
        with torch.no_grad():
            import torch.nn as nn
            head = nn.Linear(d, 1).to(device)
            head.weight.copy_(rN["head_w"].to(head.weight.dtype))
            head.bias.copy_(rN["head_b"].to(head.bias.dtype))
            attacked_logit_pre = head(unsafe_val_attacked).squeeze(-1)
            patched = rN["M_final"][None, :] * mu_benign[None, :] + (1 - rN["M_final"][None, :]) * unsafe_val_attacked
            attacked_logit_post = head(patched).squeeze(-1)
            n_attacked_pre = int((attacked_logit_pre > 0).sum().item())
            n_attacked_post = int((attacked_logit_post > 0).sum().item())
            attack_robust_corr = (n_attacked_pre - n_attacked_post) / max(n_attacked_pre, 1)
        print(f"  AUC={rN['auc']:.4f}, n_active={rN['n_active']}, corr={rN['corr']:.4f}")
        print(f"  attacked-val: pre={n_attacked_pre}, post={n_attacked_post}, "
              f"correction={attack_robust_corr:.4f}")

        rounds.append({"r": r,
                       **{k: rN[k] for k in ["auc", "n_active", "corr", "n_pre", "n_post"]},
                       "n_train": int(len(X_aug)),
                       "n_unsafe_synthetic": int(len(attacked)),
                       "n_attacked_val_pre": n_attacked_pre,
                       "n_attacked_val_post": n_attacked_post,
                       "attack_robust_corr": attack_robust_corr})
        last_round = rN

    # Save
    results = {"exp_id": args.exp_id, "lam_sparsity": args.lam_sparsity,
               "attack_eps": args.attack_eps, "rounds": rounds}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nDONE -> {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
