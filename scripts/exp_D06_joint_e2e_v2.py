#!/usr/bin/env python
"""D-6 v2: joint end-to-end training of soft-Stage-2 mask + detection head.

Per task_description_v2.md §6 D-6:
> differentiate through SAE encode -> linear detector -> soft-Stage-2
> (Gumbel-softmax over feature mask) -> mean-patch (with straight-through).
> Three losses: SAE reconstruction, detection BCE on counterfactual pairs,
> intervention quality.

This v2 uses precomputed SAE-mean features (4 hookpoints x 5120 each =
20480 dims) and trains end-to-end:
  - Detection head H : R^20480 -> R^1 (linear probe)
  - Soft Stage-2 mask M : R^20480 in [0, 1] via sigmoid(theta)
  - Patched feature: z' = M * mu_benign + (1 - M) * z (soft mean-patch)

Three losses:
  1. detect: BCE(H(z), label)         -- detection on raw features
  2. patch_classifies_safe: BCE(H(z'), 0_label) for unsafe samples only
  3. sparsity: lambda * mean(M)       -- mask should be small (select few features)

Evaluates:
  - Detection AUC vs. baseline LR
  - Intervention correction-rate: fraction of unsafe samples for which
    H(z') < threshold after applying the learned soft mask.

Outputs:
  reports/D06_joint_e2e_v2.md
  outputs/D06_joint_e2e_v2/{loss_curve.png, mask_state.pt, head_state.pt, results.json}
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
    ap.add_argument("--exp-id", default="D06_joint_e2e_v2")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lambda-sparsity", type=float, default=0.05)
    ap.add_argument("--lambda-patch", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== {args.exp_id} ===")
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")

    print("loading SAE-mean features (NSFW vs benign)")
    benign_files = sorted((REPO / "outputs/raw_coco_500/sae").glob("*.sae.pt"))[:200]
    unsafe_files = sorted((REPO / "outputs/raw_violence_n200/sae").glob("*.sae.pt"))[:200]
    print(f"  benign: {len(benign_files)}, unsafe: {len(unsafe_files)}")

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
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    # mu_benign for the mean-patch target (per-feature mean of benign samples)
    mu_benign = X_benign.mean(dim=0)
    print(f"  mu_benign shape: {mu_benign.shape}, mean: {mu_benign.mean().item():.4f}")

    # train/val split (stratified)
    rng = np.random.default_rng(args.seed)
    n_b, n_u = len(X_benign), len(X_unsafe)
    perm_b = rng.permutation(n_b)
    perm_u = rng.permutation(n_u)
    cut_b = int(0.8 * n_b)
    cut_u = int(0.8 * n_u)
    train_idx = np.concatenate([perm_b[:cut_b], perm_u[:cut_u] + n_b])
    val_idx = np.concatenate([perm_b[cut_b:], perm_u[cut_u:] + n_b])
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"  train: {X_train.shape}, val: {X_val.shape}")

    # baseline LR for AUC reference
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train.numpy(), y_train.numpy())
    s_val = clf.predict_proba(X_val.numpy())[:, 1]
    baseline_auc = float(roc_auc_score(y_val.numpy(), s_val))
    print(f"  baseline LR val AUC = {baseline_auc:.4f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    mu_benign = mu_benign.to(device)

    # joint model: theta (mask logits, 20480-dim) + linear head
    d = X.shape[1]
    theta = nn.Parameter(torch.zeros(d, device=device))  # M = sigmoid(theta)
    head = nn.Linear(d, 1).to(device)
    # init head from baseline LR coefs
    with torch.no_grad():
        head.weight.copy_(torch.tensor(clf.coef_, device=device, dtype=head.weight.dtype))
        head.bias.copy_(torch.tensor(clf.intercept_, device=device, dtype=head.bias.dtype))

    optim = torch.optim.Adam(list(head.parameters()) + [theta], lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    print(f"\ntraining joint model: {args.epochs} epochs, lr={args.lr}, "
          f"lambda_sparsity={args.lambda_sparsity}, lambda_patch={args.lambda_patch}")
    losses = {"total": [], "detect": [], "patch": [], "sparsity": []}
    val_aucs = []

    for ep in range(args.epochs):
        # forward on train
        idx = torch.randperm(len(X_train))[:args.batch_size]
        Xb, yb = X_train[idx], y_train[idx]

        # detection on raw features
        logit_raw = head(Xb).squeeze(-1)
        L_detect = bce(logit_raw, yb)

        # soft-Stage-2 mean-patch: z' = M * mu_benign + (1-M) * z
        M = torch.sigmoid(theta)
        Xb_patched = M[None, :] * mu_benign[None, :] + (1 - M[None, :]) * Xb

        # patched-as-safe: only for unsafe samples (yb==1), drive H(z') -> 0
        unsafe_mask = (yb > 0.5)
        if unsafe_mask.any():
            logit_patched = head(Xb_patched[unsafe_mask]).squeeze(-1)
            L_patch = bce(logit_patched, torch.zeros_like(logit_patched))
        else:
            L_patch = torch.tensor(0.0, device=device)

        # sparsity: encourage M small (select fewer features)
        L_sparsity = M.mean()

        L = L_detect + args.lambda_patch * L_patch + args.lambda_sparsity * L_sparsity
        optim.zero_grad()
        L.backward()
        optim.step()

        losses["total"].append(float(L))
        losses["detect"].append(float(L_detect))
        losses["patch"].append(float(L_patch))
        losses["sparsity"].append(float(L_sparsity))

        if (ep + 1) % 20 == 0 or ep == 0:
            with torch.no_grad():
                s = head(X_val).squeeze(-1).sigmoid().cpu().numpy()
                v_auc = float(roc_auc_score(y_val.cpu().numpy(), s))
                val_aucs.append((ep + 1, v_auc))
                print(f"  ep {ep+1:3d}  L={float(L):.4f}  detect={float(L_detect):.4f}  "
                      f"patch={float(L_patch):.4f}  sparsity={float(L_sparsity):.4f}  "
                      f"val_AUC={v_auc:.4f}  ||M||_1={M.sum().item():.1f}")

    # final eval
    with torch.no_grad():
        # detection AUC
        s = head(X_val).squeeze(-1).sigmoid().cpu().numpy()
        joint_auc = float(roc_auc_score(y_val.cpu().numpy(), s))

        # intervention correction rate
        M_final = torch.sigmoid(theta)
        n_active = (M_final > 0.5).sum().item()
        unsafe_val = X_val[y_val > 0.5]
        unsafe_logit_pre = head(unsafe_val).squeeze(-1)
        unsafe_patched = M_final[None, :] * mu_benign[None, :] + (1 - M_final[None, :]) * unsafe_val
        unsafe_logit_post = head(unsafe_patched).squeeze(-1)
        n_pre_unsafe = int((unsafe_logit_pre > 0).sum().item())
        n_post_unsafe = int((unsafe_logit_post > 0).sum().item())
        correction_rate = (n_pre_unsafe - n_post_unsafe) / max(n_pre_unsafe, 1)

        # benign preservation: how many benign flip to unsafe under patch?
        benign_val = X_val[y_val < 0.5]
        benign_logit_pre = head(benign_val).squeeze(-1)
        benign_patched = M_final[None, :] * mu_benign[None, :] + (1 - M_final[None, :]) * benign_val
        benign_logit_post = head(benign_patched).squeeze(-1)
        n_benign_post_unsafe = int((benign_logit_post > 0).sum().item())
        n_benign_pre_unsafe = int((benign_logit_pre > 0).sum().item())

        print("\nfinal joint result:")
        print(f"  detection val AUC: {joint_auc:.4f} (baseline LR: {baseline_auc:.4f})")
        print(f"  active mask features (M>0.5): {n_active} / {d}")
        print(f"  intervention: {n_pre_unsafe} pre-unsafe -> {n_post_unsafe} post-unsafe; "
              f"correction rate = {correction_rate:.4f}")
        print(f"  benign preservation: {n_benign_pre_unsafe} pre -> {n_benign_post_unsafe} post unsafe (benign FP shift)")

    # save
    torch.save({"theta": theta.detach().cpu(), "head_state_dict": head.state_dict(),
                "mu_benign": mu_benign.cpu()}, out_dir / "joint_state.pt")
    results = {
        "exp_id": args.exp_id,
        "baseline_lr_auc": baseline_auc,
        "joint_auc": joint_auc,
        "n_active_mask_features": n_active,
        "total_features": d,
        "intervention_correction_rate": correction_rate,
        "n_unsafe_pre_to_post": [n_pre_unsafe, n_post_unsafe],
        "benign_fp_shift": [n_benign_pre_unsafe, n_benign_post_unsafe],
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_sparsity": args.lambda_sparsity,
        "lambda_patch": args.lambda_patch,
        "seed": args.seed,
        "val_aucs_per_epoch": val_aucs,
        "final_loss": float(L),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # plot loss curve
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(losses["total"], label="total")
        axes[0].plot(losses["detect"], label="detect (BCE)")
        axes[0].plot(losses["patch"], label="patch (BCE)")
        axes[0].plot(losses["sparsity"], label=f"sparsity (mean(M))")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].legend()
        axes[0].set_title("training losses")

        eps, aucs = zip(*val_aucs) if val_aucs else ([], [])
        axes[1].plot(eps, aucs, "o-", label="joint")
        axes[1].axhline(baseline_auc, color="red", linestyle="--", label=f"baseline LR ({baseline_auc:.3f})")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("val AUC")
        axes[1].legend()
        axes[1].set_title("validation AUC")

        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png", dpi=120)
        print(f"  saved loss_curve.png")
    except Exception as e:
        print(f"  plot failed: {e}")

    print(f"\nDONE. results -> {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
