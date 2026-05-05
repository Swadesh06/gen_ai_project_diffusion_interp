#!/usr/bin/env python
"""D-6 v4: joint e2e training on cf_strategy2_seed_pairs (counterfactual).

Per task_description_v2.md §6 D-6 the spec asks for training on "5K
paired prompts". This v4 uses the cf_strategy2_seed_pairs (246 pairs;
492 images = NSFW + counterfactual benign). Smaller-scale than the
spec's 5K, but on the *correct* counterfactual data distribution.

Pipeline:
1. Load 246 cf_strategy2 pairs.
2. Encode each image via SDXL Turbo UNet at t=50 with SAE hooks,
   mean-pool over (H, W) to get (B, 20480) features.
3. Train joint mask + head with three losses (D-6 v2 protocol).
4. Save the state for downstream UDA application.

Outputs:
  reports/D06_joint_e2e_v4_cf_strategy2.md
  outputs/D06_joint_e2e_v4_cf_strategy2/{joint_state.pt, results.json}
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
F_PER_HP = 5120


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="D06_joint_e2e_v4_cf_strategy2")
    ap.add_argument("--cf-jsonl", default="outputs/cf_benchmark_v1_seed/validated.jsonl")
    ap.add_argument("--max-pairs", type=int, default=246)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lam-sparsity", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== {args.exp_id} ===")
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    out_dir = REPO / "outputs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cf_path = REPO / args.cf_jsonl
    if not cf_path.exists():
        print(f"no validated.jsonl at {cf_path}")
        return 2
    pairs = [json.loads(l) for l in cf_path.read_text().splitlines()][:args.max_pairs]
    print(f"  loaded {len(pairs)} validated pairs")

    print("loading SDXL Turbo + 4 Surkov SAEs")
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.hooks import SurkovHookManager
    from dsi.sae.load import load_surkov_sae

    pipe_w = SDXLPipelineWrapper(variant="turbo", device="cuda", dtype="fp16").load()
    sae_dict = {hp: load_surkov_sae(hp).to("cuda").eval() for hp in HOOKPOINTS}

    @torch.no_grad()
    def encode_one(img_path):
        pil = Image.open(img_path).convert("RGB").resize((512, 512))
        a = np.asarray(pil, dtype=np.float32) / 255.0
        x = torch.as_tensor((a * 2 - 1).transpose(2, 0, 1)[None], device="cuda",
                            dtype=pipe_w.vae.dtype)
        latent = pipe_w.vae.encode(x).latent_dist.sample() * pipe_w.vae.config.scaling_factor
        latent = latent.to(next(pipe_w.unet.parameters()).dtype)
        prompts = [""]
        prompt_embeds, _, pooled, _ = pipe_w.pipe.encode_prompt(
            prompt=prompts, prompt_2=prompts,
            device="cuda", num_images_per_prompt=1,
            do_classifier_free_guidance=False)
        time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], device="cuda",
                                 dtype=latent.dtype)
        with SurkovHookManager(pipe_w.unet, sae_dict, capture=True,
                                keep_inputs=False) as mgr:
            t = torch.full((1,), 50, device="cuda", dtype=torch.long)
            pipe_w.unet(latent, t, encoder_hidden_states=prompt_embeds,
                         added_cond_kwargs={"text_embeds": pooled,
                                             "time_ids": time_ids}).sample
        feats = []
        for hp in HOOKPOINTS:
            zlast = mgr.captured[hp].z[-1] if mgr.captured[hp].z else None
            if zlast is None:
                feats.append(torch.zeros(F_PER_HP))
                continue
            v = zlast.float().mean(dim=tuple(range(1, zlast.ndim - 1))).squeeze(0)
            feats.append(v.cpu())
        return torch.cat(feats, dim=0)  # (20480,)

    print(f"encoding {2 * len(pairs)} images")
    t0 = time.time()
    X_unsafe = []
    X_safe = []
    for i, p in enumerate(pairs):
        x_u = encode_one(p["flagged_path"])
        x_s = encode_one(p["unflagged_path"])
        X_unsafe.append(x_u)
        X_safe.append(x_s)
        if (i + 1) % 20 == 0:
            print(f"  encoded {i + 1}/{len(pairs)} pairs, elapsed {time.time() - t0:.0f}s")
    X_unsafe = torch.stack(X_unsafe, dim=0)
    X_safe = torch.stack(X_safe, dim=0)
    print(f"  done encoding in {time.time() - t0:.0f}s")

    X = torch.cat([X_safe, X_unsafe], dim=0)
    y = torch.cat([torch.zeros(len(X_safe)), torch.ones(len(X_unsafe))], dim=0)
    print(f"  X={X.shape}, y={y.shape}")

    mu_benign = X_safe.mean(dim=0)

    # 80/20 split (stratified)
    rng = np.random.default_rng(args.seed)
    n = len(X_safe)
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    train_idx = np.concatenate([perm[:cut], perm[:cut] + n])  # safe + unsafe halves
    val_idx = np.concatenate([perm[cut:], perm[cut:] + n])
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train.numpy(), y_train.numpy())
    s_val = clf.predict_proba(X_val.numpy())[:, 1]
    baseline_auc = float(roc_auc_score(y_val.numpy(), s_val))
    print(f"  baseline LR val AUC = {baseline_auc:.4f}")

    device = "cuda"
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    mu_benign = mu_benign.to(device)
    d = X.shape[1]

    torch.manual_seed(args.seed)
    theta = nn.Parameter(torch.zeros(d, device=device))
    head = nn.Linear(d, 1).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(clf.coef_, device=device, dtype=head.weight.dtype))
        head.bias.copy_(torch.tensor(clf.intercept_, device=device, dtype=head.bias.dtype))

    optim = torch.optim.Adam(list(head.parameters()) + [theta], lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    print(f"\ntraining: {args.epochs} epochs, lam={args.lam_sparsity}")
    for ep in range(args.epochs):
        idx = torch.randperm(len(X_train))[:args.batch_size]
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
        L = L_detect + L_patch + args.lam_sparsity * L_sparsity
        optim.zero_grad(); L.backward(); optim.step()
        if (ep + 1) % 50 == 0:
            with torch.no_grad():
                s = head(X_val).squeeze(-1).sigmoid().cpu().numpy()
                auc = float(roc_auc_score(y_val.cpu().numpy(), s))
                print(f"  ep {ep+1:3d}  L={float(L):.4f}  "
                      f"AUC={auc:.4f}  ||M||_1={M.sum().item():.0f}")

    with torch.no_grad():
        s = head(X_val).squeeze(-1).sigmoid().cpu().numpy()
        joint_auc = float(roc_auc_score(y_val.cpu().numpy(), s))
        M_final = torch.sigmoid(theta)
        n_active = int((M_final > 0.5).sum().item())
        unsafe_val = X_val[y_val > 0.5]
        unsafe_logit_pre = head(unsafe_val).squeeze(-1)
        unsafe_patched = M_final[None, :] * mu_benign[None, :] + (1 - M_final[None, :]) * unsafe_val
        unsafe_logit_post = head(unsafe_patched).squeeze(-1)
        n_pre = int((unsafe_logit_pre > 0).sum().item())
        n_post = int((unsafe_logit_post > 0).sum().item())
        corr = (n_pre - n_post) / max(n_pre, 1)

        # per-hookpoint mask
        per_hp = {}
        for i, hp in enumerate(HOOKPOINTS):
            start = i * F_PER_HP
            end = start + F_PER_HP
            per_hp[hp] = int((M_final[start:end] > 0.5).sum().item())

    print(f"\nFINAL: AUC={joint_auc:.4f}, n_active={n_active} ({per_hp}), "
          f"corr={corr:.4f} ({n_pre}->{n_post})")

    torch.save({"theta": theta.detach().cpu(),
                "head_state_dict": {"weight": head.weight.detach().cpu(),
                                     "bias": head.bias.detach().cpu()},
                "mu_benign": mu_benign.cpu(),
                "lam_sparsity": args.lam_sparsity},
                out_dir / "joint_state.pt")

    results = {
        "exp_id": args.exp_id,
        "n_pairs": len(pairs),
        "baseline_lr_auc": baseline_auc,
        "joint_auc": joint_auc,
        "n_active_total": n_active,
        "n_active_per_hp": per_hp,
        "n_pre_unsafe": n_pre,
        "n_post_unsafe": n_post,
        "correction_rate": corr,
        "epochs": args.epochs,
        "lam_sparsity": args.lam_sparsity,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nDONE -> {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
