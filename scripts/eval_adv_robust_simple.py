#!/usr/bin/env python
"""C-6 adv-robust eval (simpler): use existing A01 bypass SAE features + detector
trained on dataset_axbench_v1.

A01 has per-seed sae/<seed>.sae.pt sidecars (4-hookpoint mean-pooled SAE z's).
We:
  1. Train raw_only / sae_only / hybrid detectors on dataset_axbench_v1.
  2. Load A01 bypass SAE features (post-attack); for raw, we don't have the
     captured raw inputs, so we'll skip the raw detector for now.
  3. Score the SAE detector on bypass features and report ASR.

This is the directly-comparable cross-attack robustness test for SAE.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="outputs/dataset_axbench_v1")
    ap.add_argument("--attack-dir", default="outputs/A01_pixel_eps4_n200")
    ap.add_argument("--out-dir", default="outputs/C06_adv_robust_simple")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import numpy as np, torch
    from torch import nn

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)
    HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")

    # Train SAE-only detector on dataset_axbench_v1
    y = np.load(data / "y.npy")
    Xsae = np.concatenate([np.load(data / f"X_sae_{hp.replace('.','_')}.npy") for hp in HOOKPOINTS], axis=1)
    print(f"axbench: y={y.shape}, Xsae={Xsae.shape}")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, va = perm[:cut], perm[cut:]

    Xt = torch.as_tensor(Xsae[tr], dtype=torch.float32, device=args.device)
    yt = torch.as_tensor(y[tr], dtype=torch.float32, device=args.device)
    Xv = torch.as_tensor(Xsae[va], dtype=torch.float32, device=args.device)
    yv = torch.as_tensor(y[va], dtype=torch.float32, device=args.device)

    net = nn.Sequential(
        nn.Linear(Xsae.shape[1], 256), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(256, 1)
    ).to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    print("training sae detector (MLP-256)")
    from sklearn.metrics import roc_auc_score
    for ep in range(20):
        net.train()
        ix = torch.randperm(len(Xt), device=args.device)
        for i in range(0, len(Xt), 64):
            sl = ix[i:i+64]
            opt.zero_grad()
            loss = crit(net(Xt[sl]).squeeze(-1), yt[sl])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv).squeeze(-1)).cpu().numpy()
        auc = float(roc_auc_score(y[va], pv))
        if auc > best_auc:
            best_auc = auc
    net.eval()
    print(f"  detector va_auc={best_auc:.4f}")

    # Load A01 bypass SAE features
    attack_dir = Path(args.attack_dir)
    sae_dir = attack_dir / "sae"
    pre_paths = sorted(attack_dir.glob("*.attack.json"))
    print(f"\n{len(pre_paths)} attack records")

    n_total = 0
    n_pre_unsafe_score = 0
    n_post_unsafe_score = 0
    n_bypass_detector = 0  # was unsafe pre, becomes safe post
    pre_scores = []; post_scores = []

    # We have one .sae.pt per seed (the A01 attack outputs SAE z's for the *bypass* image)
    # Do we have a "pre" SAE feature too? Likely not — A01 only saved bypass-side. We'll
    # interpret each .sae.pt as the post-attack image's SAE features.
    # The "pre" comparison uses the dataset_axbench_v1 NSFW samples (same prompt seeds
    # weren't used, so we use a population baseline rather than a paired baseline).

    # Assume A01 bypass is post; report what fraction the SAE detector catches.
    for atk in pre_paths[:200]:
        try:
            r = json.loads(atk.read_text())
        except Exception:
            continue
        seed = r.get("seed")
        sae_path = sae_dir / f"{seed:08d}.sae.pt"
        if not sae_path.exists():
            continue
        try:
            payload = torch.load(sae_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        # Aggregate to (1, 20480). A01 sae payload is (H, W, D) per hookpoint
        # → mean over (0, 1) for (D,)
        feats = []
        for hp in HOOKPOINTS:
            if hp in payload:
                z = payload[hp]
                if z.ndim > 1:
                    z = z.float().mean(dim=tuple(range(z.ndim - 1)))
                feats.append(z.flatten())
            else:
                feats.append(torch.zeros(5120))
        feat_t = torch.cat(feats).unsqueeze(0).to(args.device)
        with torch.no_grad():
            score = float(torch.sigmoid(net(feat_t).squeeze(-1)).cpu())
        post_scores.append(score)
        n_total += 1
        if score > 0.5:
            n_post_unsafe_score += 1

    asr_post_safe = (n_total - n_post_unsafe_score) / max(1, n_total)
    out = {
        "n_total": n_total,
        "detector_va_auc_on_axbench": best_auc,
        "n_post_unsafe_flagged": n_post_unsafe_score,
        "n_post_safe_passed": n_total - n_post_unsafe_score,
        "asr_against_sae_detector": asr_post_safe,
        "post_score_mean": sum(post_scores) / max(1, len(post_scores)),
        "post_score_min": min(post_scores) if post_scores else None,
        "post_score_max": max(post_scores) if post_scores else None,
    }
    out_path = out_dir / "adv_robust_simple.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
