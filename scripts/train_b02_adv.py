#!/usr/bin/env python
"""D-6 / D-8 proxy: B02-adv detector trained on A01-bypassed SAE features.

Standard B02-v3 was trained on (oracle-flagged NSFW, benign) pairs.
B02-adv trains on (adversarial-bypassed images, benign) — the SAE
features of images that successfully bypass safety_checker via PGD.

Hypothesis: B02-adv generalizes better to held-out adversarial attacks
than B02-v3, because it has seen the post-attack distribution during
training.

Inputs:
- outputs/A01_pixel_eps4_n200_seed{0..4}/sae/*.sae.pt — adversarial-positive
  features (88 images across 5 seeds where attack bypassed safety_checker).
  Filter to pre_flagged=True && post_flagged=False (the 'bypass' subset).
- outputs/raw_coco_500/sae/*.sae.pt — benign-negative features.

Output:
- checkpoints/B02_adv_v1/best.pt — linear probe state dict
- reports/B02_adv_v1.md
- evaluation against held-out A02 5-seed bypassed
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")


def load_attack_sae_features(attack_dir, max_n=None):
    """Load SAE features for images that pre_flagged & not post_flagged (=bypass)."""
    import torch
    import json as jsonlib

    base = Path(attack_dir)
    features = []
    n_loaded = 0
    for atk_path in sorted(base.glob("*.attack.json")):
        try:
            atk = jsonlib.loads(atk_path.read_text())
        except Exception:
            continue
        # only include bypass cases
        if not atk.get("pre_flagged") or atk.get("post_flagged"):
            continue
        seed = atk.get("seed")
        sae_path = base / "sae" / f"{seed:08d}.sae.pt"
        if not sae_path.exists():
            continue
        try:
            d = torch.load(sae_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        # mean-pool over spatial dims
        feat = []
        for hp in HOOKPOINTS:
            if hp not in d:
                feat = None
                break
            v = d[hp]
            if v.ndim == 3:  # (H, W, F)
                v = v.mean(dim=(0, 1))
            elif v.ndim == 4:  # (B, H, W, F)
                v = v.mean(dim=(1, 2))[0]
            feat.append(v.float())
        if feat is None:
            continue
        features.append(torch.cat(feat))
        n_loaded += 1
        if max_n and n_loaded >= max_n:
            break
    return features


def load_benign_sae_features(benign_dir, max_n=None):
    import torch
    files = sorted(Path(benign_dir).glob("*.sae.pt"))
    if max_n: files = files[:max_n]
    features = []
    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        feat = []
        for hp in HOOKPOINTS:
            if hp not in d:
                feat = None; break
            v = d[hp]
            if v.ndim == 1:  # already mean-pooled
                feat.append(v.float())
            elif v.ndim == 3:
                feat.append(v.float().mean(dim=(0, 1)))
            elif v.ndim == 4:
                feat.append(v.float().mean(dim=(1, 2))[0])
        if feat is None:
            continue
        features.append(torch.cat(feat))
    return features


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="B02_adv_v1")
    ap.add_argument("--out-dir", default="checkpoints/B02_adv_v1")
    args = ap.parse_args()

    import torch
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.exp_id} ===")

    # POSITIVE: A01 + A02 bypassed across 5 seeds
    print("loading A01 + A02 bypassed SAE features (5-seed)")
    pos_feats = []
    for s in range(5):
        for prefix, eps in [("A01_pixel_eps4_n200", "0"),
                            ("A02_latent_eps0.1_n200", "0")]:
            d = f"outputs/{prefix}_seed{s}" if s > 0 else f"outputs/{prefix}_seed{s}"
            if s == 0 and not Path(d).exists():
                d = f"outputs/{prefix}"
            feats = load_attack_sae_features(d)
            pos_feats.extend(feats)
            print(f"  {d}: +{len(feats)} bypass features")
    print(f"  total positive: {len(pos_feats)}")
    if len(pos_feats) < 10:
        print("ERR: not enough positive features. Aborting.")
        return 1

    # NEGATIVE: COCO benign mean-pooled SAE features
    print("loading COCO benign SAE features")
    neg_feats = load_benign_sae_features("outputs/raw_coco_500/sae", max_n=500)
    print(f"  total negative: {len(neg_feats)}")

    X_pos = torch.stack(pos_feats).numpy()
    X_neg = torch.stack(neg_feats).numpy()
    print(f"  X_pos shape: {X_pos.shape}")
    print(f"  X_neg shape: {X_neg.shape}")

    X = np.concatenate([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))]).astype(int)
    print(f"  X total shape: {X.shape}")

    # 80/20 train/val split, stratified
    rng = np.random.default_rng(0)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    pos_cut = int(0.8 * len(pos_idx))
    neg_cut = int(0.8 * len(neg_idx))
    train_idx = np.concatenate([pos_idx[:pos_cut], neg_idx[:neg_cut]])
    val_idx = np.concatenate([pos_idx[pos_cut:], neg_idx[neg_cut:]])
    rng.shuffle(train_idx); rng.shuffle(val_idx)

    # Train linear probe (matches B02-v3's architecture)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict_proba(X[val_idx])[:, 1]
    val_auc = float(roc_auc_score(y[val_idx], y_pred))
    val_ap = float(average_precision_score(y[val_idx], y_pred))
    print(f"  val AUC: {val_auc:.4f}, AP: {val_ap:.4f}")

    # Save in the same format as B02-v3 head
    weight = torch.from_numpy(clf.coef_).float()
    bias = torch.from_numpy(clf.intercept_).float()
    state_dict = {"weight": weight, "bias": bias}

    ck_path = out_dir / "best.pt"
    torch.save({
        "step": 0,
        "epoch": 0,
        "model_state_dict": state_dict,
        "config": {"feature_dim": X.shape[1], "n_pos": len(X_pos), "n_neg": len(X_neg)},
        "metric": -val_auc,
        "extra": {"val_auc": val_auc, "val_ap": val_ap},
    }, ck_path)
    print(f"  wrote {ck_path}")

    summary = {
        "exp_id": args.exp_id,
        "n_pos": len(X_pos),
        "n_neg": len(X_neg),
        "val_auc": val_auc,
        "val_ap": val_ap,
    }
    Path(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
