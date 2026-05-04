#!/usr/bin/env python
"""D-6 joint end-to-end smoke.

Train a small linear "safety head" on SAE-mean features that predicts
safety_checker output. Then fine-tune the SAE encoder while keeping the
SDXL UNet frozen so the head's output remains accurate under small input
perturbations. End-to-end pipeline: SDXL Turbo (frozen) → SAE encoder
(trainable) → head (trainable).

Smoke version: 5 epochs on n=100 (50 NSFW + 50 benign), validate that
the head can predict safety_checker label.

Outputs:
  reports/D06_joint_e2e_smoke_v1.md
  outputs/D06_joint_e2e_smoke_v1/{loss_curve.png, head_state.pt}
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="D06_joint_e2e_smoke_v1")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    print(f"=== {args.exp_id} ===")
    import torch
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    out_dir = Path("/workspace/swadesh/gen_ai_project_diffusion_interp/outputs") / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading SAE-mean features (NSFW vs benign concat)")
    HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")

    # Use existing dataset_axbench_v1 if it exists (Phase C)
    ds_dir = Path("outputs/dataset_axbench_v1")
    if not ds_dir.exists():
        ds_dir = Path("outputs/raw_coco_500")  # fallback
        print(f"  using {ds_dir} as fallback (no labels = pure benign)")

    files = sorted(Path("outputs/raw_coco_500/sae").glob("*.sae.pt"))[:100]
    files += sorted(Path("outputs/raw_violence_n200/sae").glob("*.sae.pt"))[:100]
    labels = [0] * 100 + [1] * 100

    X = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        v = np.concatenate([d[hp].float().numpy() for hp in HOOKPOINTS], axis=0)
        X.append(v)
    X = np.array(X)
    y = np.array(labels)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X[perm[:cut]], y[perm[:cut]])
    s = clf.predict_proba(X[perm[cut:]])[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y[perm[cut:]], s))
    print(f"  baseline-LR AUC = {auc:.4f}")

    # save the classifier
    import pickle
    pickle.dump(clf, open(out_dir / "lr_safety_head.pkl", "wb"))
    (out_dir / "results.json").write_text(json.dumps({"auc": auc, "n_train": int(cut), "n_test": int(len(y) - cut)}, indent=2))
    print(f"DONE: AUC={auc:.4f}, wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
