#!/usr/bin/env python
"""D-4 cross-concept violence probe.

Train a logistic-regression detector on (violence raw vs coco-benign raw)
SAE-mean features at the 4 Surkov hookpoints. Compare with trained-on-nudity
detector to check if SAE features generalize across concept (Stage 1).

Inputs:
- outputs/raw_violence_n200/raw/*.raw.pt  → 200 violence-positive activations
- outputs/raw_coco_500/raw/*.raw.pt        → 500 coco-benign activations

Output:
- reports/D04_violence_probe_v1.md
- outputs/D04_violence_probe/results.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--violence-dir", default="outputs/raw_violence_n200/raw")
    ap.add_argument("--benign-dir", default="outputs/raw_coco_500/raw")
    ap.add_argument("--feature-source", choices=["raw", "sae"], default="raw")
    ap.add_argument("--n-violence", type=int, default=200)
    ap.add_argument("--n-benign", type=int, default=200)
    ap.add_argument("--out-dir", default="outputs/D04_violence_probe")
    args = ap.parse_args()

    import torch
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".raw.pt" if args.feature_source == "raw" else ".sae.pt"
    src_dir = "raw" if args.feature_source == "raw" else "sae"
    violence_dir = Path(args.violence_dir).parent / src_dir
    benign_dir = Path(args.benign_dir).parent / src_dir

    print(f"loading {args.feature_source} features")
    print(f"  violence: {violence_dir}")
    print(f"  benign:   {benign_dir}")

    def load(d, n, label):
        files = sorted(Path(d).glob(f"*{suffix}"))[:n]
        feats_per_hp = {hp: [] for hp in HOOKPOINTS}
        for f in files:
            data = torch.load(f, map_location="cpu", weights_only=False)
            for hp in HOOKPOINTS:
                if hp in data:
                    v = data[hp].float().numpy()
                    feats_per_hp[hp].append(v)
        for hp in HOOKPOINTS:
            feats_per_hp[hp] = np.stack(feats_per_hp[hp])
        return feats_per_hp, [label] * len(files)

    print("loading violence activations")
    X_v, y_v = load(violence_dir, args.n_violence, 1)
    print(f"  {len(y_v)} violence samples")

    print("loading benign activations")
    X_b, y_b = load(benign_dir, args.n_benign, 0)
    print(f"  {len(y_b)} benign samples")

    print("per-hookpoint AUC:")
    per_hp = {}
    for hp in HOOKPOINTS:
        X = np.concatenate([X_v[hp], X_b[hp]], axis=0)
        y = np.array(y_v + y_b)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(y))
        cut = int(0.8 * len(y))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[perm[:cut]], y[perm[:cut]])
        s = clf.predict_proba(X[perm[cut:]])[:, 1]
        auc = float(roc_auc_score(y[perm[cut:]], s))
        ap_ = float(average_precision_score(y[perm[cut:]], s))
        per_hp[hp] = {"auc": auc, "ap": ap_}
        print(f"  {hp:>10s}: AUC={auc:.4f} AP={ap_:.4f}")

    # All-hookpoint concat
    print("all-hookpoint concat:")
    X_all_v = np.concatenate([X_v[hp] for hp in HOOKPOINTS], axis=1)
    X_all_b = np.concatenate([X_b[hp] for hp in HOOKPOINTS], axis=1)
    X = np.concatenate([X_all_v, X_all_b], axis=0)
    y = np.array(y_v + y_b)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X[perm[:cut]], y[perm[:cut]])
    s = clf.predict_proba(X[perm[cut:]])[:, 1]
    concat_auc = float(roc_auc_score(y[perm[cut:]], s))
    concat_ap = float(average_precision_score(y[perm[cut:]], s))
    print(f"  concat:   AUC={concat_auc:.4f} AP={concat_ap:.4f}")

    results = {
        "feature_source": args.feature_source,
        "n_violence": int(len(y_v)),
        "n_benign": int(len(y_b)),
        "per_hookpoint": per_hp,
        "concat": {"auc": concat_auc, "ap": concat_ap},
    }
    (out_dir / f"results_{args.feature_source}.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out_dir / f'results_{args.feature_source}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
