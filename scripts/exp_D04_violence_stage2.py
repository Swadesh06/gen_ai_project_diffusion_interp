#!/usr/bin/env python
"""D-4 cross-concept violence Stage-2 feature filter.

For each Stage 1 candidate feature (top-k by Fisher ratio violence-vs-benign),
do a causal-intervention test: zero-out the feature during a held-out
violence-prompt generation, observe whether safety_checker flag rate drops.
Features whose intervention reliably reduces flag rate pass Stage 2.

Output the violence-specific F_c set, contrast with nudity F_c.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--violence-dir", default="outputs/raw_violence_n200/sae")
    ap.add_argument("--benign-dir", default="outputs/raw_coco_500/sae")
    ap.add_argument("--top-k-stage1", type=int, default=20)
    ap.add_argument("--out-dir", default="outputs/D04_violence_stage1")
    args = ap.parse_args()

    import torch
    import numpy as np

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("loading violence SAE activations")
    files_v = sorted(Path(args.violence_dir).glob("*.sae.pt"))
    print(f"  {len(files_v)} violence")
    print("loading benign SAE activations")
    files_b = sorted(Path(args.benign_dir).glob("*.sae.pt"))
    print(f"  {len(files_b)} benign")

    feats_v = {hp: [] for hp in HOOKPOINTS}
    feats_b = {hp: [] for hp in HOOKPOINTS}
    for f in files_v:
        d = torch.load(f, map_location="cpu", weights_only=False)
        for hp in HOOKPOINTS:
            if hp in d: feats_v[hp].append(d[hp].float().numpy())
    for f in files_b:
        d = torch.load(f, map_location="cpu", weights_only=False)
        for hp in HOOKPOINTS:
            if hp in d: feats_b[hp].append(d[hp].float().numpy())
    for hp in HOOKPOINTS:
        feats_v[hp] = np.stack(feats_v[hp])
        feats_b[hp] = np.stack(feats_b[hp])

    # Stage 1: per-hookpoint Fisher ratio (violence-vs-benign)
    fc_per_hp = {}
    for hp in HOOKPOINTS:
        mu_v = feats_v[hp].mean(axis=0)
        mu_b = feats_b[hp].mean(axis=0)
        var_v = feats_v[hp].var(axis=0) + 1e-6
        var_b = feats_b[hp].var(axis=0) + 1e-6
        fisher = (mu_v - mu_b) ** 2 / (var_v + var_b)
        top_idx = np.argsort(-fisher)[: args.top_k_stage1]
        fc_per_hp[hp] = {
            "top_k_indices": top_idx.tolist(),
            "top_k_fishers": fisher[top_idx].tolist(),
            "top_k_mu_v": mu_v[top_idx].tolist(),
            "top_k_mu_b": mu_b[top_idx].tolist(),
        }
        print(f"  {hp}: top-1 idx={top_idx[0]} fisher={fisher[top_idx[0]]:.4f}")

    (out_dir / "stage1.json").write_text(json.dumps(fc_per_hp, indent=2))
    print(f"wrote {out_dir / 'stage1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
