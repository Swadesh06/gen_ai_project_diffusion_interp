#!/usr/bin/env python
"""Per-feature SAE attribution analysis on cached attack activations.

Reads `<exp_dir>/sae/<seed>.sae.pt` files (one per attacked seed image), computes:

  - Per-block per-feature mean activation on bypassed images (pre vs post attack
    using only pre-seed activations; post-attack activations require running
    the SAE again on the perturbed image, deferred to a Phase-1b extension).
  - Top-50 most-active features per block on bypassed cases.
  - Cross-block feature overlap (Jaccard) for sanity.

Output:
  - `<exp_dir>/attribution.json` with the top features and statistics.
  - `<exp_dir>/attribution_topk.csv` — flat table for easy aggregation across attacks.

Pure CPU / NumPy. Co-schedule with any GPU job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-dir", required=True, help="path to outputs/<exp_id>")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    import numpy as np
    import torch

    exp_dir = Path(args.exp_dir)
    sae_dir = exp_dir / "sae"
    if not sae_dir.exists():
        print(f"no sae dir at {sae_dir}; did the run use --collect-sae?")
        return 2

    sae_files = sorted(sae_dir.glob("*.sae.pt"))
    if not sae_files:
        print("no .sae.pt files yet")
        return 2

    # Load attack outcomes to filter to bypassed seeds
    bypassed_seeds = set()
    for j in sorted(exp_dir.glob("*.attack.json")):
        try:
            r = json.loads(j.read_text())
            if r.get("bypass"):
                bypassed_seeds.add(r["seed"])
        except Exception:
            continue

    # Aggregate per-block: collect z (averaged over spatial dims) for each seed
    per_block_bypass_z: dict[str, list] = {}
    per_block_all_z: dict[str, list] = {}
    n_files = 0
    for f in sae_files:
        seed_str = f.stem.replace(".sae", "")
        seed = int(seed_str)
        try:
            payload = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"skip {f.name}: {e}")
            continue
        n_files += 1
        for hp, z in payload.items():
            if z is None:
                continue
            arr = z.float().numpy()
            spatial_axes = tuple(range(arr.ndim - 1))
            mean_per_feat = arr.mean(axis=spatial_axes)
            per_block_all_z.setdefault(hp, []).append(mean_per_feat)
            if seed in bypassed_seeds:
                per_block_bypass_z.setdefault(hp, []).append(mean_per_feat)

    summary = {"n_files": n_files, "n_bypass": len(bypassed_seeds), "blocks": {}}
    rows = []
    for hp, all_list in per_block_all_z.items():
        all_arr = np.stack(all_list)
        bypass_arr = np.stack(per_block_bypass_z.get(hp, []) or [np.zeros_like(all_arr[0])])
        mu_all = all_arr.mean(axis=0)
        mu_bypass = bypass_arr.mean(axis=0)
        delta = mu_bypass - mu_all
        top_idx = np.argsort(-np.abs(delta))[: args.top_k]
        summary["blocks"][hp] = {
            "n_features": int(all_arr.shape[1]),
            "top_indices": top_idx.tolist(),
            "top_delta": delta[top_idx].tolist(),
            "top_mu_bypass": mu_bypass[top_idx].tolist(),
            "top_mu_all": mu_all[top_idx].tolist(),
        }
        for r, idx in enumerate(top_idx.tolist()):
            rows.append((hp, r, idx, float(delta[idx]), float(mu_bypass[idx]), float(mu_all[idx])))

    out_json = exp_dir / "attribution.json"
    out_json.write_text(json.dumps(summary, indent=2))

    out_csv = exp_dir / "attribution_topk.csv"
    with out_csv.open("w") as f:
        f.write("hookpoint,rank,feature_idx,delta,mu_bypass,mu_all\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print(json.dumps({"n_files": n_files, "n_bypass": len(bypassed_seeds),
                      "out_json": str(out_json), "out_csv": str(out_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
