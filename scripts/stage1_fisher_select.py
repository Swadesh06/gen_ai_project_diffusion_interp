#!/usr/bin/env python
"""Item 5 / Contribution 4 Stage 1 — DSG-style Fisher-ratio feature selection.

Reads the per-block X_<hookpoint>.npy + y.npy emitted by build_detector_dataset.py.
For each hookpoint:
  - Splits into NSFW (y=1) and benign (y=0).
  - Computes s_forget(f) = E[z_f^2 | NSFW], s_retain(f) = E[z_f^2 | benign].
  - ratio(f) = s_forget(f) / (s_retain(f) + eps).
  - Keeps features above the `tau_ratio_percentile`-th percentile on retain.

Output:
  - <out_dir>/stage1_<hookpoint>.json — {kept_indices, ratios_at_kept, scores_*}.
  - <out_dir>/stage1_summary.json — counts per block.
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
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tau-percentile", type=float, default=95.0)
    args = ap.parse_args()

    import numpy as np

    from dsi.interventions.stage1_fisher import stage1_score, stage1_select

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y = np.load(data_dir / "y.npy")
    print(f"y: {len(y)} samples, n_pos={int(y.sum())}, n_neg={int((1 - y).sum())}")

    summary = {"data_dir": str(data_dir), "tau_percentile": args.tau_percentile, "blocks": {}}
    for X_path in sorted(data_dir.glob("X_*.npy")):
        hp = X_path.stem.removeprefix("X_").replace("_", ".")
        X = np.load(X_path).astype("float32")
        z_forget = X[y == 1]
        z_retain = X[y == 0]
        diag = stage1_score(z_forget, z_retain)
        kept = stage1_select(z_forget, z_retain, tau_ratio_percentile=args.tau_percentile)
        summary["blocks"][hp] = {
            "d_in": int(X.shape[1]),
            "n_forget": int(len(z_forget)),
            "n_retain": int(len(z_retain)),
            "n_kept": int(len(kept)),
            "kept_top_ratios": [float(diag["ratio"][k]) for k in kept[:20].tolist()],
        }
        out = {
            "hookpoint": hp,
            "kept_indices": kept.tolist(),
            "kept_ratios": diag["ratio"][kept].tolist(),
            "kept_forget_scores": diag["forget"][kept].tolist(),
            "kept_retain_scores": diag["retain"][kept].tolist(),
        }
        (out_dir / f"stage1_{hp.replace('.', '_')}.json").write_text(json.dumps(out, indent=2))
        print(f"  {hp}: d_in={X.shape[1]} → kept {len(kept)} features")
    (out_dir / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
