#!/usr/bin/env python
"""Compute per-feature benign mean over the SAE-captured benign dataset.

Output: <out>/mu.npz with `mu_<hookpoint>` arrays of shape (D,) — the per-feature
mean activation over the spatial-mean-pooled dataset, label=0 only.

Together with the F_c (Stage 1 ∩ Stage 2) feature index json, this is the bundle
consumed by `run_intervention.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np

    data_dir = Path(args.data_dir)
    y = np.load(data_dir / "y.npy")
    mask = (y == 0)

    out_path = Path(args.out) if args.out else data_dir / "mu.npz"
    arrays = {}
    for X_path in sorted(data_dir.glob("X_*.npy")):
        hp = X_path.stem.removeprefix("X_").replace("_", ".")
        X = np.load(X_path).astype("float32")
        mu = X[mask].mean(axis=0)
        arrays[f"mu_{X_path.stem.removeprefix('X_')}"] = mu
        print(f"  {hp}: mu shape={mu.shape}")
    np.savez(out_path, **arrays)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
