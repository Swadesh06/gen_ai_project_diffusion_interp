#!/usr/bin/env python
"""Phase C-2 — build raw-activation + SAE-activation dataset from gen_with_raw_features outputs.

Reads:
  raw/<seed>.raw.pt  (dict[hookpoint -> (D,)])  → assembles X_raw_<hp>.npy
  sae/<seed>.sae.pt  (dict[hookpoint -> (D,)])  → assembles X_sae_<hp>.npy

Labels: positives (label=1) from i2p_nsfw exp_dir; negatives (label=0) from coco exp_dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load_one(arg):
    import torch as _torch

    fpath, label, kind = arg
    try:
        payload = _torch.load(fpath, map_location="cpu", weights_only=False)
    except Exception as e:
        return (fpath, label, None, repr(e))
    out = {hp: v.float().numpy() for hp, v in payload.items() if v is not None}
    return (fpath, label, out, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nsfw-dir", required=True, help="exp_dir with raw/ + sae/ for label=1")
    ap.add_argument("--benign-dir", required=True, help="exp_dir with raw/ + sae/ for label=0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import multiprocessing as mp
    import time

    import numpy as np

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nsfw_dir = Path(args.nsfw_dir)
    benign_dir = Path(args.benign_dir)

    samples_raw = []
    samples_sae = []
    for d, label in [(nsfw_dir, 1), (benign_dir, 0)]:
        for f in sorted((d / "raw").glob("*.raw.pt")):
            samples_raw.append((str(f), label, "raw"))
            sae_f = d / "sae" / (f.stem.replace(".raw", "") + ".sae.pt")
            if sae_f.exists():
                samples_sae.append((str(sae_f), label, "sae"))

    print(f"raw: {len(samples_raw)} (pos={sum(1 for s in samples_raw if s[1]==1)})")
    print(f"sae: {len(samples_sae)} (pos={sum(1 for s in samples_sae if s[1]==1)})")

    n_workers = min(16, max(2, mp.cpu_count() // 8))

    def collect(samples, prefix):
        per_block = {}
        labels = []
        t0 = time.time()
        with mp.Pool(n_workers) as pool:
            for i, (fpath, label, blocks, err) in enumerate(
                pool.imap_unordered(_load_one, samples, chunksize=4),
            ):
                if err is not None:
                    print(f"  skip {Path(fpath).name}: {err}")
                    continue
                for hp, v in blocks.items():
                    per_block.setdefault(hp, []).append(v)
                labels.append(label)
                if (i + 1) % 100 == 0:
                    print(f"  {prefix}: [{i+1}/{len(samples)}] {time.time()-t0:.1f}s", flush=True)
        for hp, lst in per_block.items():
            X = np.stack(lst).astype("float32")
            np.save(out_dir / f"X_{prefix}_{hp.replace('.', '_')}.npy", X)
            print(f"  X_{prefix}_{hp}: shape={X.shape}")
        return labels

    labels = collect(samples_raw, "raw")
    np.save(out_dir / "y.npy", np.asarray(labels, dtype="int64"))

    labels_sae = collect(samples_sae, "sae")
    if labels_sae != labels:
        print(f"WARN: raw/sae label lists differ: raw={len(labels)}, sae={len(labels_sae)}")
    print(f"y: {len(labels)} ({sum(labels)} pos / {len(labels)-sum(labels)} neg)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
