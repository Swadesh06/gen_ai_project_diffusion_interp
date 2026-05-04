#!/usr/bin/env python
"""Build a (SAE activations, label) dataset for detector training (Item 3).

Sources:
  - Cached `*.sae.pt` from attack runs (NSFW seed images). Bypassed images carry
    label=1 (NSFW); successful safety-checker blocks carry label=1 too (still NSFW
    seed, just not bypassed).
  - Cached `*.sae.pt` from a benign SAE-capture run (the partner script
    `gen_sae_benign.py`), label=0.

Output:
  - `<out_dir>/X_<hookpoint>.npy` — concatenated activations per hookpoint, mean-pooled
    over the spatial axes (so shape (N, D)).
  - `<out_dir>/y.npy` — labels (N,).
  - `<out_dir>/meta.json` — provenance per sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def collect(sae_root: Path, label: int, source_tag: str) -> list[tuple[Path, int, str]]:
    out = []
    for f in sorted(sae_root.glob("*.sae.pt")):
        out.append((f, label, source_tag))
    return out


def _load_one(arg):
    """Module-level worker for multiprocessing.Pool."""
    import torch as _torch

    fpath, label, tag, pool = arg
    try:
        payload = _torch.load(fpath, map_location="cpu", weights_only=False)
    except Exception as e:
        return (fpath, label, tag, None, repr(e))
    out_per_block = {}
    for hp, z in payload.items():
        if z is None:
            continue
        arr = z.float().numpy()
        spatial = tuple(range(arr.ndim - 1))
        v = arr.mean(axis=spatial) if pool == "mean" else arr.max(axis=spatial)
        out_per_block[hp] = v
    return (fpath, label, tag, out_per_block, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nsfw-sae-dirs", nargs="+", required=True,
                    help="one or more `outputs/<exp>/sae` dirs from attack runs (label=1)")
    ap.add_argument("--benign-sae-dirs", nargs="+", required=True,
                    help="one or more sae dirs from benign-capture runs (label=0)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pool", default="mean", choices=["mean", "max"])
    args = ap.parse_args()

    import numpy as np
    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: list[tuple[Path, int, str]] = []
    for d in args.nsfw_sae_dirs:
        samples.extend(collect(Path(d), 1, "nsfw"))
    for d in args.benign_sae_dirs:
        samples.extend(collect(Path(d), 0, "benign"))

    print(f"collected {len(samples)} sae files")
    if not samples:
        return 2

    # Parallel read with multiprocessing (mfs latency is the bottleneck).
    import multiprocessing as mp
    import time as _time

    args_list = [(str(f), label, tag, args.pool) for (f, label, tag) in samples]
    n_workers = min(16, max(2, mp.cpu_count() // 8))
    print(f"reading {len(args_list)} sae.pt files via {n_workers} mp workers ...", flush=True)
    per_block: dict[str, list] = {}
    labels: list[int] = []
    meta: list[dict] = []
    t0 = _time.time()

    with mp.Pool(n_workers) as pool:
        for i, (fpath, label, tag, blocks, err) in enumerate(
            pool.imap_unordered(_load_one, args_list, chunksize=4),
        ):
            if err is not None:
                print(f"skip {Path(fpath).name}: {err}", flush=True)
                continue
            for hp, v in blocks.items():
                per_block.setdefault(hp, []).append(v)
            labels.append(label)
            meta.append({"file": fpath, "label": label, "source": tag})
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(args_list)}] {_time.time()-t0:.1f}s", flush=True)

    for hp, lst in per_block.items():
        X = np.stack(lst).astype("float32")
        np.save(out_dir / f"X_{hp.replace('.', '_')}.npy", X)
        print(f"  X_{hp}: shape={X.shape}")
    np.save(out_dir / "y.npy", np.asarray(labels, dtype="int64"))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"y: {len(labels)} (n_pos={sum(labels)}, n_neg={len(labels) - sum(labels)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
