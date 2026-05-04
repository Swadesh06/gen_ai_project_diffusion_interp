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

    per_block: dict[str, list] = {}
    labels: list[int] = []
    meta: list[dict] = []
    for f, label, tag in samples:
        try:
            payload = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"skip {f}: {e}")
            continue
        for hp, z in payload.items():
            if z is None:
                continue
            arr = z.float().numpy()
            spatial = tuple(range(arr.ndim - 1))
            v = arr.mean(axis=spatial) if args.pool == "mean" else arr.max(axis=spatial)
            per_block.setdefault(hp, []).append(v)
        labels.append(label)
        meta.append({"file": str(f), "label": label, "source": tag})

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
