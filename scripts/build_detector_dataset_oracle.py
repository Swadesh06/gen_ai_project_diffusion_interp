#!/usr/bin/env python
"""Like build_detector_dataset.py, but labels each sample by ORACLE judgement
on the rendered image (NudeNet+Q16+safety_checker union) rather than by
prompt origin.

Reads `<sae_dir>/<seed>.sae.pt` AND the corresponding `<pre_dir>/<seed>.png`
sidecar `*.labels.json` (cpu_worker output) + `*.safety.json` (batch_safety output)
to determine label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def collect(exp_dir: Path, sae_subdir: str = "sae", img_subdir: str = "pre",
            require_oracle: bool = True) -> list[tuple[Path, int, str]]:
    """Returns (sae_pt_path, label, source_tag)."""
    sae_root = exp_dir / sae_subdir
    img_root = exp_dir / img_subdir
    if not sae_root.exists() or not img_root.exists():
        return []
    out = []
    for sae_path in sorted(sae_root.glob("*.sae.pt")):
        seed_str = sae_path.stem.replace(".sae", "")
        png = img_root / f"{seed_str}.png"
        lbl = png.with_suffix(png.suffix + ".labels.json")
        sf = png.with_suffix(png.suffix + ".safety.json")
        nsfw = False
        seen_oracle = False
        if lbl.exists():
            try:
                r = json.loads(lbl.read_text())
                seen_oracle = True
                if r.get("flagged_any") or r.get("nudenet", {}).get("flagged") or r.get("q16", {}).get("flagged"):
                    nsfw = True
            except Exception:
                pass
        if sf.exists():
            try:
                r = json.loads(sf.read_text())
                seen_oracle = True
                if r.get("flagged"):
                    nsfw = True
            except Exception:
                pass
        if require_oracle and not seen_oracle:
            continue
        out.append((sae_path, int(nsfw), exp_dir.name))
    return out


def _load_one(arg):
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
    ap.add_argument("--exp-dirs", nargs="+", required=True,
                    help="paths to outputs/<exp_id> dirs that contain sae/ + pre/")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pool", default="mean", choices=["mean", "max"])
    args = ap.parse_args()

    import multiprocessing as mp
    import time as _time

    import numpy as np

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for d in args.exp_dirs:
        sub = collect(Path(d))
        print(f"  {d}: {len(sub)} oracle-labelled samples ({sum(s[1] for s in sub)} pos)")
        samples.extend(sub)

    if not samples:
        print("no samples")
        return 2

    args_list = [(str(f), label, tag, args.pool) for (f, label, tag) in samples]
    n_workers = min(16, max(2, mp.cpu_count() // 8))
    print(f"reading {len(args_list)} sae.pt files via {n_workers} mp workers ...", flush=True)
    per_block: dict = {}
    labels = []
    meta = []
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
