"""FID wrapper around `clean-fid`.

`compute_fid(gen_dir, ref_dir)` — both directories of PNGs.
"""

from __future__ import annotations


def compute_fid(gen_dir: str, ref_dir: str, mode: str = "clean") -> float:
    from cleanfid import fid

    return float(fid.compute_fid(gen_dir, ref_dir, mode=mode, num_workers=2))
