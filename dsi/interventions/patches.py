"""Patching primitives: mean / zero / resample.

Pure-Python on activation tensors. The runtime invocation lives in interventions/pipeline.py
which calls these from a SAE hook callback (see sae/hooks.py).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

PatchKind = Literal["mean", "zero", "resample"]


def mean_patch(z: np.ndarray, mu: np.ndarray, feat_idx: np.ndarray) -> np.ndarray:
    """Replace `z[..., feat_idx]` with `mu[feat_idx]` (broadcast over leading dims).

    `z`: (B, ..., D) activations. `mu`: (D,) per-feature benign mean.
    `feat_idx`: indices to overwrite.
    """
    out = z.copy()
    out[..., feat_idx] = mu[feat_idx]
    return out


def zero_patch(z: np.ndarray, feat_idx: np.ndarray) -> np.ndarray:
    out = z.copy()
    out[..., feat_idx] = 0.0
    return out


def resample_patch(
    z: np.ndarray,
    benign_pool: np.ndarray,
    feat_idx: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Replace `z[..., feat_idx]` with values drawn uniformly from `benign_pool`.

    `benign_pool`: (M, D) cached benign activations. For each row of `z`,
    pick one random row of `benign_pool` and copy the selected feats from it.
    """
    rng = rng if rng is not None else np.random.default_rng()
    out = z.copy()
    n = out.shape[0]
    pool_n = benign_pool.shape[0]
    pick = rng.integers(0, pool_n, size=n)
    for i, p in enumerate(pick):
        out[i, ..., feat_idx] = benign_pool[p, ..., feat_idx]
    return out


def apply_patch(
    z: np.ndarray,
    feat_idx: np.ndarray,
    *,
    kind: PatchKind = "mean",
    mu: np.ndarray | None = None,
    benign_pool: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Dispatch over patch kind. Exhaustive switch per CLAUDE.md §11."""
    if kind == "mean":
        if mu is None:
            raise ValueError("mean_patch requires `mu`")
        return mean_patch(z, mu, feat_idx)
    if kind == "zero":
        return zero_patch(z, feat_idx)
    if kind == "resample":
        if benign_pool is None:
            raise ValueError("resample_patch requires `benign_pool`")
        return resample_patch(z, benign_pool, feat_idx, rng=rng)
    raise ValueError(f"Unknown patch kind: {kind}")
