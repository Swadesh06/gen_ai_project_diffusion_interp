"""Per-feature SAE attribution math.

CPU-runnable on cached activations. Three primitives:

  - input_score(z, label):  E[z^2 | label=1] — DSG-style correlational signal (= s_forget).
  - activation_delta(z_bypass, z_clean):  per-feature mean activation difference;
    paired-sample if same prompt, else cohort-level.
  - cross_attack_overlap(top_a, top_b, k=50):  Jaccard or hits@k between top-k feature sets
    surfaced by two different attack spaces.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

ArrayLike = np.ndarray


def input_score(z: ArrayLike, label: ArrayLike | None = None, *, agg: str = "mean") -> ArrayLike:
    """Per-feature score E[z^2 | label] (or E[z^2] if no label).

    z: (N, D) activations. label: (N,) binary mask (1 = forget concept).
    Returns a length-D vector.
    """
    z = np.asarray(z)
    if z.ndim != 2:
        z = z.reshape(z.shape[0], -1)
    z2 = z * z
    if label is None:
        if agg == "mean":
            return z2.mean(axis=0)
        if agg == "max":
            return z2.max(axis=0)
        raise ValueError(f"unknown agg: {agg}")
    label = np.asarray(label).astype(bool)
    if label.sum() == 0:
        return np.zeros(z2.shape[1], dtype=z2.dtype)
    return z2[label].mean(axis=0)


def fisher_ratio(z_forget: ArrayLike, z_retain: ArrayLike, eps: float = 1e-8) -> ArrayLike:
    """DSG Theorem 3.1 stand-in: ratio of E[z^2 | forget] to E[z^2 | retain]."""
    sf = input_score(z_forget)
    sr = input_score(z_retain)
    return sf / (sr + eps)


def activation_delta(z_bypass: ArrayLike, z_clean: ArrayLike) -> ArrayLike:
    """Mean per-feature shift on bypass relative to clean. Same shape (N, D)."""
    zb = np.asarray(z_bypass)
    zc = np.asarray(z_clean)
    if zb.ndim != 2:
        zb = zb.reshape(zb.shape[0], -1)
    if zc.ndim != 2:
        zc = zc.reshape(zc.shape[0], -1)
    return zb.mean(axis=0) - zc.mean(axis=0)


def topk_features(scores: ArrayLike, k: int = 50, *, by: str = "abs") -> ArrayLike:
    """Indices of the top-k features by `by` ∈ {"value", "abs"}."""
    s = np.asarray(scores)
    if by == "abs":
        s = np.abs(s)
    elif by != "value":
        raise ValueError(f"unknown by: {by}")
    if k >= s.size:
        return np.argsort(-s)
    return np.argpartition(-s, k - 1)[:k][np.argsort(-s[np.argpartition(-s, k - 1)[:k]])]


def jaccard(set_a: Iterable[int], set_b: Iterable[int]) -> float:
    a, b = set(int(x) for x in set_a), set(int(x) for x in set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def cross_attack_overlap(top_a: ArrayLike, top_b: ArrayLike) -> float:
    """Jaccard overlap between two top-k feature sets."""
    return jaccard(top_a.tolist(), top_b.tolist())


def detector_attribution(
    z: ArrayLike,
    weights: ArrayLike,
    *,
    nonlinear_grad: ArrayLike | None = None,
) -> ArrayLike:
    """Per-feature attribution of a linear (or linearized) detector.

    For a linear probe with weights `w`, attribution = z * w.
    For an MLP head, supply `nonlinear_grad` = ∂logit/∂z (computed elsewhere).
    Returns a (N, D) matrix.
    """
    z = np.asarray(z)
    if z.ndim != 2:
        z = z.reshape(z.shape[0], -1)
    w = np.asarray(weights).reshape(-1)
    if nonlinear_grad is None:
        return z * w[None, :]
    g = np.asarray(nonlinear_grad)
    return z * g
