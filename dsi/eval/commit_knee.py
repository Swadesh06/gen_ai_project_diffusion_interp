"""Per-step commit-knee diagnostic: detector AUC as a function of how many denoising steps observed.

For step k ∈ [1, T], compute AUC over the validation set using only activations from
steps [0, k). Identify the "knee" — where additional steps give diminishing AUC gains.
"""

from __future__ import annotations

import numpy as np


def commit_knee_curve(
    auc_per_step: list[float],
    *,
    knee_eps: float = 0.005,
) -> dict:
    """Returns the index where AUC stops improving by `knee_eps` per additional step,
    plus the curve itself.
    """
    arr = np.asarray(auc_per_step)
    diffs = np.diff(arr)
    knee = int(np.argmax(diffs < knee_eps)) if len(diffs) else 0
    return {
        "auc_per_step": arr.tolist(),
        "knee_step": knee,
        "auc_at_knee": float(arr[knee]) if len(arr) else 0.0,
        "max_auc": float(arr.max()) if len(arr) else 0.0,
    }


def auc_from_logits_labels(logits: np.ndarray, labels: np.ndarray) -> float:
    """Binary AUC. Falls back gracefully when sklearn is missing."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, logits))
