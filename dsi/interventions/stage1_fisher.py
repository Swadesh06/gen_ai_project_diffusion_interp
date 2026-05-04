"""Stage 1: DSG-style Fisher-ratio feature selection.

For each feature f:  s_forget(f) / s_retain(f).  Keep features above the
`tau_ratio_percentile`-th percentile on the *retain* distribution (matches DSG's
"keep features above the noise floor on retain").
"""

from __future__ import annotations

import numpy as np

from dsi.sae.attribution import fisher_ratio, input_score


def stage1_select(
    z_forget: np.ndarray,
    z_retain: np.ndarray,
    *,
    tau_ratio_percentile: float = 95.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Returns indices of features whose Fisher ratio exceeds the percentile threshold.

    The threshold is set on the retain distribution: any feature whose ratio is in the
    top (100 - tau)% on retain is kept. This matches DSG §3.
    """
    ratio = fisher_ratio(z_forget, z_retain, eps=eps)
    cutoff = np.percentile(ratio, tau_ratio_percentile)
    return np.where(ratio >= cutoff)[0]


def stage1_score(z_forget: np.ndarray, z_retain: np.ndarray) -> dict[str, np.ndarray]:
    """Diagnostic dict: {forget, retain, ratio} per feature."""
    sf = input_score(z_forget)
    sr = input_score(z_retain)
    return {"forget": sf, "retain": sr, "ratio": sf / (sr + 1e-8)}
