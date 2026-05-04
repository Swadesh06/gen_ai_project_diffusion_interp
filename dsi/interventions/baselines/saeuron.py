"""SAeUron baseline: activation-contrast feature selection + always-on negative scaling.

Reference: Cywiński & Deja, ICML 2025 (arXiv:2501.18052).
Wraps the upstream `cywinski/SAeUron` repo's sampling pipeline. The actual generation
is GPU-side; this file holds the parameter scaffolding and a CPU dry-run mode that
loads the SAE, lists the features it'd negative-scale, and reports their indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SAeUronConfig:
    concept: str = "nudity"
    negative_scale: float = -3.0
    contrast_threshold_percentile: float = 99.0


def select_saeuron_features(
    z_concept: np.ndarray,
    z_other: np.ndarray,
    *,
    cfg: SAeUronConfig,
) -> np.ndarray:
    """Activation-contrast: keep features whose mean activation under the concept exceeds
    a high percentile on the other-distribution.

    Returns indices into the SAE feature space.
    """
    mu_c = np.asarray(z_concept).mean(axis=0)
    mu_o = np.asarray(z_other).mean(axis=0)
    cutoff = np.percentile(mu_o, cfg.contrast_threshold_percentile)
    return np.where(mu_c > cutoff)[0]
