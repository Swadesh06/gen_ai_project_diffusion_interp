"""SAeUron repurposed as a flag: any feature in the unlearned set above a threshold = unsafe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SAeUronBlockingConfig:
    feat_idx: np.ndarray = None  # indices of the SAeUron-selected features
    threshold: float = 1.0


def saeuron_block_score(z: np.ndarray, cfg: SAeUronBlockingConfig) -> np.ndarray:
    """Per-sample max activation over the SAeUron feature set."""
    if cfg.feat_idx is None or cfg.feat_idx.size == 0:
        return np.zeros(z.shape[0])
    return z[..., cfg.feat_idx].max(axis=-1)
