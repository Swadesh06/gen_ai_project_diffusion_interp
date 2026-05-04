"""Detection-triggered intervention pipeline.

Composes:
  - SAEHookManager (with intervene_fn) — runtime patching
  - detector — fires on observed activations
  - patches.apply_patch — produces the patched feature vector
  - feature set F_c — pre-selected via Stage 1 ∩ Stage 2

EM regime: detector observes the first k steps; on firing, patch from k+1 onward.
FT regime: detector observes the full trajectory; on firing, regenerate with patch on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from dsi.config import GatingMode, InterventionPatch
from dsi.interventions.patches import apply_patch

InterventionRegime = Literal["em", "ft"]


@dataclass
class FeaturePack:
    """The pre-computed F_c bundle used for intervention.

    `feat_idx_per_block`: per-hookpoint indices to patch.
    `mu_per_block`: per-hookpoint per-step benign mean (T x D) or per-hookpoint (D,).
    `pool_per_block`: per-hookpoint cached benign-pool activations for resample-patch.
    """

    feat_idx_per_block: dict[str, np.ndarray]
    mu_per_block: dict[str, np.ndarray] = field(default_factory=dict)
    pool_per_block: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class InterventionConfig:
    regime: InterventionRegime = "em"
    patch: InterventionPatch = "mean"
    gating: GatingMode = "on_detection"
    em_steps: int = 2


def make_intervene_fn(
    pack: FeaturePack,
    cfg: InterventionConfig,
    *,
    detector_fired_callback: Callable[[int], bool],
) -> Callable:
    """Build the per-step `intervene_fn(hookpoint, z, step) -> z_new | None`.

    `z` is a torch.Tensor; we move to CPU numpy to apply the patch then move back.
    """

    def fn(hookpoint: str, z, step: int):
        if cfg.gating == "on_detection" and not detector_fired_callback(step):
            return None
        if cfg.regime == "em" and step < cfg.em_steps:
            return None
        feat = pack.feat_idx_per_block.get(hookpoint)
        if feat is None or feat.size == 0:
            return None
        z_np = z.detach().cpu().numpy()
        if cfg.patch == "mean":
            mu = pack.mu_per_block[hookpoint]
            mu_t = mu[step] if mu.ndim == 2 and mu.shape[0] > step else mu.reshape(-1)
            z_new = apply_patch(z_np, feat, kind="mean", mu=mu_t)
        elif cfg.patch == "zero":
            z_new = apply_patch(z_np, feat, kind="zero")
        elif cfg.patch == "resample":
            pool = pack.pool_per_block[hookpoint]
            z_new = apply_patch(z_np, feat, kind="resample", benign_pool=pool)
        else:
            raise ValueError(f"Unknown patch kind: {cfg.patch}")
        import torch

        return torch.as_tensor(z_new, device=z.device, dtype=z.dtype)

    return fn
