"""Stage 2: Arad et al. 2025 causal-intervention output score.

For each Stage-1 surviving feature f, do a counterfactual-generation experiment:
  S_out(f, c) = Pr_clsf[c | gen(p_neutral; z_f += λ)] − Pr_clsf[c | gen(p_neutral)]

The actual generation needs a GPU and a pipeline; we expose the math here as a
score-aggregation function over already-collected counterfactual classifier outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CausalScoreInputs:
    """Per-feature classifier outputs for stage-2 scoring.

    `p_intervened`: (n_features, n_neutral_prompts) classifier P(c | intervened generation).
    `p_baseline`:   (n_neutral_prompts,) classifier P(c | un-intervened baseline generation).
    """

    p_intervened: np.ndarray
    p_baseline: np.ndarray


def stage2_score(inputs: CausalScoreInputs) -> np.ndarray:
    """Returns per-feature S_out averaged over the neutral prompt set."""
    delta = inputs.p_intervened - inputs.p_baseline[None, :]
    return delta.mean(axis=1)


def stage2_select(
    inputs: CausalScoreInputs,
    *,
    tau_out: float = 0.1,
) -> np.ndarray:
    """Returns indices where the score exceeds `tau_out`."""
    s = stage2_score(inputs)
    return np.where(s > tau_out)[0]


def two_stage_select(
    stage1_idx: np.ndarray,
    inputs_per_stage1_feature: CausalScoreInputs,
    *,
    tau_out: float = 0.1,
) -> np.ndarray:
    """Composes Stage-1 ∩ Stage-2.

    `stage1_idx` is the set of Stage-1 survivors (indices into the full feature space).
    `inputs_per_stage1_feature.p_intervened` is shaped (len(stage1_idx), n_prompts).
    Returns indices in the full feature space (subset of stage1_idx).
    """
    keep_local = stage2_select(inputs_per_stage1_feature, tau_out=tau_out)
    return stage1_idx[keep_local]
