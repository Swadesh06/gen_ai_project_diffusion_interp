"""Detection-triggered intervention pipeline + baselines.

Stage 1 = DSG Fisher-ratio. Stage 2 = Arad causal-intervention output score.
Patches: mean / zero / resample (apply_patch dispatches).
"""

from dsi.interventions.patches import apply_patch, mean_patch, resample_patch, zero_patch
from dsi.interventions.pipeline import FeaturePack, InterventionConfig, make_intervene_fn
from dsi.interventions.stage1_fisher import stage1_score, stage1_select
from dsi.interventions.stage2_causal import CausalScoreInputs, stage2_score, stage2_select, two_stage_select

__all__ = [
    "apply_patch", "mean_patch", "resample_patch", "zero_patch",
    "FeaturePack", "InterventionConfig", "make_intervene_fn",
    "stage1_score", "stage1_select",
    "CausalScoreInputs", "stage2_score", "stage2_select", "two_stage_select",
]
