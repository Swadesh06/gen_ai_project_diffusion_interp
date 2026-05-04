"""SAE plumbing: load, hook, attribution."""

from dsi.sae.attribution import (
    activation_delta,
    cross_attack_overlap,
    detector_attribution,
    fisher_ratio,
    input_score,
    jaccard,
    topk_features,
)

__all__ = [
    "activation_delta", "cross_attack_overlap", "detector_attribution",
    "fisher_ratio", "input_score", "jaccard", "topk_features",
]
