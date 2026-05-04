"""SAEmnesia baseline: supervised SAE with one-to-one concept-neuron mapping.

Reference: Cassano et al., 2025 (arXiv:2509.21379). +9.22% on UnlearnCanvas vs. SAeUron.
Wraps upstream `OPTML-Group/SAEmnesia` sampling. CPU dry-run lists which SAE features
correspond to which concept ID per their concept-mapping JSON.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SAEmnesiaConfig:
    concept_id: int = 0
    clamp_value: float = 0.0
