"""DSG (Muhamed et al., COLM 2025) adapted to diffusion: Stage-1 only + dynamic gating.

This is our ablation against "two-stage + mean patch". Selection = Fisher ratio only;
intervention = clamp to negative reference; gating = dynamic SAE-detector firing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DSGAdaptedConfig:
    tau_ratio_percentile: float = 95.0
    clamp_value: float = -1.0
