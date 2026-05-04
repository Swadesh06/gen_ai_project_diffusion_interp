"""IGD baseline: predicted-noise classifier (Yang et al., 2025, arXiv:2508.03006)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IGDConfig:
    n_steps_observed: int = 3
