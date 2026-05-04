"""FlowGuard baseline: linear latent decoding (2026, arXiv:2604.07879)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlowGuardConfig:
    decoder_dim: int = 4
