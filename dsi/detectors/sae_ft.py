"""Full-trajectory (FT) detector head: classifier consumes pooled activations across all steps.

Pooling options: mean / max / attention-pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = object  # type: ignore

PoolKind = Literal["mean", "max", "attn"]


@dataclass
class FTHeadConfig:
    in_dim: int = 65536
    pool: PoolKind = "mean"
    hidden: int = 512
    n_steps: int = 4
    n_attn_heads: int = 1


class PoolReducer(nn.Module):
    def __init__(self, kind: PoolKind, in_dim: int, n_attn_heads: int = 1):
        super().__init__()
        self.kind = kind
        if kind == "attn":
            self.attn = nn.MultiheadAttention(in_dim, n_attn_heads, batch_first=True)
            self.q = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)

    def forward(self, x):
        if self.kind == "mean":
            return x.mean(dim=1)
        if self.kind == "max":
            return x.max(dim=1).values
        if self.kind == "attn":
            B = x.shape[0]
            q = self.q.expand(B, -1, -1)
            out, _ = self.attn(q, x, x)
            return out.squeeze(1)
        raise ValueError(f"Unknown pool kind: {self.kind}")


class FTHead(nn.Module):
    """Pool over time, then MLP head."""

    def __init__(self, cfg: FTHeadConfig):
        super().__init__()
        self.pool = PoolReducer(cfg.pool, cfg.in_dim)
        self.fc1 = nn.Linear(cfg.in_dim, cfg.hidden)
        self.fc2 = nn.Linear(cfg.hidden, 1)

    def forward(self, x):
        # x: (B, T, D) — T = n_steps
        h = self.pool(x)
        return self.fc2(F.relu(self.fc1(h))).squeeze(-1)
