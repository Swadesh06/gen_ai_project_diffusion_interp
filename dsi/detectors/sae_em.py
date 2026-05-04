"""Early-monitor (EM) detector head: classifier consumes activations from the first k steps.

Architectures:
  - LinearProbe         single linear layer
  - MLPHead             2-layer MLP (default hidden=512, ReLU)
  - PerBlockEnsemble    one head per SAE hookpoint, averaged at the logit
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = object  # type: ignore


@dataclass
class EMHeadConfig:
    in_dim: int = 65536
    em_steps: int = 2
    hidden: int = 512
    pool: str = "mean"


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x))).squeeze(-1)


class PerBlockEnsemble(nn.Module):
    """One head per hookpoint. `forward(x)` expects a dict[hookpoint -> tensor]."""

    def __init__(self, in_dims: dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict({k: LinearProbe(d) for k, d in in_dims.items()})

    def forward(self, xs: dict):
        outs = [self.heads[k](xs[k]) for k in self.heads]
        return torch.stack(outs, dim=-1).mean(dim=-1)
