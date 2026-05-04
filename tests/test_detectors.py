"""Detector module smoke (CPU-only, no real data)."""

from __future__ import annotations

import torch

from dsi.detectors.sae_em import EMHeadConfig, LinearProbe, MLPHead, PerBlockEnsemble
from dsi.detectors.sae_ft import FTHead, FTHeadConfig
from dsi.detectors.train import smoke_train


def test_linear_probe_forward():
    head = LinearProbe(in_dim=8)
    x = torch.randn(4, 8)
    y = head(x)
    assert y.shape == (4,)


def test_mlp_head_forward():
    head = MLPHead(in_dim=16, hidden=32)
    y = head(torch.randn(2, 16))
    assert y.shape == (2,)


def test_per_block_ensemble_forward():
    ens = PerBlockEnsemble(in_dims={"a": 8, "b": 4})
    out = ens({"a": torch.randn(3, 8), "b": torch.randn(3, 4)})
    assert out.shape == (3,)


def test_em_config_defaults():
    c = EMHeadConfig()
    assert c.em_steps in (1, 2, 3)


def test_ft_head_pool_modes():
    for pool in ("mean", "max", "attn"):
        cfg = FTHeadConfig(in_dim=8, pool=pool, hidden=16, n_steps=3)
        head = FTHead(cfg)
        out = head(torch.randn(2, 3, 8))
        assert out.shape == (2,)


def test_smoke_train_runs():
    out = smoke_train(in_dim=8, n=32)
    assert "losses" in out
    assert len(out["losses"]) == 5
    assert out["losses"][-1] < out["losses"][0] or out["losses"][-1] < 1.0
