"""SAE module smoke."""

from __future__ import annotations

import torch

from dsi.sae.load import SAEConfig, SparseAutoencoder, _normalize_state_dict


def test_sae_forward_shapes():
    sae = SparseAutoencoder(d_in=16, d_hidden=64)
    x = torch.randn(3, 16)
    x_hat, z = sae(x)
    assert z.shape == (3, 64)
    assert x_hat.shape == (3, 16)


def test_sae_topk_sparsity():
    sae = SparseAutoencoder(d_in=8, d_hidden=32, top_k=4)
    x = torch.randn(5, 8)
    _, z = sae(x)
    n_active = (z > 0).sum(dim=-1)
    assert (n_active <= 4).all()


def test_sae_jumprelu_threshold():
    sae = SparseAutoencoder(d_in=4, d_hidden=8, jump_relu=True)
    sae.threshold.data.fill_(10.0)
    _, z = sae(torch.randn(2, 4))
    assert (z == 0).all()


def test_normalize_state_dict_surkov():
    sd = {"encoder.weight": torch.zeros(16, 64), "encoder.bias": torch.zeros(64),
          "decoder.weight": torch.zeros(64, 16), "decoder.bias": torch.zeros(16)}
    out = _normalize_state_dict(sd, "surkov")
    assert {"W_enc", "b_enc", "W_dec", "b_dec"} <= out.keys()


def test_normalize_state_dict_saeuron_passthrough():
    sd = {"W_enc": torch.zeros(16, 64), "b_enc": torch.zeros(64),
          "W_dec": torch.zeros(64, 16), "b_dec": torch.zeros(16),
          "threshold": torch.zeros(64)}
    out = _normalize_state_dict(sd, "saeuron")
    assert "threshold" in out


def test_sae_config_defaults():
    c = SAEConfig(d_in=8, d_hidden=64, backend="surkov", hookpoint="down.2.1", expansion_factor=8)
    assert c.expansion_factor == 8
