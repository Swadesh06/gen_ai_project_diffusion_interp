"""Baseline-wrapper structural tests (no real model loading)."""

from __future__ import annotations

import numpy as np

from dsi.detectors.baselines.flowguard import FlowGuardConfig
from dsi.detectors.baselines.igd import IGDConfig
from dsi.detectors.baselines.saeuron_blocking import SAeUronBlockingConfig, saeuron_block_score
from dsi.interventions.baselines.dsg_adapted import DSGAdaptedConfig
from dsi.interventions.baselines.saemnesia import SAEmnesiaConfig
from dsi.interventions.baselines.saeuron import SAeUronConfig, select_saeuron_features


def test_baseline_configs_instantiate():
    assert IGDConfig().n_steps_observed > 0
    assert FlowGuardConfig().decoder_dim > 0
    assert DSGAdaptedConfig().clamp_value < 0
    assert SAEmnesiaConfig().concept_id == 0
    assert SAeUronConfig().contrast_threshold_percentile == 99.0


def test_select_saeuron_features_shape(rng):
    z_concept = rng.standard_normal((50, 64)).astype("float32")
    z_concept[:, [3, 11, 23]] += 5.0
    z_other = rng.standard_normal((100, 64)).astype("float32")
    cfg = SAeUronConfig(contrast_threshold_percentile=95.0)
    idx = select_saeuron_features(np.abs(z_concept), np.abs(z_other), cfg=cfg)
    assert isinstance(idx, np.ndarray)
    assert {3, 11, 23}.issubset(set(idx.tolist()))


def test_saeuron_block_score_empty():
    cfg = SAeUronBlockingConfig(feat_idx=np.array([]), threshold=1.0)
    z = np.ones((4, 16))
    s = saeuron_block_score(z, cfg)
    assert s.shape == (4,)
    assert (s == 0).all()


def test_saeuron_block_score_max():
    cfg = SAeUronBlockingConfig(feat_idx=np.array([0, 5]), threshold=1.0)
    z = np.zeros((2, 8))
    z[0, 5] = 7.0
    z[1, 0] = 3.0
    s = saeuron_block_score(z, cfg)
    assert s[0] == 7.0
    assert s[1] == 3.0
