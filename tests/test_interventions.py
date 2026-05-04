"""Patch primitives + Stage-2 + pipeline glue."""

from __future__ import annotations

import numpy as np
import pytest

from dsi.interventions.patches import apply_patch, mean_patch, resample_patch, zero_patch
from dsi.interventions.stage2_causal import (
    CausalScoreInputs,
    stage2_score,
    stage2_select,
    two_stage_select,
)


def test_zero_patch_zeros_indices():
    z = np.ones((4, 8))
    out = zero_patch(z, np.array([0, 3]))
    assert (out[:, [0, 3]] == 0).all()
    assert (out[:, [1, 2, 4, 5, 6, 7]] == 1).all()


def test_mean_patch_replaces_with_mu():
    z = np.zeros((3, 5))
    mu = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    out = mean_patch(z, mu, np.array([1, 4]))
    assert out[0, 1] == 20.0 and out[0, 4] == 50.0
    assert (out[:, [0, 2, 3]] == 0).all()


def test_resample_patch_uses_pool(rng):
    z = np.zeros((4, 6))
    pool = np.tile(np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]), (8, 1))
    out = resample_patch(z, pool, np.array([0, 2]), rng=np.random.default_rng(0))
    assert (out[:, [0, 2]] != 0).all()
    assert (out[:, [1, 3, 4, 5]] == 0).all()


def test_apply_patch_dispatch():
    z = np.zeros((2, 4))
    mu = np.array([1.0, 2.0, 3.0, 4.0])
    pool = np.ones((3, 4))
    idx = np.array([1])

    out_mean = apply_patch(z, idx, kind="mean", mu=mu)
    assert out_mean[0, 1] == 2.0

    out_zero = apply_patch(z, idx, kind="zero")
    assert out_zero[0, 1] == 0.0

    out_resample = apply_patch(z, idx, kind="resample", benign_pool=pool)
    assert out_resample[0, 1] == 1.0


def test_apply_patch_missing_args():
    z = np.zeros((1, 4))
    idx = np.array([0])
    with pytest.raises(ValueError):
        apply_patch(z, idx, kind="mean")
    with pytest.raises(ValueError):
        apply_patch(z, idx, kind="resample")
    with pytest.raises(ValueError):
        apply_patch(z, idx, kind="other")  # type: ignore


def test_stage2_score_shape():
    inp = CausalScoreInputs(
        p_intervened=np.array([[0.9, 0.8], [0.1, 0.2]]),
        p_baseline=np.array([0.1, 0.1]),
    )
    s = stage2_score(inp)
    assert s.shape == (2,)


def test_stage2_select_threshold():
    inp = CausalScoreInputs(
        p_intervened=np.array([[0.9, 0.9], [0.1, 0.1]]),
        p_baseline=np.array([0.0, 0.0]),
    )
    keep = stage2_select(inp, tau_out=0.5)
    assert keep.tolist() == [0]


def test_two_stage_select_intersection():
    stage1_idx = np.array([3, 7, 11])
    inp = CausalScoreInputs(
        p_intervened=np.array([[0.9], [0.05], [0.8]]),
        p_baseline=np.array([0.1]),
    )
    keep = two_stage_select(stage1_idx, inp, tau_out=0.2)
    assert keep.tolist() == [3, 11]
