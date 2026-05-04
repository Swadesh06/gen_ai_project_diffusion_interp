"""Attribution math, Stage-1 Fisher, patches — pure-NumPy CPU tests."""

from __future__ import annotations

import numpy as np

from dsi.interventions.stage1_fisher import stage1_score, stage1_select
from dsi.sae.attribution import (
    activation_delta,
    cross_attack_overlap,
    detector_attribution,
    fisher_ratio,
    input_score,
    jaccard,
    topk_features,
)


def test_input_score_shapes(synth_activations):
    z = synth_activations(n=32, d=64)
    s = input_score(z)
    assert s.shape == (64,)
    assert (s >= 0).all()


def test_input_score_label(rng):
    z = rng.standard_normal((100, 16)).astype("float32")
    label = rng.uniform(size=100) > 0.5
    s = input_score(z, label)
    assert s.shape == (16,)


def test_fisher_ratio(benign_forget_split):
    forget, retain, feat = benign_forget_split
    ratio = fisher_ratio(forget, retain)
    rank = np.argsort(-ratio)
    top10 = set(rank[:10].tolist())
    assert len(top10 & set(feat.tolist())) >= 7


def test_stage1_select_returns_known_features(benign_forget_split):
    forget, retain, feat = benign_forget_split
    idx = stage1_select(forget, retain, tau_ratio_percentile=92.0)
    overlap = len(set(idx.tolist()) & set(feat.tolist()))
    assert overlap >= 7
    assert idx.size <= forget.shape[1]


def test_stage1_score_diagnostic(benign_forget_split):
    forget, retain, _ = benign_forget_split
    diag = stage1_score(forget, retain)
    assert {"forget", "retain", "ratio"} <= diag.keys()
    assert diag["forget"].shape == diag["retain"].shape == diag["ratio"].shape


def test_activation_delta(synth_activations):
    z_b = synth_activations(n=32, d=16) + 0.5
    z_c = synth_activations(n=32, d=16)
    d = activation_delta(z_b, z_c)
    assert d.shape == (16,)


def test_topk_basic():
    s = np.array([1.0, -3.0, 2.5, 0.1, -7.0])
    idx = topk_features(s, k=2, by="abs")
    assert set(idx.tolist()) == {1, 4}


def test_jaccard_and_overlap():
    assert jaccard([1, 2, 3], [2, 3, 4]) == 0.5
    assert jaccard([], []) == 1.0
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    assert cross_attack_overlap(a, b) == 0.5


def test_detector_attribution_linear():
    z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    w = np.array([0.5, 0.0, -1.0])
    a = detector_attribution(z, w)
    assert a.shape == z.shape
    assert np.allclose(a[0], [0.5, 0.0, -3.0])


def test_detector_attribution_nonlinear():
    z = np.ones((2, 3))
    w = np.array([1.0, 1.0, 1.0])
    g = np.array([[0.5, 1.0, 2.0], [0.0, 0.0, 0.0]])
    a = detector_attribution(z, w, nonlinear_grad=g)
    assert np.allclose(a, g)
