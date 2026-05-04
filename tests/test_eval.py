"""ASR + commit-knee tests (pure NumPy)."""

from __future__ import annotations

import numpy as np
import pytest

from dsi.eval.asr import asr_simple, asr_with_oracle
from dsi.eval.commit_knee import auc_from_logits_labels, commit_knee_curve


def test_asr_simple():
    assert asr_simple(["bypass", "bypass", "blocked", "unknown"]) == 0.5
    assert asr_simple([]) == 0.0


def test_asr_with_oracle():
    safety_flagged = [False, True, False, True]
    oracle = [True, True, False, True]
    res = asr_with_oracle(safety_flagged, oracle)
    assert res.bypassed == 1
    assert res.blocked == 2
    assert res.n == 4
    assert res.asr == 0.25


def test_asr_with_oracle_length_mismatch_raises():
    with pytest.raises(ValueError):
        asr_with_oracle([True], [True, False])


def test_commit_knee_curve_basic():
    aucs = [0.6, 0.75, 0.8, 0.81, 0.815, 0.816]
    out = commit_knee_curve(aucs, knee_eps=0.01)
    assert 0 <= out["knee_step"] <= len(aucs)
    assert out["max_auc"] == max(aucs)


def test_auc_perfect_separation():
    logits = np.array([0.1, 0.2, 0.9, 0.95])
    labels = np.array([0, 0, 1, 1])
    assert auc_from_logits_labels(logits, labels) == 1.0
