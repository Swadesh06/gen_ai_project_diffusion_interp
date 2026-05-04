"""Pytest fixtures and shared helpers."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def synth_activations(rng):
    """Synthetic SAE activations: (N, D) sparse-ish ReLU."""

    def make(n: int = 64, d: int = 128, sparsity: float = 0.05) -> np.ndarray:
        z = rng.standard_normal(size=(n, d)).astype("float32")
        mask = rng.uniform(size=z.shape) < sparsity
        out = np.where(mask, np.abs(z) * 3.0, 0.0)
        return out

    return make


@pytest.fixture
def benign_forget_split(rng, synth_activations):
    """A pair (forget, retain) with `forget` activations boosted on a known feature subset."""
    n, d = 200, 128
    forget = synth_activations(n=n, d=d, sparsity=0.05)
    retain = synth_activations(n=n, d=d, sparsity=0.05)
    forget_features = np.array([7, 11, 23, 31, 47, 59, 71, 83, 97, 113])
    forget[:, forget_features] += rng.uniform(2.0, 4.0, size=(n, forget_features.size)).astype("float32")
    return forget, retain, forget_features
