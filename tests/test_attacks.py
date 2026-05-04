"""Attack scaffold tests — purely structural."""

from __future__ import annotations

import numpy as np

from dsi.attacks import (
    AttackBatchResult,
    AttackResult,
    EmbeddingPGDConfig,
    LatentPGDConfig,
    PixelPGDConfig,
    asr_from_verdicts,
    perturbation_norm,
    run_attack,
)
from dsi.data.types import Prompt


def test_attack_configs_have_eps_grids():
    for c in [PixelPGDConfig(), LatentPGDConfig(), EmbeddingPGDConfig()]:
        assert len(c.eps_grid) >= 1
        assert c.n_steps >= 1


def test_perturbation_norm_inf():
    x_pre = np.zeros((2, 3, 4, 4))
    x_post = np.full((2, 3, 4, 4), 0.04)
    n = perturbation_norm(x_pre, x_post, p=float("inf"))
    assert abs(n - 0.04) < 1e-6


def test_asr_from_verdicts():
    assert asr_from_verdicts([]) == 0.0
    assert asr_from_verdicts(["bypass"]) == 1.0
    assert asr_from_verdicts(["blocked"]) == 0.0
    assert asr_from_verdicts(["bypass", "blocked"]) == 0.5


def test_attack_batch_result_aggregates():
    p = Prompt(text="x", source="t", label="nsfw")
    r1 = AttackResult(prompt=p, verdict="bypass", perturbation_norm=0.04, ssim=0.9)
    r2 = AttackResult(prompt=p, verdict="blocked", perturbation_norm=0.02, ssim=0.95)
    bres = AttackBatchResult(space="pixel", eps=4 / 255, results=[r1, r2])
    assert bres.asr == 0.5
    assert abs(bres.mean_perturb - 0.03) < 1e-6
    assert abs(bres.mean_ssim - 0.925) < 1e-6


def test_run_attack_with_stub():
    prompts = [Prompt(text=f"p{i}", source="t", label="nsfw") for i in range(3)]

    def step(_p):
        return {"perturbed": np.zeros(4), "perturb_norm": 0.01, "ssim": 0.99}

    def evl(_x):
        return {"verdict": "bypass", "score": 0.0}

    out = run_attack(prompts, attack_step=step, eval_safety=evl, space="pixel", eps=4 / 255)
    assert out.asr == 1.0
    assert len(out.results) == 3
