"""Intervention pipeline glue."""

from __future__ import annotations

import numpy as np
import torch

from dsi.interventions.pipeline import FeaturePack, InterventionConfig, make_intervene_fn


def test_make_intervene_fn_em_skips_early_steps():
    pack = FeaturePack(
        feat_idx_per_block={"down.2.1": np.array([0, 2])},
        mu_per_block={"down.2.1": np.array([5.0, 0.0, 7.0, 0.0])},
    )
    cfg = InterventionConfig(regime="em", patch="mean", gating="always", em_steps=2)
    fn = make_intervene_fn(pack, cfg, detector_fired_callback=lambda _s: True)
    z = torch.zeros(1, 4)
    out_step0 = fn("down.2.1", z, step=0)
    assert out_step0 is None
    out_step3 = fn("down.2.1", z, step=3)
    assert out_step3 is not None
    arr = out_step3.cpu().numpy()
    assert arr[0, 0] == 5.0
    assert arr[0, 2] == 7.0
    assert arr[0, 1] == 0.0


def test_make_intervene_fn_gating_skips_when_quiet():
    pack = FeaturePack(
        feat_idx_per_block={"down.2.1": np.array([0])},
        mu_per_block={"down.2.1": np.array([1.0, 0.0])},
    )
    cfg = InterventionConfig(regime="ft", patch="mean", gating="on_detection", em_steps=0)
    fn = make_intervene_fn(pack, cfg, detector_fired_callback=lambda _s: False)
    z = torch.zeros(1, 2)
    assert fn("down.2.1", z, step=5) is None


def test_make_intervene_fn_zero_patch():
    pack = FeaturePack(
        feat_idx_per_block={"a": np.array([1])},
    )
    cfg = InterventionConfig(regime="ft", patch="zero", gating="always", em_steps=0)
    fn = make_intervene_fn(pack, cfg, detector_fired_callback=lambda _s: True)
    z = torch.ones(2, 4)
    out = fn("a", z, step=0)
    assert out[0, 1] == 0.0
    assert out[0, 0] == 1.0
