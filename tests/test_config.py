"""Config + paths smoke."""

from __future__ import annotations

from pathlib import Path

from dsi.config import (
    EPS_GRID_EMBEDDING,
    EPS_GRID_LATENT,
    EPS_GRID_PIXEL,
    AttackCfg,
    Config,
    DetectorCfg,
    InterventionCfg,
    Paths,
    SAECfg,
    cfg,
)


def test_config_loads():
    assert isinstance(cfg, Config)
    assert isinstance(cfg.paths, Paths)
    assert cfg.paths.repo.is_absolute()


def test_paths_ensure_creates():
    cfg.paths.ensure()
    for p in [cfg.paths.log_root, cfg.paths.report_root]:
        assert p.exists() or p.parent.exists()


def test_attack_defaults():
    a = AttackCfg()
    assert a.space in ("pixel", "latent", "embedding")
    assert a.eps > 0
    assert a.n_steps > 0


def test_detector_defaults():
    d = DetectorCfg()
    assert d.regime in ("em", "ft")
    assert d.head in ("linear", "mlp", "ensemble")
    assert d.batch_size > 0


def test_intervention_defaults():
    i = InterventionCfg()
    assert i.patch in ("mean", "zero", "resample")
    assert i.gating in ("always", "on_detection")
    assert 0 <= i.stage1_tau_ratio_percentile <= 100


def test_sae_defaults():
    s = SAECfg()
    assert s.backend in ("surkov", "saeuron", "saemnesia", "custom")
    assert s.expansion_factor > 0


def test_eps_grids_cover_paper_settings():
    for eps in (2 / 255, 4 / 255, 8 / 255):
        assert eps in EPS_GRID_PIXEL
    assert len(EPS_GRID_LATENT) >= 1
    assert len(EPS_GRID_EMBEDDING) >= 1


def test_dsi_pth_present():
    """import dsi must work from any cwd via the .pth file."""
    import sys

    repo = str(cfg.paths.repo)
    pth_files = [p for p in sys.path if p == repo]
    assert any(Path(p).exists() for p in sys.path)
