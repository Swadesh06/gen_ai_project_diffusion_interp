"""Centralized configuration for DiffSafeSAE.

All paths and hyperparameter defaults live here. .env is loaded at import time.
Public API: `cfg = Config()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default)).expanduser().resolve()


def _env_str(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass
class Paths:
    repo: Path = REPO_ROOT
    data_root: Path = field(default_factory=lambda: _env_path("DATA_ROOT", "/workspace/datasets"))
    model_root: Path = field(default_factory=lambda: _env_path("MODEL_ROOT", "/workspace/.cache/huggingface"))
    sae_root: Path = field(default_factory=lambda: _env_path("SAE_ROOT", "/workspace/checkpoints/saes"))
    checkpoint_root: Path = field(default_factory=lambda: _env_path("CHECKPOINT_ROOT", "/workspace/checkpoints"))
    cache_root: Path = field(default_factory=lambda: _env_path("CACHE_ROOT", "/workspace/.cache"))
    output_root: Path = field(default_factory=lambda: _env_path("OUTPUT_ROOT", str(REPO_ROOT / "outputs")))
    log_root: Path = field(default_factory=lambda: REPO_ROOT / "logs")
    report_root: Path = field(default_factory=lambda: REPO_ROOT / "reports")
    paper_root: Path = field(default_factory=lambda: REPO_ROOT / "paper")

    def ensure(self) -> "Paths":
        for p in [
            self.data_root, self.model_root, self.sae_root, self.checkpoint_root,
            self.cache_root, self.output_root, self.log_root, self.report_root,
        ]:
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class WandbCfg:
    project: str = field(default_factory=lambda: _env_str("WANDB_PROJECT", "dsi-v1"))
    entity: str = field(default_factory=lambda: _env_str("WANDB_ENTITY", ""))
    api_key: str = field(default_factory=lambda: _env_str("WANDB_API_KEY", ""))
    mode: str = field(default_factory=lambda: _env_str("WANDB_MODE", "online"))


@dataclass
class HFCfg:
    token: str = field(default_factory=lambda: _env_str("HF_TOKEN", ""))


# ---- hyperparameter defaults ----------------------------------------------------

EPS_GRID_PIXEL: tuple[float, ...] = (2 / 255, 4 / 255, 8 / 255)
EPS_GRID_LATENT: tuple[float, ...] = (0.05, 0.1, 0.2)
EPS_GRID_EMBEDDING: tuple[float, ...] = (0.1, 0.5, 1.0)

AttackSpace = Literal["pixel", "latent", "embedding"]
DetectorRegime = Literal["em", "ft"]
InterventionPatch = Literal["mean", "zero", "resample"]
GatingMode = Literal["always", "on_detection"]


@dataclass
class AttackCfg:
    space: AttackSpace = "pixel"
    eps: float = 4 / 255
    n_steps: int = 40
    lr: float = 1.0 / 255
    batch_size: int = 4
    targeted: bool = False
    seed: int = 0


@dataclass
class DetectorCfg:
    regime: DetectorRegime = "em"
    head: Literal["linear", "mlp", "ensemble"] = "linear"
    em_steps: int = 2
    pool: Literal["mean", "max", "attn"] = "mean"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 20
    val_frac: float = 0.1
    ckpt_every_steps: int = 200
    keep_last_n_ckpts: int = 4


@dataclass
class InterventionCfg:
    patch: InterventionPatch = "mean"
    gating: GatingMode = "on_detection"
    stage1_tau_ratio_percentile: float = 95.0
    stage2_lambda_grid: tuple[float, ...] = (100.0, 250.0, 500.0)
    stage2_tau_out: float = 0.1
    benign_ref_size: int = 5000


@dataclass
class SAECfg:
    backend: Literal["surkov", "saeuron", "saemnesia", "custom"] = "surkov"
    hookpoints: tuple[str, ...] = ("down.2.1", "mid.0", "up.0.0", "up.0.1")
    expansion_factor: int = 16
    l0_target: int = 64


@dataclass
class ModelCfg:
    sdxl_turbo: str = "stabilityai/sdxl-turbo"
    sdxl_base: str = "stabilityai/stable-diffusion-xl-base-1.0"
    sd15: str = "runwayml/stable-diffusion-v1-5"
    sd15_fallback: str = "benjamin-paine/stable-diffusion-v1-5"
    sd3: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    safety_checker: str = "CompVis/stable-diffusion-safety-checker"
    clip_vision: str = "openai/clip-vit-large-patch14"


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    wandb: WandbCfg = field(default_factory=WandbCfg)
    hf: HFCfg = field(default_factory=HFCfg)
    attack: AttackCfg = field(default_factory=AttackCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    intervention: InterventionCfg = field(default_factory=InterventionCfg)
    sae: SAECfg = field(default_factory=SAECfg)
    model: ModelCfg = field(default_factory=ModelCfg)

    def __post_init__(self) -> None:
        self.paths.ensure()


cfg = Config()
