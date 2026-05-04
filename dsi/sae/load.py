"""Load Surkov / SAeUron / SAEmnesia SAE checkpoints into a uniform `SparseAutoencoder` nn.Module.

Each backend has its own state-dict layout. We expose a single class plus per-backend
constructors so downstream code does not have to branch on the source.

Reference checkpoints:
  - Surkov: https://huggingface.co/surokpro2/Unboxing_SDXL_with_SAEs
  - SAeUron: https://huggingface.co/bcywinski/SAeUron
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dsi.config import cfg

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore

SAEBackend = Literal["surkov", "saeuron", "saemnesia"]


@dataclass
class SAEConfig:
    d_in: int
    d_hidden: int
    backend: SAEBackend
    hookpoint: str
    expansion_factor: int
    l0_target: int = 64


class SparseAutoencoder(nn.Module):
    """ReLU SAE: x -> ReLU(W_e x + b_e) -> W_d z + b_d.

    Top-k variants and JumpReLU variants are loadable via a kw flag.
    """

    def __init__(self, d_in: int, d_hidden: int, *, top_k: int | None = None, jump_relu: bool = False):
        if torch is None:
            raise ImportError("PyTorch required to instantiate SparseAutoencoder")
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.top_k = top_k
        self.jump_relu = jump_relu
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.zeros(d_hidden, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if jump_relu:
            self.threshold = nn.Parameter(torch.zeros(d_hidden))

    def encode(self, x):
        z_pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        if self.jump_relu:
            z = z_pre * (z_pre > self.threshold).float()
        else:
            z = F.relu(z_pre)
        if self.top_k is not None and self.top_k > 0 and self.top_k < self.d_hidden:
            vals, idx = torch.topk(z, self.top_k, dim=-1)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, idx, 1.0)
            z = z * mask
        return z

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def load_surkov_sae(hookpoint: str, root: Path | str | None = None) -> SparseAutoencoder:
    """Load one of Surkov et al.'s SDXL Turbo SAEs.

    `hookpoint` ∈ {"down.2.1", "mid.0", "up.0.0", "up.0.1"}.
    Looks for a state-dict at `root/<hookpoint>.safetensors` or `<hookpoint>.pt`.
    """
    base = Path(root) if root else cfg.paths.sae_root / "surkov"
    return _load_from_dir(base, hookpoint, backend="surkov")


def load_saeuron_sae(hookpoint: str, root: Path | str | None = None) -> SparseAutoencoder:
    """Load one of SAeUron's SD v1.5 SAEs."""
    base = Path(root) if root else cfg.paths.sae_root / "saeuron"
    return _load_from_dir(base, hookpoint, backend="saeuron")


def _resolve_checkpoint_path(base: Path, hookpoint: str, backend: SAEBackend) -> Path:
    """Map (backend, hookpoint) to the on-disk file.

    Surkov: `base/checkpoints/unet.<hookpoint>_*` / `final/state_dict.pth`.
    SAeUron: `base/unet.<hookpoint>/sae.safetensors`.
    """
    short_to_long = {
        "down.2.1": "unet.down_blocks.2.attentions.1",
        "mid.0":    "unet.mid_block.attentions.0",
        "up.0.0":   "unet.up_blocks.0.attentions.0",
        "up.0.1":   "unet.up_blocks.0.attentions.1",
        # SAeUron's SD v1.5 hookpoints are different:
        "up.1.1":   "unet.up_blocks.1.attentions.1",
        "up.1.2":   "unet.up_blocks.1.attentions.2",
    }
    long_name = short_to_long.get(hookpoint, hookpoint)

    if backend == "surkov":
        ck_root = base / "checkpoints"
        if ck_root.exists():
            for d in ck_root.iterdir():
                if d.is_dir() and d.name.startswith(long_name + "_"):
                    p = d / "final" / "state_dict.pth"
                    if p.exists():
                        return p
        for ext in ("safetensors", "pt", "pth"):
            cand = base / f"{hookpoint}.{ext}"
            if cand.exists():
                return cand
        for ext in ("safetensors", "pt", "pth"):
            cand = base / f"{long_name}.{ext}"
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No Surkov SAE checkpoint for hookpoint={hookpoint}")

    if backend == "saeuron":
        for sub in (long_name, hookpoint, f"unet.{hookpoint}"):
            p = base / sub / "sae.safetensors"
            if p.exists():
                return p
        for ext in ("safetensors", "pt"):
            cand = base / f"{hookpoint}.{ext}"
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No SAeUron SAE checkpoint for hookpoint={hookpoint}")

    if backend == "saemnesia":
        for ext in ("safetensors", "pt"):
            cand = base / f"{hookpoint}.{ext}"
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No SAEmnesia SAE checkpoint at {base} for {hookpoint}")

    raise ValueError(f"Unknown backend: {backend}")


def _load_from_dir(base: Path, hookpoint: str, backend: SAEBackend) -> SparseAutoencoder:
    if torch is None:
        raise ImportError("PyTorch required")
    path = _resolve_checkpoint_path(base, hookpoint, backend)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(path))
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

    sd_norm = _normalize_state_dict(sd, backend)
    d_in = sd_norm["W_enc"].shape[0]
    d_hidden = sd_norm["W_enc"].shape[1]
    sae = SparseAutoencoder(d_in=d_in, d_hidden=d_hidden, jump_relu="threshold" in sd_norm)
    missing, unexpected = sae.load_state_dict(sd_norm, strict=False)
    if missing:
        sae.zero_grad()
    return sae


def _normalize_state_dict(sd: dict, backend: SAEBackend) -> dict:
    """Map per-backend key conventions to the unified {W_enc, b_enc, W_dec, b_dec, threshold} schema."""
    out = {}
    KEY_MAP = {
        "surkov": {
            "encoder.weight": "W_enc",
            "encoder.bias": "b_enc",
            "decoder.weight": "W_dec",
            "decoder.bias": "b_dec",
        },
        "saeuron": {
            "W_enc": "W_enc",
            "b_enc": "b_enc",
            "W_dec": "W_dec",
            "b_dec": "b_dec",
            "threshold": "threshold",
        },
        "saemnesia": {
            "W_enc": "W_enc",
            "b_enc": "b_enc",
            "W_dec": "W_dec",
            "b_dec": "b_dec",
        },
    }
    keymap = KEY_MAP[backend]
    for k_src, v in sd.items():
        if k_src in keymap:
            out[keymap[k_src]] = v
            continue
        # Heuristic fallbacks for the common variants we have not pinned.
        kl = k_src.lower()
        if "enc" in kl and "weight" in kl:
            out["W_enc"] = v if v.shape[0] >= v.shape[1] else v.T
        elif "enc" in kl and "bias" in kl:
            out["b_enc"] = v
        elif "dec" in kl and "weight" in kl:
            out["W_dec"] = v if v.shape[0] <= v.shape[1] else v.T
        elif "dec" in kl and "bias" in kl:
            out["b_dec"] = v
        elif "thresh" in kl:
            out["threshold"] = v
    if "W_enc" not in out:
        raise ValueError(f"Could not map state dict to W_enc for backend={backend}; saw {sorted(sd.keys())[:8]}")
    if out["W_enc"].dim() == 2 and "W_dec" in out and out["W_dec"].dim() == 2:
        if out["W_enc"].shape[0] < out["W_enc"].shape[1]:
            pass
    return out
