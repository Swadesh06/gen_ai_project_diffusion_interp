"""VAE-latent PGD: optimize the SDXL latent z (4×64×64); decode through the VAE; evaluate."""

from __future__ import annotations

from dataclasses import dataclass

from dsi.config import EPS_GRID_LATENT


@dataclass
class LatentPGDConfig:
    eps: float = 0.1
    n_steps: int = 40
    lr: float = 0.005
    targeted: bool = False
    eps_grid: tuple[float, ...] = EPS_GRID_LATENT


def pgd_step_latent(
    z_t,
    y,
    *,
    decode_fn,
    safety_logit_fn,
    eps: float,
    lr: float,
    targeted: bool = True,
):
    """One PGD step in latent space. `decode_fn(z) → image (B,3,H,W) in [0,1])`."""
    import torch
    import torch.nn.functional as F

    z = z_t.detach().clone().requires_grad_(True)
    img = decode_fn(z)
    logits = safety_logit_fn(img).float()
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, z)[0]
    sign = grad.sign()
    direction = -1.0 if targeted else 1.0
    z_new = z.detach() + direction * lr * sign
    delta = torch.clamp(z_new - z_t, min=-eps, max=eps)
    return z_t + delta


def pgd_attack_latent(z0, y, *, decode_fn, safety_logit_fn, cfg: LatentPGDConfig):
    import torch

    z = z0.detach().clone()
    for _ in range(cfg.n_steps):
        z = pgd_step_latent(
            z, y,
            decode_fn=decode_fn, safety_logit_fn=safety_logit_fn,
            eps=cfg.eps, lr=cfg.lr, targeted=cfg.targeted,
        )
    delta = (z - z0).detach()
    return z, delta.flatten(1).norm(p=float("inf"), dim=1)
