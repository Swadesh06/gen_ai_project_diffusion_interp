"""Pixel-space PGD against `CompVis/stable-diffusion-safety-checker`.

Standard ℓ∞ PGD (Madry et al.). Inputs are 3x512x512 images; the safety checker
internally pre-processes via CLIP image encoder. We attack post-preprocessing on the
raw pixel tensor, then re-render the perturbed pixels and re-evaluate.

GPU required for the inner loop; CPU dry-run reports the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dsi.config import EPS_GRID_PIXEL


@dataclass
class PixelPGDConfig:
    eps: float = 4.0 / 255
    n_steps: int = 40
    lr: float = 1.0 / 255
    targeted: bool = False
    eps_grid: tuple[float, ...] = EPS_GRID_PIXEL


def pgd_step_pixel(
    x_t,
    y,
    *,
    safety_logit_fn,
    eps: float,
    lr: float,
    targeted: bool = False,
):
    """One PGD step on pixel input. `x_t`: (B,3,H,W) torch tensor in [0,1].

    Loss: NLL on `y` (typically `y=safe`); we maximize the safe-class probability.
    """
    import torch

    x = x_t.detach().clone().requires_grad_(True)
    logits = safety_logit_fn(x)
    if targeted:
        loss = -torch.nn.functional.cross_entropy(logits, y)
    else:
        loss = torch.nn.functional.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x)[0]
    sign = grad.sign()
    x_new = x.detach() + lr * (-sign if targeted else sign)
    delta = torch.clamp(x_new - x_t, min=-eps, max=eps)
    return torch.clamp(x_t + delta, 0.0, 1.0)


def pgd_attack_pixel(
    x0,
    y,
    *,
    safety_logit_fn,
    cfg: PixelPGDConfig,
):
    """Full PGD loop. Returns the perturbed image and the perturbation norm."""
    import torch

    x = x0.detach().clone()
    for _ in range(cfg.n_steps):
        x = pgd_step_pixel(
            x, y,
            safety_logit_fn=safety_logit_fn,
            eps=cfg.eps, lr=cfg.lr, targeted=cfg.targeted,
        )
    delta = (x - x0).detach()
    return x, delta.flatten(1).norm(p=float("inf"), dim=1)


def perturbation_norm(x_pre: np.ndarray, x_post: np.ndarray, p: float = float("inf")) -> float:
    """Per-instance ℓp norm of the perturbation, averaged across the batch."""
    delta = (x_post - x_pre).reshape(x_pre.shape[0], -1)
    if p == float("inf"):
        return float(np.abs(delta).max(axis=1).mean())
    return float(np.linalg.norm(delta, ord=p, axis=1).mean())
