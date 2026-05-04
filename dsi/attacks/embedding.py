"""CLIP image-embedding PGD: attack directly on the 768-d CLIP embedding.

Most permissive attack space — no decoding step. Establishes an attack ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass

from dsi.config import EPS_GRID_EMBEDDING


@dataclass
class EmbeddingPGDConfig:
    eps: float = 0.5
    n_steps: int = 40
    lr: float = 0.05
    targeted: bool = False
    eps_grid: tuple[float, ...] = EPS_GRID_EMBEDDING


def pgd_step_embedding(
    e_t,
    y,
    *,
    safety_logit_from_embedding_fn,
    eps: float,
    lr: float,
    targeted: bool = True,
):
    """One PGD step on the CLIP image embedding.

    `y` is the desired class index. Same sign convention as pgd_step_pixel.
    """
    import torch
    import torch.nn.functional as F

    e = e_t.detach().clone().requires_grad_(True)
    logits = safety_logit_from_embedding_fn(e).float()
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, e)[0]
    sign = grad.sign()
    direction = -1.0 if targeted else 1.0
    e_new = e.detach() + direction * lr * sign
    delta = torch.clamp(e_new - e_t, min=-eps, max=eps)
    return e_t + delta


def pgd_attack_embedding(
    e0,
    y,
    *,
    safety_logit_from_embedding_fn,
    cfg: EmbeddingPGDConfig,
):
    import torch

    e = e0.detach().clone()
    for _ in range(cfg.n_steps):
        e = pgd_step_embedding(
            e, y,
            safety_logit_from_embedding_fn=safety_logit_from_embedding_fn,
            eps=cfg.eps, lr=cfg.lr, targeted=cfg.targeted,
        )
    delta = (e - e0).detach()
    return e, delta.flatten(1).norm(p=float("inf"), dim=1)
