"""End-to-end attack runner with SAE activation capture + per-image disk dump.

For each batch of prompts:
  1. SDXL Turbo generates the seed image (1 step) → x0  (B,3,512,512) in [0,1]
  2. Optionally encode to VAE latent z0 (for latent attack) or to CLIP embedding e0 (for emb attack)
  3. Run N PGD iterations of the chosen space toward `safe` against `SafetyTarget`
  4. Evaluate pre/post `SafetyTarget.flagged()` per image
  5. Persist:
       - {pre,post}/<seed>.png — the seed and perturbed images (PNG)
       - <seed>.attack.json     — pre/post safe-logit, pert norm, prompt, seed
       - <seed>.sae.pt          — captured SAE per-block residual diffs (if collect_sae)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


SpaceLit = Literal["pixel", "latent", "embedding"]


@dataclass
class AttackOutcome:
    seed: int
    prompt: str
    space: SpaceLit
    pre_safe_logit: float
    post_safe_logit: float
    pre_flagged: bool
    post_flagged: bool
    perturb_norm: float
    bypass: bool
    pre_path: str = ""
    post_path: str = ""
    sae_path: str = ""
    extra: dict = field(default_factory=dict)


def run_attack_pixel(
    *, pipe_w, target, prompts, eps: float, n_steps: int, lr: float, seed_offset: int,
    out_dir: Path, batch_size: int = 4, collect_sae=None, log_every: int = 50,
):
    return _run_attack(space="pixel", pipe_w=pipe_w, target=target, prompts=prompts,
                       eps=eps, n_steps=n_steps, lr=lr, seed_offset=seed_offset,
                       out_dir=out_dir, batch_size=batch_size,
                       collect_sae=collect_sae, log_every=log_every)


def run_attack_embedding(
    *, pipe_w, target, prompts, eps: float, n_steps: int, lr: float, seed_offset: int,
    out_dir: Path, batch_size: int = 8, collect_sae=None, log_every: int = 50,
):
    return _run_attack(space="embedding", pipe_w=pipe_w, target=target, prompts=prompts,
                       eps=eps, n_steps=n_steps, lr=lr, seed_offset=seed_offset,
                       out_dir=out_dir, batch_size=batch_size,
                       collect_sae=collect_sae, log_every=log_every)


def run_attack_latent(
    *, pipe_w, target, prompts, eps: float, n_steps: int, lr: float, seed_offset: int,
    out_dir: Path, batch_size: int = 4, collect_sae=None, log_every: int = 50,
):
    return _run_attack(space="latent", pipe_w=pipe_w, target=target, prompts=prompts,
                       eps=eps, n_steps=n_steps, lr=lr, seed_offset=seed_offset,
                       out_dir=out_dir, batch_size=batch_size,
                       collect_sae=collect_sae, log_every=log_every)


def _make_seed_images_and_sae(pipe_w, batch_prompts, seeds, *, collect_sae):
    """Generate seed images for the batch; optionally capture SAE activations during gen.

    Returns: (x0_in_01, sae_records_or_None).
    `sae_records` is dict[hookpoint -> torch.Tensor of shape (B, *spatial, D)] or None.
    """
    import torch

    gens = [torch.Generator(device=pipe_w.device).manual_seed(s) for s in seeds]
    if collect_sae is None:
        out = pipe_w.pipe(
            prompt=[p.text for p in batch_prompts],
            num_inference_steps=1, guidance_scale=0.0,
            generator=gens, height=512, width=512,
        )
        sae_records = None
    else:
        from dsi.sae.hooks import SurkovHookManager

        with SurkovHookManager(pipe_w.unet, collect_sae, capture=True, keep_inputs=False) as mgr:
            out = pipe_w.pipe(
                prompt=[p.text for p in batch_prompts],
                num_inference_steps=1, guidance_scale=0.0,
                generator=gens, height=512, width=512,
            )
        sae_records = {hp: cap.z[0] if cap.z else None for hp, cap in mgr.captured.items()}

    images = out.images
    import numpy as np

    x0 = torch.stack([
        torch.as_tensor(np.array(im, dtype=np.float32) / 255.0).permute(2, 0, 1) for im in images
    ]).to(pipe_w.device)
    return x0, sae_records, images


def _run_attack(
    *, space: SpaceLit, pipe_w, target, prompts, eps, n_steps, lr,
    seed_offset, out_dir: Path, batch_size: int, collect_sae, log_every: int,
):
    import numpy as np
    import torch
    from PIL import Image as PILImage

    out_dir.mkdir(parents=True, exist_ok=True)
    pre_dir = out_dir / "pre"
    post_dir = out_dir / "post"
    sae_dir = out_dir / "sae"
    for d in (pre_dir, post_dir):
        d.mkdir(exist_ok=True)
    if collect_sae is not None:
        sae_dir.mkdir(exist_ok=True)

    outcomes: list[AttackOutcome] = []
    safe_label_cache = {}
    t_total = time.time()

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        seeds = list(range(seed_offset + start, seed_offset + start + len(batch)))

        x0, sae_records, pil_images = _make_seed_images_and_sae(
            pipe_w, batch, seeds, collect_sae=collect_sae,
        )

        with torch.no_grad():
            pre_logits = target.pixel_to_logits(x0)
        pre_flagged = (pre_logits[:, 0] > 0).cpu().tolist()
        pre_safe = pre_logits[:, 1].detach().cpu().tolist()

        y = safe_label_cache.setdefault(
            x0.shape[0], torch.tensor([1] * x0.shape[0], device=pipe_w.device)
        )

        if space == "pixel":
            from dsi.attacks.pixel import pgd_step_pixel

            x = x0.clone()
            for _ in range(n_steps):
                x = pgd_step_pixel(x, y, safety_logit_fn=target.pixel_to_logits,
                                   eps=eps, lr=lr, targeted=True)
            x_out_for_eval = x

        elif space == "embedding":
            from dsi.attacks.embedding import pgd_step_embedding

            with torch.no_grad():
                e0 = target.pixel_to_embedding(x0)
            e = e0.clone()
            for _ in range(n_steps):
                e = pgd_step_embedding(e, y, safety_logit_from_embedding_fn=target.embedding_to_logits,
                                       eps=eps, lr=lr, targeted=True)
            x_out_for_eval = x0
            post_logits = target.embedding_to_logits(e).detach()
            embedding_outputs = e.detach().cpu()

        elif space == "latent":
            from dsi.attacks.latent import pgd_step_latent

            scale = float(getattr(pipe_w.vae.config, "scaling_factor", 0.13025))
            with torch.no_grad():
                z0 = pipe_w.vae.encode(
                    (x0.to(next(pipe_w.vae.parameters()).dtype) * 2 - 1)
                ).latent_dist.sample() * scale
            z = z0.clone()
            for _ in range(n_steps):
                z = pgd_step_latent(z, y, decode_fn=target.vae_latent_to_pixel,
                                    safety_logit_fn=target.pixel_to_logits,
                                    eps=eps, lr=lr, targeted=True)
            with torch.no_grad():
                x_out_for_eval = target.vae_latent_to_pixel(z).clamp(0, 1)
        else:
            raise ValueError(f"Unknown space: {space}")

        # Evaluate post (always in pixel space, even for embedding-PGD where there's no actual image yet)
        if space == "embedding":
            post_safe = post_logits[:, 1].detach().cpu().tolist()
            post_flagged = (post_logits[:, 0] > 0).cpu().tolist()
        else:
            with torch.no_grad():
                post_logits = target.pixel_to_logits(x_out_for_eval)
            post_safe = post_logits[:, 1].detach().cpu().tolist()
            post_flagged = (post_logits[:, 0] > 0).cpu().tolist()

        # Persist per-image
        for i, (prompt, seed, pre_img) in enumerate(zip(batch, seeds, pil_images)):
            stem = f"{seed:08d}"
            pre_path = pre_dir / f"{stem}.png"
            post_path = post_dir / f"{stem}.png"
            pre_img.save(pre_path)
            if space == "embedding":
                # No actual perturbed image; save the seed image with a `.embedding.npy` of e
                np.save(post_dir / f"{stem}.embedding.npy", embedding_outputs[i].numpy())
                post_path = post_dir / f"{stem}.embedding.npy"
                pert_norm = float(np.abs(embedding_outputs[i].numpy() - target.pixel_to_embedding(
                    x0[i:i+1]
                ).detach().cpu().numpy()[0]).max())
            else:
                arr = (x_out_for_eval[i].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
                PILImage.fromarray(arr).save(post_path)
                pert_norm = float(np.abs(
                    x_out_for_eval[i].cpu().numpy() - x0[i].cpu().numpy()
                ).max())

            sae_path = ""
            if sae_records is not None:
                import torch as T

                sae_path = str(sae_dir / f"{stem}.sae.pt")
                T.save({hp: (z[i].clone() if z is not None else None)
                        for hp, z in sae_records.items()}, sae_path)

            bypass = pre_flagged[i] and not post_flagged[i]
            outcomes.append(AttackOutcome(
                seed=seed, prompt=prompt.text, space=space,
                pre_safe_logit=pre_safe[i], post_safe_logit=post_safe[i],
                pre_flagged=pre_flagged[i], post_flagged=post_flagged[i],
                perturb_norm=pert_norm, bypass=bypass,
                pre_path=str(pre_path), post_path=str(post_path), sae_path=sae_path,
                extra={"category": prompt.category, "source": prompt.source},
            ))
            (out_dir / f"{stem}.attack.json").write_text(json.dumps({
                "seed": seed, "prompt": prompt.text, "space": space,
                "pre_safe_logit": pre_safe[i], "post_safe_logit": post_safe[i],
                "pre_flagged": pre_flagged[i], "post_flagged": post_flagged[i],
                "perturb_norm": pert_norm, "bypass": bypass,
                "eps": eps, "n_steps": n_steps, "lr": lr,
            }))

        if (start // batch_size) % max(1, log_every // batch_size) == 0:
            elapsed = time.time() - t_total
            n_done = len(outcomes)
            n_bypass = sum(o.bypass for o in outcomes)
            print(f"  [{n_done}/{len(prompts)}] {elapsed:.1f}s — bypass {n_bypass}/{n_done} "
                  f"= {100*n_bypass/max(1,n_done):.1f}%", flush=True)

    return outcomes
