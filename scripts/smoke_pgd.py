#!/usr/bin/env python
"""S03 — pixel/latent/embedding PGD smoke (3 iters × 2 prompts each).

For each attack space:
  - load SDXL Turbo + VAE + safety target
  - generate one I2P-NSFW seed image (so we have something the safety checker would flag)
  - run 3 PGD iterations
  - report peak VRAM and whether the safety_logit changed in the right direction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import cfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--n-prompts", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    import torch

    out_dir = Path(args.out_dir) if args.out_dir else cfg.paths.output_root / "S03_smoke_pgd"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading SDXL Turbo")
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper

    pipe_w = SDXLPipelineWrapper(variant="turbo", device=args.device, dtype=args.dtype).load()

    print("loading SafetyTarget (CompVis safety checker)")
    from dsi.attacks.safety_target import SafetyTarget

    target = SafetyTarget(device=args.device, dtype=args.dtype).load(vae=pipe_w.vae)

    print("loading I2P-NSFW prompts")
    from dsi.data.i2p import i2p_nsfw_subset

    prompts = i2p_nsfw_subset(limit=args.n_prompts)
    if len(prompts) < args.n_prompts:
        print(f"WARN only got {len(prompts)} I2P prompts; padding with COCO captions")
        from dsi.data.coco import load_coco_captions

        prompts += load_coco_captions(limit=args.n_prompts - len(prompts))

    summary = {"prompts": [p.text for p in prompts], "spaces": {}}

    # ---------- 1) PIXEL PGD ----------
    print("=== PIXEL PGD ===")
    torch.cuda.reset_peak_memory_stats()
    images = pipe_w.generate(prompts=[p.text for p in prompts],
                             num_inference_steps=1, guidance_scale=0.0, seed=0)
    import numpy as np
    from PIL import Image as PILImage

    x0 = torch.stack([
        torch.as_tensor(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        for img in images
    ]).to(args.device)
    pre_logits = target.pixel_to_logits(x0)
    pre_safe = pre_logits[:, 1].detach().cpu().tolist()

    from dsi.attacks.pixel import PixelPGDConfig, pgd_step_pixel

    pcfg = PixelPGDConfig(eps=4.0 / 255, n_steps=args.n_steps, lr=1.0 / 255)
    safe_label = torch.tensor([1] * len(prompts), device=args.device)  # target = "safe"
    x = x0.clone()
    t0 = time.time()
    for _ in range(args.n_steps):
        x = pgd_step_pixel(x, safe_label,
                           safety_logit_fn=target.pixel_to_logits,
                           eps=pcfg.eps, lr=pcfg.lr, targeted=True)
    pix_elapsed = time.time() - t0
    post_logits = target.pixel_to_logits(x)
    post_safe = post_logits[:, 1].detach().cpu().tolist()
    pix_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    summary["spaces"]["pixel"] = {
        "peak_vram_gb": pix_vram, "elapsed_s": pix_elapsed,
        "pre_safe_logit": pre_safe, "post_safe_logit": post_safe,
        "delta_safe": [post - pre for pre, post in zip(pre_safe, post_safe)],
    }
    for i, img_t in enumerate(x.detach().clamp(0, 1).cpu()):
        PILImage.fromarray((img_t.permute(1, 2, 0).numpy() * 255).astype("uint8")).save(
            out_dir / f"pixel_post_{i}.png"
        )

    # ---------- 2) EMBEDDING PGD ----------
    print("=== EMBEDDING PGD ===")
    torch.cuda.reset_peak_memory_stats()
    e0 = target.pixel_to_embedding(x0).detach()
    pre_emb_logits = target.embedding_to_logits(e0).detach().cpu().tolist()

    from dsi.attacks.embedding import EmbeddingPGDConfig, pgd_step_embedding

    ecfg = EmbeddingPGDConfig(eps=0.5, n_steps=args.n_steps, lr=0.05)
    e = e0.clone()
    t0 = time.time()
    for _ in range(args.n_steps):
        e = pgd_step_embedding(e, safe_label,
                               safety_logit_from_embedding_fn=target.embedding_to_logits,
                               eps=ecfg.eps, lr=ecfg.lr, targeted=True)
    emb_elapsed = time.time() - t0
    post_emb_logits = target.embedding_to_logits(e).detach().cpu().tolist()
    emb_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    summary["spaces"]["embedding"] = {
        "peak_vram_gb": emb_vram, "elapsed_s": emb_elapsed,
        "pre_safe_logit": [r[1] for r in pre_emb_logits],
        "post_safe_logit": [r[1] for r in post_emb_logits],
        "delta_safe": [r[1] - p[1] for p, r in zip(pre_emb_logits, post_emb_logits)],
    }

    # ---------- 3) LATENT PGD ----------
    print("=== LATENT PGD ===")
    torch.cuda.reset_peak_memory_stats()
    # Encode the seed image into VAE latent space
    with torch.no_grad():
        scale = float(getattr(pipe_w.vae.config, "scaling_factor", 0.13025))
        # x0 is (B, 3, 512, 512) in [0,1]; VAE expects [-1, 1]
        z0 = pipe_w.vae.encode((x0.to(target.mean.dtype) * 2 - 1)).latent_dist.sample() * scale
    pre_lat_logits = target.vae_latent_to_logits(z0).detach().cpu().tolist()

    from dsi.attacks.latent import LatentPGDConfig, pgd_step_latent

    lcfg = LatentPGDConfig(eps=0.1, n_steps=args.n_steps, lr=0.005)
    z = z0.clone()
    t0 = time.time()
    for _ in range(args.n_steps):
        z = pgd_step_latent(z, safe_label,
                            decode_fn=target.vae_latent_to_pixel,
                            safety_logit_fn=target.pixel_to_logits,
                            eps=lcfg.eps, lr=lcfg.lr, targeted=True)
    lat_elapsed = time.time() - t0
    post_lat_logits = target.vae_latent_to_logits(z).detach().cpu().tolist()
    lat_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    summary["spaces"]["latent"] = {
        "peak_vram_gb": lat_vram, "elapsed_s": lat_elapsed,
        "pre_safe_logit": [r[1] for r in pre_lat_logits],
        "post_safe_logit": [r[1] for r in post_lat_logits],
        "delta_safe": [r[1] - p[1] for p, r in zip(pre_lat_logits, post_lat_logits)],
    }

    (out_dir / "stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    fail = []
    for sp in ("pixel", "embedding", "latent"):
        s = summary["spaces"][sp]
        if s["peak_vram_gb"] > 27:
            fail.append(f"{sp} peak VRAM {s['peak_vram_gb']:.1f} > 27 GB cap")
        if min(s["delta_safe"]) <= 0 and len(s["delta_safe"]) > 0:
            print(f"WARN {sp}: at least one prompt did not move safe-logit upward in 3 iters")
    if fail:
        print("FAIL:", fail)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
