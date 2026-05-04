#!/usr/bin/env python
"""Item 4 / Contribution 3 — cross-target PGD attack.

Same pixel/latent/embedding PGD machinery as `run_attack.py`, but instead of
attacking `CompVis/stable-diffusion-safety-checker` we attack a TRAINED
SAE-activation detector. Goal: drive the detector logit toward "safe" while
keeping the underlying image NSFW (NudeNet + Q16 oracle excluded from the
attack loop).

Runs against pixel-space only for now (latent/embedding don't have a natural
gradient path through the SAE-detector head + UNet without a much larger
graph). Reports cross-target ASR and writes pre/post images for oracle
evaluation in a downstream cpu-worker pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import EPS_GRID_PIXEL, cfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector-ckpt", required=True)
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eps", type=float, default=EPS_GRID_PIXEL[1])
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1.0 / 255)
    ap.add_argument("--seed-offset", type=int, default=4_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image as PILImage

    exp_id = args.exp_id or f"C01_xtarget_pixel_eps{int(args.eps*255)}-255_n{args.n_prompts}"
    out_dir = cfg.paths.output_root / exp_id
    pre_dir = out_dir / "pre"
    post_dir = out_dir / "post"
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {exp_id} ===")
    print("loading SDXL Turbo + 4 Surkov SAEs")
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.load import load_surkov_sae

    pipe_w = SDXLPipelineWrapper(variant="turbo", device=args.device, dtype=args.dtype).load()
    sae_dict = {hp: load_surkov_sae(hp).to(args.device).eval()
                for hp in ("down.2.1", "mid.0", "up.0.0", "up.0.1")}

    print(f"loading SAE-detector target from {args.detector_ckpt}")
    from dsi.attacks.sae_detector_target import SAEDetectorTarget

    target = SAEDetectorTarget(detector_ckpt=args.detector_ckpt, pipe_w=pipe_w,
                               sae_dict=sae_dict, device=args.device, dtype=args.dtype).load()

    print("loading I2P-NSFW prompts")
    from dsi.data.i2p import i2p_nsfw_subset

    prompts = i2p_nsfw_subset(limit=args.n_prompts)
    print(f"  {len(prompts)} prompts")

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows = []
    t0 = time.time()
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        seeds = list(range(args.seed_offset + start, args.seed_offset + start + len(batch)))
        gens = [torch.Generator(device=args.device).manual_seed(s) for s in seeds]
        out = pipe_w.pipe(prompt=[p.text for p in batch], num_inference_steps=1,
                          guidance_scale=0.0, generator=gens, height=512, width=512)
        x0 = torch.stack([
            torch.as_tensor(np.array(im, dtype=np.float32) / 255.0).permute(2, 0, 1) for im in out.images
        ]).to(args.device)

        with torch.no_grad():
            pre_logit = target.x_to_logit(x0).float().detach()

        # Pixel-space PGD on the SAE-detector logit (we want "safe" = low/neg logit)
        x = x0.clone().detach().requires_grad_(True)
        for _ in range(args.n_steps):
            x_iter = x.detach().clone().requires_grad_(True)
            logit = target.x_to_logit(x_iter).float()
            # Untargeted toward "low" — minimise logit (sigmoid(logit) → P(unsafe))
            loss = logit.mean()
            grad = torch.autograd.grad(loss, x_iter)[0]
            x = x_iter.detach() - args.lr * grad.sign()
            delta = torch.clamp(x - x0, min=-args.eps, max=args.eps)
            x = torch.clamp(x0 + delta, 0.0, 1.0)

        with torch.no_grad():
            post_logit = target.x_to_logit(x).float()

        for i, (im_pre, prompt, seed) in enumerate(zip(out.images, batch, seeds)):
            stem = f"{seed:08d}"
            im_pre.save(pre_dir / f"{stem}.png")
            arr = (x[i].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
            PILImage.fromarray(arr).save(post_dir / f"{stem}.png")
            rows.append({
                "seed": seed, "prompt": prompt.text,
                "pre_logit": float(pre_logit[i]), "post_logit": float(post_logit[i]),
                "delta": float(post_logit[i] - pre_logit[i]),
            })
        if (start // args.batch_size) % 5 == 0:
            print(f"  [{start+len(batch)}/{len(prompts)}] {time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0
    summary = {
        "exp_id": exp_id, "detector_ckpt": args.detector_ckpt,
        "n_prompts": len(prompts), "eps": args.eps, "n_steps": args.n_steps,
        "lr": args.lr, "elapsed_s": elapsed, "peak_vram_gb": peak,
        "pre_logit_mean": sum(r["pre_logit"] for r in rows) / max(1, len(rows)),
        "post_logit_mean": sum(r["post_logit"] for r in rows) / max(1, len(rows)),
        "delta_mean": sum(r["delta"] for r in rows) / max(1, len(rows)),
        "n_pre_unsafe_pred": sum(1 for r in rows if r["pre_logit"] > 0),
        "n_post_unsafe_pred": sum(1 for r in rows if r["post_logit"] > 0),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
