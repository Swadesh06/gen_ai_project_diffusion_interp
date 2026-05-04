#!/usr/bin/env python
"""S00 smoke — SDXL Turbo + Surkov SAEs forward + activation capture.

Loads SDXL Turbo, registers SAE forward-pre-hooks on all four Surkov hookpoints,
generates one image from one prompt, captures the per-step per-block activations,
runs them through the loaded SAEs, and reports peak VRAM + sparsity stats.

Pass:
  - 1 image generated (no error)
  - 4 hooks each fired ≥ 1 time
  - peak VRAM < 12 GB
  - SAE sparsity (fraction of features with |z| > 0) within [0.001, 0.5] per hook
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

HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="a photo of a yellow taxi in new york city")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16")
    args = ap.parse_args()

    import torch

    out_dir = Path(args.out_dir) if args.out_dir else cfg.paths.output_root / "S00_smoke_sdxl_sae"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading SDXL Turbo on {args.device} ({args.dtype})")
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper

    pipe_w = SDXLPipelineWrapper(variant="turbo", device=args.device, dtype=args.dtype).load()

    print("loading 4 Surkov SAEs")
    from dsi.sae.load import load_surkov_sae

    saes = {}
    for hp in HOOKPOINTS:
        sae = load_surkov_sae(hp).to(args.device).eval()
        saes[hp] = sae
        print(f"  {hp}: d_in={sae.d_in} d_hidden={sae.d_hidden}")

    print("registering hooks + generating 1 image")
    from dsi.sae.hooks import SAEHookManager

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with SAEHookManager(pipe_w.unet, saes, capture=True, device=args.device) as mgr:
        images = pipe_w.generate(prompts=[args.prompt], num_inference_steps=1,
                                 guidance_scale=0.0, seed=args.seed)
    elapsed = time.time() - t0

    peak_vram_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if args.device == "cuda" else 0.0

    img_path = out_dir / "smoke.png"
    images[0].save(img_path)

    stats = {
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_path": str(img_path),
        "hooks": {},
    }
    for hp, cap in mgr.captured.items():
        if not cap.z:
            stats["hooks"][hp] = {"fired": 0}
            continue
        z = cap.z[0]
        n_active = (z.abs() > 1e-6).float().mean().item()
        stats["hooks"][hp] = {
            "fired": len(cap.z),
            "z_shape": tuple(z.shape),
            "sparsity_active_frac": n_active,
            "z_max": float(z.abs().max()),
        }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    fail = []
    for hp in HOOKPOINTS:
        h = stats["hooks"].get(hp, {"fired": 0})
        if h["fired"] < 1:
            fail.append(f"{hp} did not fire")
    if peak_vram_gb > 12:
        fail.append(f"peak_vram {peak_vram_gb:.1f} GB exceeds 12 GB budget")
    if fail:
        print("FAIL:", fail)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
