#!/usr/bin/env python
"""Generate N benign SDXL Turbo images with SAE activations captured.

Used to supply the `label=0` half of the detector training dataset (Item 3).
Roughly ~7 GB VRAM (SDXL Turbo + 4 Surkov SAEs); same speed as plain gen.
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
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=2_000_000)
    ap.add_argument("--exp-id", default="sae_benign_coco_1k")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    args = ap.parse_args()

    import torch
    from PIL import Image as PILImage

    out_dir = cfg.paths.output_root / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sae_dir = out_dir / "sae"
    pre_dir = out_dir / "pre"
    sae_dir.mkdir(exist_ok=True)
    pre_dir.mkdir(exist_ok=True)

    print(f"=== {args.exp_id} ===  n_prompts={args.n_prompts}, batch={args.batch_size}")
    print("loading SDXL Turbo")
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper

    pipe_w = SDXLPipelineWrapper(variant="turbo", device=args.device, dtype=args.dtype).load()

    print("loading 4 Surkov SAEs")
    from dsi.sae.load import load_surkov_sae

    saes = {hp: load_surkov_sae(hp).to(args.device).eval()
            for hp in ("down.2.1", "mid.0", "up.0.0", "up.0.1")}

    print("loading COCO val captions")
    from dsi.data.coco import load_coco_captions

    prompts = load_coco_captions(limit=args.n_prompts)
    print(f"  {len(prompts)} prompts")

    from dsi.sae.hooks import SurkovHookManager

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    n = 0
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        seeds = list(range(args.seed_offset + start, args.seed_offset + start + len(batch)))
        gens = [torch.Generator(device=args.device).manual_seed(s) for s in seeds]
        with SurkovHookManager(pipe_w.unet, saes, capture=True, keep_inputs=False) as mgr:
            out = pipe_w.pipe(prompt=[p.text for p in batch],
                              num_inference_steps=1, guidance_scale=0.0,
                              generator=gens, height=512, width=512)
        sae_records = {hp: cap.z[0] if cap.z else None for hp, cap in mgr.captured.items()}
        for i, (img, prompt, seed) in enumerate(zip(out.images, batch, seeds)):
            stem = f"{seed:08d}"
            img.save(pre_dir / f"{stem}.png")
            torch.save({hp: (z[i].clone() if z is not None else None)
                        for hp, z in sae_records.items()}, sae_dir / f"{stem}.sae.pt")
        n += len(batch)
        if (start // args.batch_size) % 10 == 0:
            print(f"  [{n}/{len(prompts)}] {time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0
    summary = {"exp_id": args.exp_id, "n": n, "elapsed_s": elapsed, "peak_vram_gb": peak,
               "out_dir": str(out_dir)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
