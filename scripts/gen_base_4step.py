#!/usr/bin/env python
"""SDXL Base 4-step generation for I2P-NSFW prompts.

Lifts pre-flag rate vs SDXL Turbo 1-step (which underrenders details and
produces only 7-12% safety_checker hits). 4 steps + CFG=7.5 typically gets
30-60% pre-flag on I2P-NSFW.

Output: outputs/<exp_id>/{i2p,coco}/<seed>.png with .meta.json sidecar.
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
    ap.add_argument("--exp-id", default="base_i2p_4step_n200")
    ap.add_argument("--n-i2p", type=int, default=200)
    ap.add_argument("--n-coco", type=int, default=0)
    ap.add_argument("--num-inference-steps", type=int, default=4)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=9_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    args = ap.parse_args()

    import torch
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.data.coco import load_coco_captions
    from dsi.data.i2p import i2p_nsfw_subset

    out_dir = cfg.paths.output_root / args.exp_id
    i2p_dir = out_dir / "i2p"; i2p_dir.mkdir(parents=True, exist_ok=True)
    coco_dir = out_dir / "coco"; coco_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading SDXL Base on {args.device} ({args.dtype})")
    pipe_w = SDXLPipelineWrapper(variant="base", device=args.device, dtype=args.dtype).load()

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def gen_to(prompts, dst, seed_off):
        n_done = 0
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start:start + args.batch_size]
            seeds = list(range(seed_off + start, seed_off + start + len(batch)))
            gens = [torch.Generator(device=args.device).manual_seed(s) for s in seeds]
            out = pipe_w.pipe(prompt=[p.text for p in batch],
                              num_inference_steps=args.num_inference_steps,
                              guidance_scale=args.guidance_scale,
                              generator=gens, height=512, width=512)
            for img, prompt, seed in zip(out.images, batch, seeds):
                stem = f"{seed:08d}"
                img.save(dst / f"{stem}.png")
                (dst / f"{stem}.png.meta.json").write_text(json.dumps(
                    {"prompt": prompt.text, "seed": seed,
                     "num_inference_steps": args.num_inference_steps,
                     "guidance_scale": args.guidance_scale,
                     "variant": "base"}))
                n_done += 1
            if (start // args.batch_size) % 5 == 0:
                print(f"  {n_done}/{len(prompts)}", flush=True)
        return n_done

    t0 = time.time()
    n_i2p = 0; n_coco = 0
    if args.n_i2p > 0:
        i2p_prompts = i2p_nsfw_subset(limit=args.n_i2p)
        print(f"i2p: {len(i2p_prompts)} prompts")
        n_i2p = gen_to(i2p_prompts, i2p_dir, args.seed)
    if args.n_coco > 0:
        coco_prompts = load_coco_captions(limit=args.n_coco)
        print(f"coco: {len(coco_prompts)} prompts")
        n_coco = gen_to(coco_prompts, coco_dir, args.seed + 100_000)

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0
    summary = {
        "exp_id": args.exp_id, "n_i2p": n_i2p, "n_coco": n_coco,
        "i2p_dir": str(i2p_dir), "coco_dir": str(coco_dir),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "elapsed_s": elapsed, "peak_vram_gb": peak,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
