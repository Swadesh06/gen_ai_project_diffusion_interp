#!/usr/bin/env python
"""D-9 SD3-medium nudity baseline.

Renders 100 I2P-NSFW prompts through SD3 medium 28-step, scores with
safety_checker. Compares to SDXL Turbo's 8.5%, SDXL Base 4-step's 28.6%,
to confirm whether the cross-architecture generalisation holds.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="D09_sd3_i2p_n100")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--num-inference-steps", type=int, default=28)
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusion3Pipeline
    from PIL import Image

    out_dir = Path("/workspace/swadesh/gen_ai_project_diffusion_interp/outputs") / args.exp_id
    img_dir = out_dir / "post"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {args.exp_id} ===")
    print(f"loading SD3-medium")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    from dsi.data.i2p import i2p_nsfw_subset
    prompts = i2p_nsfw_subset(limit=args.n_prompts)
    print(f"  {len(prompts)} prompts")

    print("loading safety_checker")
    from dsi.detectors.baselines.safety_checker import SafetyCheckerWrapper
    safety = SafetyCheckerWrapper(device="cuda").load()

    print("rendering")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    flagged_count = 0
    for i, p in enumerate(prompts):
        gen = torch.Generator(device="cuda").manual_seed(700_000_000 + i)
        out = pipe(prompt=p.text, num_inference_steps=args.num_inference_steps,
                   guidance_scale=7.0, generator=gen, height=512, width=512)
        img = out.images[0]
        scored = safety.score([img])[0]
        flagged = bool(scored["flagged"])
        flagged_count += int(flagged)
        img.save(img_dir / f"{i:08d}.png")
        with open(img_dir / f"{i:08d}.png.safety.json", "w") as f:
            json.dump({"flagged": flagged, "prompt": p.text}, f)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}] {time.time()-t0:.1f}s, flagged={flagged_count}, peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    summary = {
        "exp_id": args.exp_id,
        "n_prompts": len(prompts),
        "n_flagged": flagged_count,
        "flag_rate": flagged_count / len(prompts),
        "elapsed_s": elapsed,
        "peak_vram_gb": peak,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE: {flagged_count}/{len(prompts)} flagged ({flagged_count/len(prompts):.3f}), {elapsed:.1f}s, peak {peak:.2f} GB")
    return 0


if __name__ == "__main__":
    import numpy as np
    raise SystemExit(main())
