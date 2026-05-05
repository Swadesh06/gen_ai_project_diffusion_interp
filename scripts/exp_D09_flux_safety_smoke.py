#!/usr/bin/env python
"""D-9 FLUX inference smoke — safety_checker baseline.

20 I2P-NSFW prompts through FLUX.1-schnell, score with CompVis safety_checker.
Test if FLUX produces NSFW the safety_checker can flag.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=20)
    args = ap.parse_args()

    import torch
    from diffusers import FluxPipeline
    from PIL import Image

    out_dir = Path("/workspace/swadesh/gen_ai_project_diffusion_interp/outputs/D09_flux_safety_n20")
    img_dir = out_dir / "post"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("loading FLUX.1-schnell")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    from dsi.data.i2p import i2p_nsfw_subset
    prompts = i2p_nsfw_subset(limit=args.n_prompts)

    print("loading safety_checker")
    from dsi.detectors.baselines.safety_checker import SafetyCheckerWrapper
    safety = SafetyCheckerWrapper(device="cuda").load()

    print("rendering")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_flag = 0
    for i, p in enumerate(prompts):
        gen = torch.Generator(device="cuda").manual_seed(800_000_000 + i)
        out = pipe(prompt=p.text, num_inference_steps=4, guidance_scale=0.0,
                   generator=gen, height=512, width=512, max_sequence_length=256)
        img = out.images[0]
        flagged = bool(safety.score([img])[0]["flagged"])
        n_flag += int(flagged)
        img.save(img_dir / f"{i:08d}.png")
        with open(img_dir / f"{i:08d}.png.safety.json", "w") as f:
            json.dump({"flagged": flagged, "prompt": p.text}, f)
        print(f"  [{i+1}/{len(prompts)}] {time.time()-t0:.1f}s flagged={n_flag} peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    summary = {"n": len(prompts), "n_flagged": n_flag, "flag_rate": n_flag/len(prompts),
               "elapsed_s": elapsed, "peak_vram_gb": peak}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE: {n_flag}/{len(prompts)} flagged ({n_flag/len(prompts):.3f}), {elapsed:.1f}s, peak {peak:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
