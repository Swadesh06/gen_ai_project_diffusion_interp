#!/usr/bin/env python
"""SAeUron baseline reproduction — streamlined version on UnlearnDiffAtk-nudity.

Uses the SAeUron upstream code from /workspace/datasets/SAeUron and the
released checkpoint at `bcywinski/SAeUron_coco`. Renders a chosen
UnlearnDiffAtk concept split with SD v1.4 + SAeUron's SAE-feature
intervention, then oracle-scores via the cpu-worker pool.

References:
- Cywiński & Deja, ICML 2025 (arXiv:2501.18052)
- Repo: https://github.com/cywinski/SAeUron

Usage:
    python scripts/repro_saeuron_streamlined.py \
        --concept nudity --feature-idx 11627 --multiplier -3.0 --n-prompts 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Upstream SAeUron repo
SAE_URON_PATH = Path("/workspace/datasets/SAeUron")
sys.path.insert(0, str(SAE_URON_PATH))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concept", default="nudity",
                    choices=["nudity", "violence", "vangogh", "church", "parachute"])
    ap.add_argument("--feature-idx", type=int, default=11627,
                    help="SAE feature index identified by SAeUron for the concept")
    ap.add_argument("--multiplier", type=float, default=-3.0)
    ap.add_argument("--hookpoint", default="unet.up_blocks.1.attentions.1")
    ap.add_argument("--hub-name", default="bcywinski/SAeUron_coco")
    ap.add_argument("--model-name", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=70_000_000)
    ap.add_argument("--exp-id", default="repro_saeuron_nudity_n100")
    args = ap.parse_args()

    import torch

    from dsi.config import cfg

    out_dir = cfg.paths.output_root / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_dir = out_dir / "no_intervention"
    post_dir = out_dir / "with_saeuron"
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    csv_path = Path("/workspace/datasets/Diffusion-MU-Attack/prompts") / f"{args.concept}.csv"
    if not csv_path.exists():
        print(f"missing prompt csv at {csv_path}")
        return 2
    import csv

    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            if len(rows) >= args.n_prompts:
                break
    print(f"loaded {len(rows)} prompts from {args.concept}")

    # Load SD v1.4 + SAeUron SAE
    print(f"loading {args.model_name} via SAeUron's HookedStableDiffusionPipeline")
    from SAE.sae import Sae
    from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
    import utils.hooks as hooks

    dtype = torch.float16
    pipe = HookedStableDiffusionPipeline.from_pretrained(
        args.model_name, torch_dtype=dtype, safety_checker=None,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True) if hasattr(pipe, "set_progress_bar_config") else None

    print(f"loading SAE from {args.hub_name} hookpoint={args.hookpoint}")
    sae = Sae.load_from_hub(args.hub_name, hookpoint=args.hookpoint, device="cuda").to(dtype)

    intervention_hook = hooks.SAEFeatureInterventionHook(
        sae=sae, feature_idx=args.feature_idx, multiplier=args.multiplier,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for i, r in enumerate(rows):
        case_id = r.get("case_number", str(i))
        seed = args.seed + i
        gen = torch.Generator(device="cuda").manual_seed(seed)

        # Render WITHOUT intervention
        gen = torch.Generator(device="cuda").manual_seed(seed)
        baseline_imgs = pipe.run_with_hooks(
            prompt=r["prompt"],
            generator=gen,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            position_hook_dict={},
        )
        gen = torch.Generator(device="cuda").manual_seed(seed)
        intervened_imgs = pipe.run_with_hooks(
            prompt=r["prompt"],
            generator=gen,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            position_hook_dict={args.hookpoint: intervention_hook},
        )
        stem = f"{int(case_id):08d}_{seed:08d}"
        baseline_imgs[0].save(pre_dir / f"{stem}.png")
        intervened_imgs[0].save(post_dir / f"{stem}.png")
        for d in (pre_dir, post_dir):
            (d / f"{stem}.png.meta.json").write_text(json.dumps({
                "case_number": case_id, "prompt": r["prompt"],
                "seed": seed, "concept": args.concept,
                "feature_idx": args.feature_idx, "multiplier": args.multiplier,
                "hookpoint": args.hookpoint,
            }))
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            print(f"  {i+1}/{len(rows)}  elapsed={elapsed:.1f}s peak_vram={peak:.1f}GB", flush=True)

    elapsed = time.time() - t0
    summary = {
        "exp_id": args.exp_id, "concept": args.concept, "n": len(rows),
        "feature_idx": args.feature_idx, "multiplier": args.multiplier,
        "hookpoint": args.hookpoint, "model": args.model_name,
        "no_intervention_dir": str(pre_dir),
        "with_saeuron_dir": str(post_dir),
        "elapsed_s": elapsed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
