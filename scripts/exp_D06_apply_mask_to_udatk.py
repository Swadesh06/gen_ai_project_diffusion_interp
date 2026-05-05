#!/usr/bin/env python
"""D-6 follow-up: apply joint-trained 41-feature mask to UDA-nudity rendering.

Loads joint_state.pt from D06_joint_e2e_v3_sparsity_sweep (lambda=5.0),
splits the 20480-dim mask into 4 hookpoints (5120 each), and uses the
M>0.5 features as the F_c set for the existing intervention pipeline.

Hookpoint order in concat: down.2.1, mid.0, up.0.0, up.0.1 (alphabetical
for sorted iteration).

This validates whether the cached-feature 100% correction translates to
image-rendered flag-rate reduction.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import cfg

HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")
F_PER_HP = 5120


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="D06_joint_mask_udatk")
    ap.add_argument("--prompts-csv",
                    default="/workspace/datasets/Diffusion-MU-Attack/prompts/nudity.csv")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--joint-state",
                    default="outputs/D06_joint_e2e_v2/joint_state.pt")
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--seed-offset", type=int, default=42_500_000)
    ap.add_argument("--num-inference-steps", type=int, default=4)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    args = ap.parse_args()

    import numpy as np
    import torch

    out_dir = cfg.paths.output_root / args.exp_id
    pre_dir = out_dir / "pre"
    post_dir = out_dir / "post"
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {args.exp_id} ===")

    # Load joint state
    print(f"loading joint state from {args.joint_state}")
    state = torch.load(args.joint_state, map_location="cpu", weights_only=False)
    theta = state["theta"]                  # (20480,)
    mu_benign = state["mu_benign"]          # (20480,)
    M = torch.sigmoid(theta)                # mask in [0,1]
    print(f"  mask shape: {M.shape}, mean: {M.mean():.4f}")

    # Split into hookpoints
    feat_set = {}
    mu_hp = {}
    for i, hp in enumerate(HOOKPOINTS):
        start = i * F_PER_HP
        end = start + F_PER_HP
        M_hp = M[start:end]
        mu_hp[hp] = mu_benign[start:end]
        idx = (M_hp > args.mask_threshold).nonzero(as_tuple=False).squeeze(-1).tolist()
        feat_set[hp] = idx
        print(f"  {hp}: {len(idx)} active features (M > {args.mask_threshold})")
    total_active = sum(len(v) for v in feat_set.values())
    print(f"  total active: {total_active} / 20480 ({100 * total_active / 20480:.2f}%)")

    # Save the F_c json for reproducibility
    feat_json = out_dir / "F_c_joint_mask.json"
    feat_json.write_text(json.dumps(feat_set, indent=2))
    print(f"  saved {feat_json}")

    # Load prompts
    prompts = []
    with open(args.prompts_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            prompts.append(row["prompt"].strip())
    prompts = prompts[:args.n_prompts]
    print(f"  loaded {len(prompts)} UDA-nudity prompts")

    print("loading SDXL Base + SAEs + safety_checker")
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.hooks import SurkovHookManager
    from dsi.sae.load import load_surkov_sae
    from dsi.detectors.baselines.safety_checker import SafetyCheckerWrapper

    pipe_w = SDXLPipelineWrapper(variant="base", device="cuda", dtype="fp16").load()
    saes = {hp: load_surkov_sae(hp).to("cuda").eval() for hp in HOOKPOINTS}
    sc = SafetyCheckerWrapper(device="cuda").load()

    feat_t = {hp: torch.tensor(feat_set[hp], device="cuda", dtype=torch.long)
              for hp in HOOKPOINTS}
    mu_t = {hp: mu_hp[hp].to("cuda").to(next(saes[hp].parameters()).dtype)
            for hp in HOOKPOINTS}

    def intervene_fn(hp, z, step):
        idx = feat_t.get(hp)
        if idx is None or len(idx) == 0:
            return None
        z_new = z.clone()
        z_new[..., idx] = mu_t[hp][idx].view(*([1] * (z.ndim - 1)), -1)
        return z_new

    print("rendering pre + post")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_pre_flag = 0
    n_post_flag = 0
    n_corrected = 0
    n_new_fp = 0  # benign pre, flagged post
    rows = []
    for i, prompt in enumerate(prompts):
        gen_pre = torch.Generator(device="cuda").manual_seed(args.seed_offset + i)
        gen_post = torch.Generator(device="cuda").manual_seed(args.seed_offset + i)

        # Pre (no intervention)
        out_pre = pipe_w.pipe(prompt=prompt, num_inference_steps=args.num_inference_steps,
                               guidance_scale=args.guidance_scale, generator=gen_pre,
                               height=512, width=512)
        img_pre = out_pre.images[0]
        flag_pre = bool(sc.score([img_pre])[0]["flagged"])
        if flag_pre:
            n_pre_flag += 1
        img_pre.save(pre_dir / f"{i:04d}.png")

        # Post (with F_c hook)
        with SurkovHookManager(pipe_w.unet, saes,
                                capture=False, intervene_fn=intervene_fn):
            out_post = pipe_w.pipe(prompt=prompt,
                                    num_inference_steps=args.num_inference_steps,
                                    guidance_scale=args.guidance_scale,
                                    generator=gen_post, height=512, width=512)
        img_post = out_post.images[0]
        flag_post = bool(sc.score([img_post])[0]["flagged"])
        if flag_post:
            n_post_flag += 1
        img_post.save(post_dir / f"{i:04d}.png")

        if flag_pre and not flag_post:
            n_corrected += 1
        if (not flag_pre) and flag_post:
            n_new_fp += 1

        rows.append({"i": i, "prompt": prompt[:80],
                     "flag_pre": flag_pre, "flag_post": flag_post,
                     "corrected": flag_pre and not flag_post,
                     "new_fp": (not flag_pre) and flag_post})

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}] pre={n_pre_flag} post={n_post_flag} "
                  f"corr={n_corrected} fp={n_new_fp} elapsed={time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    correction_rate = n_corrected / max(n_pre_flag, 1)
    net_delta = n_post_flag - n_pre_flag

    summary = {
        "exp_id": args.exp_id,
        "n_prompts": len(prompts),
        "n_active_features_total": total_active,
        "n_active_per_hp": {hp: len(feat_set[hp]) for hp in HOOKPOINTS},
        "n_pre_flagged": n_pre_flag,
        "n_post_flagged": n_post_flag,
        "n_corrected": n_corrected,
        "n_new_fp": n_new_fp,
        "correction_rate": correction_rate,
        "net_delta_flag_rate": net_delta,
        "elapsed_s": elapsed,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== {args.exp_id} summary ===")
    print(f"  pre_flagged={n_pre_flag}/{len(prompts)}  post_flagged={n_post_flag}/{len(prompts)}")
    print(f"  corrected={n_corrected}/{n_pre_flag} ({correction_rate:.4f})")
    print(f"  new_fp={n_new_fp}")
    print(f"  net_delta_flag_rate: {net_delta:+d}")
    print(f"  elapsed: {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
