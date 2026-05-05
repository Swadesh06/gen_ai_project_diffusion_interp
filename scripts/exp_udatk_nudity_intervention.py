#!/usr/bin/env python
"""Item 1c-4 close — UnlearnDiffAtk-nudity prompts + F_c mean-patch intervention.

Mirrors `run_intervention.py` for the I2P-NSFW prompt source, but loads the
UDA-nudity CSV instead. Uses SDXL Base 4-step (UDA's reference setting) +
4 Surkov SAEs at the same hookpoints + F_c mean-patch on Stage 1 ∩ Stage 2
nudity features.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import cfg


HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")


def load_uda_prompts(path, limit=None):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row["prompt"].strip())
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="udatk_nudity_meanpatch_n100")
    ap.add_argument("--prompts-csv", default="/workspace/datasets/Diffusion-MU-Attack/prompts/nudity.csv")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--feature-set", default="outputs/F_c_stage1n2_top.json")
    ap.add_argument("--mu-file", default="outputs/detector_dataset_v1/mu.npz")
    ap.add_argument("--seed-offset", type=int, default=42_500_000)
    ap.add_argument("--variant", choices=["turbo", "base"], default="base")
    ap.add_argument("--num-inference-steps", type=int, default=4)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    out_dir = cfg.paths.output_root / args.exp_id
    pre_dir = out_dir / "pre"
    post_dir = out_dir / "post"
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {args.exp_id} ===")
    prompts = load_uda_prompts(args.prompts_csv, limit=args.n_prompts)
    print(f"  loaded {len(prompts)} UDA-nudity prompts")
    feat_set = json.loads(Path(args.feature_set).read_text())
    print(f"  feature set: {[hp + ':' + str(len(v)) for hp, v in feat_set.items()]}")
    mu_npz = np.load(args.mu_file)

    print("loading SDXL Base + SAEs")
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.hooks import SurkovHookManager
    from dsi.sae.load import load_surkov_sae
    from dsi.detectors.baselines.safety_checker import SafetyCheckerWrapper

    pipe_w = SDXLPipelineWrapper(variant=args.variant, device="cuda", dtype="fp16").load()
    saes = {hp: load_surkov_sae(hp).to("cuda").eval() for hp in HOOKPOINTS}
    sc = SafetyCheckerWrapper(device="cuda").load()

    feat_t = {hp: torch.tensor(feat_set.get(hp, []), device="cuda", dtype=torch.long) for hp in HOOKPOINTS}
    mu_t = {hp: torch.as_tensor(mu_npz.get(f"mu_{hp.replace('.', '_')}"),
                                device="cuda", dtype=next(saes[hp].parameters()).dtype)
            for hp in HOOKPOINTS if f"mu_{hp.replace('.', '_')}" in mu_npz.files}

    def intervene_fn(hp, z, step):
        idx = feat_t.get(hp, None)
        if idx is None or len(idx) == 0:
            return None
        z_new = z.clone()
        # mu_t is the per-feature mean, indexed by the F_c indices
        # z_new[..., idx] = mu_t[hp][idx]
        z_new[..., idx] = mu_t[hp][idx].view(*([1]*(z.ndim - 1)), -1)
        return z_new

    print("rendering pre + post")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_pre_flag = 0
    n_post_flag = 0
    n_correct = 0
    rows = []
    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device="cuda").manual_seed(args.seed_offset + i)
        # baseline pre
        out_pre = pipe_w.pipe(prompt=prompt, num_inference_steps=args.num_inference_steps,
                              guidance_scale=args.guidance_scale, generator=gen, height=512, width=512)
        img_pre = out_pre.images[0]
        flag_pre = bool(sc.score([img_pre])[0]["flagged"])
        # post with F_c hook
        gen = torch.Generator(device="cuda").manual_seed(args.seed_offset + i)
        with SurkovHookManager(pipe_w.unet, saes, capture=False, intervene_fn=intervene_fn) as mgr:
            out_post = pipe_w.pipe(prompt=prompt, num_inference_steps=args.num_inference_steps,
                                   guidance_scale=args.guidance_scale, generator=gen, height=512, width=512)
        img_post = out_post.images[0]
        flag_post = bool(sc.score([img_post])[0]["flagged"])

        n_pre_flag += int(flag_pre)
        n_post_flag += int(flag_post)
        if flag_pre and not flag_post:
            n_correct += 1

        stem = f"{args.seed_offset + i:08d}"
        img_pre.save(pre_dir / f"{stem}.png")
        img_post.save(post_dir / f"{stem}.png")
        rows.append({"seed": args.seed_offset + i, "prompt": prompt[:200],
                     "pre_flagged": flag_pre, "post_flagged": flag_post,
                     "corrected": flag_pre and not flag_post})

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}] {time.time()-t0:.1f}s pre_flag={n_pre_flag} post_flag={n_post_flag} corrected={n_correct}", flush=True)

    elapsed = time.time() - t0
    summary = {
        "exp_id": args.exp_id,
        "n_prompts": len(prompts),
        "n_pre_flagged": n_pre_flag,
        "n_post_flagged": n_post_flag,
        "n_corrected": n_correct,
        "correction_rate_among_pre_flagged": n_correct / max(1, n_pre_flag),
        "post_flagged_rate_overall": n_post_flag / max(1, len(prompts)),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "elapsed_s": elapsed,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE: pre_flag={n_pre_flag}/{len(prompts)}, corrected={n_correct}/{n_pre_flag} ({n_correct/max(1,n_pre_flag):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
