#!/usr/bin/env python
"""Item 5 / Contribution 4 — detection-triggered correction pipeline.

For each prompt:
  1. Run SDXL Turbo unconditionally → seed image.
  2. Run *with* SAE-detector hook + intervene_fn that on detector firing patches
     the Stage-2-survivor features at this hookpoint to their per-feature benign
     mean (mean-patch), at every spatial position, from step k+1 onwards.
  3. Save both the un-intervened and intervened images.
  4. Score with Q16 / NudeNet / safety_checker oracles.

Compares:
  - "no defense" baseline (un-intervened image safety)
  - "two-stage + mean patch" intervention (the proposed Contribution 4)

Output goes to outputs/D01_<concept>_intervened/{pre,post}/<seed>.png + summary.json.
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
    ap.add_argument("--exp-id", default="D01_two_stage_meanpatch_n100")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--feature-set", required=True,
                    help="JSON file with {hookpoint: [feat_idx, ...]} per Surkov hookpoint")
    ap.add_argument("--mu-file", required=True,
                    help="Numpy .npz with mu_<hookpoint> arrays = (D,) benign-mean per hookpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--seed-offset", type=int, default=3_000_000)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    import numpy as np
    import torch

    out_dir = cfg.paths.output_root / args.exp_id
    pre_dir = out_dir / "pre"
    post_dir = out_dir / "post"
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {args.exp_id} ===")
    feat_set = json.loads(Path(args.feature_set).read_text())
    mu_npz = np.load(args.mu_file)
    print(f"loaded feature set: {[hp + ':' + str(len(v)) for hp, v in feat_set.items()]}")
    print(f"loaded benign means: {list(mu_npz.files)}")

    print("loading SDXL Turbo + SAEs")
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.hooks import SurkovHookManager
    from dsi.sae.load import load_surkov_sae

    pipe_w = SDXLPipelineWrapper(variant="turbo", device=args.device, dtype=args.dtype).load()
    saes = {hp: load_surkov_sae(hp).to(args.device).eval() for hp in feat_set}

    # Pre-compute per-hookpoint feat tensors and mu tensors on device
    feat_t = {hp: torch.tensor(v, device=args.device, dtype=torch.long) for hp, v in feat_set.items()}
    mu_t = {hp: torch.as_tensor(mu_npz[f"mu_{hp.replace('.', '_')}"],
                                device=args.device, dtype=next(saes[hp].parameters()).dtype)
            for hp in feat_set if f"mu_{hp.replace('.', '_')}" in mu_npz.files}

    def intervene(hp, z, step):
        idx = feat_t.get(hp)
        mu = mu_t.get(hp)
        if idx is None or mu is None:
            return None
        z_new = z.clone()
        # Replace the selected features at every spatial location with the benign mean
        z_new[..., idx] = mu[idx]
        return z_new

    print("loading I2P-NSFW prompts")
    from dsi.data.i2p import i2p_nsfw_subset

    prompts = i2p_nsfw_subset(limit=args.n_prompts)
    print(f"  {len(prompts)} prompts")

    print("loading SafetyTarget for oracle scoring")
    from dsi.attacks.safety_target import SafetyTarget

    target = SafetyTarget(device=args.device, dtype=args.dtype).load(vae=pipe_w.vae)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    n_pre_flagged = 0
    n_post_flagged = 0
    n_corrected = 0
    rows = []
    t0 = time.time()

    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        seeds = list(range(args.seed_offset + start, args.seed_offset + start + len(batch)))
        gens = [torch.Generator(device=args.device).manual_seed(s) for s in seeds]

        # 1. un-intervened
        out = pipe_w.pipe(prompt=[p.text for p in batch], num_inference_steps=1,
                          guidance_scale=0.0, generator=gens, height=512, width=512)
        pre_imgs = out.images
        # 2. intervened — same seeds for fair comparison
        gens2 = [torch.Generator(device=args.device).manual_seed(s) for s in seeds]
        with SurkovHookManager(pipe_w.unet, saes, capture=False, intervene_fn=intervene,
                               device=args.device):
            out_int = pipe_w.pipe(prompt=[p.text for p in batch], num_inference_steps=1,
                                  guidance_scale=0.0, generator=gens2, height=512, width=512)
        post_imgs = out_int.images

        # 3. score both via the safety_target (matches our attack-side metric)
        import numpy as np
        x_pre = torch.stack([
            torch.as_tensor(np.array(im, dtype=np.float32) / 255.0).permute(2, 0, 1) for im in pre_imgs
        ]).to(args.device)
        x_post = torch.stack([
            torch.as_tensor(np.array(im, dtype=np.float32) / 255.0).permute(2, 0, 1) for im in post_imgs
        ]).to(args.device)
        with torch.no_grad():
            l_pre = target.pixel_to_logits(x_pre).float()
            l_post = target.pixel_to_logits(x_post).float()
        pre_flag = (l_pre[:, 0] > 0).cpu().tolist()
        post_flag = (l_post[:, 0] > 0).cpu().tolist()

        for i, (im_pre, im_post, prompt, seed) in enumerate(zip(pre_imgs, post_imgs, batch, seeds)):
            stem = f"{seed:08d}"
            im_pre.save(pre_dir / f"{stem}.png")
            im_post.save(post_dir / f"{stem}.png")
            n_pre_flagged += int(pre_flag[i])
            n_post_flagged += int(post_flag[i])
            if pre_flag[i] and not post_flag[i]:
                n_corrected += 1
            rows.append({
                "seed": seed, "prompt": prompt.text,
                "pre_flagged": pre_flag[i], "post_flagged": post_flag[i],
                "pre_safe_logit": float(l_pre[i, 1]), "post_safe_logit": float(l_post[i, 1]),
                "corrected": pre_flag[i] and not post_flag[i],
            })
        if (start // args.batch_size) % 5 == 0:
            print(f"  [{start+len(batch)}/{len(prompts)}] {time.time()-t0:.1f}s "
                  f"pre_flagged={n_pre_flagged} post_flagged={n_post_flagged} corrected={n_corrected}",
                  flush=True)

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0
    summary = {
        "exp_id": args.exp_id,
        "n_prompts": len(prompts),
        "n_pre_flagged": n_pre_flagged,
        "n_post_flagged": n_post_flagged,
        "n_corrected": n_corrected,
        "correction_rate_among_pre_flagged": n_corrected / max(1, n_pre_flagged),
        "post_flagged_rate_overall": n_post_flagged / max(1, len(prompts)),
        "peak_vram_gb": peak,
        "elapsed_s": elapsed,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
