#!/usr/bin/env python
"""Apply joint mask to I2P-NSFW rendering (SDXL Turbo, the joint mask's
training distribution architecture).

Variant of exp_D06_apply_mask_to_udatk.py for SDXL Turbo + I2P-NSFW
prompts. n=200 default; use n=100 for speed."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from dsi.config import cfg

HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")
F_PER_HP = 5120


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="D06_joint_mask_i2p_n100")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--joint-state",
                    default="outputs/D06_lambda5_state/joint_state.pt")
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--seed-offset", type=int, default=43_000_000)
    args = ap.parse_args()

    import numpy as np
    import torch

    out_dir = cfg.paths.output_root / args.exp_id
    pre_dir = out_dir / "pre"
    post_dir = out_dir / "post"
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {args.exp_id} ===")

    state = torch.load(args.joint_state, map_location="cpu", weights_only=False)
    M = torch.sigmoid(state["theta"])
    mu_benign = state["mu_benign"]

    feat_set, mu_hp = {}, {}
    for i, hp in enumerate(HOOKPOINTS):
        s, e = i * F_PER_HP, (i + 1) * F_PER_HP
        M_hp = M[s:e]
        mu_hp[hp] = mu_benign[s:e]
        idx = (M_hp > args.mask_threshold).nonzero(as_tuple=False).squeeze(-1).tolist()
        feat_set[hp] = idx
        print(f"  {hp}: {len(idx)} active features")
    total = sum(len(v) for v in feat_set.values())
    print(f"  total: {total} / 20480 ({100 * total / 20480:.2f}%)")

    print("loading SDXL Turbo + SAEs + safety_checker")
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.hooks import SurkovHookManager
    from dsi.sae.load import load_surkov_sae
    from dsi.detectors.baselines.safety_checker import SafetyCheckerWrapper

    pipe_w = SDXLPipelineWrapper(variant="turbo", device="cuda", dtype="fp16").load()
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

    print("loading I2P-NSFW prompts")
    from dsi.data.i2p import i2p_nsfw_subset
    prompts = i2p_nsfw_subset(limit=args.n_prompts)
    print(f"  {len(prompts)} prompts")

    print("rendering pre + post")
    t0 = time.time()
    n_pre = n_post = n_corr = n_fp = 0
    rows = []
    for i, p in enumerate(prompts):
        gen_pre = torch.Generator(device="cuda").manual_seed(args.seed_offset + i)
        gen_post = torch.Generator(device="cuda").manual_seed(args.seed_offset + i)
        out_pre = pipe_w.pipe(prompt=[p.text], num_inference_steps=1,
                               guidance_scale=0.0, generator=gen_pre,
                               height=512, width=512)
        img_pre = out_pre.images[0]
        flag_pre = bool(sc.score([img_pre])[0]["flagged"])
        if flag_pre:
            n_pre += 1
        img_pre.save(pre_dir / f"{i:04d}.png")

        with SurkovHookManager(pipe_w.unet, saes, capture=False, intervene_fn=intervene_fn):
            out_post = pipe_w.pipe(prompt=[p.text], num_inference_steps=1,
                                    guidance_scale=0.0, generator=gen_post,
                                    height=512, width=512)
        img_post = out_post.images[0]
        flag_post = bool(sc.score([img_post])[0]["flagged"])
        if flag_post:
            n_post += 1
        img_post.save(post_dir / f"{i:04d}.png")

        if flag_pre and not flag_post:
            n_corr += 1
        if (not flag_pre) and flag_post:
            n_fp += 1

        rows.append({"i": i, "prompt": p.text[:80],
                     "flag_pre": flag_pre, "flag_post": flag_post,
                     "corrected": flag_pre and not flag_post,
                     "new_fp": (not flag_pre) and flag_post})

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(prompts)}] pre={n_pre} post={n_post} "
                  f"corr={n_corr} fp={n_fp} elapsed={time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    cr = n_corr / max(n_pre, 1)
    summary = {"exp_id": args.exp_id, "n_prompts": len(prompts),
               "n_active_features_total": total,
               "n_pre_flagged": n_pre, "n_post_flagged": n_post,
               "n_corrected": n_corr, "n_new_fp": n_fp,
               "correction_rate": cr,
               "net_delta_flag_rate": n_post - n_pre,
               "elapsed_s": elapsed,
               "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== summary ===")
    print(f"  pre={n_pre}, post={n_post}, corrected={n_corr}/{n_pre} ({cr:.4f})")
    print(f"  new_fp={n_fp}, net_delta={n_post - n_pre:+d}, elapsed={elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
