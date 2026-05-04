#!/usr/bin/env python
"""Single-script entrypoint for Item 2.1 / 2.2 / 2.3 attack runs.

Usage:
  python scripts/run_attack.py --space pixel --eps 0.0157 --n-steps 40 --n-prompts 50 \
                               --collect-sae --exp-id A01_pixel_eps4_50p

Defaults match `task_descriptions/task_description_v1.md` §3 / §5 Item 2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import EPS_GRID_EMBEDDING, EPS_GRID_LATENT, EPS_GRID_PIXEL, cfg  # noqa: E402

DEFAULT_LR = {"pixel": 1.0 / 255, "latent": 0.005, "embedding": 0.05}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--space", choices=["pixel", "latent", "embedding"], required=True)
    ap.add_argument("--eps", type=float, default=None,
                    help="ℓ∞ epsilon; defaults to per-space middle of EPS_GRID_*")
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--collect-sae", action="store_true",
                    help="capture SAE activations during seed generation")
    ap.add_argument("--exp-id", default=None,
                    help="explicit experiment id; auto if omitted")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    args = ap.parse_args()

    eps_default = {
        "pixel": EPS_GRID_PIXEL[1],          # 4/255
        "latent": EPS_GRID_LATENT[1],        # 0.1
        "embedding": EPS_GRID_EMBEDDING[1],  # 0.5
    }
    eps = args.eps if args.eps is not None else eps_default[args.space]
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.space]

    exp_id = args.exp_id or f"A0{ {'pixel': 1, 'latent': 2, 'embedding': 3}[args.space] }_{args.space}_eps{int(eps*255):d}-255_n{args.n_prompts}"
    out_dir = cfg.paths.output_root / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {exp_id} ===")
    print(f"  space={args.space}, eps={eps:.6f}, n_steps={args.n_steps}, lr={lr:.6f}")
    print(f"  n_prompts={args.n_prompts}, batch={args.batch_size}, collect_sae={args.collect_sae}")

    print("loading SDXL Turbo")
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper

    pipe_w = SDXLPipelineWrapper(variant="turbo", device=args.device, dtype=args.dtype).load()

    print("loading SafetyTarget")
    from dsi.attacks.safety_target import SafetyTarget

    target = SafetyTarget(device=args.device, dtype=args.dtype).load(vae=pipe_w.vae)

    collect_sae = None
    if args.collect_sae:
        print("loading 4 Surkov SAEs for activation capture")
        from dsi.sae.load import load_surkov_sae

        collect_sae = {
            hp: load_surkov_sae(hp).to(args.device).eval()
            for hp in ("down.2.1", "mid.0", "up.0.0", "up.0.1")
        }

    print("loading I2P-NSFW prompts")
    from dsi.data.i2p import i2p_nsfw_subset

    prompts = i2p_nsfw_subset(limit=args.n_prompts)
    print(f"  {len(prompts)} prompts")

    import torch

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    if args.space == "pixel":
        from dsi.attacks.runner import run_attack_pixel

        outcomes = run_attack_pixel(
            pipe_w=pipe_w, target=target, prompts=prompts,
            eps=eps, n_steps=args.n_steps, lr=lr,
            seed_offset=args.seed_offset, out_dir=out_dir,
            batch_size=args.batch_size, collect_sae=collect_sae,
        )
    elif args.space == "embedding":
        from dsi.attacks.runner import run_attack_embedding

        outcomes = run_attack_embedding(
            pipe_w=pipe_w, target=target, prompts=prompts,
            eps=eps, n_steps=args.n_steps, lr=lr,
            seed_offset=args.seed_offset, out_dir=out_dir,
            batch_size=args.batch_size, collect_sae=collect_sae,
        )
    elif args.space == "latent":
        from dsi.attacks.runner import run_attack_latent

        outcomes = run_attack_latent(
            pipe_w=pipe_w, target=target, prompts=prompts,
            eps=eps, n_steps=args.n_steps, lr=lr,
            seed_offset=args.seed_offset, out_dir=out_dir,
            batch_size=args.batch_size, collect_sae=collect_sae,
        )
    else:
        raise ValueError(args.space)

    elapsed = time.time() - t0
    peak_vram = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if args.device == "cuda" else 0.0

    n = len(outcomes)
    n_pre_flagged = sum(o.pre_flagged for o in outcomes)
    n_post_flagged = sum(o.post_flagged for o in outcomes)
    n_bypass = sum(o.bypass for o in outcomes)
    asr = n_bypass / max(1, n_pre_flagged)
    asr_all = n_bypass / max(1, n)
    pre_safe_mean = sum(o.pre_safe_logit for o in outcomes) / max(1, n)
    post_safe_mean = sum(o.post_safe_logit for o in outcomes) / max(1, n)
    pert_norm_mean = sum(o.perturb_norm for o in outcomes) / max(1, n)

    summary = {
        "exp_id": exp_id, "space": args.space, "eps": eps, "n_steps": args.n_steps, "lr": lr,
        "n_prompts": n,
        "n_pre_flagged": n_pre_flagged, "n_post_flagged": n_post_flagged, "n_bypass": n_bypass,
        "asr_among_pre_flagged": asr, "asr_among_all": asr_all,
        "pre_safe_logit_mean": pre_safe_mean, "post_safe_logit_mean": post_safe_mean,
        "perturb_norm_mean": pert_norm_mean,
        "peak_vram_gb": peak_vram, "elapsed_s": elapsed,
        "out_dir": str(out_dir),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
