#!/usr/bin/env python
"""Phase D-7 — mechanistic feature-firing trajectory plot (canonical paper figure).

For 3-5 cherry-picked bypass cases:
  1. Pull the original prompt + bypass image (saved during attack).
  2. Re-render with SDXL Base 4-step capturing SAE activations at every
     UNet step at all 4 Surkov hookpoints.
  3. Pull the post-attack image's UNet trace via VAE-encode + multi-step
     inversion (or just one VAE-encode + 4-step renderer with the
     perturbed image as conditioning).
  4. Plot top-10 (by max |Δ|) features' activation magnitude per step for
     both clean and attacked. Where they diverge tells the story of when
     the unsafe content materialises.

Output:
  outputs/D07_mechanistic_trajectory/<seed>/{trajectory.json, plot.pdf}
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

HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-id", default="D07_mechanistic_trajectory")
    ap.add_argument("--attack-dir", default="outputs/A01_pixel_eps4_n200",
                    help="dir with pre/<seed>.png + <seed>.attack.json")
    ap.add_argument("--n-cases", type=int, default=5)
    ap.add_argument("--num-inference-steps", type=int, default=4)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k-features", type=int, default=10)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    out_dir = cfg.paths.output_root / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    attack_dir = Path(args.attack_dir)

    print("loading SDXL Base + 4 Surkov SAEs", flush=True)
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.hooks import SurkovHookManager
    from dsi.sae.load import load_surkov_sae

    pipe_w = SDXLPipelineWrapper(variant="base", device=args.device, dtype="fp16").load()
    sae_dict = {hp: load_surkov_sae(hp).to(args.device).eval() for hp in HOOKPOINTS}

    # Pick 5 successful bypass seeds: pre_flagged=True AND post_flagged=False
    print("collecting bypass cases", flush=True)
    cases: list[dict] = []
    for atk in sorted(attack_dir.glob("*.attack.json")):
        try:
            r = json.loads(atk.read_text())
        except Exception:
            continue
        if r.get("pre_flagged") and (not r.get("post_flagged")):
            cases.append(r)
        if len(cases) >= args.n_cases:
            break
    print(f"  {len(cases)} bypass cases", flush=True)

    pre_dir = attack_dir / "pre"
    post_dir = attack_dir / "post"
    sae_root = attack_dir / "sae"

    @torch.no_grad()
    def trace_steps(prompt: str, *, latent_init=None, seed=0):
        """Run SDXL Base for n_inference_steps with SurkovHookManager active.
        Optionally start from a specific latent (for re-tracing the post-attack image).
        """
        gen = torch.Generator(device=args.device).manual_seed(seed)
        with SurkovHookManager(pipe_w.unet, sae_dict, capture=True, keep_inputs=False) as mgr:
            kwargs = dict(
                prompt=[prompt],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
                height=512, width=512,
            )
            if latent_init is not None:
                kwargs["latents"] = latent_init
            _ = pipe_w.pipe(**kwargs)
        # Per-step per-hookpoint mean-pooled feature activation
        per_step = {}
        for hp in HOOKPOINTS:
            zs = mgr.captured[hp].z
            steps = []
            for z in zs:
                a = z.float()
                # spatial mean → (D,) per step
                spatial = tuple(range(1, a.ndim - 1))
                v = a.mean(dim=spatial)[0].cpu().numpy()
                steps.append(v)
            per_step[hp] = np.stack(steps) if steps else np.zeros((0, 1))
        return per_step

    @torch.no_grad()
    def img_to_latent(pil_img):
        import numpy as np
        a = np.asarray(pil_img.convert("RGB").resize((512, 512)), dtype=np.float32) / 255.0
        a = a * 2.0 - 1.0
        x = torch.as_tensor(a.transpose(2, 0, 1)[None], device=args.device, dtype=pipe_w.vae.dtype)
        latent = pipe_w.vae.encode(x).latent_dist.sample()
        latent = latent * pipe_w.vae.config.scaling_factor
        # Resize to expected SDXL Base resolution at 4-step (the pipeline handles 512 internally)
        return latent

    out_summary: list[dict] = []
    for case in cases:
        seed = case["seed"]
        prompt = case["prompt"]
        case_dir = out_dir / f"seed_{seed:08d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        # Trajectory A: re-render from prompt only (clean trajectory)
        print(f"\n=== seed {seed} ===  clean trajectory")
        clean_traj = trace_steps(prompt, latent_init=None, seed=seed)
        # Trajectory B: re-render with the post-attack image as starting latent
        post_path = post_dir / f"{seed:08d}.png"
        if not post_path.exists():
            print(f"  no post image at {post_path}; skipping attacked trajectory")
            attacked_traj = None
        else:
            post_pil = Image.open(post_path).convert("RGB")
            post_lat = img_to_latent(post_pil)
            print(f"  attacked trajectory from {post_path}")
            attacked_traj = trace_steps(prompt, latent_init=post_lat, seed=seed)

        # Per-hookpoint top-k features by |clean - attacked| at any step
        per_hp_top: dict = {}
        for hp in HOOKPOINTS:
            cs = clean_traj.get(hp)
            if cs is None or cs.size == 0:
                continue
            if attacked_traj is None:
                top = np.argsort(-cs.max(axis=0))[: args.top_k_features]
            else:
                ab = attacked_traj[hp]
                ndiff = np.abs(cs - ab).max(axis=0)
                top = np.argsort(-ndiff)[: args.top_k_features]
            per_hp_top[hp] = {
                "feature_idx": top.tolist(),
                "clean": cs[:, top].tolist(),
                "attacked": (attacked_traj[hp][:, top].tolist() if attacked_traj is not None else None),
            }

        # Save trajectory
        (case_dir / "trajectory.json").write_text(json.dumps({
            "seed": seed, "prompt": prompt, "n_steps": args.num_inference_steps,
            "hookpoints": list(HOOKPOINTS), "top_k": args.top_k_features,
            "per_hp": per_hp_top,
        }))
        # Render plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            for ax, hp in zip(axes.flatten(), HOOKPOINTS):
                rec = per_hp_top.get(hp)
                if rec is None:
                    ax.set_title(f"{hp}: (no data)")
                    continue
                clean_arr = np.asarray(rec["clean"])
                attacked_arr = (np.asarray(rec["attacked"])
                                if rec["attacked"] is not None else None)
                steps = np.arange(clean_arr.shape[0])
                for j in range(clean_arr.shape[1]):
                    f = rec["feature_idx"][j]
                    ax.plot(steps, clean_arr[:, j], "-", alpha=0.6, label=f"clean f{f}")
                    if attacked_arr is not None:
                        ax.plot(steps, attacked_arr[:, j], "--", alpha=0.6, label=f"atk f{f}")
                ax.set_title(f"{hp}: top-{args.top_k_features} divergent features")
                ax.set_xlabel("denoising step")
                ax.set_ylabel("mean feature activation")
            fig.suptitle(f"seed {seed:08d}\nprompt: {prompt[:80]}", fontsize=10)
            fig.tight_layout()
            fig.savefig(case_dir / "trajectory.png", dpi=120, bbox_inches="tight")
            fig.savefig(case_dir / "trajectory.pdf", bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"  plot failed: {e}")
        out_summary.append({"seed": seed, "prompt": prompt,
                            "case_dir": str(case_dir), "n_features": args.top_k_features})

    (out_dir / "summary.json").write_text(json.dumps(out_summary, indent=2))
    print(f"\nwrote {len(out_summary)} cases to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
