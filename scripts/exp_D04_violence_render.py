#!/usr/bin/env python
"""D-4 violence-concept activation render.

Renders SDXL Turbo on violence-tagged I2P-MU-Attack prompts (n=200) with SAE
hooks at the 4 Surkov hookpoints; saves raw + sae mean-pooled activations and
PNG images. Output mirrors `outputs/raw_coco_500/`'s schema.
"""
from __future__ import annotations

import argparse, csv, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import cfg  # noqa: E402

HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")


def load_violence_prompts(limit=200):
    path = "/workspace/datasets/Diffusion-MU-Attack/prompts/violence.csv"
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row["prompt"].strip())
    return rows[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-id", default="raw_violence_n200")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed-offset", type=int, default=42_000_000)
    args = ap.parse_args()

    import torch

    out_dir = cfg.paths.output_root / args.exp_id
    raw_dir = out_dir / "raw"
    sae_dir = out_dir / "sae"
    pre_dir = out_dir / "pre"
    for d in (raw_dir, sae_dir, pre_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"=== {args.exp_id} ===")

    prompts = load_violence_prompts(args.n_prompts)
    print(f"loaded {len(prompts)} violence prompts")

    print("loading SDXL Turbo")
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    pipe_w = SDXLPipelineWrapper(variant="turbo", device="cuda", dtype="fp16").load()

    print("loading 4 Surkov SAEs")
    from dsi.sae.load import load_surkov_sae
    saes = {hp: load_surkov_sae(hp).to("cuda").eval() for hp in HOOKPOINTS}

    from dsi.sae.hooks import HOOKPOINT_TO_GETTER
    captured = {hp: {"raw": [], "z": []} for hp in HOOKPOINTS}
    handles = []

    def _retrieve(io):
        if io is None:
            return None
        if isinstance(io, tuple):
            return io[0]
        if hasattr(io, "sample"):
            return io.sample
        return io

    def make_hook(hp):
        sae = saes[hp]
        def post_hook(module, args_, kwargs_, output):
            inp = _retrieve(args_ if args_ else kwargs_.get("hidden_states"))
            out = _retrieve(output)
            if inp is None or out is None:
                return None
            diff = (out - inp).detach()
            x_bhwc = diff.permute(0, 2, 3, 1)
            with torch.no_grad():
                z = sae.encode(x_bhwc.to(next(sae.parameters()).device).to(next(sae.parameters()).dtype))
            captured[hp]["raw"].append(x_bhwc.float().mean(dim=tuple(range(1, x_bhwc.ndim - 1))).cpu())
            captured[hp]["z"].append(z.float().mean(dim=tuple(range(1, z.ndim - 1))).cpu())
            return None
        return post_hook

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_done = 0

    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        seeds = list(range(args.seed_offset + start, args.seed_offset + start + len(batch_prompts)))
        gens = [torch.Generator(device="cuda").manual_seed(s) for s in seeds]

        for hp in HOOKPOINTS:
            captured[hp]["raw"].clear()
            captured[hp]["z"].clear()
        for hp in HOOKPOINTS:
            target = HOOKPOINT_TO_GETTER[hp](pipe_w.unet)
            handles.append(target.register_forward_hook(make_hook(hp), with_kwargs=True))

        out = pipe_w.pipe(prompt=batch_prompts, num_inference_steps=1,
                          guidance_scale=0.0, generator=gens, height=512, width=512)
        for h in handles:
            h.remove()
        handles.clear()

        for i, (img, seed) in enumerate(zip(out.images, seeds)):
            stem = f"{seed:08d}"
            img.save(pre_dir / f"{stem}.png")
            raw = {hp: captured[hp]["raw"][0][i].clone() for hp in HOOKPOINTS}
            z = {hp: captured[hp]["z"][0][i].clone() for hp in HOOKPOINTS}
            torch.save(raw, raw_dir / f"{stem}.raw.pt")
            torch.save(z, sae_dir / f"{stem}.sae.pt")
            n_done += 1

        if start % 16 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_done}/{len(prompts)}] {elapsed:.1f}s, peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"DONE: {n_done} renders, {elapsed:.1f}s, peak VRAM {peak:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
