#!/usr/bin/env python
"""C-4 multi-concept defense — generate violence-category I2P prompts with raw + SAE capture.

Pulls violence-category prompts from I2P (n=300 default), generates SDXL Turbo
1-step + raw + SAE capture at the four Surkov hookpoints. Output:
outputs/raw_violence_<n>/{i2p, raw/, sae/}.

Used for:
  - Cross-concept Stage 1 ∩ Stage 2 F_c selection (violence-specific F_c)
  - Concept-overlap analysis (does the nudity F_c also flag violence?)
  - Multi-concept simultaneous defense (union of nudity + violence F_c)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="raw_violence_300")
    ap.add_argument("--n-prompts", type=int, default=300)
    ap.add_argument("--category", default="violence")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed-offset", type=int, default=8_000_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from dsi.config import cfg
    from dsi.data.i2p import load_i2p
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.hooks import SurkovHookManager
    from dsi.sae.load import load_surkov_sae

    out_dir = cfg.paths.output_root / args.exp_id
    img_dir = out_dir / "i2p"; raw_dir = out_dir / "raw"; sae_dir = out_dir / "sae"
    for d in (img_dir, raw_dir, sae_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"loading I2P prompts category={args.category}")
    all_prompts = load_i2p("full", limit=4703)
    cat_prompts = [p for p in all_prompts if args.category in p.category.lower()]
    cat_prompts = cat_prompts[: args.n_prompts]
    print(f"  {len(cat_prompts)} prompts in '{args.category}'")

    print("loading SDXL Turbo + 4 Surkov SAEs")
    pipe_w = SDXLPipelineWrapper(variant="turbo", device=args.device, dtype="fp16").load()
    saes = {hp: load_surkov_sae(hp).to(args.device).eval()
            for hp in ("down.2.1", "mid.0", "up.0.0", "up.0.1")}
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    n_done = 0
    for start in range(0, len(cat_prompts), args.batch_size):
        batch = cat_prompts[start:start + args.batch_size]
        seeds = list(range(args.seed_offset + start, args.seed_offset + start + len(batch)))
        gens = [torch.Generator(device=args.device).manual_seed(s) for s in seeds]
        with SurkovHookManager(pipe_w.unet, saes, capture=True, keep_inputs=True) as mgr:
            out = pipe_w.pipe(prompt=[p.text for p in batch], num_inference_steps=1,
                              guidance_scale=0.0, generator=gens, height=512, width=512)
        for i, (img, prompt, seed) in enumerate(zip(out.images, batch, seeds)):
            stem = f"{seed:08d}"
            img.save(img_dir / f"{stem}.png")
            (img_dir / f"{stem}.png.meta.json").write_text(json.dumps(
                {"prompt": prompt.text, "seed": seed, "category": prompt.category}))
            raw_payload = {}; sae_payload = {}
            for hp in ("down.2.1", "mid.0", "up.0.0", "up.0.1"):
                if mgr.captured[hp].inputs:
                    rin = mgr.captured[hp].inputs[0][i]   # (C, H, W)
                    raw_payload[hp] = rin.float().mean(dim=(1, 2)).cpu()  # (C,)
                if mgr.captured[hp].z:
                    z = mgr.captured[hp].z[0][i]   # (H, W, D) or (D,)
                    if z.ndim > 1:
                        z = z.float().mean(dim=tuple(range(z.ndim - 1)))
                    sae_payload[hp] = z.cpu()
            torch.save(raw_payload, raw_dir / f"{seed:08d}.raw.pt")
            torch.save(sae_payload, sae_dir / f"{seed:08d}.sae.pt")
            n_done += 1
        if (start // args.batch_size) % 5 == 0:
            print(f"  [{n_done}/{len(cat_prompts)}] {time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0
    summary = {"exp_id": args.exp_id, "category": args.category, "n_done": n_done,
               "elapsed_s": elapsed, "peak_vram_gb": peak}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
