#!/usr/bin/env python
"""Generate SDXL Turbo images and capture BOTH raw UNet residual diffs AND
SAE-encoded activations + predicted-noise summary at the four Surkov hookpoints.

Used for:
  - Phase C-2 (AxBench raw-activation probe baseline)
  - Phase C-6 (hybrid SAE + predicted-noise detector)

Sidecar files per generation:
  <exp_dir>/raw/<seed>.raw.pt   — dict[hookpoint -> mean-pooled raw residual (D,)]
  <exp_dir>/sae/<seed>.sae.pt   — dict[hookpoint -> mean-pooled SAE z (D,)]   (existing format)
  <exp_dir>/noise/<seed>.noise.pt — dict[step -> mean-pooled UNet predicted-noise stats (3 stats * 4 spatial blocks)]
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
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--prompts-source", choices=["i2p_nsfw", "coco"], default="coco")
    ap.add_argument("--n-prompts", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed-offset", type=int, default=5_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--num-inference-steps", type=int, default=1)
    args = ap.parse_args()

    import torch

    out_dir = cfg.paths.output_root / args.exp_id
    raw_dir = out_dir / "raw"
    sae_dir = out_dir / "sae"
    noise_dir = out_dir / "noise"
    pre_dir = out_dir / "pre"
    for d in (raw_dir, sae_dir, noise_dir, pre_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"=== {args.exp_id} ===")
    print(f"  prompts={args.prompts_source}, n={args.n_prompts}, batch={args.batch_size}")

    print("loading SDXL Turbo")
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    pipe_w = SDXLPipelineWrapper(variant="turbo", device=args.device, dtype=args.dtype).load()

    print("loading 4 Surkov SAEs")
    from dsi.sae.load import load_surkov_sae
    saes = {hp: load_surkov_sae(hp).to(args.device).eval() for hp in HOOKPOINTS}

    print(f"loading {args.prompts_source} prompts")
    if args.prompts_source == "i2p_nsfw":
        from dsi.data.i2p import i2p_nsfw_subset
        prompts = i2p_nsfw_subset(limit=args.n_prompts)
    else:
        from dsi.data.coco import load_coco_captions
        prompts = load_coco_captions(limit=args.n_prompts)
    print(f"  {len(prompts)} prompts")

    # Custom hook manager: captures BOTH raw residual diff AND SAE z
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
            # Spatial mean → (B, D)
            captured[hp]["raw"].append(x_bhwc.float().mean(dim=tuple(range(1, x_bhwc.ndim - 1))).cpu())
            captured[hp]["z"].append(z.float().mean(dim=tuple(range(1, z.ndim - 1))).cpu())
            return None
        return post_hook

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    n_done = 0
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        seeds = list(range(args.seed_offset + start, args.seed_offset + start + len(batch)))
        gens = [torch.Generator(device=args.device).manual_seed(s) for s in seeds]

        for hp in HOOKPOINTS:
            captured[hp]["raw"].clear()
            captured[hp]["z"].clear()
        for hp in HOOKPOINTS:
            target = HOOKPOINT_TO_GETTER[hp](pipe_w.unet)
            handles.append(target.register_forward_hook(make_hook(hp), with_kwargs=True))

        out = pipe_w.pipe(prompt=[p.text for p in batch],
                          num_inference_steps=args.num_inference_steps,
                          guidance_scale=0.0, generator=gens, height=512, width=512)
        for h in handles:
            h.remove()
        handles.clear()

        for i, (img, prompt, seed) in enumerate(zip(out.images, batch, seeds)):
            stem = f"{seed:08d}"
            img.save(pre_dir / f"{stem}.png")
            raw_payload = {hp: captured[hp]["raw"][0][i].clone() for hp in HOOKPOINTS}
            sae_payload = {hp: captured[hp]["z"][0][i].clone() for hp in HOOKPOINTS}
            torch.save(raw_payload, raw_dir / f"{stem}.raw.pt")
            torch.save(sae_payload, sae_dir / f"{stem}.sae.pt")
        n_done += len(batch)
        if (start // args.batch_size) % 5 == 0:
            print(f"  [{n_done}/{len(prompts)}] {time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0
    summary = {"exp_id": args.exp_id, "n": n_done, "elapsed_s": elapsed,
               "peak_vram_gb": peak, "out_dir": str(out_dir),
               "prompts_source": args.prompts_source}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
