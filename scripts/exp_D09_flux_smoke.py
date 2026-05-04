#!/usr/bin/env python
"""Phase D-9 — FLUX cross-architecture smoke + activation collection.

Step 1 of D-9: load FLUX.1-schnell, hook its DiT transformer blocks,
generate a small batch of I2P + COCO prompts, and capture per-block
post-attention residual activations. Saves them as numpy arrays for
downstream SAE training.

Architecture note: FLUX uses a Multi-modal DiT (MM-DiT) that runs a
joint text+image transformer. The double-stream blocks (0..18) and
single-stream blocks (0..37) each have a residual stream we can hook.
We sample 4 layers at structurally analogous depths to SDXL's
{down.2.1, mid.0, up.0.0, up.0.1} → FLUX
{transformer.transformer_blocks.4, .9, .14, .18}.

Output: outputs/D09_flux_activations/<prompt_split>/<seed>.flux.pt
        (dict[hookpoint -> tensor]).
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

# 4 hookpoint indices into FLUX double-stream blocks
HOOKPOINT_INDICES = (4, 9, 14, 18)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-id", default="D09_flux_smoke")
    ap.add_argument("--n-i2p", type=int, default=20)
    ap.add_argument("--n-coco", type=int, default=20)
    ap.add_argument("--num-inference-steps", type=int, default=20)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=900_000_000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    import numpy as np
    import torch

    out_dir = cfg.paths.output_root / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    activations_dir = out_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("loading FLUX.1-schnell", flush=True)
    from diffusers import StableDiffusion3Pipeline
    td = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=td,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    # Discover available transformer blocks
    n_blocks = len(pipe.transformer.transformer_blocks)
    print(f"  FLUX transformer has {n_blocks} double-stream blocks", flush=True)
    actual_indices = [i for i in HOOKPOINT_INDICES if i < n_blocks]
    print(f"  hooking indices: {actual_indices}", flush=True)

    captured: dict[int, list] = {i: [] for i in actual_indices}

    handles = []

    def make_hook(layer_idx: int):
        def hook(module, args_t, output):
            # Output is a tuple typically (encoder_hidden, hidden) or just hidden
            if isinstance(output, tuple):
                h = output[-1] if hasattr(output[-1], "shape") else output[0]
            else:
                h = output
            if hasattr(h, "shape"):
                # Spatial-mean-pool to (B, D)
                v = h.float().mean(dim=tuple(range(1, h.ndim - 1)))
                captured[layer_idx].append(v.detach().cpu())
        return hook

    for idx in actual_indices:
        block = pipe.transformer.transformer_blocks[idx]
        h = block.register_forward_hook(make_hook(idx))
        handles.append(h)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Load prompts
    from dsi.data.i2p import i2p_nsfw_subset
    from dsi.data.coco import load_coco_captions
    nsfw_prompts = i2p_nsfw_subset(limit=args.n_i2p)
    coco_prompts = load_coco_captions(limit=args.n_coco)

    # Generate
    rendered = []
    t0 = time.time()
    for source, label, prompts in (("i2p_nsfw", 1, nsfw_prompts), ("coco", 0, coco_prompts)):
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start:start + args.batch_size]
            seed = args.seed + start
            gens = [torch.Generator(device=args.device).manual_seed(seed + i) for i in range(len(batch))]
            try:
                # clear captured between batches
                for k in captured:
                    captured[k] = []
                out = pipe(
                    prompt=[p.text for p in batch],
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=7.0,  # SD3 prefers CFG ≈ 7
                    height=args.height, width=args.width,
                    generator=gens,
                )
                imgs = out.images
            except Exception as e:
                print(f"  error in batch: {type(e).__name__}: {e}", flush=True)
                continue
            for j, (img, prompt) in enumerate(zip(imgs, batch)):
                cur_seed = seed + j
                stem = f"{source}_{cur_seed:08d}"
                img.save(images_dir / f"{stem}.png")
                # Save activations: stack per-step per-layer mean-pooled features
                payload = {}
                for k, vs in captured.items():
                    if vs:
                        # Take last batch element across timesteps
                        # captured[k] has [step] x (B, D); we want this batch element only
                        per_step = torch.stack([v[j] for v in vs if j < v.shape[0]], dim=0)  # (T, D)
                        payload[f"flux_block_{k}"] = per_step.float()
                if payload:
                    torch.save(payload, activations_dir / f"{stem}.flux.pt")
                rendered.append({"source": source, "label": label, "seed": cur_seed,
                                 "prompt": prompt.text[:200]})
            print(f"  rendered {len(rendered)} ({source})  elapsed={time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0
    summary = {
        "exp_id": args.exp_id,
        "n_rendered": len(rendered),
        "n_blocks_total": n_blocks,
        "hookpoint_indices": actual_indices,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram,
        "rendered": rendered,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rendered"}, indent=2))

    for h in handles:
        h.remove()
    return 0


if __name__ == "__main__":
    sys.exit(main())
