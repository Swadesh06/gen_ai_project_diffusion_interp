#!/usr/bin/env python
"""Counterfactual benchmark Strategy 2 — same-prompt different-seed pairs.

v2 §3 Item 1c-0 Strategy 2. Identify ~100 I2P prompts that flag stochastically
(NudeNet positive on ~50% of seeds). Generate 8 seeds per prompt at SDXL Base
4-step. Pair flagged-seed and unflagged-seed generations from the IDENTICAL
prompt — prompt distribution is exactly held constant.

Output: outputs/cf_benchmark_v1_seed/
    seed_pairs.jsonl
    by_prompt/<prompt_id>/seed_<S>.png
    validated.jsonl after labels exist
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


def cmd_build(args) -> int:
    from dsi.data.i2p import load_i2p
    import torch
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper

    out_dir = cfg.paths.output_root / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    nsfw = [p for p in load_i2p("full", limit=None) if p.label == "nsfw"]
    chosen = nsfw[: args.n_prompts]
    print(f"chosen {len(chosen)} I2P NSFW prompts", flush=True)

    pipe_w = SDXLPipelineWrapper(variant="base", device=args.device, dtype=args.dtype).load()

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rendered: list[dict] = []
    t0 = time.time()
    for pi, prompt in enumerate(chosen):
        pid = f"p{pi:04d}"
        p_dir = out_dir / "by_prompt" / pid
        p_dir.mkdir(parents=True, exist_ok=True)
        # Save the prompt text for reference
        (p_dir / "prompt.txt").write_text(prompt.text)
        for sb in range(0, args.n_seeds, args.batch_size):
            batch_seeds = list(range(args.seed + pi * 1000 + sb, args.seed + pi * 1000 + min(sb + args.batch_size, args.n_seeds)))
            gens = [torch.Generator(device=args.device).manual_seed(s) for s in batch_seeds]
            imgs = pipe_w.pipe(
                prompt=[prompt.text] * len(batch_seeds),
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=gens,
                height=512,
                width=512,
            ).images
            for img, s in zip(imgs, batch_seeds):
                p_png = p_dir / f"seed_{s:08d}.png"
                img.save(p_png)
                (p_dir / f"seed_{s:08d}.png.meta.json").write_text(json.dumps(
                    {"prompt": prompt.text, "seed": s, "prompt_id": pid,
                     "category": prompt.category}
                ))
                rendered.append({"prompt_id": pid, "seed": s, "path": str(p_png)})
        if pi % 10 == 0:
            elapsed = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0
            print(f"  prompts {pi+1}/{len(chosen)}  elapsed={elapsed:.1f}s  peak_vram={peak:.1f}GB", flush=True)

    rendered_path = out_dir / "rendered.jsonl"
    with rendered_path.open("w") as f:
        for r in rendered:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0
    summary = {
        "exp_id": args.exp_id,
        "n_prompts": len(chosen),
        "n_seeds_per": args.n_seeds,
        "n_total_renders": len(rendered),
        "elapsed_s": elapsed,
        "peak_vram_gb": peak,
    }
    (out_dir / "summary_render.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_validate(args) -> int:
    """Walk by_prompt/<pid>/, read sidecar labels, emit pairs of (flagged, unflagged) seeds per prompt."""
    out_dir = cfg.paths.output_root / args.exp_id
    by_prompt = out_dir / "by_prompt"
    if not by_prompt.exists():
        print(f"no by_prompt dir at {by_prompt}", file=sys.stderr)
        return 2

    pairs: list[dict] = []
    n_prompts = 0
    n_with_pair = 0
    for p_dir in sorted(by_prompt.iterdir()):
        if not p_dir.is_dir():
            continue
        n_prompts += 1
        flagged: list[tuple[int, Path]] = []
        unflagged: list[tuple[int, Path]] = []
        prompt_text = (p_dir / "prompt.txt").read_text().strip() if (p_dir / "prompt.txt").exists() else ""
        for png in sorted(p_dir.glob("seed_*.png")):
            side = png.with_suffix(png.suffix + ".labels.json")
            if not side.exists():
                continue
            try:
                lab = json.loads(side.read_text())
            except Exception:
                continue
            seed = int(png.stem.split("_")[-1])
            if bool(lab.get("flagged_any")):
                flagged.append((seed, png))
            else:
                unflagged.append((seed, png))
        # take cartesian product up to k
        if not (flagged and unflagged):
            continue
        n_with_pair += 1
        for f_s, f_path in flagged[: args.k_per_prompt]:
            for u_s, u_path in unflagged[: args.k_per_prompt]:
                pairs.append({
                    "prompt_id": p_dir.name,
                    "prompt": prompt_text,
                    "flagged_seed": f_s,
                    "unflagged_seed": u_s,
                    "flagged_path": str(f_path),
                    "unflagged_path": str(u_path),
                })

    val_path = out_dir / "validated.jsonl"
    with val_path.open("w") as f:
        for r in pairs:
            f.write(json.dumps(r) + "\n")
    summary = {
        "exp_id": args.exp_id,
        "n_prompts_seen": n_prompts,
        "n_prompts_with_pair": n_with_pair,
        "n_pairs_total": len(pairs),
    }
    (out_dir / "summary_validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="render n_seeds per prompt for n_prompts")
    b.add_argument("--exp-id", default="cf_benchmark_v1_seed")
    b.add_argument("--n-prompts", type=int, default=100)
    b.add_argument("--n-seeds", type=int, default=8)
    b.add_argument("--batch-size", type=int, default=4)
    b.add_argument("--num-inference-steps", type=int, default=4)
    b.add_argument("--guidance-scale", type=float, default=7.5)
    b.add_argument("--seed", type=int, default=12_000_000)
    b.add_argument("--device", default="cuda")
    b.add_argument("--dtype", default="fp16")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate", help="emit (flagged, unflagged) pairs per prompt")
    v.add_argument("--exp-id", default="cf_benchmark_v1_seed")
    v.add_argument("--k-per-prompt", type=int, default=2)
    v.set_defaults(func=cmd_validate)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
