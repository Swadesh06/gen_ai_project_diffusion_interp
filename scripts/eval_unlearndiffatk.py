#!/usr/bin/env python
"""UnlearnDiffAtk evaluation — Item 1c-4 headline migration.

Move UnlearnDiffAtk-nudity / -violence to the headline ASR table by
generating SDXL Base 4-step images for each prompt and oracle-scoring
with NudeNet (nudity) and Q16 (violence). Comparable apples-to-apples
with SAeUron's published Table 1.

Subcommands:
    gen      generate SDXL Base 4-step images for the chosen split
    score    NudeNet/Q16/safety_checker scoring on rendered images
    summary  consolidate metrics into a JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load_prompts(split: str, limit: int = 0) -> list[dict]:
    base = Path("/workspace/datasets/Diffusion-MU-Attack/prompts")
    csv_path = base / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    import csv

    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def cmd_gen(args) -> int:
    import torch
    from dsi.config import cfg
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper

    out_dir = cfg.paths.output_root / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = out_dir / "renders"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(args.split, limit=args.limit)
    print(f"loaded {len(prompts)} prompts from {args.split}")

    pipe_w = SDXLPipelineWrapper(variant="base", device="cuda", dtype="fp16").load()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    n_done = 0
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start:start + args.batch_size]
        seed_offsets = [args.seed + start + i for i in range(len(batch))]
        gens = [torch.Generator(device="cuda").manual_seed(s) for s in seed_offsets]
        imgs = pipe_w.pipe(
            prompt=[r["prompt"] for r in batch],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gens, height=512, width=512,
        ).images
        for r, img, s in zip(batch, imgs, seed_offsets):
            case_id = r.get("case_number", "?")
            stem = f"{int(case_id):08d}_{s:08d}"
            img.save(rendered_dir / f"{stem}.png")
            (rendered_dir / f"{stem}.png.meta.json").write_text(json.dumps({
                "case_number": case_id,
                "prompt": r["prompt"],
                "seed": s,
                "categories": r.get("categories", ""),
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "split": args.split,
            }))
            n_done += 1
        if (start // args.batch_size) % 5 == 0:
            elapsed = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            print(f"  {n_done}/{len(prompts)}  elapsed={elapsed:.1f}s  peak_vram={peak:.1f}GB", flush=True)

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    summary = {
        "exp_id": args.exp_id, "split": args.split, "n": n_done,
        "render_dir": str(rendered_dir),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "elapsed_s": elapsed, "peak_vram_gb": peak,
    }
    (out_dir / "summary_render.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_summary(args) -> int:
    """Compute ASR-on-oracle from labels.json sidecars in the renders dir."""
    from dsi.config import cfg

    out_dir = cfg.paths.output_root / args.exp_id
    rendered_dir = out_dir / "renders"
    if not rendered_dir.exists():
        print(f"no rendered dir at {rendered_dir}")
        return 2
    pngs = sorted(rendered_dir.glob("*.png"))
    n_total = len(pngs)
    n_labelled = 0
    n_nudity_flag = 0
    n_q16_flag = 0
    n_safety_flag = 0
    n_any_flag = 0
    by_concept: dict = {}
    for p in pngs:
        side = p.with_suffix(p.suffix + ".labels.json")
        if not side.exists():
            continue
        try:
            r = json.loads(side.read_text())
        except Exception:
            continue
        n_labelled += 1
        is_nudity = bool(r.get("nudenet", {}).get("flagged", False))
        is_q16 = bool(r.get("q16", {}).get("flagged", False))
        is_safety = bool(r.get("safety_checker", {}).get("flagged", False))
        is_any = bool(r.get("flagged_any", is_nudity or is_q16 or is_safety))
        if is_nudity: n_nudity_flag += 1
        if is_q16: n_q16_flag += 1
        if is_safety: n_safety_flag += 1
        if is_any: n_any_flag += 1
        # By categories
        meta_path = p.with_suffix(p.suffix + ".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            cat = meta.get("categories", "unknown")
            by_concept.setdefault(cat, {"n": 0, "n_flag": 0})
            by_concept[cat]["n"] += 1
            if is_any:
                by_concept[cat]["n_flag"] += 1

    summary = {
        "exp_id": args.exp_id, "n_total": n_total, "n_labelled": n_labelled,
        "asr_nudity": n_nudity_flag / max(1, n_labelled),
        "asr_q16": n_q16_flag / max(1, n_labelled),
        "asr_safety_checker": n_safety_flag / max(1, n_labelled),
        "asr_any": n_any_flag / max(1, n_labelled),
        "by_concept": {k: {**v, "asr": v["n_flag"] / max(1, v["n"])} for k, v in by_concept.items()},
    }
    (out_dir / "summary_eval.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="render UnlearnDiffAtk prompts with SDXL Base")
    g.add_argument("--exp-id", default="udatk_nudity_n143")
    g.add_argument("--split", default="nudity",
                   choices=["nudity", "violence", "vangogh", "church", "parachute", "garbage_truck",
                            "tench", "i2p", "illegal", "coco_10k"])
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--num-inference-steps", type=int, default=4)
    g.add_argument("--guidance-scale", type=float, default=7.5)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--seed", type=int, default=42_000_000)
    g.set_defaults(func=cmd_gen)

    s = sub.add_parser("summary", help="aggregate oracle labels into ASR metrics")
    s.add_argument("--exp-id", required=True)
    s.set_defaults(func=cmd_summary)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
