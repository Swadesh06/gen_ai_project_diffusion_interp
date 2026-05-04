#!/usr/bin/env python
"""Counterfactual benchmark Strategy 1 — prompt-edit pairs.

v2 §3 Item 1c-0 Strategy 1. Apply the safety substitution dictionary to all
I2P prompts; render pre and post with SDXL Base 4-step at identical seed;
cpu-workers oracle-label both; pairs where pre flags and post does not become
the validated set.

Output: outputs/cf_benchmark_v1/
    pairs.jsonl                      one row per substitution candidate
    pre/<pair_id>.png                rendered pre image
    post/<pair_id>.png               rendered post image
    *.png.labels.json sidecars are written by cpu_worker.py asynchronously.
    validated.jsonl                  written by --validate after labels exist.
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
from dsi.data.counterfactual import build_strategy1_pairs  # noqa: E402
from dsi.data.i2p import load_i2p  # noqa: E402


def _read_label(p: Path) -> dict | None:
    """Combine NudeNet/Q16 (.labels.json from cpu_worker) + safety_checker (.safety.json from batch_safety_checker)."""
    rec = {}
    seen = False
    side = p.with_suffix(p.suffix + ".labels.json")
    if side.exists():
        try:
            r = json.loads(side.read_text())
            seen = True
            for k in ("nudenet", "q16"):
                rec[k] = r.get(k, {})
            if rec["nudenet"].get("flagged") or rec["q16"].get("flagged"):
                rec["flagged_any"] = True
        except Exception:
            pass
    safe_side = p.with_suffix(p.suffix + ".safety.json")
    if safe_side.exists():
        try:
            r = json.loads(safe_side.read_text())
            seen = True
            rec["safety_checker"] = r
            if r.get("flagged"):
                rec["flagged_any"] = True
        except Exception:
            pass
    if not seen:
        return None
    rec.setdefault("flagged_any", False)
    return rec


def cmd_build(args) -> int:
    out_dir = cfg.paths.output_root / args.exp_id
    pre_dir = out_dir / "pre"
    post_dir = out_dir / "post"
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    print("loading I2P prompts ...", flush=True)
    prompts = load_i2p("full", limit=None)
    texts = [p.text for p in prompts if p.text]
    print(f"  {len(texts)} prompts", flush=True)

    pairs = build_strategy1_pairs(texts)
    print(f"  {len(pairs)} substitution candidates", flush=True)

    pairs_path = out_dir / "pairs.jsonl"
    with pairs_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p.to_dict()) + "\n")
    print(f"wrote {pairs_path}", flush=True)

    if args.no_render:
        return 0

    if args.limit:
        pairs = pairs[: args.limit]
        print(f"limited to {len(pairs)} for render", flush=True)

    import torch
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper

    print(f"loading SDXL Base on {args.device}", flush=True)
    pipe_w = SDXLPipelineWrapper(variant="base", device=args.device, dtype=args.dtype).load()

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    n_done = 0
    t0 = time.time()
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        seeds = [args.seed + start + i for i in range(len(batch))]
        gens = [torch.Generator(device=args.device).manual_seed(s) for s in seeds]

        # render PRE (unsafe)
        pre_imgs = pipe_w.pipe(
            prompt=[p.pre_prompt for p in batch],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gens,
            height=512,
            width=512,
        ).images
        # render POST (safe) at the SAME seeds — identical noise → counterfactual
        gens = [torch.Generator(device=args.device).manual_seed(s) for s in seeds]
        post_imgs = pipe_w.pipe(
            prompt=[p.post_prompt for p in batch],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gens,
            height=512,
            width=512,
        ).images

        for pair, pre_img, post_img, seed in zip(batch, pre_imgs, post_imgs, seeds):
            pre_path = pre_dir / f"{pair.pair_id}.png"
            post_path = post_dir / f"{pair.pair_id}.png"
            pre_img.save(pre_path)
            post_img.save(post_path)
            (pre_dir / f"{pair.pair_id}.png.meta.json").write_text(
                json.dumps({"prompt": pair.pre_prompt, "seed": seed,
                            "pair_id": pair.pair_id, "side": "pre",
                            "cluster": pair.cluster})
            )
            (post_dir / f"{pair.pair_id}.png.meta.json").write_text(
                json.dumps({"prompt": pair.post_prompt, "seed": seed,
                            "pair_id": pair.pair_id, "side": "post",
                            "cluster": pair.cluster})
            )
            n_done += 1
        if (start // args.batch_size) % 10 == 0:
            elapsed = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0
            print(f"  rendered {n_done}/{len(pairs)}  elapsed={elapsed:.1f}s  peak_vram={peak:.1f}GB", flush=True)

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0
    summary = {
        "exp_id": args.exp_id,
        "n_pairs_total": len(pairs),
        "n_rendered": n_done,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak,
    }
    (out_dir / "summary_render.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_validate(args) -> int:
    """Read all pre/<pair_id>.png and post/<pair_id>.png; check oracle sidecars; emit validated.jsonl."""
    out_dir = cfg.paths.output_root / args.exp_id
    pre_dir = out_dir / "pre"
    post_dir = out_dir / "post"
    pairs_path = out_dir / "pairs.jsonl"
    if not pairs_path.exists():
        print(f"no pairs.jsonl at {pairs_path}", file=sys.stderr)
        return 2
    pairs = [json.loads(l) for l in pairs_path.read_text().splitlines()]

    rows: list[dict] = []
    n_total = 0
    n_pre_labelled = n_post_labelled = 0
    n_pre_flagged = n_post_unflagged = 0
    n_validated = 0
    for p in pairs:
        pid = p["pair_id"]
        pre_png = pre_dir / f"{pid}.png"
        post_png = post_dir / f"{pid}.png"
        if not (pre_png.exists() and post_png.exists()):
            continue
        n_total += 1
        pre_lab = _read_label(pre_png)
        post_lab = _read_label(post_png)
        if pre_lab is None or post_lab is None:
            continue
        n_pre_labelled += 1
        n_post_labelled += 1
        pre_flag = bool(pre_lab.get("flagged_any"))
        post_flag = bool(post_lab.get("flagged_any"))
        if pre_flag:
            n_pre_flagged += 1
        if not post_flag:
            n_post_unflagged += 1
        validated = pre_flag and (not post_flag)
        if validated:
            n_validated += 1
        rows.append({
            **p,
            "pre_flagged": pre_flag,
            "post_flagged": post_flag,
            "validated": validated,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    val_path = out_dir / "validated.jsonl"
    with val_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {
        "exp_id": args.exp_id,
        "n_pairs_jsonl": len(pairs),
        "n_with_renders": n_total,
        "n_with_labels": min(n_pre_labelled, n_post_labelled),
        "n_pre_flagged": n_pre_flagged,
        "n_post_unflagged": n_post_unflagged,
        "n_validated": n_validated,
        "validation_rate": n_validated / max(1, n_total),
    }
    (out_dir / "summary_validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="apply substitution dictionary + render pre/post")
    b.add_argument("--exp-id", default="cf_benchmark_v1")
    b.add_argument("--no-render", action="store_true")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--num-inference-steps", type=int, default=4)
    b.add_argument("--guidance-scale", type=float, default=7.5)
    b.add_argument("--batch-size", type=int, default=4)
    b.add_argument("--seed", type=int, default=11_000_000)
    b.add_argument("--device", default="cuda")
    b.add_argument("--dtype", default="fp16")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate", help="cross-check oracle sidecars; emit validated.jsonl")
    v.add_argument("--exp-id", default="cf_benchmark_v1")
    v.set_defaults(func=cmd_validate)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
