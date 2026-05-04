#!/usr/bin/env python
"""Item 5 / Contribution 4 Stage 2 — Arad-style causal-intervention output score.

For each Stage-1 surviving feature f:
  S_out(f, c) = E_p_neutral [Pr_clsf(c | gen(p; z_f += λ)) − Pr_clsf(c | gen(p))]

Implementation:
  - Iterates over Stage-1 survivor features per hookpoint (loaded from `stage1_*.json`).
  - For each feature, monkey-patches the Surkov hook to add `λ` to that single feature
    at every spatial position; runs SDXL Turbo on a fixed neutral-prompt set; scores the
    output image with the chosen classifier (Q16 by default; configurable).
  - Aggregates Δ probability of the concept across the neutral set; writes
    stage2_<hookpoint>.json with per-feature score.

Heavy: scales as O(|Stage-1 survivors| × N_neutral × 1 SDXL gen). Co-schedule with CPU work.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-dir", required=True,
                    help="dir with stage1_<hookpoint>.json files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hookpoint", required=True,
                    help="which Surkov hookpoint to score")
    ap.add_argument("--n-neutral", type=int, default=8,
                    help="number of neutral COCO captions per feature")
    ap.add_argument("--lam", type=float, default=250.0)
    ap.add_argument("--max-features", type=int, default=128,
                    help="cap survivors to score (heavy compute)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--classifier", choices=["q16", "nudenet", "safety_checker"], default="q16")
    args = ap.parse_args()

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 1 survivors for this hookpoint
    stage1_path = Path(args.stage1_dir) / f"stage1_{args.hookpoint.replace('.', '_')}.json"
    s1 = json.loads(stage1_path.read_text())
    feat_idx = s1["kept_indices"]
    if args.max_features and len(feat_idx) > args.max_features:
        # Take the highest-ratio features
        ratios = s1["kept_ratios"]
        order = sorted(range(len(feat_idx)), key=lambda i: -ratios[i])[: args.max_features]
        feat_idx = [feat_idx[i] for i in order]
    print(f"scoring {len(feat_idx)} features at {args.hookpoint}")

    print("loading SDXL Turbo")
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper

    pipe_w = SDXLPipelineWrapper(variant="turbo", device=args.device, dtype=args.dtype).load()

    print("loading Surkov SAE for hookpoint")
    from dsi.sae.load import load_surkov_sae

    sae = load_surkov_sae(args.hookpoint).to(args.device).eval()

    print("loading classifier")
    if args.classifier == "q16":
        from dsi.detectors.baselines.q16 import Q16Wrapper

        scorer = Q16Wrapper(device=args.device).load()

        def score_pil(img):
            r = scorer.score_image(img)
            return float(r["score"]), bool(r["flagged"])

    elif args.classifier == "nudenet":
        from dsi.detectors.baselines.nudenet import NudeNetWrapper

        nn = NudeNetWrapper().load()

        def score_pil(img):
            tmp = out_dir / "_tmp_nn.png"
            img.save(tmp)
            r = nn.score_path(str(tmp))
            return float(r["score"]), bool(r["flagged"])
    else:
        from dsi.detectors.baselines.safety_checker import SafetyCheckerWrapper

        sc = SafetyCheckerWrapper(device=args.device).load()

        def score_pil(img):
            r = sc.score([img])[0]
            return float(r["score"]), bool(r["flagged"])

    print("loading neutral prompts (COCO val)")
    from dsi.data.coco import load_coco_captions

    prompts = load_coco_captions(limit=args.n_neutral)
    print(f"  {len(prompts)} neutral prompts")

    # Hook closure that adds λ to a SINGLE feature at every spatial position
    from dsi.sae.hooks import HOOKPOINT_TO_GETTER

    target_block = HOOKPOINT_TO_GETTER[args.hookpoint](pipe_w.unet)

    cur_feat = [None]
    cur_lam = [args.lam]

    def hook(module, args_, kwargs_, output):
        f = cur_feat[0]
        if f is None:
            return None
        inp = args_[0]
        out = output if not isinstance(output, tuple) else output[0]
        if hasattr(out, "sample"):
            out = out.sample
        diff = out - inp
        x_bhwc = diff.permute(0, 2, 3, 1)
        with torch.no_grad():
            z = sae.encode(x_bhwc)
            z[..., f] = z[..., f] + cur_lam[0]
            rec = sae.decode(z)
        rec_bchw = rec.permute(0, 3, 1, 2).to(out.dtype)
        new_out = inp + rec_bchw
        if isinstance(output, tuple):
            return (new_out,) + output[1:]
        if hasattr(output, "sample"):
            output.sample = new_out
            return output
        return new_out

    handle = target_block.register_forward_hook(hook, with_kwargs=True)

    # Baseline scores (cur_feat = None → no-op)
    print("baseline pass")
    cur_feat[0] = None
    base_scores = []
    for i, p in enumerate(prompts):
        gen = torch.Generator(device=args.device).manual_seed(1000 + i)
        out = pipe_w.pipe(prompt=[p.text], num_inference_steps=1, guidance_scale=0.0,
                          generator=gen, height=512, width=512)
        base_scores.append(score_pil(out.images[0]))
    base_avg = sum(s for s, _ in base_scores) / len(base_scores)
    print(f"  baseline mean classifier score: {base_avg:.3f}")

    # Per-feature scoring
    feat_records = {}
    t0 = time.time()
    for fi, f in enumerate(feat_idx):
        cur_feat[0] = f
        scores = []
        for i, p in enumerate(prompts):
            gen = torch.Generator(device=args.device).manual_seed(1000 + i)
            out = pipe_w.pipe(prompt=[p.text], num_inference_steps=1, guidance_scale=0.0,
                              generator=gen, height=512, width=512)
            scores.append(score_pil(out.images[0]))
        avg = sum(s for s, _ in scores) / len(scores)
        delta = avg - base_avg
        feat_records[int(f)] = {"avg_score": avg, "delta": delta,
                                "flagged_frac": sum(1 for _, fl in scores if fl) / len(scores)}
        if fi % max(1, len(feat_idx) // 10) == 0:
            print(f"  [{fi+1}/{len(feat_idx)}] feature {f} delta={delta:+.4f} elapsed={time.time()-t0:.1f}s",
                  flush=True)
    handle.remove()

    summary = {
        "hookpoint": args.hookpoint, "lam": args.lam, "n_neutral": len(prompts),
        "classifier": args.classifier, "baseline_avg": base_avg,
        "n_features_scored": len(feat_records),
        "features": feat_records,
    }
    out_path = out_dir / f"stage2_{args.hookpoint.replace('.', '_')}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
