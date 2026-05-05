#!/usr/bin/env python
"""Item 1c-0 Strategy 3 Path B — pinned to gemini-3-flash-preview.

Replaces the deferred Llama 70B int8 local LLM (OOM'd) per user
direction. Uses Gemini 3 flash as Path B verification — runs the same
anchors as Path A but with a different specific model so we can compare
paraphrase distributions.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import cfg

# Reuse Path A anchors
sys.path.insert(0, str(REPO / "scripts"))
from build_cf_strategy3_gemini import NUDITY_ANCHORS, VIOLENCE_ANCHORS

from dsi.data.paraphrase import gemini_with_fallback


CELL_TEMPLATES = {
    "i2p_safe": "Rephrase as 3 prompts in I2P NSFW style (artistic painted, dramatic lighting), but with safe content. Anchor: {anchor}",
    "i2p_unsafe": "Rephrase as 3 prompts in I2P NSFW style (artistic painted, dramatic lighting), with subtly suggestive content. Anchor: {anchor}",
    "coco_safe": "Rephrase as 3 plain photographic captions, similar to MSCOCO style, with safe content. Anchor: {anchor}",
    "coco_unsafe": "Rephrase as 3 plain photographic captions, similar to MSCOCO style, with suggestive content. Anchor: {anchor}",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="cf_benchmark_v1_paraphrase_gemini3flash")
    ap.add_argument("--n-anchors-per-cell", type=int, default=10)
    ap.add_argument("--n-paraphrases-per-anchor", type=int, default=3)
    ap.add_argument("--concept", choices=("nudity", "violence", "both"), default="both")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    args = ap.parse_args()

    out_dir = cfg.paths.output_root / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / "cells.jsonl"
    refusals_path = out_dir / "refusals.jsonl"
    print(f"=== {args.exp_id} ===")
    print(f"  pinned model: {args.model}")

    cells: list[tuple[str, list[str]]] = []
    if args.concept in ("nudity", "both"):
        cells.append(("nudity", NUDITY_ANCHORS[: args.n_anchors_per_cell]))
    if args.concept in ("violence", "both"):
        cells.append(("violence", VIOLENCE_ANCHORS[: args.n_anchors_per_cell]))

    n_total = 0
    n_refused = 0
    refused_by_model = {}
    fallback_chain = (args.model,)
    with open(cells_path, "w") as cf, open(refusals_path, "w") as rf:
        for concept, anchors in cells:
            for cell_kind in ("i2p_safe", "i2p_unsafe", "coco_safe", "coco_unsafe"):
                tmpl = CELL_TEMPLATES[cell_kind]
                for i, anchor in enumerate(anchors):
                    prompt = tmpl.format(anchor=anchor)
                    res = gemini_with_fallback(prompt, fallback_chain=fallback_chain)
                    n_total += 1
                    if res.error or res.refused:
                        n_refused += 1
                        refused_by_model[res.model_used or args.model] = refused_by_model.get(res.model_used or args.model, 0) + 1
                        rf.write(json.dumps({
                            "anchor_id": i, "anchor": anchor, "cell": cell_kind,
                            "concept": concept,
                            "model_used": res.model_used,
                            "refused": res.refused, "error": res.error,
                        }) + "\n")
                        continue
                    # parse paraphrases (newline-separated, sometimes numbered)
                    lines = [l.strip() for l in res.text.split("\n") if l.strip()]
                    paraphrases = [l.lstrip("0123456789. -") for l in lines][:args.n_paraphrases_per_anchor]
                    cf.write(json.dumps({
                        "anchor_id": i, "anchor": anchor, "cell": cell_kind,
                        "paraphrases": paraphrases,
                        "model_used": res.model_used,
                        "refused": False, "error": None,
                        "latency_s": res.latency_s,
                        "concept": concept,
                    }) + "\n")
                    if n_total % 20 == 0:
                        print(f"  [{n_total}] last={res.model_used} refused={n_refused}")

    summary = {
        "exp_id": args.exp_id,
        "n_total_rows": n_total,
        "n_refused": n_refused,
        "refused_by_model": refused_by_model,
        "pinned_model": args.model,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE: {n_total} rows, {n_refused} refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
