#!/usr/bin/env python
"""Counterfactual benchmark Strategy 3 Path B — local Llama 3.1 70B int8 paraphrase.

v2 §3 Item 1c-0 Strategy 3 Path B. Same 4-cell matrix as Path A, but generated
by a local Llama 70B int8 model. Runs alongside Path A; the local model handles
whatever Gemini refused (Llama has no usage policy at the weights level for
content paraphrasing).

Output: outputs/cf_benchmark_v1_paraphrase_llama/
    cells.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import cfg  # noqa: E402
from dsi.data.paraphrase_local import LlamaParaphraser  # noqa: E402

# Reuse anchor definitions from Path A
sys.path.insert(0, str(REPO / "scripts"))
from build_cf_strategy3_gemini import NUDITY_ANCHORS, VIOLENCE_ANCHORS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-id", default="cf_benchmark_v1_paraphrase_llama")
    ap.add_argument("--n-anchors-per-cell", type=int, default=10)
    ap.add_argument("--n-paraphrases-per-anchor", type=int, default=3)
    ap.add_argument("--concept", choices=("nudity", "violence", "both"), default="both")
    ap.add_argument("--model-id", default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    ap.add_argument("--load-in-8bit", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = cfg.paths.output_root / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / "cells.jsonl"

    cells: list[tuple[str, list[str]]] = []
    if args.concept in ("nudity", "both"):
        cells.append(("nudity", NUDITY_ANCHORS[: args.n_anchors_per_cell]))
    if args.concept in ("violence", "both"):
        cells.append(("violence", VIOLENCE_ANCHORS[: args.n_anchors_per_cell]))

    print(f"loading {args.model_id} (int8={args.load_in_8bit}) ...", flush=True)
    p = LlamaParaphraser(model_id=args.model_id, load_in_8bit=args.load_in_8bit).load()

    n_total = 0
    n_refused = 0
    with cells_path.open("w") as cf:
        for concept, anchors in cells:
            for cell_kind in ("i2p_safe", "i2p_unsafe", "coco_safe", "coco_unsafe"):
                rows = p.paraphrase_anchors(anchors, cell_kind,
                                            n_per_anchor=args.n_paraphrases_per_anchor)
                for r in rows:
                    r["concept"] = concept
                    cf.write(json.dumps(r) + "\n")
                    n_total += 1
                    if r.get("refused"):
                        n_refused += 1
                print(f"  {concept}/{cell_kind}: {len(rows)} anchors -> {sum(len(r['paraphrases']) for r in rows)} paraphrases", flush=True)

    summary = {"exp_id": args.exp_id, "n_total_rows": n_total, "n_refused": n_refused}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
