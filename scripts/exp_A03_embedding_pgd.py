#!/usr/bin/env python
"""Item 2.3 / Contribution 1 — CLIP image-embedding PGD against CompVis safety checker.

Most permissive attack — no decoding step; the CLIP image embedding is the safety
checker's input directly. Establishes an attack ceiling.
"""

from __future__ import annotations

import sys

from _common import base_parser, echo_dry, resolve_exp_id, write_dry_report

from dsi.attacks import EmbeddingPGDConfig
from dsi.config import EPS_GRID_EMBEDDING, cfg
from dsi.data.i2p import i2p_nsfw_subset


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--n-steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.05)
    args = p.parse_args()

    exp_id = resolve_exp_id(args, prefix="A03_embedding_pgd")
    prompts = i2p_nsfw_subset(limit=args.n_prompts)

    plan = {
        "exp_id": exp_id,
        "task": "Item 2.3 — CLIP image-embedding PGD",
        "attack_space": "embedding",
        "config": EmbeddingPGDConfig(eps=args.eps, n_steps=args.n_steps, lr=args.lr,
                                     eps_grid=EPS_GRID_EMBEDDING),
        "n_prompts": len(prompts),
        "device": args.device,
        "expected_vram_gb": 3.0,
        "co_scheduling": "GPU: 3GB — easily co-located with latent-PGD or detector training",
        "outputs": {
            "bypass_images_dir": str(cfg.paths.output_root / exp_id / "bypass"),
            "report_md": str(cfg.paths.report_root / f"{exp_id}.md"),
        },
    }

    if args.dry_run:
        echo_dry(plan)
        write_dry_report(exp_id, plan)
        return 0

    if args.device == "cpu":
        print("ERROR: embedding PGD requires GPU.")
        return 2
    print("GPU run not implemented in this stub.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
