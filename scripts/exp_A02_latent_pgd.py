#!/usr/bin/env python
"""Item 2.2 / Contribution 1 — VAE-latent PGD against CompVis safety checker.

PGD on 4x64x64 SDXL VAE latent; decode through VAE; evaluate. Gradient checkpointing
through the VAE keeps VRAM under the 27 GB cap.
"""

from __future__ import annotations

import sys

from _common import base_parser, echo_dry, resolve_exp_id, write_dry_report

from dsi.attacks import LatentPGDConfig
from dsi.config import EPS_GRID_LATENT, cfg
from dsi.data.i2p import i2p_nsfw_subset


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--n-steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--collect-sae", action="store_true")
    args = p.parse_args()

    exp_id = resolve_exp_id(args, prefix="A02_latent_pgd")
    prompts = i2p_nsfw_subset(limit=args.n_prompts)

    plan = {
        "exp_id": exp_id,
        "task": "Item 2.2 — VAE-latent PGD",
        "attack_space": "latent",
        "config": LatentPGDConfig(eps=args.eps, n_steps=args.n_steps, lr=args.lr,
                                  eps_grid=EPS_GRID_LATENT),
        "n_prompts": len(prompts),
        "device": args.device,
        "expected_vram_gb": 12.0,
        "co_scheduling": "GPU: latent-PGD ~12GB; co-locate with embedding-PGD ~3GB on same device",
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
        print("ERROR: latent PGD requires GPU.")
        return 2
    print("GPU run not implemented in this stub.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
