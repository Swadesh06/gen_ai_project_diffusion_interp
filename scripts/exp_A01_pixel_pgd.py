#!/usr/bin/env python
"""Item 2.1 / Contribution 1 — pixel-space PGD against CompVis safety checker.

Defaults: ε=4/255, 40 PGD steps, 50 I2P-NSFW prompts (smoke). Scale to 500 via --n-prompts 500.
SAE activations are captured during attack runs (per Item 2.5) and persisted for Items 3 / 4.
"""

from __future__ import annotations

import sys

from _common import base_parser, echo_dry, resolve_exp_id, write_dry_report

from dsi.attacks import PixelPGDConfig
from dsi.config import EPS_GRID_PIXEL, cfg
from dsi.data.i2p import i2p_nsfw_subset


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--eps", type=float, default=4.0 / 255)
    p.add_argument("--n-steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=1.0 / 255)
    p.add_argument("--targeted", action="store_true")
    p.add_argument("--collect-sae", action="store_true",
                   help="record SAE activations during attack (per Item 2.5)")
    args = p.parse_args()

    exp_id = resolve_exp_id(args, prefix="A01_pixel_pgd")
    prompts = i2p_nsfw_subset(limit=args.n_prompts)

    pcfg = PixelPGDConfig(eps=args.eps, n_steps=args.n_steps, lr=args.lr,
                          targeted=args.targeted, eps_grid=EPS_GRID_PIXEL)
    plan = {
        "exp_id": exp_id,
        "task": "Item 2.1 — pixel PGD (Contribution 1)",
        "attack_space": "pixel",
        "config": pcfg,
        "n_prompts": len(prompts),
        "device": args.device,
        "co_scheduling": "GPU: pixel-PGD ~14-16GB; CPU: NudeNet/Q16 labelling of bypass images concurrent",
        "expected_vram_gb": 14.0,
        "pass_criteria": {"asr_at_eps_8_255_on_i2p_nsfw_min": 0.7},
        "outputs": {
            "bypass_images_dir": str(cfg.paths.output_root / exp_id / "bypass"),
            "sae_activations_dir": str(cfg.paths.output_root / exp_id / "sae"),
            "report_md": str(cfg.paths.report_root / f"{exp_id}.md"),
        },
        "collect_sae": bool(args.collect_sae),
    }

    if args.dry_run:
        echo_dry(plan)
        write_dry_report(exp_id, plan)
        return 0

    if args.device == "cpu":
        print("ERROR: pixel PGD requires GPU. --device cuda required.")
        return 2

    print("GPU run not implemented in this stub; finish in Phase 1b.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
