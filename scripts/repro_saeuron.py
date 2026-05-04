#!/usr/bin/env python
"""Reproduce SAeUron (Cywiński & Deja, ICML 2025) on UnlearnCanvas as a baseline.

Wraps the upstream `cywinski/SAeUron` repo's sampling pipeline. The baseline locks in
the diffusion-native unlearning SOTA against which our two-stage + mean-patch method competes.
"""

from __future__ import annotations

import sys

from _common import base_parser, echo_dry, resolve_exp_id, write_dry_report

from dsi.config import cfg


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--concept", type=str, default="van_gogh")
    p.add_argument("--negative-scale", type=float, default=-3.0)
    args = p.parse_args()

    exp_id = resolve_exp_id(args, prefix=f"repro_saeuron_{args.concept}")
    plan = {
        "exp_id": exp_id,
        "task": "baseline reproduction — SAeUron",
        "upstream": "https://github.com/cywinski/SAeUron",
        "concept": args.concept,
        "negative_scale": args.negative_scale,
        "device": args.device,
        "expected_vram_gb": 10.0,
        "outputs": {
            "report_md": str(cfg.paths.report_root / f"{exp_id}.md"),
            "samples_dir": str(cfg.paths.output_root / exp_id),
        },
    }
    if args.dry_run:
        echo_dry(plan)
        write_dry_report(exp_id, plan)
        return 0
    print("Real run requires GPU.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
