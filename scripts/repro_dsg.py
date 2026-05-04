#!/usr/bin/env python
"""Reproduce DSG (Muhamed et al., COLM 2025) adapted to diffusion safety.

Stage 1 (Fisher ratio) feature selection + dynamic classifier-gated clamp to negative
reference. This is the LLM SOTA transferred to T2I; an ablation of our full two-stage
+ mean-patch pipeline.
"""

from __future__ import annotations

import sys

from _common import base_parser, echo_dry, resolve_exp_id, write_dry_report

from dsi.config import cfg
from dsi.interventions.baselines.dsg_adapted import DSGAdaptedConfig


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--tau-percentile", type=float, default=95.0)
    p.add_argument("--clamp-value", type=float, default=-1.0)
    p.add_argument("--concept", type=str, default="nudity")
    args = p.parse_args()

    exp_id = resolve_exp_id(args, prefix=f"repro_dsg_{args.concept}")
    dcfg = DSGAdaptedConfig(tau_ratio_percentile=args.tau_percentile, clamp_value=args.clamp_value)
    plan = {
        "exp_id": exp_id,
        "task": "baseline reproduction — DSG-adapted",
        "upstream": "arXiv:2504.08192 (LLM unlearning); adapted to diffusion",
        "config": dcfg,
        "device": args.device,
        "expected_vram_gb": 8.0,
        "outputs": {
            "report_md": str(cfg.paths.report_root / f"{exp_id}.md"),
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
