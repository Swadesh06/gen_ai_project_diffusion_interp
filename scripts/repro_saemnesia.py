#!/usr/bin/env python
"""Reproduce SAEmnesia (Cassano et al., 2025) — one-to-one supervised SAE; +9.22% over SAeUron.

NOTE: As of Phase 1a download, no public checkpoint or repo is released. PLAN.md flags this
as `reproduce-from-scratch`: train supervised SAE on labelled UnlearnCanvas concepts and
report alongside SAeUron. Until then, this script remains a dry-run stub.
"""

from __future__ import annotations

import sys

from _common import base_parser, echo_dry, resolve_exp_id, write_dry_report

from dsi.config import cfg


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--concept", type=str, default="van_gogh")
    args = p.parse_args()

    exp_id = resolve_exp_id(args, prefix=f"repro_saemnesia_{args.concept}")
    plan = {
        "exp_id": exp_id,
        "task": "baseline reproduction — SAEmnesia (queued: reproduce-from-scratch)",
        "upstream": "arXiv:2509.21379 (no public repo as of Phase 1a)",
        "device": args.device,
        "expected_vram_gb": 16.0,
        "outputs": {
            "report_md": str(cfg.paths.report_root / f"{exp_id}.md"),
        },
        "TODO": [
            "Train supervised SAE (one-to-one concept-neuron mapping) on labelled UnlearnCanvas concepts",
            "Match Cassano et al. §3 architecture: ReLU SAE w/ supervised concept loss",
            "Compare on UnlearnCanvas accuracy + FID against SAeUron",
        ],
    }
    if args.dry_run:
        echo_dry(plan)
        write_dry_report(exp_id, plan)
        return 0
    print("Real run requires GPU and trained checkpoint.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
