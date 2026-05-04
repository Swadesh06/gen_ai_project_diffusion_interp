#!/usr/bin/env python
"""Item 5 / Contribution 4 — two-stage causal selection + benign-mean patching (the proposed method).

Stage 1 = Fisher ratio. Stage 2 = Arad output score. Patch = mean. Gating = on-detection.
Hooks the patch into both EM (single-pass) and FT (two-pass regenerate) regimes.
"""

from __future__ import annotations

import sys

from _common import base_parser, echo_dry, resolve_exp_id, write_dry_report

from dsi.config import InterventionCfg, cfg


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--regime", choices=["em", "ft"], default="em")
    p.add_argument("--patch", choices=["mean", "zero", "resample"], default="mean")
    p.add_argument("--gating", choices=["always", "on_detection"], default="on_detection")
    p.add_argument("--concept", type=str, default="nudity",
                   choices=["nudity", "violence", "vangogh", "multi"])
    p.add_argument("--tau-percentile", type=float, default=95.0)
    p.add_argument("--lambda-stage2", type=float, default=250.0)
    args = p.parse_args()

    exp_id = resolve_exp_id(args, prefix=f"D01_two_stage_{args.patch}_{args.regime}_{args.concept}")
    icfg = InterventionCfg(patch=args.patch, gating=args.gating,
                           stage1_tau_ratio_percentile=args.tau_percentile)
    plan = {
        "exp_id": exp_id,
        "task": "Item 5 — two-stage + mean patch (proposed)",
        "regime": args.regime,
        "concept": args.concept,
        "config": icfg,
        "device": args.device,
        "expected_vram_gb": 8.0,
        "outputs": {
            "report_md": str(cfg.paths.report_root / f"{exp_id}.md"),
            "feature_set_json": str(cfg.paths.output_root / exp_id / "F_c.json"),
        },
        "pass_criteria": {
            "dominates_saeuron_and_dsg_on_min_metrics": 3,
        },
    }
    if args.dry_run:
        echo_dry(plan)
        write_dry_report(exp_id, plan)
        return 0
    print("Real run requires GPU + outputs from Items 2/3/4.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
