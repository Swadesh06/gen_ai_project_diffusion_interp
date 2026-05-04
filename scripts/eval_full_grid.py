#!/usr/bin/env python
"""Run one row of the §3.4 evaluation grid; aggregate to reports/PHASE_1_FINAL.md.

12 rows × 5 seeds × multi-metric. Use --row to select; --aggregate to assemble the headline table.
"""

from __future__ import annotations

import sys

from _common import base_parser, echo_dry, resolve_exp_id, write_dry_report

from dsi.config import cfg

ROWS = [
    "no_defense",
    "safety_checker",
    "nudenet",
    "q16",
    "saeuron_repro",
    "saemnesia_repro",
    "dsg_adapted",
    "attribution_only",
    "stage1_only_meanpatch",
    "two_stage_meanpatch",        # proposed
    "two_stage_int_attribution",
    "two_stage_union_attribution",
    "zero_patch_two_stage",
    "resample_patch_two_stage",
]


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--row", type=str, choices=ROWS + ["all"], required=True)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--aggregate", action="store_true",
                   help="produce the headline reports/PHASE_1_FINAL.md from prior row outputs")
    args = p.parse_args()

    exp_id = resolve_exp_id(args, prefix=f"grid_row_{args.row}")
    plan = {
        "exp_id": exp_id,
        "task": "evaluation grid",
        "row": args.row,
        "rows_total": len(ROWS),
        "n_seeds": args.n_seeds,
        "metrics_per_cell": [
            "ASR_i2p_naive", "ASR_i2p_adversarial",
            "ASR_mma_text", "ASR_mma_image", "ASR_unlearndiff",
            "FID_coco_5k", "CLIPscore_coco",
            "UnlearnCanvas_acc_retain",
            "latency_ms",
            "kl_collateral_per_feature",
        ],
        "co_scheduling": "12 rows: pair reproductions on GPU; CPU does FID + CLIP-score on cached images",
        "outputs": {
            "row_report_md": str(cfg.paths.report_root / f"grid_row_{args.row}.md"),
            "headline_md": str(cfg.paths.report_root / "PHASE_1_FINAL.md"),
        },
    }
    if args.dry_run:
        echo_dry(plan)
        write_dry_report(exp_id, plan)
        return 0
    print("Real grid requires GPU + outputs from Items 2/3/4/5.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
