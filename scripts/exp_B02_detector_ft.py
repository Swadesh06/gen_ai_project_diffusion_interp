#!/usr/bin/env python
"""Item 3.x / Contribution 2 — full-trajectory (FT) detector training.

Pools activations across all denoising steps (mean / max / attention-pool), then
classifies. Powers the two-pass regenerate correction (Contribution 4).
"""

from __future__ import annotations

import sys

from _common import base_parser, echo_dry, resolve_exp_id, write_dry_report

from dsi.config import DetectorCfg, cfg


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--pool", choices=["mean", "max", "attn"], default="mean")
    p.add_argument("--head", choices=["linear", "mlp", "ensemble"], default="mlp")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    exp_id = resolve_exp_id(args, prefix="B02_detector_ft")
    dcfg = DetectorCfg(regime="ft", head=args.head, pool=args.pool, epochs=args.epochs, lr=args.lr)

    plan = {
        "exp_id": exp_id,
        "task": "Item 3 — FT detector",
        "config": dcfg,
        "device": args.device,
        "expected_vram_gb": 4.0,
        "co_scheduling": "GPU: ~4GB — pair with EM detector training or attack run",
        "outputs": {
            "ckpt_dir": str(cfg.paths.checkpoint_root / exp_id),
            "report_md": str(cfg.paths.report_root / f"{exp_id}.md"),
        },
    }

    if args.dry_run:
        echo_dry(plan)
        write_dry_report(exp_id, plan)
        return 0
    print("Real training requires real activations; finish in Phase 1b.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
