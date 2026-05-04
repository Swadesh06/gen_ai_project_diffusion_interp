#!/usr/bin/env python
"""Item 3.x / Contribution 2 — early-monitor (EM) detector training.

Trains the detector on activations collected from steps 0..k-1 of SDXL Turbo
generations, labelled as NSFW (I2P) or benign (COCO). Heads: linear / mlp / ensemble.
"""

from __future__ import annotations

import sys

from _common import base_parser, echo_dry, resolve_exp_id, write_dry_report

from dsi.config import DetectorCfg, cfg
from dsi.detectors.train import smoke_train


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--head", choices=["linear", "mlp", "ensemble"], default="linear")
    p.add_argument("--em-steps", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--include-adversarial", action="store_true",
                   help="mix in red-team bypasses from Item 2 (Item 3.4)")
    args = p.parse_args()

    exp_id = resolve_exp_id(args, prefix="B01_detector_em")

    dcfg = DetectorCfg(regime="em", head=args.head, em_steps=args.em_steps,
                       epochs=args.epochs, lr=args.lr)

    plan = {
        "exp_id": exp_id,
        "task": "Item 3 — EM detector",
        "config": dcfg,
        "include_adversarial": args.include_adversarial,
        "device": args.device,
        "expected_vram_gb": 4.0,
        "co_scheduling": "GPU: ~4GB — co-locate with attack runs or SAEuron repro",
        "pass_criteria": {
            "auc_i2p_naive_min": 0.95,
            "auc_i2p_adversarial_min": 0.85,
            "auc_mma_text_min": 0.80,
        },
        "outputs": {
            "ckpt_dir": str(cfg.paths.checkpoint_root / exp_id),
            "report_md": str(cfg.paths.report_root / f"{exp_id}.md"),
            "commit_knee_plot": str(cfg.paths.output_root / exp_id / "commit_knee.png"),
        },
    }

    if args.dry_run:
        echo_dry(plan)
        write_dry_report(exp_id, plan)
        return 0

    if args.dry_smoke:
        out = smoke_train(in_dim=64, n=128)
        print(f"smoke losses: {out['losses']}")
        return 0

    print("Real training requires real activations; finish in Phase 1b.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
