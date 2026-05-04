#!/usr/bin/env python
"""D-8 adversarial training smoke — PGD with intervention-aware loss.

Trains a PGD perturbation that maximizes safety_checker bypass *and*
SAE-Stage1∩Stage2 F_c-mean-patch bypass simultaneously. A successful
A01-style perturbation (passes safety_checker) might still trip the
F_c surgery defense; this run tests whether F_c surgery survives an
adversary that knows about it.

Concretely: instead of `loss = -safety_logit`, use `loss = -safety_logit -
patched_safety_logit` where `patched_safety_logit` is computed on the
post-mean-patch generation. If both terms drive to zero, the attack
defeats both layers.

Smoke: 30 prompts, ε=4/255, 40 steps. Compare to A01 baseline
ASR_among_pre_flagged on safety alone.

Outputs:
  reports/D08_advtrain_smoke_v1.md
  outputs/D08_advtrain_smoke/
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="D08_advtrain_smoke_v1")
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--eps", type=float, default=4/255)
    ap.add_argument("--lr", type=float, default=1/255)
    args = ap.parse_args()

    print(f"=== {args.exp_id} ===")
    print(f"  n={args.n_prompts}, n_steps={args.n_steps}, eps={args.eps:.4f}, lr={args.lr:.4f}")
    print()
    print("This is a stub for D-8 — full adversarial training requires")
    print("integrating the intervention pipeline as a second loss term, which")
    print("doubles the attack's compute. For Phase 1c the simpler proxy is the")
    print("static defensive eval: take A01 attack outputs, apply mean-patch")
    print("intervention at F_c, score with safety_checker. The bypass rate")
    print("of A01-with-defense compares to A01-no-defense.")
    print()
    print("Implementation: scripts/eval_A01_defense_static.py (proxy).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
