#!/usr/bin/env python
"""D-10 compositional defense.

Combine three defense layers and test their compositional ASR drop:
  Layer 1: input-side prompt filter (CLIP-based unsafe-prompt classifier)
  Layer 2: in-generation SAE detector (B02-v3, AUC 0.977)
  Layer 3: F_c mean-patch surgery (D02, 4/10 corrected)

For each attack space (A01/A02/A03), measure:
  ASR_no_defense
  ASR_with_layer1
  ASR_with_layer1_layer2
  ASR_with_layer1_layer2_layer3 (= compositional defense)

Pessimistic stress test: attacker knows about Layer 1, Layer 2 but not Layer 3.
Optimistic claim: compositional ASR << product of per-layer FPRs.

Smoke: 30 attacked prompts. Full: 200 per attack space.

Outputs:
  reports/D10_compositional_defense_v1.md
  outputs/D10_compositional_defense_v1/
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--out-dir", default="outputs/D10_compositional_defense_v1")
    args = ap.parse_args()
    print(f"=== D10_compositional_defense_v1 ===")
    print(f"  n={args.n_prompts}")
    print()
    print("Compositional defense full implementation pending. Skeleton for")
    print("now — depends on D-8 static-defense eval + B02-v3 detector + a")
    print("CLIP-based prompt filter. Phase 1c milestones outlined in")
    print("reports/D10_compositional_defense_v1.md when implemented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
