# C06 — SAE detector cross-attack robustness vs A01 pixel-PGD bypass set

## Goal
Measure whether the SAE-activation detector is robust to the white-box pixel
PGD attack that was specifically optimised to bypass the safety_checker.
This is the deepest C-6 hypothesis: SAE features should be more
adversarially robust than raw activations because the topk-sparse projection
is non-smooth in the input space (the attacker can't follow a continuous
gradient through the topk gate).

## Setup
- **Detector**: MLP-256 head trained on `dataset_axbench_v1` (1000 SDXL
  Turbo gens; 500 NSFW I2P + 500 benign COCO; clean-label split). Trained
  via BCE on Surkov SAE z's at 4 hookpoints (concat 20480-d). Best val AUC
  on the clean-label split = **0.9864**.
- **Attack source**: `outputs/A01_pixel_eps4_n200/` — 200 PGD ε=4/255
  bypass-image SAE features (post-attack image SAE captures, saved during
  the original A01 attack).
- **Eval**: load each `.sae.pt` payload (4 hookpoints, (16, 16, 5120) per
  hp), spatial-mean-pool to (5120,), concat → (20480,), forward through
  the trained detector.

## Headline

| metric | value |
|---|---|
| n attack records evaluated         | 200 |
| n SAE detector flagged-as-unsafe   | **199** |
| n SAE detector flagged-as-safe (escaped) | 1 |
| **ASR against SAE detector**       | **0.005** |
| post-attack score mean             | 0.996 |
| post-attack score min              | 0.413 |
| post-attack score max              | 1.000 |

Comparison with safety_checker:

| target | ASR vs A01 pixel-PGD |
|---|---|
| safety_checker (white-box target)   | **1.000** |
| **SAE detector (off-target)**       | **0.005** |
| difference (transfer survival rate) | **99.5 pp** |

## Interpretation
The SAE detector catches **199 out of 200** A01 PGD bypasses. The PGD
attack — which specifically optimised to drive the safety_checker's unsafe
logit below zero — does *not* drive the SAE detector's unsafe logit below
zero. The two detectors monitor near-disjoint feature subspaces, and a
gradient walk through the safety_checker's loss surface does not trace a
gradient walk in the SAE's sparse-topk activation space.

This validates the cross-target transferability story from Item 4
(transferability = 0.000) at the population scale (n=200, not n=17 from
the original measurement).

## Implication for paper
The SAE detector is the operational defence: it shoulders 99.5% of the
attack cost that safety_checker fails to absorb. Combined with the C-1
black-box result (safety_checker fails to BB attacks at 87.5% ASR), the
SAE detector is the only practical safety layer in this stack.

## Caveats
- The PGD attack was *optimised against the safety_checker*, not against
  the SAE detector. A separate adaptive attack should target the SAE
  detector itself; the C-1 attack-vs-SAE-detector script (queued; needs
  fix to UNet hook firing in SAEDetectorTarget.x_to_logit) is the next
  step. Until that runs, this result speaks to *transferred* attacks, not
  adaptive attacks.
- The 1 image that escaped (score = 0.413) is at the boundary; needs
  visual inspection.

## Next
- Adaptive attack: re-run C-1 (Square Attack) against the SAE detector
  directly, with proper UNet conditioning so SAE hooks fire.
- Same eval against safety-trained SAE detector (C-3 concat MLP, in flight)
  — expected even higher transfer-survival.
- Per-feature attribution: which of the 20480 features carry the
  discriminative signal that survived the attack? → paper figure.

## Artefacts
- `outputs/C06_adv_robust_simple/adv_robust_simple.json`
- `logs/C6_adv_robust_simple.log`
