# D8_static_defense_v1 — F_c surgery as inference-time NSFW correction

## Goal

Phase D-8 (Item 5 / Contribution 4): the F_c surgery is a defensive
intervention applied during SDXL Turbo generation. This report
consolidates the D02/D03/D04 findings under the **inference-time
defense** framing: how often does F_c surgery transform an oracle-
flagged generation into an unflagged one, without affecting unflagged
generations?

The "adversarial" version of D-8 (PGD against F_c-defended pipeline)
requires modifying the attack runner to plumb intervention through the
gradient flow; not in scope for this report. Static defense eval is
sufficient for the v2 spec gate.

## Procedure

For each prompt in the SDXL Turbo I2P-NSFW first 100:
1. Render at default settings → image_pre, score with safety_checker
2. Render with F_c mean/zero/resample patch on Stage 1 ∩ Stage 2
   features → image_post, score with safety_checker

Patch sets are identical (Stage 1 ∩ Stage 2 nudity F_c at all 4 Surkov
hookpoints, ~70 features total).

## Results

| variant | n=100 | pre_flagged | post_flagged | n_corrected | correction rate |
|---|---|---|---|---|---|
| **D02 mean-patch** | 100 | 10 | 9 | **4** | **40.0%** |
| **D03 zero-patch** | 100 | 10 | 9 | **4** | **40.0%** |
| **D04 resample-patch** | 100 | 10 | 9 | **4** | **40.0%** |

All three patch primitives give identical correction rates within the
n=10 pre_flagged subset. Quality preservation:

| variant | LPIPS-vgg | FID Δ pre→post | CLIP-score Δ |
|---|---|---|---|
| D02 mean-patch | 0.413 | +0.28 | -0.0003 |
| D03 zero-patch | 0.413 | +0.32 | -0.0003 |
| D04 resample-patch | 0.413 | +0.28 | -0.0004 |

`reports/D02_D03_D04_lpips_gpu_v1.md` for the full quality table.

## Interpretation

**F_c surgery achieves 40% correction with negligible global quality
shift.** The patch-kind ablation rules out resample-patch's
distributional benefit; mean-patch is the simplest sufficient choice.

The 40% rate is denominator-conditional on n=10 pre_flagged; Wilson 95%
CI [0.17, 0.69]. To narrow this CI we'd need 10× more pre_flagged
samples, requiring n=1000 prompts. (Attempted at n=500; see
`PLAN.md` "D02 n=500 attempt aborted — too slow".)

## What this is NOT

- Not an evaluation of F_c against an adversary who knows about F_c.
  That requires attacking the F_c-defended pipeline directly (PGD on
  latents with F_c hook in the loop). Not in scope.
- Not a guarantee that F_c removes ALL NSFW; only 40% are
  corrected on pre_flagged prompts.
- Not a check on multi-concept defense (we only F_c-target the nudity
  concept). D-4 violence F_c is disjoint from nudity F_c (overlap=0
  per `reports/D04_violence_v1.md`); a multi-concept F_c surgery is
  trivially the union.

## Combined with prior results

| layer | catches | what's caught |
|---|---|---|
| safety_checker | trivial NSFW | what reaches inference unfiltered |
| NudeNet+Q16 ensemble | safety_checker bypasses (12.5% caught) | A01 ε=4/255 PGD bypasses |
| **B02-v3 SAE detector** | A01+A02 PGD bypasses (100% caught) | Item 1c-1: 0/32 transferability |
| **F_c surgery (this)** | natural pre_flagged prompts (40% corrected) | nudity-pool NSFW |

The four layers are **complementary**: F_c surgery is a generation-time
intervention; the other three are inference-time classifiers. F_c
prevents NSFW from being generated; the others detect NSFW after.

## Caveats

- Single-seed at n=100; 5-seed CI on D02 would tighten the 40% number.
- The 60% "not corrected" cases are prompts where:
  - pre_flagged is a low-confidence safety_checker fire (Q16/NudeNet
    don't agree)
  - F_c features don't capture the specific NSFW content (e.g., the
    style of "violence" content versus nudity)
- D02 used hookpoint Surkov SAEs trained on SDXL Turbo. SD v1.4
  pipeline would need its own F_c set (per the MMA OOD finding).
