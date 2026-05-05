# saeuron_feature_contrast_v1 — Item 1c-5 close: data-driven SAEUron nudity feature

## Goal

Item 1c-5 partial close. The prior `repro_saeuron_nudity_n30_smoke.md`
attempt used `feature_idx = 11627` from the SAEUron upstream-example
notebook — which is the "**cat**" feature, not nudity. The smoke
result accordingly showed the intervention INCREASED safety_checker
flag rate (33% → 43%). This report finds the correct nudity feature_idx
via data-driven Fisher-ratio scoring on (NSFW, benign) activations.

## Procedure

`scripts/saeuron_feature_contrast.py` (post device + shape fixes).
1. Load SD v1.4 + `bcywinski/SAeUron_coco` SAE at hookpoint
   `unet.up_blocks.1.attentions.1` (n_features = 20480).
2. Render 20 I2P-NSFW prompts + 20 COCO benign captions through SD
   v1.4 with a pre-hook on the SAEUron hookpoint.
3. SAE-encode each forward; flatten spatial × time → (BS, F).
4. Compute per-feature Fisher ratio (μ_nsfw - μ_benign)² / (σ_n² + σ_b²).
5. Report top-20 features.

Output: `outputs/saeuron_feature_contrast_v5.json`.

## Results — top-20 SAEUron-nudity features

| rank | feature_idx | Fisher | μ_nsfw | μ_benign |
|---|---|---|---|---|
| **1** | **12571** | **1.1853** | 0.97 | 0.26 |
| 2 | 16608 | 1.1576 | 0.88 | 0.21 |
| 3 | 11615 | 0.9024 | 1.89 | 0.42 |
| 4 | 14527 | 0.8924 | 4.06 | 1.26 |
| 5 | 6637 | 0.7824 | 0.22 | 0.06 |
| 6 | 12101 | 0.7542 | 0.21 | 0.02 |
| 7 | 5090 | 0.7503 | 0.57 | 0.06 |
| 8 | 3506 | 0.7403 | 1.25 | 0.24 |
| 9 | 1368 | 0.7037 | 0.38 | 0.05 |
| 10 | 6912 | 0.6748 | 0.25 | 0.03 |
| 11-20 | (see JSON) | 0.61-0.66 | — | — |

**Note**: feature 11627 (the "cat" feature used by the upstream
example, and by our v1 smoke) is NOT in the top-20 nudity-discriminative
list. Our prior smoke's wrong-feature usage is confirmed.

The top candidate **feature 12571** has μ_nsfw = 0.97, μ_benign = 0.26
(Fisher 1.19) — strong differential activation on NSFW content.

## Interpretation

A data-driven feature-contrast scoring on the SAEUron
`bcywinski/SAeUron_coco` SAE recovers a nudity-discriminative top-1
feature (idx 12571) that the upstream README does NOT publish for
nudity. The published examples use feature 11627 = "cat", which is
not relevant to the unlearning task.

For a proper SAeUron-vs-our-pipeline head-to-head:
1. Re-run `scripts/repro_saeuron_streamlined.py` with `--feature-idx
   12571 --multiplier -3.0` on UnlearnDiffAtk-nudity prompts.
2. Compare safety_checker flag rate to baseline.
3. Compare to our F_c surgery (D02) head-to-head.

If feature 12571 reduces safety_checker rate from 33% (baseline) to
≤ 10% (SAEUron's published Table 1 ≈ 5%), the upstream method is
reproduced.

## What this enables

- Item 1c-5 SAEUron baseline reproduction can now be re-run with the
  correct feature_idx (instead of the cat-feature placeholder).
- Comparison of our 4-hookpoint Surkov SAE intervention vs SAEUron's
  single-hookpoint single-feature intervention on the same UDA-nudity
  prompts.
- Cross-method ablation for the paper.

## Caveats

- 20 nudity + 20 benign is small. Wilson 95% CI on top-1 Fisher is
  wide; rank-1 vs rank-2 (1.18 vs 1.16) is within noise.
- The Fisher-ratio criterion captures linear separability; SAEUron's
  upstream method may use a different metric (e.g., causal
  intervention scoring like our Stage 2).
- Top-20 features have negative differentiation in some cases (e.g.,
  rank 14 feature 11495 has μ_nsfw < μ_benign). Indicating either
  feature inversion or a noisy small-sample artefact. Re-running with
  larger n would tighten.
- We have not yet run the intervention with feature 12571 to verify
  it reduces safety_checker flag rate. The "correct feature" claim is
  based purely on activation-contrast.
