# Patch-kind ablation — mean / zero / resample at the same F_c

## Setup
Stage 1 ∩ Stage 2 F_c (12 + 0 + 32 + 25 = 69 features across the four Surkov
hookpoints). 100 I2P-NSFW prompts, batch=4, SDXL Turbo 1-step. Same prompts +
same seeds across all three runs for a fair comparison.

## Headline (safety_checker)

| run | F_c source | patch | n_pre_flagged | n_corrected | correction rate | post_flagged_rate_overall | wall (s) |
|---|---|---|---|---|---|---|---|
| D01 | Stage-1 only (top-32 × 4)         | mean      | 6  | 1 | 0.167 | 0.140 | 148  |
| **D02** | Stage 1 ∩ Stage 2            | **mean**  | 10 | 4 | **0.400** | 0.090 | 280 |
| **D03** | Stage 1 ∩ Stage 2            | **zero**  | 10 | 4 | **0.400** | 0.100 | 305 |
| **D04** | Stage 1 ∩ Stage 2            | **resample** | 10 | 4 | **0.400** | 0.100 | 302 |

All three patch kinds at the proper Stage 1 ∩ Stage 2 F_c achieve the same
correction rate of 0.40. Mean wins by 1pp on overall post-flagged rate
(disrupts un-flagged generations 1pp less).

## Interpretation
The spec's predicted mean ≫ zero on ASR is **not seen** at this scale — the
three patch primitives are interchangeable for safety-checker correction
when the F_c features are well chosen by Stage 1 ∩ Stage 2. The mean
advantage is expected to shift to the FID side (mean preserves the benign
distribution structure, zero introduces an off-manifold artefact, resample
samples a single benign image's value at the feature). FID measurements
on the post images are queued and will sharpen the mean-vs-zero claim.

The headline robustness result is that **F_c quality dominates patch primitive**:
moving from Stage-1-only top-32 (D01, 17% correction) to Stage 1 ∩ Stage 2
(D02-D04, 40% correction) is a 23-pp improvement, while the patch-kind choice
is at most 1 pp on this slice.

## Artefacts
- `outputs/D02_stage1n2_meanpatch_n100/` — 200 PNGs (pre + post) + summary.
- `outputs/D03_stage1n2_zeropatch_n100/`  — same.
- `outputs/D04_stage1n2_resamplepatch_n100/` — same.

## Next
- FID + CLIP-score on each {pre, post} dir (computing in parallel; first
  result D02-pre has cached COCO val2017 statistics).
- LPIPS + DreamSim across all three (queued).
- Combine with B02 oracle detector trained on rendered-image labels for a
  final headline table (Item 6 redo).
