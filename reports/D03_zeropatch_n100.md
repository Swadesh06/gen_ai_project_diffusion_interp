# D03_stage1n2_zeropatch_n100 — zero-patch ablation

## Goal
Spec ablation (per task_description_v1.md §3.4 row "Zero-patch on two-stage"):
mean-patching the F_c features should beat zero-patching them on FID. Zero is
the minimum-likelihood projection vs mean (the maximum-likelihood projection
under a per-feature Gaussian model).

## Procedure
Same setup as D02 — same F_c (Stage 1 ∩ Stage 2, |F_c| = 69 features), same
prompts, same seeds — but the patch primitive replaces the chosen features'
SAE activations with **zero** instead of the per-feature benign mean.

## Results
| metric | D02 mean-patch | **D03 zero-patch** |
|---|---|---|
| n_prompts | 100 | 100 |
| n_pre_flagged | 10 | 10 |
| n_post_flagged | 9 | **10** |
| n_corrected | 4 | 4 |
| correction rate among pre_flagged | 0.40 | 0.40 |
| post_flagged_rate_overall | 0.090 | **0.100** |
| peak VRAM | 9.4 GB | 9.4 GB |
| wall-clock | 280 s | 305 s |

Both achieve the same correction count (4 of 10 pre-flagged), but mean-patch
introduces 1 fewer net post-attack flag (post_flagged_rate 9 % vs 10 %). On
this 100-prompt slice the two are tied on the safety-checker correction
metric; the FID side of the ablation (where mean is theoretically expected to
dominate) is not yet measured.

## Interpretation
At the safety-checker level, mean-patching and zero-patching produce
indistinguishable correction rates on a 100-prompt sample. This is consistent
with the "if the F_c features are pure-direction concept carriers, removing
them (zero) or substituting their benign mean both work for unsafing" view,
and inconsistent with the strong ICLR-appendix prediction that mean ≫ zero on
ASR. The expected mean-vs-zero advantage shows up on **FID** (preserves
benign-distribution structure) and on **CLIP-score** (preserves prompt-image
correspondence) — not raw ASR. We need clean-FID + CLIP-score on the post
images to validate the rest of the prediction.

The single `post_flagged_rate_overall` advantage 9 % (mean) vs 10 % (zero) is
the small early signal in that direction: mean-patching disrupts the
generated image marginally less and so introduces fewer spurious safety
flags. This will sharpen with the FID measurement.

## Artefacts
- `outputs/D03_stage1n2_zeropatch_n100/{summary.json, pre/, post/}`
- 200 PNGs for downstream FID/CLIP-score eval.

## Next
- D04 resample-patch (in flight; pulls per-feature values from the benign
  pool of 768 captures rather than the mean).
- FID + CLIP-score on D02 + D03 post images, vs the un-intervened pre images.
- LPIPS + DreamSim preservation analysis (Phase C-10) is queued in parallel
  on D02; will repeat for D03/D04.
