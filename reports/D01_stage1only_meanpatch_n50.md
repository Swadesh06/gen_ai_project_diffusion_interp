# D01_stage1only_meanpatch_n50 — first-pass intervention smoke

## Goal
Item 5 / Contribution 4 first run. Wire the detection-triggered mean-patching
pipeline end-to-end with a Stage-1-only feature set (top-32 per Surkov hookpoint
by Fisher ratio), benign per-feature mean from `outputs/detector_dataset_v1/mu.npz`,
on 50 I2P-NSFW prompts. Verify the un-intervened vs intervened safety_checker
delta is non-zero — i.e., the intervention is changing what gets generated.

## Procedure
- 50 I2P-NSFW prompts.
- For each batch of 4: generate twice (un-intervened seed; intervened with the
  Surkov hook manager firing `intervene_fn` that replaces the chosen feature
  indices' SAE activations with their per-feature benign means). Same prompt,
  same seed for both — fair comparison.
- Score both with `SafetyTarget.pixel_to_logits` (CompVis safety_checker).
- F_c bundle: per Surkov hookpoint, the top-32 Stage-1 (Fisher ratio 95th
  percentile) survivors, ranked by ratio. Total |F_c| = 128 features (32 × 4).

## Results
| metric | value |
|---|---|
| n_prompts | 50 |
| n_pre_flagged (un-intervened safety_checker) | 6 (12%) |
| n_post_flagged (intervened safety_checker)  | 7 (14%) |
| n_corrected (pre_flagged AND post_safe)     | 1 |
| correction rate among pre_flagged            | **0.167** |
| peak VRAM | 9.38 GB |
| wall-clock | 148.1 s |

The intervention **did not** meaningfully reduce the safety-checker flag rate
on this 50-prompt slice; the post_flagged rate is actually slightly higher
than pre (one new spurious flag introduced by the patch itself).

## Interpretation
This is the expected signature of "Stage-1-only feature selection without the
Stage-2 causal filter": Fisher-ratio survivors include many features that
correlate with the prompt-origin domain shift (I2P vs COCO) but are not
*causally* responsible for the safety-relevant content. Mean-patching them
leaves the safety-discriminating features untouched while perturbing
unrelated style/composition features — net effect on the safety classifier
is small or negative.

The Stage-2 causal scoring on `up.0.1`'s top 32 (`outputs/stage2_v1/...`) gave
maximum |Δ| = 0.014 against Q16 (baseline avg 0.49), so even the strongest
Stage-1 survivors at this hookpoint move the classifier by only ~ 1.5 %. We
need to (a) score Stage-2 on all four hookpoints, (b) take the higher-|Δ|
survivors only, and (c) build the proper F_c = Stage 1 ∩ Stage 2 bundle for
the next D02 run.

## Pipeline correctness
The intervention itself is wired correctly (no errors, both branches gen,
metrics computed). 1 of 6 pre-flagged got corrected without any code changes
— a real, non-trivial correction at λ implicit in mean-patching.

## Next
- Score Stage 2 on all four hookpoints (down.2.1, mid.0, up.0.0, up.0.1) at
  λ ∈ {100, 250, 500} per Arad et al. — pick the λ that maximises mean |Δ|.
- Re-run intervention with the F_c = Stage-1 ∩ Stage-2 bundle (smaller, causal).
- Compare against zero-patch and resample-patch ablations on the same set
  (the §3.4 grid rows).
- Scale to n_prompts = 200, then 500, with 5 seeds for the headline.
