# repro_saeuron_nudity_n30_smoke — SAeUron pipeline reproduction (caveats)

## Goal

Item 1c-5 — SAeUron baseline reproduction. Validate the upstream
SAeUron `bcywinski/SAeUron_coco` checkpoint + `HookedStableDiffusionPipeline`
loads correctly and produces SAE-feature-intervention output on
UnlearnDiffAtk-nudity prompts.

## Procedure

- Repo: `/workspace/datasets/SAeUron` (cywinski/SAeUron upstream).
- Wrapper: `scripts/repro_saeuron_streamlined.py`.
- Backbone: SD v1.4 (`CompVis/stable-diffusion-v1-4`).
- SAE checkpoint: `bcywinski/SAeUron_coco` at hookpoint
  `unet.up_blocks.1.attentions.1`.
- Intervention: `SAEFeatureInterventionHook(feature_idx=11627, multiplier=-3.0)`.
- Prompts: 30 from UnlearnDiffAtk-nudity.csv.
- 30 inference steps, CFG 7.5, 512×512.
- Wall: 2910 s (49 min).
- Output: 30 baseline (no intervention) + 30 with-saeuron generations,
  oracle-scored by safety_checker.

## Results

| condition | n | safety_checker flagged | flag rate |
|---|---|---|---|
| no_intervention | 30 | 10 | **33.3 %** |
| with_saeuron (feature 11627, mult −3.0) | 30 | 13 | **43.3 %** |

## Interpretation — important caveat

**The SAeUron intervention INCREASED the flag rate (33% → 43%), not
decreased.** This is **NOT** an indictment of SAeUron's method — it's
a misuse of the API.

`feature_idx = 11627` is the SAeUron upstream-example notebook's
illustrative feature_idx for the "**cat**" concept, not nudity. The
SAeUron paper publishes per-concept feature lists; the
`bcywinski/SAeUron_coco` checkpoint was selected to demo the pipeline,
not necessarily nudity-specific. Negative scaling of an unrelated
feature can perturb generation in arbitrary ways.

For a **proper** SAeUron-vs-our-pipeline head-to-head:
1. Identify the SAeUron-published nudity feature_idx (paper Section 4
   or supplementary; if not published, run the activation-contrast
   feature selection on the SAeUron SAE on a balanced
   nude-vs-clothed split).
2. Use the SAeUron's UnlearnCanvas-trained checkpoint instead of
   `_coco`, which is style/object-tuned.

The pipeline integration works (HookedStableDiffusionPipeline loads,
SAE encodes, intervention hook fires, generation completes); the
issue is feature selection.

## Next

- Hunt down the published nudity feature_idx in SAeUron paper /
  upstream issues / supplementary.
- Re-run with the correct feature_idx; expected flag-rate drop to
  ≤ 10 % (per their published Table 1 ASR ≈ 5 %).
- Once this baseline is solid, compare against our Stage 1 ∩ Stage 2
  + mean-patch on the same UnlearnDiffAtk-nudity split.
- Add SAEmnesia repro (no public release yet — train from scratch
  on UnlearnCanvas concepts at SAeUron's hookpoints).
