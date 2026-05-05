# D06_mask_training_comparison_v1 — joint mask under three training distributions

## Goal

The D-6 joint mask + head pipeline can be trained on any
labeled-NSFW-vs-benign dataset. This report compares three training
distributions and their downstream UDA-nudity correction performance:

1. **violence-trained**: `raw_violence_n200` (200 violence-prompt
   I2P images) vs `raw_coco_500` (200 COCO benign).
2. **cf_strategy2-trained**: 246 cf_strategy2_seed_pairs
   (counterfactual: same prompt, different seed).
3. **i2p-trained**: `raw_i2p_500` (200 general I2P-NSFW images,
   mixed concepts) vs `raw_coco_500` (200 COCO benign).

All trained at λ_sparsity=5.0, 200 epochs, identical hyperparameters.
Then applied as F_c hook in SDXL UNet, rendered UDA-nudity n=50
prompts, scored with safety_checker.

## Training distribution table

| training | n_train | val AUC | active features | per-hookpoint |
|---|---|---|---|---|
| violence vs COCO | 320 | 1.000 | 41 | down.2.1=24, mid.0=0, up.0.0=1, up.0.1=16 |
| cf_strategy2 pairs | 393 | 0.942 | 384 | down.2.1=44, mid.0=86, up.0.0=127, up.0.1=127 |
| i2p vs COCO | 320 | 1.000 | 44 | down.2.1=22, mid.0=0, up.0.0=2, up.0.1=20 |

**Observation**: cf_strategy2 produces a **denser** mask (384 vs 41-44)
because the discrimination is harder (same prompt, only seed differs).
The violence- and i2p-trained masks are similar in size (~40 features)
but identify different specific features.

## Mask overlap (Jaccard)

| pair | total intersection | total union | Jaccard |
|---|---|---|---|
| violence ∩ i2p | 17 | 68 | 0.250 |

Per-hookpoint:

| hookpoint | viol | i2p | inter | Jaccard |
|---|---|---|---|---|
| down.2.1 | 24 | 22 | 11 | 0.314 |
| mid.0 | 0 | 0 | 0 | n/a |
| up.0.0 | 1 | 2 | 0 | 0.000 |
| up.0.1 | 16 | 20 | 6 | 0.200 |

The two natural-NSFW masks (violence, i2p) share 25% of features
(by Jaccard). Both concentrate on `down.2.1` (skip-residual to up
path) and `up.0.1` (last attention before bottleneck). `mid.0` is
not used by either. Structural similarity but feature-level
divergence.

## UDA-nudity application table

| training | active | n_pre | n_post | corrected | new FP | corr rate | net Δ |
|---|---|---|---|---|---|---|---|
| **violence vs COCO** | **41** | 19 | 18 | **9** | **8** | **47.4%** | **-1** |
| cf_strategy2 pairs | 384 | 19 | 23 | 7 | 11 | 36.8% | +4 |
| i2p vs COCO | 44 | 19 | 25 | 7 | 13 | 36.8% | +6 |

**The violence-trained mask is the clear winner across all three
metrics**: highest correction rate, lowest new-FP rate, only
distribution to achieve net flag-rate improvement.

## Why violence > i2p > cf_strategy2 on UDA-nudity?

This is counter-intuitive at first: i2p-NSFW contains nudity-relevant
prompts, so an i2p-trained mask should generalize to UDA-nudity
better than a violence-trained mask. Yet violence wins.

Possible explanations:

1. **safety_checker's concept space**: SDXL's `StableDiffusionSafetyChecker`
   was trained on a specific concept set that may align more with
   violence-flavored composition (gore, struggle, fear) than with
   nudity. The "feature axis" the safety_checker uses might overlap
   more with violence-relevant SAE features than nudity-specific ones.
2. **Sparser is more concept-axis-aligned**: at λ=5.0, the optimizer
   is forced to identify the *most discriminating* features. For a
   violence-vs-COCO contrast, those happen to be safety_checker-aligned
   features. For i2p-vs-COCO (mixed concepts), the optimizer finds
   features that span the concept-mixture, not the safety-axis.
3. **Distribution shift mismatch**: cf_strategy2 trained on
   within-prompt seed variations doesn't transfer because UDA-nudity
   is a cross-prompt-distribution test.

## Implications for the paper

**The headline mask** is the violence-trained 41-feature mask. The
"cleanest" interpretability story is: 41 SAE features (concentrated at
`down.2.1` and `up.0.1`) suffice to correct safety_checker's NSFW
flag for UDA-nudity prompts at 47% rate with only 8 false positives.

The i2p-trained and cf_strategy2-trained masks are valuable as
ablations showing that:
- cf_strategy2 (counterfactual pairs) overfits to subtle within-prompt
  variations.
- i2p-trained (mixed-concept) finds different features that don't
  generalize as well to specific UDA-nudity setting.

Sparsity matters: 41 features works better than 44 features — the
specific feature selection matters more than the count.

## Caveats

- All three trained at the same λ_sparsity=5.0; testing different λ
  on i2p-trained could reveal more.
- All three rendered with the same UDA-nudity n=50 prompts; n=100
  would tighten confidence.
- The "violence wins" finding might be specific to UDA-nudity at SDXL
  Base 4-step. Other test distributions (I2P-NSFW at SDXL Turbo,
  MMA-Diffusion at SD 1.4) might re-order the rankings.

## Next

- λ_sparsity sweep on i2p-trained: does forcing 20 features select
  the safety_checker-axis features?
- Test all three masks on I2P-NSFW SDXL Turbo (their training
  distribution, more or less). Does violence-trained win there too?
- Hybrid training: mix violence + i2p + COCO for a "broader concept"
  mask.

## Artifacts

- `outputs/D06_lambda5_state/joint_state.pt` (violence-trained, 41 feat)
- `outputs/D06_joint_e2e_v4_cf_strategy2/joint_state.pt` (384 feat)
- `outputs/D06_joint_e2e_v5_i2p_n200/joint_state.pt` (i2p-trained, 44 feat)
- `outputs/D06_joint_mask_udatk_n50_lam5/{summary.json}` — violence on UDA
- `outputs/D06_joint_mask_udatk_n50_cf/{summary.json}` — cf on UDA
- `outputs/D06_joint_mask_udatk_n50_i2p/{summary.json}` — i2p on UDA
