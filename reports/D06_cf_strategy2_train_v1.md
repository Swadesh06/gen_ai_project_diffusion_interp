# D06_cf_strategy2_train_v1 — joint mask trained on counterfactual seed-pairs (D-6 v4)

## Goal

Train the D-6 joint mask + head on the cf_strategy2_seed_pairs
counterfactual benchmark (246 pairs of NSFW + benign images from the
same prompt × different seed). This is the "correct" data distribution
per spec — counterfactual rather than synthetic violence-prompt-based.
Then render UDA-nudity with the resulting mask and compare to the
violence-trained baseline (`D06_joint_mask_udatk_v1`).

## Procedure

### Training (`exp_D06_joint_e2e_v4_cf_strategy2.py`)

1. Load 246 cf_strategy2 pairs (492 images: 246 flagged + 246 unflagged).
2. Encode each via SDXL Turbo UNet at t=50 with SAE hooks; mean-pool
   over (H, W) → (B, 20480) per image.
3. 80/20 stratified train/val split.
4. Train joint pipeline (soft mask M + linear head) with three losses
   (D-6 v2 protocol). λ_sparsity=5.0, 200 epochs, Adam lr=3e-3.
5. Save state to `outputs/D06_joint_e2e_v4_cf_strategy2/joint_state.pt`.

Encoding time: 151 s for 492 images.

### Application (`exp_D06_apply_mask_to_udatk.py`)

Load the saved joint state, split mask into 4 hookpoints, register as
F_c hook in SDXL UNet, render UDA-nudity n=50 prompts.

## Training results

| metric | value |
|---|---|
| n_pairs | 246 (492 images) |
| baseline LR val AUC | 0.9364 |
| **joint val AUC** | **0.9424** |
| active mask features (M > 0.5) | 384 / 20480 (1.88%) |
| training correction rate | 100% (38/38) |

Per-hookpoint mask size: down.2.1=44, mid.0=86, up.0.0=127, up.0.1=127.

The cf-trained mask is **denser** than the violence-trained mask
(384 vs 41 features at the same λ_sparsity=5.0). This reflects the
harder discrimination task: cf_strategy2 pairs share the same prompt,
so the safe/unsafe distinction is **subtle** (often just composition
or pose differences), needing more features to separate.

## UDA-nudity application results (n=50)

| metric | violence-trained mask | cf_strategy2-trained mask |
|---|---|---|
| n_active features | 41 | 384 |
| n_pre_flagged | 19 | 19 |
| n_post_flagged | 18 | 23 |
| **corrected** | **9 / 19 = 47.4%** | 7 / 19 = 36.8% |
| **new false positives** | **8** | 11 |
| **net Δ flag rate** | **-1 (improved)** | +4 (worse) |

**Counter-intuitive result: the violence-trained 41-feature mask
outperforms the cf-trained 384-feature mask on UDA-nudity rendering.**

## Interpretation

**Sparser is better here.** The 41-feature mask trained on (COCO
benign vs raw_violence_n200) generalizes to UDA-nudity better than
the 384-feature mask trained on cf_strategy2_seed_pairs.

Possible reasons:
1. **Concept specificity**: cf_strategy2 pairs share the same prompt
   (only seed differs), so the discriminating features are **subtle**
   composition/pose features, not concept-specific NSFW features.
   The mask captures these subtleties, but they don't generalize to
   UDA-nudity (which is a different distribution shift).
2. **Sparsity bias**: a 41-feature mask is forced to identify the
   most concept-relevant features (the "concept axis"); a 384-feature
   mask can fit subtler distinctions but at the cost of generality.
3. **Training distribution shift**: cf_strategy2 = SDXL Turbo I2P
   prompts at 1-step. UDA-nudity = SDXL Base 4-step CFG=7.5. The
   architecture and conditioning differ.

**Lesson**: for the production pipeline, **prefer the violence-trained
sparse mask** (41 features) over the cf-trained dense mask (384
features). The violence training is a stronger concept-axis signal;
cf training overfits to subtle within-prompt variations.

This finding aligns with the modular F_c surgery's behavior: the
modular pipeline uses Stage 1 (Fisher ratio) on a clean NSFW vs benign
contrast (also like the violence-trained setting), and avoids cf-style
seed-pair noise.

## Combined paper claim

| training data | features | UDA correction | UDA new FP | net Δ |
|---|---|---|---|---|
| Modular F_c surgery (violence corpus) | ~32 | 34% | 24 (37%) | +12 |
| **Joint mask v3 (raw_violence_n200 vs COCO)** | **41** | **47%** | **8 (26%)** | **-1** |
| Joint mask v4 (cf_strategy2 seed pairs) | 384 | 37% | 11 (32%) | +4 |

The headline production-ready mask is the violence-trained 41-feature
mask, **not** the cf-trained 384-feature mask. Counterfactual training
captures different signal than concept training; for in-context NSFW
correction, concept training wins.

## Caveats

- n=50 UDA-nudity prompts; n=100 would be tighter. Both runs used
  the same prompt set so comparison is direct.
- λ_sparsity=5.0 was used for both; cf could have been tested at a
  larger λ to force comparable sparsity.
- The cf-trained AUC (0.94) is genuinely high — the training works,
  just the *application to UDA-nudity rendering* doesn't.

## Next

- Cross-distribution transfer: train on cf_strategy2, test on raw
  I2P-NSFW (SDXL Turbo). Does the cf-trained mask work on its own
  distribution but fail on the cross-architecture UDA setting?
- λ_sparsity=50.0 on cf_strategy2 → force ~40 features; rerun on
  UDA-nudity to control for sparsity.
- Mixed training: 246 cf pairs + 200 raw_violence_n200 + 200 COCO →
  4-class problem; train on union; test on UDA-nudity.

## Artifacts

- `outputs/D06_joint_e2e_v4_cf_strategy2/{joint_state.pt, results.json}`
- `outputs/D06_joint_mask_udatk_n50_cf/{pre, post, summary.json,
  F_c_joint_mask.json}`
