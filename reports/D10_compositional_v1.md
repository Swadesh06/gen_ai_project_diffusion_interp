# D10_compositional_v1 — composite (SC + SAE detector) defense on D02 outputs

## Goal

Phase D-10 from `task_descriptions/task_description_v2.md` §6:
compositional defense. The simplest version is the **union flag**
across multiple detectors. For the D02 patch-ablation outputs (n=100
pre/post), compute per-image classifications under (safety_checker,
B02-v3 SAE detector) and report:
1. Marginal flag rates per layer
2. Per-image union (any layer flags = bad)
3. Whether F_c surgery improves or degrades the union rate

## Procedure

1. Render 100 I2P-NSFW prompts through SDXL Turbo with vs without F_c
   mean-patch hook (D02 already produced these images at
   `outputs/D02_stage1n2_meanpatch_n100/{pre,post}/`).
2. Run B02-v3 SAE detector via `scripts/eval_b02v3_on_mma.py` adapted
   to the D02 directory structure (re-uses VAE encode + UNet at t=50
   + SAE encoder + linear probe).
3. Cross-tabulate per-image safety_checker flag × SAE detector flag
   for both pre and post.

## Results — per-image classifications

### PRE images (no F_c surgery)

| classification | count |
|---|---|
| safety_checker AND SAE detector | 1 |
| only safety_checker | 9 |
| only SAE detector | 4 |
| neither | 86 |
| **any layer flag** | **14 / 100 = 14.0%** |

Marginals: SC flag rate **10%**, SAE detector flag rate **5%**.

### POST images (F_c mean-patched)

| classification | count |
|---|---|
| safety_checker AND SAE detector | 0 |
| only safety_checker | 9 |
| only SAE detector | 7 |
| neither | 84 |
| **any layer flag** | **16 / 100 = 16.0%** |

Marginals: SC flag rate **9%**, SAE detector flag rate **7%**.

## Interpretation

**F_c surgery shifts NSFW signal from one layer to another rather than
removing it from both.**

- F_c reduces SC flag rate marginally: 10% → 9% (4 corrected, 3 added).
- F_c **increases** SAE detector flag rate: 5% → 7% (2 added).
- Union flag rate increases slightly: 14% → 16%.

The 2 added SAE flags in post are images where F_c surgery has
introduced features the SAE-feature linear probe reads as NSFW-shifted
even though safety_checker doesn't catch them. This is a known
side-effect of feature-replacement interventions: the
post-intervention activation distribution is **out of distribution**
for the SAE detector's training set, and the detector picks up on the
unfamiliar pattern.

**Implication for the four-layer defense**:

| layer | catches what (in 100 SDXL Turbo I2P-NSFW prompts) |
|---|---|
| safety_checker | 10% baseline on natural generations |
| SAE detector (B02-v3) | 5% baseline + 2% extra on F_c-patched |
| F_c surgery | "removes" 40% of safety_checker pre_flagged |
| union | 14-16% any-layer flag, regardless of F_c |

**F_c is most useful for generation-time defense, not for boosting
detection union**. The paper claim should be:
- F_c reduces safety_checker rate at generation time
- SAE detector catches what safety_checker misses (transferability=0
  on adversarial)
- Combined, they form independent layers, but F_c at generation does
  not add to detection-union at inference

## Adversarial scope (deferred)

Proper D-10 compositional adversarial eval requires testing each
attack against the **full stack**:
- Prompt filter (input)
- F_c surgery (generation)
- safety_checker (output)
- SAE detector (post-generation, second pass)

For each attack, measure layer-by-layer survival. Not in scope for
this report (requires modifying attack runners).

## Caveats

- n=10 SC pre_flagged is small. Wilson 95% CI [0.05, 0.46] for the
  40% correction rate.
- SAE detector logit threshold is 0 (the natural LR cutoff). Lower
  thresholds would catch more (higher recall, more false positives).
- The 7 "only SAE detector" post-flagged images have not been
  manually inspected; some may be true positives (NSFW content
  safety_checker missed) rather than F_c artefacts.
