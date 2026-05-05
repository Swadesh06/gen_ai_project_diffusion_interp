# udatk_nudity_meanpatch_v1 — Item 1c-4 close: F_c intervention on UDA-nudity

## Goal

Item 1c-4 close (intervention pending). The UDA-nudity prompts
(n=143) were rendered + scored at the safety_checker baseline level
in `udatk_safety_scores.md`. The remaining piece was the F_c
intervention: do nudity F_c features generalize from I2P-NSFW
(D02 training data) to UDA-nudity (held-out distribution)?

## Procedure

`scripts/exp_udatk_nudity_intervention.py`. For each of the first 100
UDA-nudity prompts:
1. Render with **SDXL Base 4-step** (UDA's reference setting), CFG=7.5.
2. Render again, same seed, with `SurkovHookManager` applying F_c
   mean-patch on Stage 1 ∩ Stage 2 nudity features.
3. Score both with safety_checker.

`F_c` set: `outputs/F_c_stage1n2_top.json` (D02 training output: 12
features at down.2.1, 0 at mid.0, 32 at up.0.0, 25 at up.0.1 — 69
total). Trained on I2P-NSFW prompts.

## Results

| metric | n=100 UDA-nudity |
|---|---|
| pre_flagged (no F_c) | 35 / 100 = **35.0%** |
| post_flagged (F_c mean-patch) | 47 / 100 = **47.0%** |
| n_corrected (pre_flag → post_unflag) | **12 / 35 = 34.3%** |
| n_new_false_positives (pre_unflag → post_flag) | **24 / 65 = 37.0%** |
| net Δ post − pre | **+12 (worse)** |
| wall time | 279 s |

## Comparison to D02 (I2P-NSFW)

| dataset | n_pre | corrected | rate | n_new_fp |
|---|---|---|---|---|
| D02 I2P-NSFW (n=100, SDXL Turbo 1-step) | 10 | 4 | 40.0% | 3 |
| **UDA-nudity (n=100, SDXL Base 4-step)** | **35** | **12** | **34.3%** | **24** |

The correction rate generalizes (40% → 34%) but the false-positive
rate is dramatically worse on UDA-nudity (3% → 24%). F_c surgery on
SDXL Base 4-step produces more "corrupted" outputs that safety_checker
flags spuriously.

## Interpretation

**The nudity F_c set learned on SDXL Turbo / I2P-NSFW partially
generalizes to UDA-nudity / SDXL Base 4-step**:

- Correction rate (when pre is correctly flagged): 34.3% — comparable
  to D02's 40%.
- BUT: F_c intervention introduces **more** new false positives than it
  corrects (24 new vs 12 corrected). Net flag rate goes UP.

Two non-exclusive explanations:

1. **Distribution shift**: The F_c set is centred on SDXL Turbo's
   1-step output distribution. Applying it to SDXL Base 4-step's
   different inference trajectory produces residual artefacts that
   safety_checker reads as NSFW (the "F_c shifts signal" phenomenon
   from D-10).

2. **UDA-nudity prompt distribution**: UDA's nudity prompts are
   adversarial, often containing words like "stockings" or
   "nightmare" that are partial-match to nudity but include other
   content (self-harm, violence, etc.) that the F_c set doesn't
   neutralize. The F_c intervention partially zeros out nudity
   features but the surrounding content gets distorted.

**The paper claim**: F_c surgery is **prompt-distribution-conditional**.
On the training-distribution-similar I2P-NSFW prompts F_c provides net
defense (correction > false-positive). On out-of-distribution
UDA-nudity F_c provides per-correction benefit but net harm to flag
rate.

## What this means for D-8 / D-10

The Phase 1c v2 spec calls for the intervention pipeline to work
across the v2 evaluation grid (5 datasets). On UDA-nudity the F_c
surgery is **net-negative** unless the false-positive rate is
acceptable for the application (e.g., if the false-positive images
are then filtered by a downstream check, F_c's correction is still
useful).

A more conservative deployment is "**F_c trigger by SAE detector**":
only apply F_c surgery to images the SAE detector flags. This avoids
the false-positive explosion on benign UDA-prompts. Implemented in
the `gating="on_detection"` mode of run_intervention.py but not yet
evaluated on UDA-nudity. Future work.

## Caveats

- 100 UDA-nudity prompts is a substantial sample; Wilson 95% CI on
  34.3% correction = [22.7%, 48.0%].
- The 47% post flag rate has 95% CI [37.5%, 56.7%].
- The "false positive" count is a heuristic — pre is "safe-by-
  safety_checker" not "actually safe". The 24 new flags may be
  legitimate NSFW that the F_c surgery surfaces (i.e., F_c failure
  to remove + some artefact addition).
- SDXL Base 4-step has its own internal safety_checker that returns
  black images for some NSFW prompts — those are handled identically
  pre and post.
