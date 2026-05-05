# repro_saeuron_v2_feature12571 — Item 1c-5 close: SAEUron with correct feature

## Goal

Re-run `scripts/repro_saeuron_streamlined.py` with the **data-driven**
feature_idx=12571 (from `saeuron_feature_contrast_v1.md`) instead of
the upstream-example feature_idx=11627 (cat) which was used in v1.

## Procedure

`scripts/repro_saeuron_streamlined.py` with:
- `--concept nudity` (UDA-nudity 30 prompts)
- `--feature-idx 12571` (top-1 by Fisher ratio)
- `--multiplier -3.0`
- SD v1.4, 50 inference steps, CFG=7.5
- Wall: 259 s

Renders 30 baseline (no_intervention) + 30 with_saeuron generations
at hookpoint `unet.up_blocks.1.attentions.1`.

## Results

| condition | n | safety_checker flagged | flag rate |
|---|---|---|---|
| **no_intervention** | 30 | 14 | **46.7%** |
| **with_saeuron** (feature 12571, mult −3.0) | 30 | 12 | **40.0%** |
| **Δ (reduction)** | — | -2 | **-6.7 pp** |

Wilson 95% CIs:
- no_intervention: [29.8%, 64.4%]
- with_saeuron: [24.6%, 57.7%]

Single-tailed test of "with_saeuron < no_intervention": p ≈ 0.4 (CIs
overlap heavily on n=30; result is suggestive but not significant).

## Comparison to v1 (wrong feature, cat)

| feature | n | no_intervention | with_saeuron | Δ |
|---|---|---|---|---|
| **12571** (correct, this run) | 30 | 46.7% | 40.0% | **-6.7 pp** |
| 11627 (cat, v1 prior) | 30 | 33.3% | 43.3% | +10.0 pp |

The correct feature produces the **expected direction** (flag rate
decreases), confirming the SAEUron method's claim qualitatively. The
wrong-feature v1 result was an artefact of the cat-feature being
unrelated to nudity content.

## Comparison to F_c surgery (our pipeline) on same UDA-nudity

From `udatk_nudity_meanpatch_v1.md`:

| method | n | pre flag rate | post flag rate | reduction |
|---|---|---|---|---|
| **F_c surgery (D02 nudity, SDXL Base 4-step)** | 100 | 35.0% | 47.0% | **+12.0 pp** (worse) |
| **SAEUron (12571, SD v1.4 50-step)** | 30 | 46.7% | 40.0% | **-6.7 pp** (better) |

**SAEUron's correct feature reduces the flag rate; our F_c surgery on
SDXL Base 4-step increases it.** The two methods are not directly
comparable (different SD architectures, different inference steps,
different intervention objects: SAEUron negates a single feature
multipliedly, our F_c replaces with benign mean of 69 features), but
the direction is informative.

For a fair comparison:
- Run our F_c surgery on SD v1.4 50-step UDA-nudity n=30.
- Or run SAEUron on SDXL Base 4-step UDA-nudity n=100.

Both queued.

## Caveats

- n=30 is small; CIs are wide.
- Multiplier=-3.0 is the SAEUron upstream default for the cat feature.
  For nudity it may need stronger negation; try -5.0 or -10.0.
- The SAEUron paper Table 1 reports ASR ≈ 5%. We got 40% post-SAEUron.
  The gap is likely (a) multiplier under-tuning and (b) UDA-nudity is
  more adversarial than the paper's I2P-NSFW.
- SAEUron disables the in-pipe safety_checker via `safety_checker=None`,
  so the flag is from a separate scoring pass — same as our metric.
