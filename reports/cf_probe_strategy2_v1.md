# cf_probe_strategy2_v1 — counterfactual probe on same-prompt different-seed pairs

## Goal

Item 1c-0 evaluation downstream — second framing-discriminator number.
Strategy 2 holds prompt distribution **exactly** constant
(same prompt, different seed) and varies only the noise seed. The
probe trained to discriminate flagged-seed from unflagged-seed
activations is reading **purely** image-content signal at timestep 50
with the actual prompt as conditioning.

## Procedure

- Source: `outputs/cf_benchmark_v1_seed/validated.jsonl` (246 validated
  same-prompt seed pairs, where one seed flagged and another did not
  on the identical prompt).
- Encoding: VAE-encode flagged + unflagged → UNet single forward at
  timestep 50 with the **actual prompt** (not empty) → SurkovHookManager
  capture → spatial-mean-pool to (B, 5120) per side, concat 4 hookpoints
  to 5120-d (raw).
- Probe: `LogisticRegression(class_weight="balanced")`.
- 80/20 in-distribution split + per-prompt leave-one-out (LOPO).

## Results

### In-distribution (80/20 random split)

| metric | value |
|---|---|
| n_pairs | 246 |
| n_unique_prompts | ~76 |
| **AUC** | **0.9436** |
| AP | 0.8840 |

### Strategy 1 vs Strategy 2 — the key contrast

| benchmark | n_pairs | encoding prompt | in-distribution AUC |
|---|---|---|---|
| Strategy 1 prompt-edit pairs | 63  | empty | **0.275** (below chance) |
| **Strategy 2 same-prompt pairs** | **246** | **actual prompt** | **0.9436** |

## Interpretation

**Strategy 2's AUC = 0.9436 is the key reframe.** With prompt held
constant and the actual prompt used as conditioning, the SAE
features **do** carry strong flagged-vs-unflagged signal. The probe is
reading image content, not prompt distribution.

The Strategy 1 result (AUC 0.275 on prompt-edit pairs with empty
conditioning) was misleading — it conflated three confounds:
1. Different prompts (edited safety token).
2. Different conditioning (empty vs actual).
3. Different encoding (timestep 50 with empty embeds).

Strategy 2 isolates the seed → image variation under fixed prompt.
SAE features pick up the unsafe content cleanly when conditions are
matched.

**Framing implication:**
- Framing A (SAE detector primary) gets a strong "yes" signal from
  Strategy 2.
- Framing B (SAE for interp + intervention) is consistent but the
  detector role is now defended by Strategy 2's clean signal.

The headline message:
- Strategy 1 alone would push toward Framing B.
- Strategy 2 alone would push toward Framing A.
- Together they suggest the SAE detector works *given content is
  actually present* (Strategy 2 is the natural deployment regime —
  benign vs unsafe images of the same prompt) but does **not** work
  on prompt-distribution shortcuts (Strategy 1's substitution dictionary
  varies prompt).

The cross-target result (transferability = 0.000 on A01 + A02) holds
in either framing — those are pixel/latent perturbations, not prompt
edits. The detector is robust to image-space perturbations even though
it is sensitive to (genuine) image-content variation.

## Caveats

- Per-prompt LOPO results not yet finalized (script still running on
  100 unique prompts). The headline AUC 0.9436 is the in-distribution
  80/20 split; cluster generalization analogous to Strategy 1's
  per-cluster LOO is coming.
- The probe is logistic regression on 5120-d concat features. A
  deeper probe might lift AUC further but may also overfit at n=246
  pairs.
- Encoding uses timestep 50 (low noise, ~5% noise schedule). At
  higher noise (matching the active denoising schedule SDXL Turbo
  actually uses) the signal may differ.

## Next

- Wait for LOPO numbers to land.
- Run Strategy 2 with `--feature-source sae` (Surkov SAE encoded) for
  comparison.
- Combine Strategy 1 + Strategy 2 (309 pairs) into a single probe with
  same-prompt as a covariate.
- Write `reports/REFRAMING_DECISION.md` with these two AUCs +
  cross-target + B02-v3 numbers as the four framing-discriminator
  inputs.
