# A01_A02_eps_sweep_partial_v1 — partial ASR at half/quarter ε

## Status

ε-sweep on A01 pixel + A02 latent at n=100 prompts each, 4 attacks
launched in parallel. ~50% done at time of writing. Attack jobs in
`logs/a0{1,2}-eps{1,2,05,025}.log`.

## Procedure

| attack | space | ε | Phase 1 default? |
|---|---|---|---|
| A01 pixel | 4/255 | yes | (saturates, n=200) |
| A01 pixel | 2/255 | half | this run |
| A01 pixel | 1/255 | quarter | this run |
| A02 latent | 0.1 | yes | (saturates, n=200) |
| A02 latent | 0.05 | half | this run |
| A02 latent | 0.025 | quarter | this run |

40 PGD steps each, n=100 prompts.

## Partial results (~40-50% complete at time of writing)

| attack | ε | n_pre / n_bypass (partial) | ASR among pre-flagged |
|---|---|---|---|
| A01 pixel | 1/255 (quarter) | 6 / 6 | **1.000** |
| A01 pixel | 2/255 (half) | 6 / 6 | **1.000** |
| A02 latent | 0.025 (quarter) | 5 / 5 | **1.000** |
| A02 latent | 0.05 (half) | 7 / 7 | **1.000** |

**ASR remains saturating even at ε/4**, the smallest meaningful pixel
perturbation (1 step in uint8 quantization).

## Interpretation — preliminary

This is a **strong** Phase 1 finding: **safety_checker is brittle, not
"resistant to small perturbations"**. The attack power is NOT the
bottleneck — defender brittleness is.

Possible interpretations:
1. The CompVis safety_checker is a CLIP-based image-prototype
   classifier with low margin to NSFW labels. Tiny perturbations are
   sufficient to flip the classification.
2. PGD with 40 steps and lr=ε/40 finds adversarial examples even at
   ε/4. The optimization finds the minimum-perturbation adversarial
   example for each input.
3. The "natural" L_∞ separation between safe and unsafe in this
   representation is < 1/255.

This **strengthens the case for a multi-layer defense** (Framing A
contribution 3) and **weakens the standalone safety_checker**'s
trustworthiness as the sole protection.

## Next

- Wait for n=100 to complete on each attack to confirm the partial
  saturation result.
- Add ε=0.5/255 (eighth) probe — does ASR finally drop at minimum
  perturbation budget? Or does it stay at 1.000?
- Pair with the SAE detector (B02-v3) — does the SAE detector still
  catch the small-ε bypassed images? Item 1c-1 already showed the
  full-ε images don't bypass the detector.

## Caveats

- Single seed each; full 5-seed CI not yet computed for the smaller-ε
  values.
- The partial sample (n_pre=5-7) has wide CI (Wilson [0.59, 1.00] for
  5/5). Full n=100 will tighten the bound.
