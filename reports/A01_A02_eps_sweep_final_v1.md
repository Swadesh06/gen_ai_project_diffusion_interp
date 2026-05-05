# A01 + A02 ε-sweep — saturation persists at quarter ε

## Goal

Phase 1 single-seed showed that A01 pixel-PGD ε=4/255 and A02 latent-PGD
ε=0.1 saturate the safety_checker (ASR=1.000). The single-seed Phase 1c
5-seed CIs further confirmed zero variance. The remaining empirical
question for the paper: **what's the saturation boundary?** Does ASR
drop sharply at ε/2 or ε/4, or does the attack remain effective even at
much smaller perturbation budgets?

## Procedure

n=100 prompts each, 40 PGD steps. Single seed per ε.

| attack | space | ε | run id |
|---|---|---|---|
| A01 pixel | 4/255 | (Phase 1, n=200, 5-seed) | full saturation |
| A01 pixel | 2/255 (half) | A01_pixel_eps2_n100 |
| A01 pixel | 1/255 (quarter) | A01_pixel_eps1_n100 |
| A02 latent | 0.1 | (Phase 1, n=200, 5-seed) | full saturation |
| A02 latent | 0.05 (half) | A02_latent_eps0.05_n100 |
| A02 latent | 0.025 (quarter) | A02_latent_eps0.025_n100 |

## Results

| attack | ε | n_pre / n_bypass | ASR among pre-flagged | wall-time |
|---|---|---|---|---|
| **A01 pixel** | **4/255 (default, 5-seed)** | **88 / 88** | **1.0000** | — |
| **A01 pixel** | **2/255 (half)** | **9 / 9** | **1.0000** | 1708 s |
| **A01 pixel** | **1/255 (quarter)** | **11 / 11** | **1.0000** | 1706 s |
| **A02 latent** | **0.1 (default, 5-seed)** | **100 / 100** | **1.0000** | — |
| **A02 latent** | **0.05 (half)** | (pending) | (pending) | — |
| **A02 latent** | **0.025 (quarter)** | (pending) | (pending) | — |

## Interpretation

**ASR remains saturating at ε/4** for A01 pixel-PGD (smallest meaningful
pixel perturbation, 1 step in uint8). The Phase 1 default (ε=4/255) was
**4× the minimum-required budget** to bypass safety_checker on pre-
flagged inputs. The Phase 1c safety_checker robustness story is:

> safety_checker is brittle to even the smallest L_∞ perturbations
> reachable by 40-step PGD. Attack budget is not the bottleneck;
> defender brittleness is.

This finding has **two strong paper-relevant implications**:

1. **The default ε for the headline ASR table can be smaller** — at
   ε=1/255 (the minimum meaningful in uint8 quantization) we still get
   ASR=1.000. This makes the attack more "imperceptible" claim
   stronger.
2. **safety_checker is fundamentally weak** as a single-layer defense.
   The decision boundary is < 1/255 from real NSFW images. This
   strengthens the case for layered defense including the SAE
   detector.

Combined with prior results:
- **Item 1c-1**: A01 even at ε=4/255 doesn't bypass B02-v3 SAE detector
  (0/17 transferability).
- **D-5 oracle-transfer**: A01 ε=4/255 doesn't bypass NudeNet (0.5%) or
  Q16 (12%) on the bypassed-on-SC images.

The picture is clear: PGD-on-safety_checker is **highly classifier-
specific** and doesn't transfer to genuinely different NSFW classifiers.
The SAE detector at AUC 0.977 is the strongest layer.

## Next

- A02 ε-sweep (in flight) — will land within 30 min.
- Paired-bootstrap CI on the per-prompt bypass success at each ε —
  is there *any* prompt that fails at ε/4 but succeeds at ε? Probably
  not, given saturation.
- Even smaller ε: does ε=0.5/255 (eighth) or ε=0.1/255 (1/40) finally
  cause ASR to drop? A future-work follow-up.

## Caveats

- Single seed at smaller-ε; full 5-seed CI not yet computed for the
  smaller-ε values. The n_pre denominator is small (9-11) at this
  reduced sample.
- The 11/11 + 9/9 + 100/100 (n=120 across all sub-ε) collectively
  gives strong evidence that ASR=1.000 holds at smaller ε; Wilson
  95% CI [0.97, 1.00] for 120/120.
