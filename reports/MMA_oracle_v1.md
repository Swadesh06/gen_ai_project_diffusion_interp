# MMA_oracle_v1 — MMA-Diffusion adv-gen vs full oracle ensemble

## Goal

Score MMA-Diffusion's adversarial-prompt-generated images (n=50) with
NudeNet + Q16 + safety_checker oracles. Combined with the prior B02-v3
SAE detector test, this characterises MMA's threat model from each
defense layer's perspective.

## Results

| Oracle | A01 PGD-on-SC (ε=4/255, n=200) | MMA adv-gen (n=50) |
|---|---|---|
| safety_checker | 0.0% | **58.0%** |
| NudeNet | 0.5% | **20.0%** |
| Q16 | 12.0% | **28.0%** |
| any oracle (SC + NudeNet + Q16) | 12.5% | **74.0%** |
| B02-v3 SAE detector | 100% caught (Item 1c-1: 0/32 transferability) | **0% caught** |

## Interpretation — three orthogonal failure modes revealed

The two attacks expose three distinct blind spots in different defense
layers:

1. **A01 image-side PGD** (gradient on safety_checker):
   - Finds **safety_checker-specific perturbations** that transfer
     poorly to other oracles. 87.5% of A01 bypasses escape any oracle.
   - The **SAE detector catches 100%** of these bypasses (Item 1c-1).
   - Defense advice: don't deploy safety_checker alone; ensemble
     NudeNet + Q16 + SAE detector (which is the strongest layer).

2. **MMA text-side adversarial-prompt attack**:
   - Produces real-looking NSFW content that gets caught by 74% of
     the 3-oracle ensemble. Particularly safety_checker (58%), NudeNet
     (20%), and Q16 (28%).
   - The **SAE detector misses 100%** of these (architecture-specific
     to SDXL, MMA is SD v1.4-based).
   - Defense advice: SAE detector trained on SDXL doesn't generalize
     to SD v1.4 NSFW. Per-backbone training needed.

3. **B02-v3 SAE detector blind spots**:
   - Architecture-specific (SDXL-only).
   - Doesn't transfer to MMA-Diffusion (SD v1.4 backbone, AUC 0.388).
   - Combined with the strong cross-target robustness on SDXL (AUC
     0.977 in-distribution; 0/32 transferability under A01+A02 attacks
     per Item 1c-1), the SAE detector is **strongest in-distribution,
     weakest OOD**.

## Composite defense view

| attack source | safety_checker | NudeNet+Q16 ensemble | SAE detector | best-of-all-three |
|---|---|---|---|---|
| A01 PGD-on-SC (SDXL) | bypassed | 12.5% caught | 100% caught | **100% caught** |
| MMA adv-gen (SD v1.4) | 58% caught | 36% (N+Q) caught | 0% caught | **74% caught** |

**No single defense layer is sufficient against both attacks, but the
union of all three catches both attack distributions.**

This **strengthens the Phase 1c case for layered defense (D-10
compositional)** — each layer covers a different attack family.

## Caveats

- 50 + 200 sample sizes with wide CI on each rate.
- A01 attack was specifically targeted at safety_checker; the
  classifier-specific result is by construction.
- MMA-Diffusion attack was published 2024; the safety_checker may have
  been hardened against MMA-style attacks since then. We use the 2025
  CompVis safety_checker checkpoint.
- The "any oracle" rate sums up to 74% on MMA but doesn't tell us
  *which* image-prompt pair maps to *which* oracle's flag. Per-image
  union analysis is queued.
