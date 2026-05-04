# C — Internal structure of F_c

## Goal
Probe the internal structure of the 69-feature F_c bundle (Stage 1 ∩ Stage 2)
across the four Surkov hookpoints. Asks: are the 69 features independent,
redundant, or low-rank?

## Setup
- F_c bundle: down.2.1=12, mid.0=0, up.0.0=32, up.0.1=25 → 69 total.
- Activations: SAE z's at the corresponding hookpoint indices on the
  `dataset_axbench_v1` pool (1000 samples, 500 NSFW + 500 benign).
- Compute: per-feature class-mean separation; pairwise correlation; SVD
  for effective rank.

## Headline

| metric | value |
|---|---|
| n features | 69 |
| pairwise \|corr\| off-diag mean | 0.030 |
| pairwise \|corr\| off-diag max  | 0.999 |
| effective rank (∑σ)² / ∑σ²    | 23.62 / 69 = 34% |

Interpretation:
- The 69 features are **near-uncorrelated on average** (mean |r| = 0.030).
  No global redundancy — each feature contributes some independent signal.
- One pair has |r| ≈ 1.0 — likely two near-duplicate features the SAE
  topk-hashed into separate slots. A pruning pass could drop one.
- Effective rank is **23.6 out of 69** — i.e. the F_c subspace has the
  variance of ~24 truly independent directions. The remaining ~45
  features carry a sparse / quasi-redundant slice of the same information.

## Implication
The F_c bundle is **compressible**. A safety SAE specifically trained to
allocate ~24 dimensions to the same concept-mass should match the current
F_c with a smaller intervention surface (fewer features patched per call →
lower latency + cleaner intervention). C-3 (in flight) tests exactly this.

## Next
- Apply k-means / structured pruning to F_c to drop the redundant features
  and re-run the intervention; expect identical safety correction rate at
  smaller |F_c|.
- Rebuild Stage 1 ∩ Stage 2 selection on the safety SAE features (when C-3
  finishes) — should yield a smaller F_c with higher per-feature causal
  score per appendix prediction.
- Per-feature attribution map (per-image which F_c features fire) →
  visual paper figure.

## Artefacts
- `outputs/F_c_analysis/fc_structure.json`
- `outputs/F_c_analysis/fc_pairwise_corr.npy` (69×69 matrix)
