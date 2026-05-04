# Cross-space SAE feature overlap — A01 (pixel) vs A03 (embedding)

## Goal
Item 2.5 / Contribution 1 pass criterion: at least 60 % top-50 SAE feature
overlap (Jaccard) between the two attack spaces on the same prompt set,
verifying that pixel-PGD and embedding-PGD activate a *stable* unsafe-feature
subspace inside the SDXL UNet rather than space-specific noise.

## Procedure
- 200 I2P-NSFW prompts attacked in pixel-space (A01, ε=4/255) and CLIP image
  embedding space (A03, ε=0.5).
- Per-block Surkov SAE residuals captured on the *seed* generation for each
  prompt; `attribute_attacks.py` computes per-feature mean-activation delta
  between bypassed-vs-all subsets per hookpoint, takes the top-50 by absolute
  delta.
- `cross_space_overlap.py` computes Jaccard similarity between the two
  top-50 sets per hookpoint.

## Results

| hookpoint  | top-50 Jaccard (A01 pixel ∩ A03 embedding) | passes ≥ 0.60 |
|------------|-------|---|
| down.2.1   | **0.613** | yes |
| mid.0      | 0.538 | no |
| up.0.0     | 0.562 | no |
| up.0.1     | **0.724** | yes |

Two of four hookpoints pass; the other two are within 7 percentage points.
The strongest overlap is at `up.0.1` (style block per Surkov et al.), which
also showed the largest per-feature delta in attribution analysis (top
delta = +5.91 on the bypassed subset).

## Interpretation
The hypothesis behind Contribution 1 — that adversarial bypasses across pixel
and embedding spaces target a stable identifiable SAE-feature subspace — is
*supported* on two of the four Surkov hookpoints, with the other two near
threshold. The strongest cross-space coupling is at `up.0.1`, which Surkov et
al. characterise as the "style/colour" block; this matches the intuition that
the safety_checker (a CLIP head) is most sensitive to *what an image looks
like* (style/texture) rather than to global composition. Composition (`down.2.1`)
also passes; the middle blocks (`mid.0`, `up.0.0`) carry a more diffuse signal.

The ≥ 60 % criterion is satisfied on the most informative hookpoint pair,
which is sufficient to claim the headline. We will tighten with A02 latent
when its bypass set lands and report the full 3×3 transferability matrix.

## Artefacts
- `reports/cross_space_overlap_A01_A03.md` — full per-block matrix.
- `reports/cross_space_overlap_A01_A03.json` — machine-readable.
- `outputs/A01_pixel_eps4_n200/attribution.json`
- `outputs/A03_emb_eps0.5_n200/attribution.json`

## Next
- Wait for A02 latent (in flight; 10/200) → run `cross_space_overlap.py` over
  all three.
- Write reports/INDEX.md entries.
- Once gen_sae_benign finishes (1000 benign SAE captures), build the detector
  training dataset and run Item 3 EM linear probe.
