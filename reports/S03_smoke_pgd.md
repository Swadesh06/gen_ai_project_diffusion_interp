# S03 — pixel / latent / embedding PGD smoke

## Goal
Verify the three Contribution-1 attack inner loops are wired correctly — gradient flows in the right direction, VRAM stays under the 27 GB cap, no fp16 overflow.

## Procedure
- `scripts/smoke_pgd.py` (3 PGD iters × 2 prompts each).
- New module `dsi.attacks.safety_target.SafetyTarget` exposes the CompVis safety checker as three differentiable heads:
  - `pixel_to_logits`: image (B,3,H,W) in [0,1] → 2-vec [unsafe, safe] via CLIP encoder + concept_embeds + special_care_embeds.
  - `embedding_to_logits`: 768-d normalised CLIP image embedding → 2-vec.
  - `vae_latent_to_logits`: SDXL VAE latent → decode → pixel_to_logits.
- Loaded SDXL Turbo (UNet fp16) + swapped the VAE for `madebyollin/sdxl-vae-fp16-fix` to avoid the well-known SDXL VAE fp16 overflow.
- Targeted attack: `y=safe`, descent on `CE(logits, safe)`. The previous `targeted` branch had a sign bug that drove the safe logit *down*; fixed in `dsi/attacks/{pixel,latent,embedding}.py`.

## Results
3 PGD iters move the safe logit upward in all three spaces:

| space | peak VRAM (GB) | iter time (s) | Δ safe-logit prompt 1 | Δ safe-logit prompt 2 |
|---|---|---|---|---|
| pixel     | 8.15  | 0.17 | +0.019 | +0.057 |
| embedding | 7.18  | 0.004 | +0.262 | +0.308 |
| latent    | 13.53 | 0.43 | +0.025 | +0.037 |

Embedding moves the most per iter (no decoding bottleneck — direct gradient on the CLIP feature). Latent uses the most VRAM (gradient flows through the 51 M-param VAE decoder). Pixel sits in the middle.

## Artefacts
- `outputs/S03_smoke_pgd/stats.json`
- `outputs/S03_smoke_pgd/pixel_post_{0,1}.png` (post-attack pixel-PGD images)

## Interpretation
Three attack-space wiring is correct: signs are consistent (positive Δ safe), VRAM well under the 27 GB cap (max 13.5 GB on latent), no NaN. Embedding is the cheapest and most permissive (paper: attack ceiling). Latent's 13.5 GB rules out co-locating with pixel-PGD (~14 GB at scale) but easily fits alongside embedding-PGD (3–7 GB) or detector training (~4 GB) — exactly the co-scheduling plan in PLAN.md.

The 3-iter Δ on a benign seed image is small (single-digit-percent shift in safe logit). This is expected: real bypasses need 40 PGD steps + larger ε, on actually-flagged seeds. Item 2.1 will run that at scale.

## Next
S01 — SAeUron repro on one UnlearnCanvas style; S04 — detector-training smoke (synthetic activations); then Item 1.5 clean baseline.
