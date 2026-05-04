# D09_pixart_sigma_smoke — cross-architecture activation collection on PixArt-Sigma

## Goal

Phase D-9 from `task_description_v2.md` §6 — "Cross-architecture
generalization to FLUX / SD3". Both FLUX.1-schnell and SD3-medium are
gated on this pod's HF token. PixArt-Sigma-XL-2-1024-MS is an open
DiT-style alternative (28 transformer blocks, hidden dim 1152) — used
as the cross-architecture target instead.

This is the smoke step: load PixArt, generate a small batch of I2P
and COCO prompts, hook 4 transformer blocks at structurally-analogous
depths to SDXL UNet's {down.2.1, mid.0, up.0.0, up.0.1}, capture
per-step per-block residual mean-pooled activations.

## Procedure

- Model: `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (open, gated=false).
- Pipeline: `diffusers.PixArtSigmaPipeline`, bfloat16.
- Prompts: 5 I2P-NSFW + 5 COCO captions.
- Generation: 20 inference steps, CFG 7.0, 512×512 (down from default
  1024 to fit VRAM alongside in-flight attacks).
- Hookpoints: transformer blocks 4, 9, 14, 18 (rough depths analogous
  to SDXL UNet's down.2.1 / mid / up.0.0 / up.0.1).
- Capture: forward hooks on each block; spatial-mean-pool the output to
  per-step (B, 1152) → stack to (T=20, 1152) per (sample, block).
- Wall: 369 s for 10 generations.
- Peak VRAM: 12.6 GB.

## Results

| metric | value |
|---|---|
| n_rendered | 10 |
| n_transformer_blocks_total | 28 |
| hookpoints used | [4, 9, 14, 18] |
| activation tensor shape per (sample, block) | (20, 1152) |
| elapsed (s) | 369 |
| peak VRAM (GB) | 12.6 |
| sec / image (PixArt 512×512, 20 steps) | 36.9 |

`outputs/D09_pixart_sigma_smoke/{images, activations, summary.json}`.

## Interpretation

Smoke confirms the cross-architecture activation-collection plumbing
works on a Transformer-DiT (vs the SDXL UNet plumbing in
`dsi/sae/hooks.py`). Activations are 1152-d (dense Transformer hidden
state) vs SDXL UNet's 1280-d residual stream — comparable scale but
different distributional characteristics (DiT has joint text+image
tokens in the same stream; SDXL UNet has separate cross-attn).

PixArt-Sigma is **NOT** identical to FLUX or SD3 architecturally — the
core distinction is full Transformer joint stream (PixArt) vs MM-DiT
double-stream (FLUX). The cross-architecture story will need to be
qualified ("open DiT cross-architecture transfer to PixArt-Sigma-XL")
rather than "FLUX cross-architecture".

## Next

- Scale to 500 I2P + 500 COCO renders for SAE training data.
- Train an SAE on each of the 4 PixArt hookpoints (TopK x16 expansion,
  k=64) using the cached (T, 1152) activations.
- Compute Procrustes-aligned cosine similarity between SDXL Surkov
  SAE decoder columns and PixArt SAE decoder columns at structurally
  analogous depths.
- Attempt `bcywinski/SAeUron` checkpoints on SD v1.4 as a separate
  cross-architecture leg (different backbone, same SDXL-like hooks).
- File HF access requests for FLUX.1-dev and SD3-medium so Phase D-9
  can include the originally-spec'd MM-DiT models as a follow-up.
