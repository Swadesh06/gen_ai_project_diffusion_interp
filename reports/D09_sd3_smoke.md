# D09_sd3_smoke — cross-architecture activation collection on SD3-medium

## Goal

Phase D-9 — cross-architecture transfer. SD3-medium is a Multi-modal
DiT (MM-DiT) Transformer with 24 double-stream blocks at hidden dim
1536. After receiving access to the gated repo
(`stabilityai/stable-diffusion-3-medium-diffusers`), this is the first
SD3 cross-architecture data collection.

## Procedure

- Model: `stabilityai/stable-diffusion-3-medium-diffusers`, fp16.
- Pipeline: `diffusers.StableDiffusion3Pipeline`.
- Prompts: 10 I2P-NSFW + 10 COCO captions.
- Generation: 20 inference steps, CFG 7.0, 512×512.
- Hookpoints: blocks 4, 9, 14, 18 (analogous depth to SDXL UNet's
  down.2.1 / mid.0 / up.0.0 / up.0.1).
- Wall: 554 s for 20 generations (≈ 28 s/image with 4 SAE-style
  hooks active).
- Peak VRAM: 15.1 GB.

## Results

| metric | value |
|---|---|
| n_rendered | 20 |
| n_transformer_blocks_total | 24 |
| hookpoints used | [4, 9, 14, 18] |
| activation tensor shape per (sample, block) | (20, 1536) |
| elapsed (s) | 554 |
| peak VRAM (GB) | 15.1 |
| sec / image (SD3 512×512, 20 steps) | 27.7 |

`outputs/D09_sd3_n20/{images, activations, summary.json}`.

## Interpretation

SD3-medium (MM-DiT, 24 blocks × 1536-d) is the **closer** match to
FLUX architecturally than PixArt-Sigma (DiT, 28 blocks × 1152-d). The
hidden dim 1536 vs SDXL UNet's 1280 is slightly larger but
comparable scale.

Cross-architecture transfer experiments will use SD3 as the primary
"different but comparable" target backbone. PixArt remains the open
DiT alternative.

Combined with the new FLUX.1-schnell access (also gated, just
granted), the D-9 experiment matrix becomes:

| backbone | architecture | hookpoints | status |
|---|---|---|---|
| SDXL Turbo | UNet | 4 Surkov SAEs | trained (Phase 1) |
| SDXL Base 4-step | UNet | same | inferred from Surkov SAEs |
| SD v1.4 | UNet | SAeUron `bcywinski/SAeUron_coco` | repro running |
| **SD3-medium** | **MM-DiT** | **4 hookpoints captured** | **smoke done** |
| **FLUX.1-schnell** | **MM-DiT** | (in flight, slow load on shared GPU) | loading |
| PixArt-Sigma | DiT | 4 hookpoints captured | smoke done |

## Next

- Train SAEs on SD3 activations (200 sample target via larger batch run).
- Train SAEs on FLUX activations once that loads.
- Compute feature alignment between SDXL Surkov SAE decoder columns
  and SD3 SAE decoder columns at the 4 analogous hookpoints
  (Procrustes-aligned cosine similarity per Phase D-9 spec).
- Test: does the Stage-1 ∩ Stage-2 nudity feature set from SDXL
  Turbo identify a corresponding feature set in SD3?
