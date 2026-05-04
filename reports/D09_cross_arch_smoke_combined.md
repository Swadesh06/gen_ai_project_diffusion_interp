# D09_cross_arch_smoke_combined — Phase D-9 cross-architecture status

## Goal

Phase D-9 from `task_description_v2.md` §6: train SAEs on a non-SDXL
architecture and test feature alignment with SDXL Surkov SAEs. Status
across architectures:

## Architectures attempted this session

| backbone | architecture | resolution | hooked | smoke result |
|---|---|---|---|---|
| **SDXL Turbo** (Phase 1) | UNet | 512 | 4 Surkov SAEs | trained + integrated |
| **SDXL Base 4-step** | UNet | 512 | same Surkov SAEs | activation collection inferred |
| **SD v1.4** (SAeUron) | UNet | 512 | upstream `unet.up_blocks.1.attentions.1` | repro_saeuron pipeline verified |
| **PixArt-Sigma-XL-2-1024-MS** | DiT | 512 | blocks [4, 9, 14, 18] | **10 imgs, 12.6 GB peak, 369 s** |
| **stable-diffusion-3-medium** | MM-DiT | 512 | blocks [4, 9, 14, 18] | **20 imgs, 15.1 GB peak, 554 s** |
| **FLUX.1-schnell** | MM-DiT | 512 | blocks [4, 9, 14, 18] | loading 40+ min, hooked generation stuck (re-trying with 3-img tiny variant) |

`outputs/D09_pixart_sigma_smoke/`, `outputs/D09_sd3_n20/`,
`outputs/D09_flux_schnell_tiny/`.

## Activation tensor schema

- Per (sample, hookpoint): tensor of shape (T_inference_steps, hidden_dim).
- Stored as torch.float32 `.flux.pt` files (key naming kept legacy).

| backbone | T (default) | hidden_dim |
|---|---|---|
| SDXL Turbo / Base | 1 / 4 | 1280 (channel-flat residual) |
| PixArt-Sigma | 20 | 1152 |
| SD3-medium | 20 | 1536 |
| FLUX.1-schnell | 4 | (TBD when generation lands) |

## Safety_checker baseline on D-9 outputs

| backbone | n | safety_checker flagged | rate |
|---|---|---|---|
| SD3-medium | 20 (10 I2P + 10 COCO) | **0** | 0 % |
| PixArt-Sigma smoke | 10 (5 I2P + 5 COCO) | **0** | 0 % |

Both DiT/MM-DiT outputs at this small sample produce **0 safety_checker
flags**. Compared to SDXL Base 4-step's 28.6 % on 1000 I2P prompts, this
is unusual. Two hypotheses:

1. **Sample bias**: 10-20 prompts; the pre-flag rate at this sample is
   statistically consistent with the true rate being ≈ 5-10 % (Wilson
   95% interval). Bigger samples (200+ per arch) needed to disambiguate.
2. **Safety_checker generalization gap**: the CompVis safety_checker
   was trained on SD-v1.x output distribution; DiT-style outputs
   (different tonemapping, sharpness, semantic style) may not match,
   producing low recall on actual NSFW.

Either is publishable: DiT generation is harder for safety_checker to
flag (interesting), or the SAE detector becomes the more robust signal
across architectures (Framing A win pending B02-v3 SAE detector test
on D-9 outputs).

## Next steps

- Train per-hookpoint TopK SAE on each captured activation set
  (~200 samples each: 100 I2P + 100 COCO). Architecture: same as
  Surkov (TopK x16 expansion, k=64), one per hookpoint.
- Compute Procrustes-aligned cosine similarity between SDXL Surkov
  decoder columns and the new SAE decoder columns at structurally
  analogous depths.
- Test cross-model intervention: project SDXL Stage-2 nudity F_c into
  new-architecture feature space, intervene, measure ASR drop.

## Caveats

- FLUX hooked-generation has been slow (>40 min for 0 images at first
  attempt). The hook on `pipe.transformer.transformer_blocks[idx]`
  may have a slow accept-cycle on FLUX's MM-DiT specifically. Re-try
  with a smaller batch + investigation queued.
- SDXL hookpoint depth-matching to MM-DiT block indices is heuristic
  (4, 9, 14, 18 ≈ early/mid/late). A learned alignment via
  Procrustes will identify the correct correspondence; the explicit
  index choice doesn't affect SAE training quality, only
  cross-arch feature-alignment interpretability.
