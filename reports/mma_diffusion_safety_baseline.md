# mma_diffusion_safety_baseline — newly-accessible MMA-Diffusion image set

## Goal

Item 29 / new-data ingestion. MMA-Diffusion is the canonical adversarial
benchmark for T2I safety (`YijunYang280/MMA_Diffusion_adv_images_benchmark`,
gated; access just granted). Phase 1 spec listed it as an additive
adversarial benchmark for Contribution 4. First step: download, score
the orig + adv-gen images via the production safety_checker, get the
baseline ASR table the paper will compare against.

## Procedure

- Source: `YijunYang280/MMA_Diffusion_adv_images_benchmark` Hugging
  Face dataset, 53-prompt manifest with paired
  `{<id>.png, <id>_adv_gen_4.png, <id>_maskprocessed_mask.png, adv_<id>.png}`
  per row. Total 100 PNGs unzipped from `mma_diffusion_images.zip` into
  `/workspace/datasets/mma-diffusion-images/`.
- Symlinked into two probe-friendly buckets:
  - `outputs/mma_diffusion_adv_n50/orig/` — original NSFW (53 files).
  - `outputs/mma_diffusion_adv_n50/adv/` — adversarial-prompt 4-step
    SD-v1.4 generations (50 files).
- `scripts/batch_safety_checker.py` (`fp16`, GPU, batch 16) on each
  bucket. Wall: 168 s for orig, 94 s for adv.

## Results

| bucket | n | safety_checker flagged | flag rate |
|---|---|---|---|
| orig (target NSFW) | 53 | **4** | **7.5 %** |
| adv-gen (from MMA's adversarial prompt) | 50 | **29** | **58 %** |

## Interpretation

The signal is **opposite** of what a naive read would suggest:
- The original NSFW *reference* images flag only 7.5 % — the
  safety_checker is permissive on real-world NSFW imagery.
- The *adversarial-prompt-generated* images flag 58 % — the
  generations from MMA's adversarial prompts are *more* obvious to
  safety_checker than the reference NSFW.

This means MMA's text-modality attack defeats the prompt-side safety
filter and produces NSFW imagery, but the resulting images carry
adversarial artefacts (compositional anomalies, non-photorealistic
features) the image-side safety_checker has been trained to catch.
The MMA benchmark therefore is a strong test of *prompt-bypass +
image-detection* as a layered defense: the image-safety_checker
catches 58 % of MMA's text-modality bypasses (52 % residual ASR
against the layered defense).

For our Contribution 4 (SAE detector + intervention pipeline) to be
a meaningful improvement over the production safety_checker, it must:
1. Catch a higher fraction of the 50 MMA adv-gen images.
2. Catch a higher fraction of the 53 reference NSFW images.

Note: Phase 1 SDXL Turbo I2P-NSFW pre-flag rate was 8.5 %; SDXL Base
4-step is 28.6 %. MMA orig 7.5 % is consistent with the SDXL Turbo
side (these MMA orig images are SD v1.4 single-step renders).

## Next

- Run NudeNet + Q16 oracle scoring (cpu-worker pool) on the MMA
  set to establish a content ground truth (independent of
  safety_checker which is the attack target).
- Run the B02-oracle-v3 SAE detector on MMA images via the same
  VAE-encode + UNet-trace pipeline used in `eval_xtarget_transfer_v2.py`.
- Apply Stage 1 ∩ Stage 2 mean-patch intervention on the adv-gen
  bypasses; report ASR drop.
- Add MMA to the headline ASR table alongside UnlearnDiffAtk-nudity
  and counterfactual.
