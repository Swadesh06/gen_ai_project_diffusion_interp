# C01_xtarget_v2_A01_vs_B01 — Item 1c-1 detector logits bug fix verification

## Goal

Phase 1 C01 reported `transferability_safety_to_detector = 0.000` with the
suspicious property that `detector_pre_logit == detector_post_logit` to
floating-point exactness for all 200 rows. The v1 evaluation called
`pipe_w.pipe(prompt=prompts, ...)` for both pre and post — same prompt,
same seed → same generation, same SAE trace, same detector logit.
The detector never actually saw the perturbed image's content.

Item 1c-1 fix: VAE-encode the actual pre and post images, then run a
single UNet forward pass on each latent at low noise (timestep 50, noise
strength 0.05) and capture SAE features through SurkovHookManager.

## Procedure

- Source: `outputs/A01_pixel_eps4_n200` (200 records of pixel-PGD
  attacks against `CompVis/stable-diffusion-safety-checker`).
- Detector: `/workspace/checkpoints/B01_em_linear_v1/best.pt` —
  20480-d (4-hookpoint × 5120) linear probe.
- Image-conditioned trace path:
  1. VAE-encode `pre/<seed>.png` and `post/<seed>.png` separately.
  2. Run `pipe.unet(latent, t=50, encoder_hidden_states=text_emb)` once
     under `SurkovHookManager(capture=True)` for each side.
  3. Spatial-mean-pool each captured z to (B, D), concat across the
     four hookpoints to (B, 20480).
- Detector logit = `head(features)`.

## Results

| metric | value |
|---|---|
| n_total | 200 |
| n_logit_identical_pre_post | **0** (bug fixed) |
| n_safety_pre_flagged | 17 |
| n_safety_post_flagged | 0 |
| n_detector_pre_flagged | 157 |
| n_detector_post_flagged | 158 |
| n_safety_bypass | 17 |
| n_detector_bypass | 3 |
| **n_bypass_both** | **0** |
| asr_safety_among_pre_flagged | 1.000 |
| **transferability_safety_to_detector** | **0.000** |
| noise_strength | 0.05 |
| timestep | 50 |
| wall (s) | 924 |
| peak GPU (GB) | ~6 |

`outputs/C01_xtarget_v2_A01_vs_B01/transferability.json` has the per-row data.

## Interpretation

The bug is real: all 200 pre/post pairs now produce *different* detector
logits (`n_logit_identical_pre_post = 0`). The image-conditioned trace
exposes the perturbed image content to the SAE in a way the v1 script
did not.

The original conclusion holds under the fix. Of 17 cases where the
pixel-PGD attack flips the safety_checker from "unsafe" → "safe",
**zero** also flip the SAE detector. The cross-target transferability
is 0.0 — a stronger result now that we can rule out the artefact.

The detector flags 157/200 pre images and 158/200 post images, both
much higher than the 17/200 safety_checker pre-flag rate. B01 is
heavily biased by the prompt-origin label leak from Phase 1 (it almost
always says "I2P prompt"). The meaningful re-test against B02-oracle-v3
(now built; 1544 samples, oracle labels) is the next step.

3 cases of `detector_bypass` (pre flagged, post not flagged) confirm
the detector reads the image content non-trivially.

## Next

- Re-run xtarget v2 against `B02_oracle_v3_*` checkpoints (training in
  flight). Expect detector flag rate to drop substantially (no longer
  prompt-origin biased).
- Re-run against `safety_saes_v2` detector (AUC 1.0 on in-distribution
  axbench) once that detector ckpt is dumped.
- Repeat for A02 latent and A03 embedding attacks (`outputs/A02_*`,
  `outputs/A03_*`) — the full 3×2 cross-target matrix needs all three.
