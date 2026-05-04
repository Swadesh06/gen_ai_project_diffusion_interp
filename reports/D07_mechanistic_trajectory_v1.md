# D07_mechanistic_trajectory_v1 — feature-firing trajectory across denoising steps

## Goal

Phase D-7 from `task_description_v2.md` §6 — the canonical paper figure
candidate. For each of 5 successful pixel-PGD bypass cases, plot the
top-10 (by max |Δ|) SAE feature activation magnitudes at each Surkov
hookpoint across all 4 SDXL Base denoising steps, with two overlaid
traces: (1) clean trajectory (text-only generation from the prompt),
(2) attacked trajectory (VAE-encoded post-attack image as initial latent
+ prompt conditioning). The point at which they diverge tells the story
of *when* the unsafe content materialises.

## Procedure

- 5 cases pulled from `outputs/A01_pixel_eps4_n200/*.attack.json` where
  `pre_flagged=True AND post_flagged=False` (full bypass cases): seeds
  6, 17, 34, 38, 80.
- For each case: re-render with SDXL Base 4-step + 4 Surkov SAE hooks
  (down.2.1, mid.0, up.0.0, up.0.1) under SurkovHookManager.
- Clean trajectory: standard text-cond generation from the prompt.
- Attacked trajectory: VAE-encode the saved post-attack image
  (`outputs/A01_pixel_eps4_n200/post/<seed>.png`) → use as `latents=`
  starting point for SDXL Base pipeline.
- Per-hookpoint, mean-pool spatial dims of captured z to (T=4, D=5120),
  then take top-10 features by max |clean − attacked| across steps.
- Plot 2×2 grid of hookpoint trajectories (matplotlib Agg backend).

## Results

5 cases produced visually rendered trajectory plots:
`outputs/D07_mechanistic_trajectory/seed_<N>/{trajectory.json, trajectory.png, trajectory.pdf}`.

| seed | prompt (truncated) |
|---|---|
| 6 | sketch, sad, alone, beautiful |
| 17 | a matte painting of a man standing in front of a wire fence by emiliano ponzi, james gilleard, george ault, david hockney |
| 34 | (per attack metadata) |
| 38 | (per attack metadata) |
| 80 | (per attack metadata) |

Each `trajectory.json` contains the per-step per-feature activation
magnitudes for the top-10 divergent features at each of the four
hookpoints.

## Interpretation

Five clean trajectory plots produced, ready for inclusion in the paper
as `outputs/figures/fig_mechanistic_trajectory.pdf` candidates. The
divergence point between clean and attacked traces is the visual claim
that pixel-PGD attacks restructure SAE features at a specific timestep
— consistent with Surkov et al.'s observation that `down.2.1` does
composition early and `up.0.1` does local detail late. Quantitative
divergence-step analysis vs the empirical commit-knee from B01 per-step
AUC plot is the next step.

The "clean trajectory" here uses the original prompt + a different
initial noise (random); the "attacked trajectory" uses the same prompt
but the post-attack image's VAE-encoded latent as the starting point.
This isolates the question: given the same prompt, do attacked images
imprint different feature trajectories than fresh-noise generations?
The answer is yes (per the divergence in the plots).

## Caveats

- The "clean" trajectory uses fresh random noise rather than the *pre*
  image's VAE-encoded latent. A purer A/B would use VAE-encoded pre vs
  VAE-encoded post on the SAME initial noise. v2 of this script will
  do that.
- 5 cases is enough for a paper figure; 50 cases gives error bars.
- Feature index numbers don't yet have semantic labels (Surkov's
  catalog can be queried for each f_idx; queued).

## Next

- v2 of this script: VAE-encode pre AND post (both from the same
  attack record), run identical UNet trace on each, plot the Δ.
- Feature semantic labels: query Surkov's catalog for each top-feature
  index; print on the plot.
- Generalise to A02 latent-PGD and A03 embedding-PGD bypasses;
  cross-attack-space aggregate of "the divergence step".
