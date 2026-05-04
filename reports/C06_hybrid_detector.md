# C06 — Hybrid raw||SAE detector (Phase C-6)

## Goal
Phase C-6 from the appendix: combine the SAE-activation detector with the
raw-residual signal (the same signal IGD uses) to test whether the two views
carry complementary information.

## Procedure
- Dataset: `outputs/dataset_axbench_v1` (1000 samples, 500 NSFW SDXL Turbo
  I2P + 500 benign SDXL Turbo COCO; same seeds across the two pools).
- For each Surkov hookpoint we have:
  - `X_raw_<hp>.npy` (1000, 1280): mean-pooled raw UNet residual diff.
  - `X_sae_<hp>.npy` (1000, 5120): mean-pooled Surkov SAE z (k=64 / D=5120).
- All four hookpoints concatenated: raw → 5120-d, sae → 20480-d, hybrid → 25600-d.
- 80/20 train/val split, MLP-256 head, BCE with auto pos-weight, 30 epochs.

## Results

| variant | val AUC | val AP |
|---|---|---|
| raw_only      | 0.9995 | 0.9995 |
| sae_only      | 0.9866 | 0.9887 |
| **hybrid (raw‖sae)** | **1.0000** | **1.0000** |

- hybrid − raw = +0.0005 (+0.05 pp) — within noise.
- hybrid − sae = +0.0134 (+1.34 pp) — meaningful.
- Pass criterion (hybrid > max(raw, sae) + 1pp): **NO** on the raw-vs-hybrid
  side. The raw signal already saturates this in-distribution NSFW-vs-benign
  task, so concatenation has nowhere to climb.

## Interpretation
On *in-distribution* NSFW-vs-benign labels, raw mean-pooled residuals are a
near-perfect probe target (4 hookpoints × 1280 dims is enough). The Surkov
SAE drops a fraction of a pp of separability vs raw because TopK/k=64
discards bulk variance, but it remains > 0.98. Hybrid saturates.

Where the C-6 hypothesis still has bite is *adversarial* generalization:
under PGD-attacked inputs the raw signal is what the attack drives down, so
SAE-side activations should be the more robust feature — that test goes via
the bypass-image evaluation (re-running detector on attack-bypass outputs;
queued).

## Next
- Re-evaluate raw-only / sae-only / hybrid on the A01 pixel-PGD bypass set
  (ε=4/255, n=200) to test adversarial robustness.
- Compute the per-feature SAE attribution lift (which SAE features survive
  attacks vs which raw channels survive) — reframe the hybrid as a per-
  hookpoint mask rather than a flat concat.

## Artefacts
- `outputs/dataset_axbench_v1/C06_hybrid_results.json`
- `logs/C6_hybrid.log`
