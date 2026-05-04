# C09_transcoder_v2 — transcoder reconstruction error as detector signal

## Goal

Phase C-9 from `task_description_v1_appendix.md` §G C-9. Train a small
MLP transcoder (Dunefsky et al. 2024) per adjacent-block pair on the
benign half of `dataset_axbench_v1`. Hypothesis: NSFW images have
qualitatively different cross-block residual relationships → benign-only
transcoder reconstructs benign well and NSFW poorly → the per-sample
reconstruction error functions as an in-generation detector.

## Procedure

- Data: `outputs/dataset_axbench_v1` raw activations, 1000 samples,
  500 NSFW + 500 benign labels from prior session.
- Pairs: `down.2.1 → mid.0`, `mid.0 → up.0.0`, `up.0.0 → up.0.1`.
- Architecture: `Linear(d_src, 512) → ReLU → Linear(512, d_dst)`.
- Trained on benign only (80/20 within-benign train/val split, n=400/100).
- Optimizer: Adam lr=1e-3 wd=1e-4, MSE, 30 epochs.
- AUC computed on the held-out 100 NSFW + 100 benign per-sample MSE.
- Hardware: cuda; ~1 minute total wall on the new RTX PRO 6000 Blackwell
  (was queued as CPU on the old pod).

## Results

| pair | val_mse (benign) | AUC | AP |
|---|---|---|---|
| down.2.1 → mid.0  | 0.675 | 0.976 | 0.956 |
| mid.0 → up.0.0    | 1.059 | 0.963 | 0.947 |
| up.0.0 → up.0.1   | 1.464 | **0.991** | **0.991** |

`outputs/C09_transcoder/transcoder_results.json`. Five-seed CIs and
counterfactual-set evaluation queued under Items 1c-3 / 1c-6.

## Interpretation

Transcoder reconstruction error provides a per-sample NSFW detection
signal at AUC ≥ 0.96 across all three adjacent-block pairs, with the
best signal at the late-decoder pair `up.0.0 → up.0.1` (AUC 0.991).
This matches Surkov et al.'s observation that `up.0.1` carries
local-detail / colour features — NSFW content drives qualitatively
different local-detail feature relationships compared to benign,
detectable as a cross-block transcoder mismatch.

The signal is in-distribution (the dataset already saturates raw
linear probes at AUC 1.000 in C-2). Whether it survives the
counterfactual benchmark (Item 1c-0) and white-box attack (Item 1c-9)
is the meaningful test queued.

## Next

- Re-evaluate on counterfactual-pair held-out cells (Item 1c-0).
- Combine into hybrid `[raw ‖ sae ‖ transcoder]` detector input
  (Phase C-6 extension).
- Mechanistic plot: which features drive the up.0.0 → up.0.1
  reconstruction failure on NSFW (D-1 / D-7 candidate).
