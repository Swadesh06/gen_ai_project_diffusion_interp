# C03_safety_sae_v2 — expansion 16 + 32 sweep, all four Surkov hookpoints

## Goal

Per `task_description_v2.md` §3 Item 1c-10: scale C-3 safety-trained
SAE training from the v1 expansion=8 baseline (`outputs/safety_saes_v1/`)
to expansion 16 and 32 across all four Surkov hookpoints. The 96 GB pod
makes both feasible to train concurrently. L0 sparsity sweep is queued
as a follow-up at this resolution.

## Procedure

- Data: `outputs/raw_i2p_500/raw` (NSFW) + `outputs/raw_coco_500/raw`
  (benign), 1000 mean-pooled hookpoint vectors total.
- Architecture: `dsi.sae.load.SurkovTopKSAE` per hookpoint, k=64,
  expansion ∈ {16, 32} → d_hidden ∈ {20480, 40960}.
- Optimizer: AdamW lr=3e-4 wd=0, MSE recon, 30 epochs, batch 128.
- 8 jobs launched in parallel under tmux `sae-<hp>-x{16,32}`; each
  used ~1 GB GPU peak. Wall-clock 30-60 s each.

## Results

| hookpoint | expansion | d_hidden | recon_mse | active_frac |
|---|---|---|---|---|
| down.2.1 | 16 | 20480 | 0.290 | 0.0031 |
| down.2.1 | 32 | 40960 | 0.363 | 0.0016 |
| mid.0    | 16 | 20480 | **0.119** | 0.0031 |
| mid.0    | 32 | 40960 | 0.152 | 0.0016 |
| up.0.0   | 16 | 20480 | 0.188 | 0.0031 |
| up.0.0   | 32 | 40960 | 0.237 | 0.0016 |
| up.0.1   | 16 | 20480 | 0.299 | 0.0031 |
| up.0.1   | 32 | 40960 | 0.367 | 0.0016 |

Checkpoints under `outputs/safety_saes_v2/safety_sae_<hp>_x{16,32}_k64/`.

## Interpretation

At fixed k=64, raising expansion from 16 → 32 worsens reconstruction
across every hookpoint by ≈ 25%. The dictionary has more atoms but the
same number are active per sample → effective sparsity drops, less
information captured per sample. The L0 budget is the primary lever,
not expansion. Mid.0 has the best reconstruction (lowest MSE 0.119 at
×16) and is the candidate for the most semantically-clean concept
features.

## Next

- L0 sweep at fixed expansion 16: k ∈ {32, 64, 128, 256} per hookpoint
  (16 jobs, all parallelizable on the new GPU).
- Stage-1 / Stage-2 selection on these new SAEs to test whether
  Stage-2-survivor causal score beats v1 expansion=8 set by ≥ 1.5×.
- Detector probe AUCs on raw / surkov / safety-v1 / safety-v2 to see
  if expansion 16 closes the 1.21pp gap that v1 expansion=8 left
  vs raw (per `reports/C03_safety_sae_v1.md`).
