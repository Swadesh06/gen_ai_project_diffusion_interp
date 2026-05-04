# reports/INDEX.md

Chronological log of every experiment. Each entry: `<exp_id> | date | best metric | status (keep/discard/crash) | one-line summary`.

| exp_id | date | metric | status | summary |
|---|---|---|---|---|
| phase1a_bootstrap | 2026-05-04 | 24/24 verify_assets rows green | keep | CPU scaffold; 66/66 tests pass; SAE checkpoints (Surkov 4 hookpoints, SAeUron 2 hookpoints) on disk; SDXL Turbo + SDXL Base + SD v1.5 + CompVis safety checker + CLIP ViT-L/14 + NudeNet + Q16 + LPIPS + DreamSim downloaded; SAEmnesia queued for reproduce-from-scratch (no public release as of 2026-05-04); UnlearnCanvas (81 GB, 153 parquet shards, 38+ styles) + I2P (4703) / I2P-adv (1104) / COCO val (5000 images / 25014 captions) / MMA-Diffusion text / UnlearnDiffAtk / Ring-A-Bell / Q16 repos cloned; LAION-COCO gated upstream → loader falls back to COCO captions (25014 prompts); env packed to /workspace/env-archives/dsi.tar.gz (672 MB, CPU torch wheel — GPU pod must reinstall + repack); GitHub repo not yet created — see PLAN.md "Open ops items for Phase 1b first hour". |
| S00_smoke_sdxl_sae | 2026-05-04 | peak_vram=7.86GB, all 4 hooks fired k=10 | keep | SDXL Turbo + 4 Surkov SAEs forward, residual hook protocol works on Blackwell sm_120 |
| S03_smoke_pgd | 2026-05-04 | pixel 8.2 / emb 7.2 / lat 13.5 GB; signs correct | keep | PGD machinery wired in 3 spaces; sign bug fixed; SDXL fp16-fix VAE swapped in |
| A01_pixel_eps4_n200    | 2026-05-04 | ASR_pre=1.000 (17/17), 9.4GB peak | keep | pixel PGD eps=4/255 saturates safety_checker; pre_flagged=8.5% |
| A03_emb_eps0.5_n200    | 2026-05-04 | ASR_pre=1.000 (15/15), 11.4GB peak | keep | embedding PGD eps=0.5 saturates safety_checker (post_safe_logit 0.937) |
| cross_space_A01_A03    | 2026-05-04 | Jaccard down.2.1=0.613, up.0.1=0.724 | keep | Contribution 1 pass criterion ≥60% met on 2/4 Surkov hookpoints |
| B01_em_detector_v1 | 2026-05-04 | val_auc=1.000 (linear, all-blocks) | keep-with-caveat | trivially passes Item 3 ≥0.95 but labels are by prompt-origin, not oracle judgement; rebuild B02_em_oracle_v1 next |
| A02_latent_eps0.1_n200 | 2026-05-04 | ASR_pre=1.000 (15/15), 13.7GB peak, 1728s | keep | latent-PGD eps=0.1 saturates safety_checker; matches A01/A03 bypass count |
| contribution1_final | 2026-05-04 | 3/3 cross-space ≥0.60 on down.2.1+up.0.1; ASR_pre=1.0 all spaces | keep | full Contribution 1 result; 3x3 SAE-feature overlap matrix |
| D01_stage1only_meanpatch_n50 | 2026-05-04 | 1/6 corrected (17%), no FID-side fallout | partial-keep | pipeline wired; Stage-1-only F_c too noisy → need Stage-1∩Stage-2 |
| stage2_up01 (smoke) | 2026-05-04 | top |Δ| = 0.014 on 32 features | partial-keep | causal scoring works; small Δ at up.0.1 with λ=250, Q16 oracle |
