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
| C01_xtarget_A01_vs_B01 | 2026-05-04 | transferability=0.000 (17/17 bypass safety, 0/17 bypass detector) | keep | Item 4 pass; off-diagonal cell at 0% — safety_checker and SAE detector monitor different feature subspaces |
| D02_stage1n2_meanpatch_n100 | 2026-05-04 | 4/10 corrected (40%), |F_c|=69 | keep | Stage1∩Stage2 doubles correction vs Stage1-only (D01 17%); validates Stage-2 contribution |
| PHASE_1_FINAL | 2026-05-04 | all 4 contributions evidenced | keep | headline tables for Items 1-5; carry-overs for Phase C |
| D03_stage1n2_zeropatch_n100 | 2026-05-04 | 4/10 corrected (40%) tied with D02 mean | keep | mean ≈ zero on safety_checker; FID ablation needed |
| B02_em_oracle_v2_balanced | 2026-05-04 | va_auc 0.852, va_ap 0.356 (linear, balanced BCE) | keep | balancing brought AP up vs unbalanced 0.344 |
| D04_stage1n2_resamplepatch_n100 | 2026-05-04 | 4/10 corrected (40%) tied with D02/D03 | keep | resample-patch matches mean/zero on safety_checker; FID arbiter pending |
| D02_D03_D04_patch_ablation | 2026-05-04 | mean = zero = resample on safety; F_c quality dominates | keep | patch-kind ablation; FID measurements pending |
| B02_em_oracle_v2_mlp_balanced | 2026-05-04 | va_auc=0.891 va_ap=0.421 (MLP-256 + balanced BCE) | keep | best oracle-relabelled detector so far; +4pp over linear |
| FID_D02_pre_n100 | 2026-05-04 | FID=234.93 vs COCO val 5K (n=100) | partial | FID is expected high w/ Turbo 1-step + small sample; full FID series running |
| C06_hybrid_detector | 2026-05-04 | hybrid AUC=1.0 vs raw=0.999 vs sae=0.987 | keep | raw saturates in-distribution; hybrid edges sae by 1.34pp |
| FID_D02_post_n100 | 2026-05-04 | post=235.21 (Δpre=+0.28) | keep | mean-patch FID delta < 0.5 vs pre |
| FID_D03_post_n100 | 2026-05-04 | post=235.25 (Δpre=+0.32) | keep | zero-patch FID delta within 0.05 of mean |
| C03_safety_sae_v1 | 2026-05-04 | recon_mse 0.09-0.23 4 hookpoints | keep | TopK x8 k=64 on 1000 mixed I2P+COCO; first-hp probe AUC=1.0 ties raw |
| base_i2p_4step_n200 | 2026-05-04 | 200 imgs, 522s, 7.6GB peak | keep | SDXL Base 4-step CFG=7.5; safety scoring queued |
