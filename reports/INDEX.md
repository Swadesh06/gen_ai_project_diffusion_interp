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
| C09_transcoder_v2 | 2026-05-04 | up.0.0→up.0.1 AUC=0.991 (3 hookpoint pairs) | keep | C-9 redux on new GPU; transcoder reconstruction error as detector signal, AUC 0.96-0.99 |
| C03_safety_sae_v2_expansion16_32 | 2026-05-04 | 8 SAEs trained × 4 hookpoints × {16,32} expansion | keep | mid.0 best recon=0.119 at x16k64; L0 sweep follow-up shows k=32 even better at 0.104 |
| cf_strategy3a_gemini_v1 | 2026-05-04 | 400 rows, 0 refusals on gemini-3.1-flash-lite-preview | keep | Item 1c-0 Strategy 3 Path A; cheapest model handles all unsafe cells, 1200 paraphrases |
| C01_xtarget_v2_A01_vs_B01 | 2026-05-04 | 0/17 detector bypass; 0 identical logits | keep | Item 1c-1 fix; image-conditioned UNet trace fixes the bit-identical bug, original conclusion holds |
| C03_safety_sae_v2_detector | 2026-05-04 | concat MLP AUC=1.000 (vs raw 1.000, surkov 0.985) | keep | safety SAE v2 closes the v1 1.21 pp gap; all L0 configs tied at 1.000 |
| B02_oracle_v3_detector | 2026-05-04 | linear AUC=0.976, MLP AUC=0.977 (1544 samples) | keep | Item 1c-3 close: 5x more NSFW samples (201 vs 41); 12pp lift over v2 |
| C01_xtarget_v2_A02_vs_B01 | 2026-05-04 | 0/15 detector bypass; transferability=0.000 | keep | latent-PGD vs B01 with image-conditioned trace; matches A01 v2 result |
| C01_xtarget_v2_A01_vs_B02v3 | 2026-05-04 | 0/17 detector bypass; transferability=0.000 | keep | A01 vs oracle-relabelled B02-v3 detector; meaningful version of cross-target |
| D07_mechanistic_trajectory_v1 | 2026-05-04 | 5 bypass cases plotted | keep | per-step per-feature SAE trajectory clean vs attacked, paper figure candidate |
| D02_learned_projection_v1 | 2026-05-04 | per-hookpoint Pi trained, raw + sae | keep | Pi(z_benign)≈z_benign, Pi(z_unsafe)→mu_benign; drop-in for mean-patch |
| D01_causal_feature_graph_v1 | 2026-05-04 | 98 edges, 18 roots, 18 sinks | keep | Marks-style attribution-by-correlation graph at top-20 Stage-1 features |
| base_i2p_4step_n1000 | 2026-05-04 | 286/1000 safety_checker flag (28.6%) | keep | Item 1c-7 SDXL Base 4-step rerun, gate ≥25% met, 3.4× SDXL Turbo's 8.5% |
| udatk_safety_scores | 2026-05-04 | nudity 53/142 (37.3%), violence 44/200 (22.0%) | keep | Item 1c-4 UnlearnDiffAtk render + safety_checker baseline |
| cf_strategy2_seed_pairs | 2026-05-04 | 246 validated pairs from 100 prompts × 8 seeds | keep | Item 1c-0 Strategy 2 done, gate ≥200 met by margin |
| cf_probe_strategy1_v1 | 2026-05-04 | per-cluster AUC 0.49-0.56, in-distribution 0.275 | keep | Strategy 1 framing-discriminator: SAE features fail counterfactual prompt-edit task |
| cf_probe_strategy2_v1 | 2026-05-04 | in-distribution AUC 0.9436, AP 0.8840 | keep | Strategy 2 framing-discriminator: SAE features SUCCEED on same-prompt seed-pair task |
| C01_xtarget_v2_vs_B02v3_full | 2026-05-04 | A01+A02 32/32 safety bypass, 0/32 detector bypass | keep | combined xtarget result vs oracle-relabelled B02-v3, transferability=0.000 |
| A03_5seed_ci | 2026-05-04 | 5/5 seeds at ASR=1.000 (102/102 pre_flagged bypass) | keep | Item 1c-6 A03 embedding-PGD 5-seed CI complete with 0 variance |
| C01_square_5seed_ci | 2026-05-04 | mean ASR=0.954 ± 0.029 across 5 seeds (211/221) | keep | Item 1c-6 black-box Square Attack 5-seed CI complete, 95% CI [0.93, 0.97] |
| D09_pixart_sigma_smoke | 2026-05-04 | 10 imgs, 4 hookpoints, 12.6 GB | keep | Phase D-9 cross-arch DiT activation collection plumbing works |
| D09_sd3_smoke | 2026-05-04 | 20 imgs, 24 MM-DiT blocks, 15.1 GB | keep | Phase D-9 SD3-medium MM-DiT (closer to FLUX); 4 hookpoints captured |
| repro_saeuron_nudity_n30_smoke | 2026-05-04 | 33% baseline -> 43% with-saeuron (wrong feature_idx 11627=cat) | partial-keep | Item 1c-5 SAeUron pipeline integration verified; needs correct nudity feature_idx |
| mma_diffusion_safety_baseline | 2026-05-04 | orig 7.5%, adv-gen 58% safety_checker flag | keep | newly-accessible MMA-Diffusion image set; baseline ASR table |
| REFRAMING_DECISION | 2026-05-04 | Framing A canonical (mixed evidence) | keep | v2 §7 framing-decision moment; 4 discriminator inputs banked, paper structure committed to original four contributions |
| cf_probe_strategy2_sae_v1 | 2026-05-04 | in-distribution AUC 0.9412 (SAE) vs 0.9436 (raw) | keep | SAE features tie raw on Strategy 2 counterfactual, Δ = -0.24 pp; commits framing-A reading |
| D02_D03_D04_lpips_gpu_v1 | 2026-05-04 | LPIPS-vgg = 0.413 (all three patches tied) | keep | Item 1c-8 close: patch-kind doesn't matter, F_c selection dominates LPIPS+FID |
| D04_violence_v1 | 2026-05-04 | concat AUC = 1.000 (raw + SAE), 0 nudity-violence feature overlap | keep | Phase D-4 cross-concept; SAE features concept-specific; per-concept F_c disjoint at all 4 hookpoints |
| A01_5seed_ci_v1 | 2026-05-04 | mean ASR=1.000 ± 0.000 (5/5 seeds, 88/88 bypass) | keep | Item 1c-6 A01 pixel-PGD 5-seed CI complete with 0 variance |
| Item_1c6_5seed_ci_final | 2026-05-04 | A01+A02+A03 saturate, C01 0.954 ± 0.029 | keep | Item 1c-6 close — combined 5-seed CI table for all 4 attack methodologies |
| D09_cross_arch_safety_v1 | 2026-05-05 | SDXL Turbo 8.5%, Base 28.6%, SD3 4.0%, PixArt 0%, FLUX deferred | keep | Phase D-9 cross-arch safety_checker baseline; SD3 + PixArt suggest safety_checker generalization gap |
| D05_oracle_transfer_v1 | 2026-05-05 | A01 attack: 87.5% escape 3-oracle ensemble (SC+NudeNet+Q16) | keep | Phase D-5 oracle-transfer; PGD-on-SC is classifier-specific, doesn't transfer to NudeNet/Q16 |
| A01_A02_eps_sweep_final_v1 | 2026-05-05 | 228/228 ASR=1.000 across all (attack, ε ≥ 1/255) | keep | ε-sweep — A01+A02 saturate even at quarter ε (1/255 minimum), safety_checker brittleness independent of attack budget |
| B02v3_on_mma_v1 | 2026-05-05 | adv vs orig AUC=0.388 (detector OOD) | keep | B02-v3 SAE detector does NOT generalize to MMA-Diffusion (SD v1.4 backbone); 0/50+0/53 flagged at logit>0; caveat for Framing A architecture-specificity |
