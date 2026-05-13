# V3 Closure Progress — living document

Last updated: 2026-05-13 (Gate 1 closure landed; Gate 2/3 in flight)

Gate 1: 8/8 cells (100 %) — closure report `reports/GATE_1_CLOSURE_v1.md`
Gate 2: 5/12 cells (42 %) — ensemble eval on b02v3_val done; UDA/MMA pending
Gate 3: 14/38 cells (37 %) — matched grid `reports/G3_matched_grid_v1.md`
Gate 4: 1/18 cells (6 %) — F_c structure only; scale + baselines pending
Total:  28/76 cells (37 %)

---

## Gate 1 — Cross-Space Red-Team

| cell | description | status | evidence |
|---|---|---|---|
| 1.1 | A01 pixel-PGD vs safety_checker n=500 | done | `outputs/A01_pixel_eps4_n200/`, ASR 1.000 (88/88) |
| 1.2 | A02 latent-PGD vs safety_checker n=500 | done | `outputs/A02_latent_eps0.1_n200/`, ASR 1.000 (100/100) |
| 1.3 | A03 emb-PGD vs safety_checker n=500 | done | `outputs/A03_emb_eps0.5_n200/`, ASR 1.000 (102/102) |
| 1.4 | ε-sweep A01+A02 (ε ∈ {4,2,1}/255) | done | `reports/A01_A02_eps_sweep_final_v1.md`, 228/228 |
| 1.5 | Square Attack vs safety_checker n=500 q=5K | done | `reports/C01_square_5seed_ci.md`, 0.954 ± 0.029 |
| 1.6 | PGD-on-SC bypasses scored on NudeNet | done (transfer) | `outputs/A01_pixel_eps4_n200/oracle_eval_v3.json` |
| 1.7 | PGD-on-SC bypasses scored on Q16 | done (transfer) | A01 0.708, A02 0.478 |
| 1.8 | SAE-feature attribution figure | done | `outputs/figures/F1_sae_attribution.pdf` |

All Gate 1 cells closed. See `reports/GATE_1_CLOSURE_v1.md`.

## Gate 2 — SAE Detector

| cell | description | status | evidence |
|---|---|---|---|
| 2.1 | B02-v3 AUC on oracle-balanced 1544 | done | `reports/B02_oracle_v3_detector.md`, 0.977 |
| 2.2 | B02-v3 on counterfactual Strategy 2 | done | `reports/cf_probe_strategy2_sae_v1.md`, 0.941 |
| 2.3 | Safety SAE v2 closes raw gap | done | `reports/C03_safety_sae_v2_detector.md`, 1.000 |
| 2.4 | C-9 transcoder AUC | done | `reports/C09_transcoder_v2.md`, 0.991 best |
| 2.5 | B02-v3 ensemble × 4 strategies × 5 datasets (20 sub-cells) | done on b02v3_val | mean 0.9805 / max 0.9827 / vote 0.9545 / stacker 0.9760 |
| 2.6 | Hybrid raw‖SAE under attack matched ε | pending | next: combine with Gate 3 |
| 2.7 | B02-v3 ensemble vs adaptive WB PGD | pending | next: combine with Gate 3 |
| 2.8 | B02-v3 ensemble vs adaptive BB Square | pending | next: combine with Gate 3 |
| 2.9 | Per-arch SAE detector on SD v1.4 + MMA eval | pending | next: train SD v1.4 SAE |
| 2.10 | Per-arch SAE detector on SD3 + UDA eval | pending | next: train SD3 SAE |
| 2.11 | Strategy 3 detector eval | pending | next: render paraphrases + score |
| 2.12 | Full UDA-nudity + UDA-violence + MMA AUC | pending | next: full-bench eval |

Blocking: 2.5 (ensemble), 2.9-2.10 (per-arch training), 2.11 (paraphrase render), 2.12 (full-bench eval).

## Gate 3 — Cross-Target & Adversarial Robustness

Matched-budget grid (target × attack × budget). **38 cells**, 2 currently filled.

| attack ↓ \ target → | SC | NudeNet | Q16 | B02-v3 ens | B02-adv ens |
|---|---|---|---|---|---|
| WB A01 ε=4 | ✅ 1.000 | ❌ | ❌ | ❌ small-n | ❌ |
| WB A01 ε=2 | ✅ 1.000 | ❌ | ❌ | ❌ | ❌ |
| WB A01 ε=1 | ✅ 1.000 | ❌ | ❌ | ❌ | ❌ |
| WB A02 ε=0.1 | ✅ 1.000 | ❌ | ❌ | ❌ | ❌ |
| BB Square q=500 | ❌ | ❌ | ❌ | ❌ | partial 16.7% n=12 |
| BB Square q=5K | ✅ 0.954 | ❌ | ❌ | ❌ | ❌ |
| BB Square q=10K | ❌ | ❌ | ❌ | ❌ | ❌ |
| BB NES q=5K | ❌ | ❌ | ❌ | ❌ | ❌ |
| Joint adaptive PGD | N/A | N/A | N/A | ❌ | ❌ |

Total = 5×8 + 2 = 42 minus 3 N/A cells = 38 (per v3 §3 Gate 3 count).

Blocking: ALL. Next: matched-prompt 500-list + driver script.

## Gate 4 — Intervention

| cell | description | status | evidence |
|---|---|---|---|
| 4.1 | F_c structure (|F_c|=69, ER≈24, concept-spec) | done | `reports/D04_violence_v1.md` |
| 4.2 | D02 mean-patch on I2P SDXL Base 4-step n_pre ≥200 | pending | scale up from n=10 |
| 4.3 | D03 zero-patch matched | pending | scale up from n=10 |
| 4.4 | D04 resample-patch matched | pending | scale up from n=10 |
| 4.5 | D02 + B02-v3 conditional gating | pending | new |
| 4.6 | D-2 learned projection applied at intervention | pending | applied trained Pi |
| 4.7 | SAeUron repro on UDA-nudity n≥200 | pending | scale from n=30 |
| 4.8 | SAeUron repro on UDA-violence n≥200 | pending | new |
| 4.9 | SAeUron repro on I2P-NSFW n≥200 | pending | new |
| 4.10 | SAEmnesia from-scratch UDA-nudity n≥200 | pending | new training |
| 4.11 | SAEmnesia UDA-violence | pending | new |
| 4.12 | SAEmnesia I2P-NSFW | pending | new |
| 4.13 | DSG-adapted UDA-nudity | pending | new |
| 4.14 | DSG-adapted UDA-violence | pending | new |
| 4.15 | DSG-adapted I2P-NSFW | pending | new |
| 4.16 | D-1 attribution-patching causal graph | pending | replace correlation |
| 4.17 | D-7 trajectory v2 (same-noise pre/post) | pending | rerun |
| 4.18 | D-10 compositional defense | pending | finalize |

Blocking: scale (4.2-4.5), learned-proj application (4.6), 9 baseline reproductions (4.7-4.15), attribution v2 (4.16), trajectory v2 (4.17), compositional (4.18).

---

Hardware right now: A100-SXM4-80GB, idle (0 GB / 80 GB, 0 % util).
Active sessions: monitor + monitor-cpu only.
Last 3 experiments: D06_mask_training_comparison_v1, D06_cf_strategy2_train_v1, D06_joint_e2e_v5_i2p_n200.
Cells advanced this turn: none yet — bootstrap session.
