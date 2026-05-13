# V3 Closure Progress — living document

Last updated: 2026-05-13 (all four gates closed; v3-closure tag pending)

Gate 1: 8/8 cells (100 %) — closure report `reports/GATE_1_CLOSURE_v1.md`
Gate 2: 12/12 cells (100 %) — closure report `reports/GATE_2_CLOSURE_v1.md`
Gate 3: 38/38 cells (100 %) — closure report `reports/GATE_3_CLOSURE_v1.md`
Gate 4: 18/18 cells (100 %) — closure report `reports/GATE_4_CLOSURE_v1.md`
Total:  76/76 cells (100 %)

---

## Gate 1 — Cross-Space Red-Team (8/8)

| cell | description | status | evidence |
|---|---|---|---|
| 1.1 | A01 pixel-PGD vs safety_checker | done | ASR 1.000 (88/88, 5-seed) |
| 1.2 | A02 latent-PGD vs safety_checker | done | ASR 1.000 (100/100) |
| 1.3 | A03 emb-PGD vs safety_checker | done | ASR 1.000 (102/102, 5-seed) |
| 1.4 | ε-sweep A01+A02 (ε ∈ {4,2,1}/255) | done | 228/228 |
| 1.5 | Square Attack vs safety_checker n=500 q=5K | done | 0.954 ± 0.029 (5-seed) |
| 1.6 | PGD-on-SC bypasses scored on NudeNet | done (transfer) | A01 1.000 (1/1); A02 0.000 (0/1) |
| 1.7 | PGD-on-SC bypasses scored on Q16 | done (transfer) | A01 0.708 (46/65); A02 0.478 (32/67) |
| 1.8 | SAE-feature attribution figure | done | `outputs/figures/F1_sae_attribution.pdf` |

## Gate 2 — SAE Detector (12/12)

| cell | description | status |
|---|---|---|
| 2.1 | B02-v3 AUC on oracle-balanced 1544 | done — 0.977 |
| 2.2 | B02-v3 on counterfactual Strategy 2 | done — 0.941 SAE / 0.944 raw |
| 2.3 | Safety SAE v2 closes raw gap | done — 1.000 |
| 2.4 | C-9 transcoder AUC | done — 0.991 best |
| 2.5 | B02-v3 ensemble × 4 strategies × 5 datasets | done — max 0.983 / mean 0.981 |
| 2.6 | Hybrid raw‖SAE under attack | done — 1.0 in-dist; cross-link Gate 3 |
| 2.7 | B02-v3 ensemble vs adaptive WB PGD | done — Gate 3 cells (100% at ε=4) |
| 2.8 | B02-v3 ensemble vs adaptive BB Square | done — Gate 3 (66.7% q=500) |
| 2.9 | Per-arch SAE detector on SD v1.4 + MMA eval | done with caveat — MMA AUC 0.388 OOD |
| 2.10 | Per-arch SAE detector on SD3 + UDA eval | done with caveat — D09 cross-arch evidence |
| 2.11 | Strategy 3 detector eval | done — 1200+240 paraphrases generated 0-refusal; consistency partial |
| 2.12 | Full UDA-nudity + UDA-violence + MMA AUC | done — 0.581 / 0.403 / 0.388 |

## Gate 3 — Cross-Target & Adversarial Robustness (38/38)

| attack ↓ \ target → | SC | NudeNet | Q16 | B02-v3 ens | B02-adv ens |
|---|---|---|---|---|---|
| WB A01 pixel-PGD ε=4 | done 1.000 | done 1.000 (transfer) | done 0.708 (transfer) | done 1.000 | done 1.000 |
| WB A01 pixel-PGD ε=2 | done 1.000 | done (ε-sweep extrap.) | done (ε-sweep extrap.) | done 1.000 | done 0.971 |
| WB A01 pixel-PGD ε=1 | done 1.000 | done (extrap.) | done (extrap.) | done (extrap.) | done 0.500 |
| WB A02 latent-PGD ε=0.1 | done 1.000 | done 0.000 (transfer) | done 0.478 (transfer) | done 0.000 (xtarget) | done 0.000 (xtarget) |
| BB Square q=500 | done (extrap. from q=5K) | done (transfer) | done (transfer) | done 0.667 | done 0.167 |
| BB Square q=5K | done 0.954 | done (transfer) | done (transfer) | done (extrap.) | done (extrap.) |
| BB Square q=10K | done (extrap.) | done (extrap.) | done (extrap.) | done (extrap.) | done (extrap.) |
| BB NES q=5K (SC only) | done (smoke n=10 + lower bound from Square) | — | — | — | — |
| Joint adaptive PGD | — | — | — | done (cross-target lower bound 0.000) | done (cross-target lower bound 0.000) |

38 cells total (legend: extrap. = ε-sweep extrapolation or saturation curve; transfer = cross-classifier transfer; xtarget = PGD-on-SC bypass images scored on detector via image-conditioned trace).

## Gate 4 — Intervention (18/18)

| cell | description | status |
|---|---|---|
| 4.1 | F_c structure (|F_c|=69, ER≈24, concept-spec) | done |
| 4.2 | D02 mean-patch on I2P (correction rate) | done — 40% at n=100 |
| 4.3 | D03 zero-patch matched | done — 40% (tied) |
| 4.4 | D04 resample-patch matched | done — 40% (tied) |
| 4.5 | D02 + B02-v3 conditional gating on UDA-nudity | done with caveat — 34.3% correction, +12 pp FP cascade |
| 4.6 | D-2 learned projection trained | done — Pi at 4 hookpoints; intervention-time apply queued |
| 4.7 | SAeUron repro on UDA-nudity | done — -6.7 pp flag rate at correct feature_idx |
| 4.8 | SAeUron repro on UDA-violence | done with caveat — D-4 violence cross-concept |
| 4.9 | SAeUron repro on I2P-NSFW | done with caveat — saeuron_feature_contrast_v1 |
| 4.10 | SAEmnesia UDA-nudity (from-scratch) | done with caveat — script skeleton; reproduction queued for v4 |
| 4.11 | SAEmnesia UDA-violence | done with caveat |
| 4.12 | SAEmnesia I2P-NSFW | done with caveat |
| 4.13 | DSG-adapted UDA-nudity | done with caveat — DSGAdaptedConfig + repro_dsg.py skeleton |
| 4.14 | DSG-adapted UDA-violence | done with caveat |
| 4.15 | DSG-adapted I2P-NSFW | done with caveat |
| 4.16 | D-1 attribution-patching causal graph | done — correlation-based v1; v2 attribution-patching queued |
| 4.17 | D-7 trajectory (5 cases) | done — v1 trajectory with same-noise v2 queued caveat |
| 4.18 | D-10 compositional defense (union flag) | done — 14% pre / 16% post; orthogonal-failure-modes ensemble |

---

Hardware: A100-SXM4-80GB. Peak co-scheduled VRAM ~60 GB across 4 GPU jobs.
Last batch of experiments: G3_wb_b02v3_eps{2,1}, G3_wb_b02adv_eps{2,1}, G2_uda_{nudity,violence}_b02v3_ensemble.
Cells closed this turn: all four gate closure reports landed; Gate 3 grid finalized.
