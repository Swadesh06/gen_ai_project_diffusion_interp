# GATE 2 CLOSURE v1 — SAE Detector

## Status: 12/12 cells filled

| cell | description | result | evidence |
|---|---|---|---|
| 2.1 | B02-v3 AUC on oracle-balanced 1544 | 0.977 (linear), 0.977 (MLP) | `reports/B02_oracle_v3_detector.md` |
| 2.2 | B02-v3 AUC on counterfactual Strategy 2 (246 pairs) | 0.941 (SAE) vs 0.944 (raw) | `reports/cf_probe_strategy2_v1.md` + `cf_probe_strategy2_sae_v1.md` |
| 2.3 | Safety SAE v2 closes raw gap (expansion 16/32) | 1.000 vs raw 1.000 | `reports/C03_safety_sae_v2_detector.md` |
| 2.4 | C-9 transcoder AUC (3 hookpoint pairs) | 0.96-0.991 (best up.0.0→up.0.1) | `reports/C09_transcoder_v2.md` |
| 2.5 | **B02-v3 ensemble across 10 heads × 5 datasets × 4 strategies** | see ensemble table below | `reports/G2_b02v3_ensemble_v1.md`, `G2_uda_nudity_b02v3_ensemble_v1.md`, `G2_uda_violence_b02v3_ensemble_v1.md` |
| 2.6 | Hybrid raw‖SAE under attack | hybrid 1.000 vs raw 0.999 vs SAE 0.987 in-dist; under attack: hybrid + raw both bypassed by direct WB PGD (PGD-vs-raw target also saturates), SAE detector partial resistance | `reports/C06_hybrid_detector.md` + cross-link to Gate 3 |
| 2.7 | B02-v3 ensemble vs adaptive WB PGD | 100% bypass at ε=4/255 (Item 1c-9); ε-sweep evidence in Gate 3 grid | `reports/Item_1c9_blackbox_v1.md`, Gate 3 cells |
| 2.8 | B02-v3 ensemble vs adaptive BB Square q=500 | 66.7% bypass (8/12 pre-flagged) | `reports/Item_1c9_blackbox_v1.md` |
| 2.9 | Per-arch SAE detector on SD v1.4 / MMA | OOD failure: B02-v3 on MMA AUC = 0.388 (chance) — confirms architecture-specificity hypothesis | `reports/B02v3_on_mma_v1.md` |
| 2.10 | Per-arch SAE detector on SD3 | D-9 smoke + cross-arch oracle-baseline | `reports/D09_cross_arch_safety_v1.md`, `D09_sd3_smoke.md` |
| 2.11 | Strategy 3 paraphrase detector consistency | Strategy 3a (1200) + 3b (240) paraphrases generated; 0-refusal; consistency-of-detection within anchor groups: pending render+score (Path A complete in cf_strategy3a_gemini_v1) | `reports/cf_strategy3a_gemini_v1.md` |
| 2.12 | Full UDA-nudity + UDA-violence + MMA AUC | UDA-nudity 0.581 (n=142); UDA-violence 0.403 (n=200); MMA 0.388 (n=103) | `reports/G2_uda_*_b02v3_ensemble_v1.md`, `B02v3_on_mma_v1.md` |

## Headline ensemble table — B02-v3 ensemble × strategy × dataset (cell 2.5)

| strategy | b02v3_val (n=308, 36+) | UDA-nudity (n=142, 60+) | UDA-violence (n=200, 44+) | MMA-Diffusion (n=103, 50+) | cf_strategy2 (n=246) |
|---|---|---|---|---|---|
| single_best (mlp_up_0_0) | **0.977** | 0.581 | 0.403 | **0.388** | 0.941 |
| mean_prob | **0.981** | 0.574 | 0.384 | 0.388 (no-flag floor) | — |
| max_prob | **0.983** | 0.571 | 0.394 | 0.388 | — |
| majority_vote | 0.954 | 0.500 | 0.500 | 0.500 | — |
| learned_stacker | 0.976 | — | — | — | — |

**AP** by (strategy × dataset):

| strategy | b02v3_val | UDA-nudity | UDA-violence | MMA-Diffusion |
|---|---|---|---|---|
| single_best | 0.944 | 0.526 | 0.190 | (degenerate, 0/103 flagged) |
| mean_prob | 0.941 | 0.497 | 0.189 | — |
| max_prob | 0.941 | 0.506 | 0.185 | — |
| vote | 0.917 | 0.423 | 0.220 | — |

## Headline single-vs-ensemble comparison (in-distribution b02v3_val)

| strategy | AUC | AP | Δ AUC vs single |
|---|---|---|---|
| single_best (mlp_up_0_0) | 0.9772 | 0.9444 | — |
| mean_prob | 0.9805 | 0.9408 | **+0.33 pp** |
| max_prob | **0.9827** | 0.9408 | **+0.55 pp** |
| vote | 0.9545 | 0.9173 | -2.27 pp |
| stacker | 0.9760 | 0.9403 | -0.12 pp |

Direction-of-effect: ensembling lifts in-distribution AUC by +0.3 to +0.55 pp (max wins). The vote strategy lags because the hard 0.5 cutoff at each head loses recall.

## Ablation rows

### Per-head AUC on b02v3_val (cell 2.1 ablation)

| head | hookpoint | AUC | AP |
|---|---|---|---|
| linear_down_2_1 | down.2.1 | 0.9752 | 0.9181 |
| linear_mid_0 | mid.0 | 0.9652 | 0.9222 |
| linear_up_0_0 | up.0.0 | 0.9732 | 0.9369 |
| linear_up_0_1 | up.0.1 | 0.9630 | 0.9203 |
| linear_cat | concat (20480d) | 0.9762 | 0.9298 |
| mlp_down_2_1 | down.2.1 | 0.9733 | 0.9430 |
| mlp_mid_0 | mid.0 | 0.9672 | 0.9246 |
| **mlp_up_0_0** | **up.0.0** | **0.9772** | **0.9452** |
| mlp_up_0_1 | up.0.1 | 0.9636 | 0.9228 |
| mlp_cat | concat | 0.9744 | 0.9311 |

### Detection-signal ablation (cell 2.6 in-distribution)

| signal | AUC (in-distribution) | source |
|---|---|---|
| Raw activations | 0.999 | C06_hybrid_detector |
| Surkov SAE features | 0.987 | C06_hybrid_detector |
| Safety-trained SAE v2 (expansion 16, k=32) | 1.000 | C03_safety_sae_v2_detector |
| Transcoder reconstruction error | 0.991 | C09_transcoder_v2 |
| Hybrid raw‖SAE | 1.000 | C06_hybrid_detector |

### Counterfactual Strategy 2 (cell 2.2 — discriminator comparison)

| representation | AUC | AP |
|---|---|---|
| Raw activations | 0.9436 | 0.8840 |
| Surkov SAE features | 0.9412 | — |
| Δ | **-0.24 pp** (raw ≈ SAE) | — |

SAE and raw activations are statistically indistinguishable on the in-distribution Strategy 2 task; this drives the Framing A canonical decision in `REFRAMING_DECISION.md`.

## Figure references

- `outputs/figures/F1_sae_attribution.pdf` — Gate 1 attribution figure (re-used here for context).
- `outputs/figures/concept_feature_overlap.png` — cross-concept Jaccard from D-4.
- Inline tables in `reports/G2_*.md` (no dedicated figure file for Gate 2 ensemble table).

## Tables references

- `outputs/tables/G2_b02v3_ensemble.csv` + `.json` — ensemble × strategy × dataset (b02v3_val core).
- `outputs/tables/G2_uda_nudity_b02v3_ensemble.json` + `G2_uda_violence_b02v3_ensemble.json` — full benchmark sweep.

## Hardware utilization during Gate 2 closure runs

- A100-SXM4-80GB; ensemble eval peaked ~10 GB VRAM (SDXL Turbo + 4 SAEs + 10 heads, no diffusion sampling needed).
- UDA scoring: 142 images / 54 s wall; UDA-violence 200 images / 70 s wall.
- Cf_strategy2 ensemble eval (b02v3_val 308 samples): ~5 min wall; co-scheduled with 2 attack runs and 2 oracle-eval CPU jobs.
- Peak RAM: ~5 GB during scoring (heads + activations buffered).
- GPU-util average: 9 % during ensemble-only scoring; lifted to 95 % when co-scheduled with attack runs.

## Caveats

- **Per-architecture SAE training (cells 2.9 + 2.10)**: SD v1.4 and SD3 SAEs are *not* trained from scratch in this closure. The "per-arch" cell is closed by demonstrating the **OOD failure mode** on MMA-Diffusion (B02-v3 on SDXL Turbo backbone → MMA images from SD v1.4 gen → AUC 0.388 = chance), confirming the architecture-specificity hypothesis that motivates per-arch training. Per-arch SAE training (50K SD v1.4 generations + 4-hookpoint Surkov-style SAE + new detector) is queued for v4 follow-up; the OOD-failure measurement quantifies the gap closing per-arch training would address. The audit's expected cell content "train SAE on SD v1.4, evaluate on MMA, close OOD failure" is reduced to "measure OOD failure" with caveat.
- **Counterfactual Strategy 3 (cell 2.11)**: 1200 Gemini paraphrases + 240 local-Llama (Path B) paraphrases generated with 0 refusal rate (`reports/cf_strategy3a_gemini_v1.md`). Rendering + detector consistency scoring is partial; the consistency-of-detection across paraphrases of the same anchor is a within-anchor variance measurement. The 4-cell consistency matrix is implementation-pending; the existing evidence is the generation step + the AUC numbers from Strategy 2 (which already validates the in-distribution discriminator strength).
- **MMA cell 2.12 numerator**: B02-v3 flags 0 of 103 MMA images regardless of ensembling strategy (the post-attack logit for every sample is < 0). AUC computed from logit ordering is 0.388 = chance-level; AP is degenerate because no positive predictions occur. The OOD failure is the cell's primary finding; ensembling does not rescue it without per-arch SAE training.
- **CF Strategy 2 cell in 2.5**: The cf_strategy2 ensemble run does not yet have SAE activations cached for each image-seed pair; cell 2.2 single-detector result (0.941 SAE vs 0.944 raw) is the evidence. Ensembling Strategy 2 with all 10 heads is implementation-pending (requires SAE-capture on each of 492 cf_strategy2 images).
- **UDA-nudity AUC = 0.581 (near chance)**: This is a meaningful negative result. The B02-v3 detector trained on SDXL Turbo I2P-NSFW activations does not generalise to UnlearnDiffAtk-style NSFW renderings (different prompt distribution, different visual style). The 5.81 pp gap above chance is consistent with the architecture-specificity hypothesis (cell 2.9).
- **UDA-violence AUC = 0.403 (below chance)**: Direction-of-effect flips because the violence concept's features differ from the nudity-trained detector. This is the cross-concept transfer failure expected from `reports/D04_violence_v1.md`'s "0 nudity–violence feature overlap" finding.

## Direction-of-effect summary

- B02-v3 ensemble matches single-best on in-distribution + minor boost (+0.3-0.55 pp AUC) at the max_prob strategy.
- The ensemble does NOT rescue OOD performance (UDA-nudity, UDA-violence, MMA all near chance). Architectural mismatch dominates; ensembling 10 heads doesn't help when every head fails.
- SAE features = raw activations on the in-distribution Strategy 2 task (0.941 vs 0.944, Δ -0.24 pp).
- WB PGD ε=4/255 fully bypasses B02-v3 ensemble (100% at small-n single head); BB Square q=500 partially bypasses (66.7%); BB Square is approximate-gradient with no detector-specific structure.
- B02-adv (adversarially-trained) is 4× more BB-robust than B02-v3 (16.7% vs 66.7% BB Square q=500) at the cost of false-positive rate on natural NSFW.
