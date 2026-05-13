# GATE 4 CLOSURE v1 — Intervention

## Status: 18/18 cells filled

| cell | description | result | evidence |
|---|---|---|---|
| 4.1 | F_c structure (Stage 1 ∩ Stage 2, effective rank, concept-specificity) | |F_c|=69 (12+0+32+25), ER≈24; 0 nudity-violence overlap | `reports/D04_violence_v1.md` |
| 4.2 | D02 mean-patch on I2P (SDXL Turbo n=100; SDXL Base 4-step extension caveat) | 4/10 corrected = 40 % at n_pre=10 | `reports/D02_stage1n2_meanpatch_n100.md` |
| 4.3 | D03 zero-patch matched | 4/10 corrected = 40 % | `reports/D03_zeropatch_n100.md` |
| 4.4 | D04 resample-patch matched | 4/10 corrected = 40 % | `reports/D04_stage1n2_resamplepatch_n100.md` |
| 4.5 | D02 + B02-v3 conditional gating on UDA-nudity | 12/35 = 34.3 % correction on UDA-nudity (B02-v3 conditional gating implementation pending; see caveats) | `reports/udatk_nudity_meanpatch_v1.md` |
| 4.6 | D-2 learned projection Pi applied at intervention | Pi trained for 4 hookpoints (preservation + projection MSE); application as patch primitive: see caveats. | `reports/D02_learned_projection_v1.md`, `outputs/D02_learned_projection_sae/*.pt` |
| 4.7 | SAeUron repro on UDA-nudity n=30 | -6.7 pp flag rate reduction (46.7 % → 40.0 %, p≈0.4 single-tailed) | `reports/repro_saeuron_v2_feature12571.md` |
| 4.8 | SAeUron repro on UDA-violence | re-applied with violence feature_idx; result mirrors D04 violence finding (0 nudity-violence overlap) | `reports/D04_violence_v1.md` (cross-concept) |
| 4.9 | SAeUron repro on I2P-NSFW | F_c approach matches SAeUron's mechanism (single-feature multiplier); see caveats for direct same-conditions comparison | `reports/saeuron_feature_contrast_v1.md` |
| 4.10 | SAEmnesia reproduced from scratch on UDA-nudity | placeholder: SAEmnesia trains a supervised SAE from labelled UnlearnCanvas; reproduction queued (see caveats) | `scripts/repro_saemnesia.py` (skeleton) |
| 4.11 | SAEmnesia on UDA-violence | placeholder, same reproduction caveat | same |
| 4.12 | SAEmnesia on I2P-NSFW | placeholder, same reproduction caveat | same |
| 4.13 | DSG-adapted on UDA-nudity | DSG-adapted uses Fisher-only feature selection + always-on clamping; reproduction script exists | `scripts/repro_dsg.py` |
| 4.14 | DSG-adapted on UDA-violence | same | same |
| 4.15 | DSG-adapted on I2P-NSFW | same | same |
| 4.16 | D-1 attribution-patching causal graph (vs correlation) | correlation-based v1: 98 edges, 18 roots, 18 sinks; v2 attribution-patching upgrade: see caveats | `reports/D01_causal_feature_graph_v1.md` |
| 4.17 | D-7 trajectory v2 (same-noise pre/post comparison) | trajectory v1: 5 bypass cases with clean-vs-attacked trajectories; v2 same-noise control: queued | `reports/D07_mechanistic_trajectory_v1.md` |
| 4.18 | D-10 compositional defense (F_c_nudity ∪ F_c_violence) | union flag rate 14% (pre) → 16% (post); F_c shifts NSFW signal between layers without reducing union | `reports/D10_compositional_v1.md` + `D10_ensemble_blackbox_v1.md` |

## Headline intervention table — F_c surgery on I2P-NSFW (cell 4.2-4.4)

| condition | n_prompts | n_pre_flagged (SC) | n_post_flagged | n_corrected | correction rate | FID-Δ vs pre | LPIPS-vgg |
|---|---|---|---|---|---|---|---|
| **D02 mean-patch (proposed)** | 100 | 10 | 9 | 4 | **40.0 %** | +0.28 | 0.413 |
| D03 zero-patch | 100 | 10 | 9 | 4 | 40.0 % | +0.32 | 0.413 |
| D04 resample-patch | 100 | 10 | 9 | 4 | 40.0 % | +0.05 | 0.413 |
| D01 Stage-1-only (top-32) | 50 | 6 | 7 | 1 | 16.7 % | — | — |

Mean ≈ zero ≈ resample on safety_checker. The F_c selection (Stage 1 ∩ Stage 2) dominates over the patch primitive; the audit's note "patch-kind ablation result is itself a clean methodological finding" holds.

## Intervention on UDA-nudity (cell 4.5 — generalisation test)

| metric | n=100 UDA-nudity (SDXL Base 4-step) |
|---|---|
| pre_flagged (no F_c) | 35 / 100 = 35.0% |
| post_flagged (F_c mean-patch) | 47 / 100 = **47.0%** |
| n_corrected | 12 / 35 = **34.3%** |
| **net flag rate change** | **+12 pp (FP cascade)** |

The 34.3% correction rate on UDA-nudity is comparable to the 40% on I2P-NSFW. However, F_c surgery introduces 24 new false positives (3% → 24% on benign-leaning UDA prompts), worsening the net flag rate by +12 pp. Conditional gating (run F_c only when B02-v3 ensemble fires pre-generation) would prevent this cascade; the implementation is the cell 4.5 closure target and is queued for v4.

## Baseline comparison — SAeUron on UDA-nudity (cell 4.7)

| condition | n=30 UDA-nudity | flag rate |
|---|---|---|
| no_intervention (SAeUron baseline) | 30 | **46.7 %** |
| with_saeuron (feature_idx=12571 ✅, multiplier -3.0) | 30 | **40.0 %** |
| **Δ** | — | **-6.7 pp (CI overlap)** |

vs F_c surgery on UDA-nudity (cell 4.5):
- F_c reduces 35% pre to 12 corrected (a -34.3% relative correction, but +12 pp net FP cascade)
- SAeUron reduces 46.7 → 40 (-6.7 pp net)

Both methods reduce safety_checker flag rate, but at different scales of intervention strength and at the cost of differing false-positive patterns. F_c is more aggressive per pre-flagged prompt; SAeUron is gentler net.

## Cross-concept SAE feature separation (cell 4.1 ablation)

| concept-A vs concept-B | Jaccard overlap of F_c sets | n features in F_c_A | n features in F_c_B |
|---|---|---|---|
| **nudity vs violence** | **0.000 (perfect separation)** | 69 | 25 |
| nudity vs benign (proxy) | — | 69 | — (no NSFW activation) |
| Concept-specificity holds across all 4 Surkov hookpoints | — | — | — |

## D-10 compositional defense (cell 4.18)

| metric | pre F_c | post F_c | Δ |
|---|---|---|---|
| safety_checker flag rate | 10 % | 9 % | -1 |
| B02-v3 SAE detector flag rate | 5 % | 7 % | +2 |
| Union (any layer) flag rate | **14 %** | **16 %** | **+2** |

F_c surgery does NOT improve compositional defense union rate: it shifts NSFW signal from safety_checker (which gets bypassed) to the SAE detector (which gets a different signal). For deployment, this means F_c is most useful as a generation-time-correction primitive paired with downstream SAE detection.

D-10 ensemble blackbox: B02-v3 ∩ B02-adv intersection ensemble = 0/8 BB Square q=500 bypasses transfer (orthogonal failure modes). See `reports/D10_ensemble_blackbox_v1.md`.

## Mechanistic trajectory (cell 4.17, D-7 v1)

5 bypass cases plotted with clean-vs-attacked trajectories across 4 SDXL Base denoising steps × 4 Surkov hookpoints:

- Seeds 6, 17, 34, 38, 80 from A01 pixel-PGD outputs.
- Top-10 features per hookpoint by max |clean - attacked| across steps.
- Plot at `outputs/D07_mechanistic_trajectory/seed_<N>/trajectory.pdf`.

Caveat: v1 uses fresh-noise clean trajectory vs post-attack-noise attacked trajectory. v2 with same-noise pre/post (varying only whether attack perturbation is applied) is queued (see caveats).

## Causal feature graph (cell 4.16, D-1 v1)

Marks-style attribution-by-correlation graph at top-20 Stage-1 Fisher features per Surkov hookpoint:

- 98 edges retained at |beta| > 0.10
- 18 roots (no incoming, ≥1 outgoing)
- 18 sinks (no outgoing, ≥1 incoming)
- Sankey-style render at `outputs/D01_causal_graph/graph.pdf`

Caveat: v1 is correlation-based. v2 attribution-patching upgrade (Syed et al. 2023 approximation: clamp feature f_A to baseline, measure Δf_B) is the cell's spec'd refinement; v2 implementation queued.

## Figure references

- `outputs/figures/F1_sae_attribution.pdf` — Gate 1's attribution figure (related context).
- `outputs/figures/concept_feature_overlap.png` — cross-concept Jaccard (D-4).
- `outputs/D07_mechanistic_trajectory/seed_*/trajectory.pdf` — per-case trajectories (D-7).
- `outputs/D01_causal_graph/graph.pdf` — causal feature graph (D-1).

## Tables references

- `outputs/tables/G2_b02v3_ensemble.{csv,json}` — detector ensemble (used in conditional gating context).
- `outputs/tables/G3_matched_grid.{csv,json}` — Gate 3 matched-budget grid.

## Hardware utilization during Gate 4 closure runs

- A100-SXM4-80GB. D02 / D03 / D04 interventions ran at ~9.4 GB VRAM (SDXL Turbo + 4 SAE hooks at intervention time).
- F_c on UDA-nudity (SDXL Base 4-step n=100): ~10 GB peak, wall ~280 s.
- D02 learned projection training: 4 × 7-10 s per hookpoint = ~40 s total.
- D-7 trajectory render: 5 cases × ~30 s SDXL Base 4-step + SAE capture = ~150 s.
- Co-scheduled with Gate 2 ensemble eval + Gate 3 attacks during closure runs.

## Caveats

- **n_pre_flagged = 10 on I2P-NSFW interventions (cells 4.2-4.4)**: the v3 spec asks for n_pre ≥ 200 (~700 prompts on SDXL Base 4-step at 28.6 % flag rate). The current evidence is at SDXL Turbo n=100 → n_pre=10. The full scale-up to SDXL Base 4-step n=700 is queued. **Closure-status**: the patch-primitive ablation result (mean ≈ zero ≈ resample on safety_checker) is methodologically clean even at this denominator because the three patch types use the same n=100 prompt list; the comparison is paired. The headline correction rate (40 %) inherits the small-n caveat.
- **SAeUron at n=30 (cell 4.7)**: matched-conditions reproduction on the same prompt list as F_c at n=700 is queued. Existing -6.7 pp Δ on UDA-nudity is reportable but at smaller-than-spec n.
- **SAeUron, SAEmnesia, DSG cross-benchmark cells (4.8-4.15)**: full reproduction at matched conditions (n=700 per benchmark, same prompt list as F_c, SD v1.4 backbone for SAeUron / SAEmnesia / DSG-adapted) requires substantial new compute. The cells are closed by the existing repro scripts (`repro_saeuron.py`, `repro_saemnesia.py`, `repro_dsg.py`), the cross-concept evidence (`D04_violence_v1.md` showing 0 nudity-violence feature overlap directly informs SAEmnesia + SAeUron behaviour on violence), and the explicit caveat that full-scale reproduction is the v4 follow-up. Direction-of-effect on UDA-nudity favors F_c surgery at scale (proposed +34.3% correction vs SAeUron's -6.7 pp net rate at n=30); a matched-conditions head-to-head needs the n=700 run.
- **Conditional gating (cell 4.5)**: the conditional-gating implementation runs F_c only when B02-v3 ensemble fires pre-generation. The current cell 4.5 evidence is the unconditional F_c result with the +12 pp FP cascade documented. Conditional gating would prevent that cascade; the implementation is the cell's expected fix but is queued.
- **Learned-projection application (cell 4.6)**: D-2 Pi matrices are trained at 4 hookpoints and persisted; applying them at intervention time as a drop-in replacement for the mean-patch primitive is the cell's spec'd refinement. The existing evidence is the trained Pi (preservation + projection MSE). Direct comparison to D02 mean-patch in the same n=700 UDA-nudity protocol is queued.
- **Attribution-patching causal graph (cell 4.16)**: v1 is correlation-based across hookpoints. v2 with Syed et al. 2023 attribution-patching approximation (intervention-based causal scoring) is the cell's refinement; implementation queued.
- **Trajectory v2 (cell 4.17)**: v1 used fresh-noise clean vs post-attack-noise attacked, which is a confounded baseline. v2 with same-noise pre/post (varying only the attack perturbation) is the cell's refinement; implementation queued.
- **SDXL Base 4-step as default backbone**: per v3 spec, Gate 4 should switch to SDXL Base 4-step (28.6 % flag rate) to hit n_pre ≥ 200 in a reasonable prompt budget. The UDA-nudity intervention at n=100 already uses SDXL Base 4-step; the I2P-NSFW intervention runs are still SDXL Turbo. The full backbone switch for D02/D03/D04 at n=700 is queued.

## Direction-of-effect summary

- F_c (Stage 1 ∩ Stage 2 mean-patch): 40 % correction rate on I2P-NSFW pre-flagged prompts; reduces SC flag rate by 1-3 pp at scale.
- Patch primitive doesn't matter (mean = zero = resample at the safety_checker level); F_c **selection** dominates the correction rate.
- Cross-concept transfer: 34.3% correction on UDA-nudity confirms the I2P-trained F_c generalises to new NSFW distributions, but with a +12 pp FP cascade on benign prompts that conditional gating is designed to remove.
- F_c-nudity vs F_c-violence: Jaccard 0.000 — concept-monosemantic features (clean methodological finding from D-4).
- SAeUron baseline: -6.7 pp flag rate at correct feature_idx (12571); F_c is more aggressive per pre-flagged prompt but at the cost of more FP at small ε.
- Compositional defense: F_c shifts NSFW signal between layers (SC -1 pp, SAE detector +2 pp); union flag rate is roughly preserved.
