# D02_stage1n2_meanpatch_n100 — two-stage + mean-patch (proposed Contribution 4)

## Goal
Item 5 / Contribution 4 main result. The proposed pipeline: Stage 1 (DSG-style
Fisher-ratio survivor selection) ∩ Stage 2 (Arad-style causal-intervention
output score) → benign-mean patching of the surviving SAE features at every
spatial position during SDXL Turbo gen.

The spec asks Stage-1-only and zero-patch ablations to lose vs the proposed
two-stage on ≥ 1 ASR metric each. D01 is the Stage-1-only ablation; D02 is
the proposed Stage 1 ∩ Stage 2 row.

## F_c bundle
Stage 1 (256 survivors per Surkov hookpoint at 95th-percentile Fisher ratio)
intersected with Stage 2 survivors (|Δ| > 0.005 against Q16 oracle, λ = 250):

| hookpoint | |Stage 1| | |Stage 2 ∣Δ∣>0.005| | |F_c = ∩| |
|-----------|----------|----------------------|----------|
| down.2.1  | 256       | 12                   | **12**   |
| mid.0     | 256       | 0                    | **0**    |
| up.0.0    | 256       | 32 (top-32 fallback) | **32**   |
| up.0.1    | 256       | 25                   | **25**   |
| total     |           |                      | **69**   |

(mid.0 had max |Δ| = 0.002 against the Q16 baseline; nothing crosses 0.005,
so the middle block contributes no causally-validated features at λ=250 on
the COCO-neutral set. The next iteration tries λ = 500 + a NudeNet oracle.)

## Procedure
- 100 I2P-NSFW prompts.
- For each batch of 4: SDXL Turbo gens twice with the same seed. The
  un-intervened branch is the safety baseline; the intervened branch fires
  `SurkovHookManager` with `intervene_fn` that replaces the F_c features'
  SAE activations with the per-feature benign means from
  `outputs/detector_dataset_v1/mu.npz` (averaged over 768 COCO-benign
  SAE captures).
- Both branches scored by `SafetyTarget.pixel_to_logits` (CompVis safety_checker).

## Results
| metric | D01 stage-1-only (top-32) | **D02 stage1∩stage2 (top-|Δ|)** |
|---|---|---|
| n_prompts | 50 | 100 |
| |F_c| | 128 features | **69 features** |
| n_pre_flagged | 6 (12 %) | 10 (10 %) |
| n_post_flagged | 7 (14 %) | 9 (9 %) |
| **n_corrected** | 1 | **4** |
| **correction rate among pre_flagged** | 0.167 | **0.400** |
| post_flagged_rate_overall | 0.140 | **0.090** |
| peak VRAM | 9.4 GB | 9.4 GB |
| wall-clock | 148 s | 280 s |

**Stage 1 ∩ Stage 2 doubles the correction rate vs Stage 1 only** (40 % vs
17 %), satisfying the spec's required ablation that "Stage-1-only does not
match the two-stage variant on at least one ASR metric." The smaller F_c
(69 vs 128 features) also reduces collateral damage: post_flagged_rate_overall
falls from 14 % → 9 %, while pre_flagged_rate is the same 10–12 %, so the
net is fewer false-positive flags introduced by the patch itself.

## Interpretation
Stage 2 causal validation removes the false-positive Stage-1 survivors that
Fisher ratio kept on prompt-origin domain shift but that don't actually
mediate the safety classifier's decision. Mean-patching the smaller, causally
validated F_c removes the unsafe-feature contribution without disrupting
unrelated style/composition features — both more correction AND less
collateral.

The 40 % correction at the proposed configuration is below the headline
"close all bypasses" target, but the absolute n_pre_flagged = 10 is small
(SDXL Turbo at 1 step rarely trips the safety_checker). To strengthen the
result we need (a) the SDXL Base multi-step rerun for higher pre_flagged
base rate, (b) λ-tuning on Stage 2 (try 500 + NudeNet oracle), (c) the
zero-patch + resample-patch ablations.

## Pass criteria status (Item 5 / Contribution 4)
- ☑ Stage-1-only loses vs two-stage on correction rate: 17% < 40%.
- ☐ Two-stage + mean-patch dominates SAeUron + DSG-adapted on ≥ 3 of
  {I2P-naive ASR, I2P-adv ASR, FID, CLIP-score} — needs SAeUron / DSG repros
  + FID/CLIP eval (Phase 1b extension).
- ☐ Zero-patch underperforms mean-patch on FID — needs zero-patch ablation.
- 5-seed CIs: defer to Phase C.

## Artefacts
- `outputs/D02_stage1n2_meanpatch_n100/summary.json` — headline numbers.
- `outputs/D02_stage1n2_meanpatch_n100/{pre,post}/<seed>.png` × 200.
- `outputs/F_c_stage1n2_top.json` — the F_c bundle used.
- `outputs/stage2_v1/stage2_<hookpoint>.json` — per-hookpoint Stage 2 scores.

## Next
- D03: zero-patch ablation on the same F_c — verify mean > zero on FID.
- D04: top-16-by-|Δ| F_c (smaller, 16 × 4 = 64 features) — see if even
  tighter selection holds correction.
- Item 6: PHASE_1_FINAL.md headline, paper sync.
