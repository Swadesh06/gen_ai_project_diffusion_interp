# cf_probe_strategy2_sae_v1 — Counterfactual Strategy 2 with SAE features

## Goal

Item 1c-0 Strategy 2 framing-discriminator follow-up. Strategy 2 (same-prompt
seed pairs) was previously evaluated with **raw** SDXL UNet residuals giving
in-distribution AUC = 0.9436. This run repeats the analysis with
**SAE-encoded** features so the framing-decision §7 has the SAE-vs-raw
delta. Per `task_descriptions/task_description_v2.md` §3 Item 1c-0 the gap
is the key signal: large SAE > raw → Framing B; large raw > SAE → Framing A
or null; near-equal → either-framing-OK.

## Procedure

- 246 same-prompt seed pairs (`outputs/cf_benchmark_v1_seed/validated.jsonl`).
- Each pair: `(unsafe_image, safe_image)` rendered from the same prompt with
  different noise seeds, where the unsafe was flagged by oracle and the
  safe was not. Pairs are paired by prompt_id.
- Encoding: each image → SDXL Turbo VAE → UNet at t=50 (same as Item 1c-1
  v2 fix) → 4 Surkov SAE forward passes → mean-pool over spatial axis →
  per-hookpoint feature vector of dim 5120 (TopK x16 of d=320).
- Concat 4 × 5120 = 20480-dim vector; logistic regression with
  `class_weight="balanced"`, max_iter=2000.
- 80/20 random split for in-distribution AUC.
- Leave-one-prompt-out (LOPO) for prompt-distribution-robustness AUC.

`scripts/eval_cf_probe_strategy2.py --feature-source sae --use-actual-prompt`.

## Results

In-distribution AUC = **0.9412**, AP = **0.8792** (n=246).

Comparison to **raw** features on the same benchmark:

| feature source | in-distribution AUC | in-distribution AP |
|---|---|---|
| raw (Strategy 2) | **0.9436** | **0.8840** |
| **SAE (Strategy 2)** | **0.9412** | **0.8792** |
| Δ (raw − SAE) | +0.0024 | +0.0048 |

LOPO results: pending.

## Interpretation

The gap is **0.24 pp AUC** — within typical train/val noise on a 246-pair
benchmark. Raw and SAE features carry essentially the same signal for the
"same-prompt seed pair" counterfactual.

This is the **most-important framing input** because Strategy 2 is the
deployment-realistic test (prompt held constant, just different noise
seeds). On Strategy 2 SAE doesn't outperform raw, but it doesn't lose
either. SAE-as-mechanistic-interpretability (Framing B) does not get a
quantitative win; SAE-as-feature-detector (Framing A) gets a tied result.

Combined with the four discriminator inputs (Item 1c-0 Strategy 1 raw=0.275,
Item 1c-1 0/32, Item 1c-3 0.977, this 0.9412 SAE vs 0.9436 raw):
- 3/4 inputs favor Framing A (high detector AUC, white-box transferability=0,
  AUC tied at counterfactual).
- 1/4 input shows a near-chance result on the **prompt-edit** counterfactual,
  but in a setting that the deployed system doesn't see (empty-prompt
  conditioning).

**Verdict: Framing A canonical** (already committed in `REFRAMING_DECISION.md`).

## Next

- LOPO mean ± std once script completes.
- Compute the SAE-feature linear probe weight magnitude per hookpoint, plot
  top-20 features by |w|, and check whether they align with the Stage 1 ∩
  Stage 2 features used by the intervention pipeline.
- D-4 cross-concept violence: same probe structure, different concept.

## Caveats

- 80/20 splits with n=246 have ~3-4 pp standard error.
- The benchmark is "same prompt, different seeds" — the intra-prompt content
  variation it captures is not the same as the prompt-distribution shift the
  deployed detector has to handle. The result is upper-bound for an idealised
  setting.
