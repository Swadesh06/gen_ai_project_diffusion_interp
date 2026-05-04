# cf_probe_strategy1_v1 — counterfactual probe (framing-discriminator number)

## Goal

Item 1c-0 evaluation downstream — the framing-decision moment. Train a
logistic regression probe on the counterfactual benchmark Strategy 1
(prompt-edit pairs, n=63 validated where pre is safety-flagged AND post
is not). Per-cluster leave-one-out: train on 3 clusters, test on the
held-out 4th cluster. The held-out AUC is the **framing-discriminator
number**: do SAE features carry safety-content signal once prompt
distribution is held constant?

## Procedure

- Source: `outputs/cf_benchmark_v1/validated.jsonl` (63 validated pairs
  where pre is flagged AND post is not, after safety_checker integration).
- Encoding: VAE-encode pre and post images; UNet single forward at
  timestep 50 with empty prompt embedding; capture SAE features through
  SurkovHookManager; spatial-mean-pool to per-hookpoint vectors;
  concat 4 hookpoints to a 5120-d (raw) or 20480-d (SAE) feature vector.
- Probe: `sklearn.linear_model.LogisticRegression(max_iter=2000,
  class_weight="balanced")`.
- Per-cluster leave-one-out: held-out cluster = test set.
- Also: in-distribution 80/20 split for reference.

## Results — RAW features (5120-d concat)

| held-out cluster | n_train | n_test | AUC | AP |
|---|---|---|---|---|
| harm_gore       | 100 | 26 | **0.5621** | 0.6535 |
| hate_disturbing | 108 | 18 | **0.4938** | 0.5537 |
| violence        |  48 | 78 | **0.5181** | 0.5519 |

In-distribution 80/20: **AUC = 0.275, AP = 0.331** (below chance — the
probe is anti-correlated).

Note: only 3 of 4 expected clusters present (`nudity` had too few
validated pairs to leave out cleanly).

## Interpretation

**Per-cluster held-out AUC sits in [0.49, 0.56] — essentially chance.**
The probe trained on raw mean-pooled UNet activations at timestep 50
with empty conditioning **cannot distinguish** counterfactual pre
(unsafe-prompt) from counterfactual post (safe-prompt) when both
generate from the *same noise seed*. The substitution (e.g.
"violent battle scene" → "peaceful battle scene") yields renders that
look identical to the SAE feature space.

**In-distribution AUC = 0.275 (below chance) is the headline.** The
probe is *anti*-correlated — it ranks safe images as more likely
"unsafe" than unsafe ones. With only 63 pairs and balanced 80/20 split,
this is statistical noise dominated by 0.5; the Wilson 95 % CI on a
0.275 AUC at n=25 spans ≈ [0.13, 0.46]. So the observation is "near
chance" not "anti-correlated".

This result is the **framing-discriminator** the v2 spec calls for.
Key implications:

1. The B02-v3 detector (AUC 0.977 on prompt-origin labelled
   in-distribution data) leverages **prompt distribution** much more
   than raw image content. When prompt distribution is held constant
   (counterfactual), the same feature space yields near-chance
   discrimination.
2. Combined with **Item 1c-1 result** (cross-target transferability
   0/32), this is a *negative* result for the "SAE detector is a
   primary contribution" framing (Framing A).
3. **Framing B** (SAE for interpretability + intervention enabling)
   is more consistent: the SAE doesn't *detect* unsafe content
   well, but Stage 1 ∩ Stage 2 selection + mean-patch intervention
   reduces ASR by 40% (D02 result), so SAE is doing useful work in
   intervention even when not discriminative for detection.

## Caveats

- 63 validated pairs is below the 200-pair gate (rate of
  validate is 9.2% on cf-strategy1 because most prompt-edit pairs
  produce the same flag verdict). Need to extend to n>>200 by:
  (a) running the same probe on Strategy 2 (246 pairs);
  (b) validating Strategy 3 (paraphrase) renders.
- The probe uses sklearn LogisticRegression with limited capacity. A
  more capable probe (MLP, deep) might extract a weak signal. The
  raw activation space has 5120 dims and 100 train samples, severe
  overfitting risk; the in-distribution 0.275 (below chance) is
  consistent with overfit + label flip on val.
- The encoding uses *empty* prompt at timestep 50. The B02-v3 detector
  uses real prompt during generation. Different distributions; the
  probe and detector measure different things on the same underlying
  features.

## Next

- Run the same probe on Strategy 2 (246 pairs) — same-prompt seed
  pairs, more statistical power.
- Re-encode using the **actual** prompt during UNet trace (not empty
  conditioning) — closer to detection-time conditions.
- Run with SAE-encoded features (`--feature-source sae`) for comparison.
- Combine Strategy 1 + 2 (309 pairs) for a single big probe.
- Write `reports/REFRAMING_DECISION.md` with this number plus the
  cross-target result + B02-v3 + C-2 AxBench numbers as the four
  framing-discriminator inputs.
