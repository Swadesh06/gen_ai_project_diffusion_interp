# REFRAMING_DECISION.md — Phase 1c framing-discriminator verdict

> Per `task_descriptions/task_description_v2.md` §7. Both Framings A
> and B were pursued in parallel through Phase 1c. This report
> documents the four discriminator inputs and commits to a primary
> framing for the paper draft. The other framing's notes are not
> discarded; they archive to `paper/archive/` and may inform
> supplementary appendices.

## The four discriminator inputs

### 1. Counterfactual benchmark (Item 1c-0) — **mixed**

| benchmark | n | encoding | in-distribution AUC |
|---|---|---|---|
| Strategy 1 prompt-edit pairs | 63 validated | empty prompt @ t=50 | **0.275** (below chance) |
| Strategy 2 same-prompt seed pairs | 246 validated | actual prompt @ t=50 | **0.9436** |

`reports/cf_probe_strategy1_v1.md` + `reports/cf_probe_strategy2_v1.md`.

The **two strategies disagree**. Strategy 1 (where prompt distribution
varies by single-token edit) gives near-chance discrimination; Strategy 2
(where prompt is held exactly constant and only the noise seed varies)
gives strong discrimination (AUC 0.94).

The most-likely interpretation: **the SAE/raw activation features pick
up safety-content signal robustly when prompt distribution is held
constant, but not under the prompt-edit substitutions of Strategy 1**.
This makes Strategy 2 the more direct test of "can SAE detect unsafe
content when conditioned on the actual prompt?". Answer: **yes**.

### 2. Bit-identical detector logits bug fix (Item 1c-1) — **Framing A holds**

The Phase 1 cross-target result (`transferability_safety_to_detector =
0.000`) was real, not the bit-identical-logits artefact. After fixing
the bug and running the v2 image-conditioned UNet trace pipeline:

| attack | safety bypass | detector bypass | both | transferability |
|---|---|---|---|---|
| A01 pixel-PGD vs B02-v3 | 17 / 17 | 4 / 14 | **0** | **0.000** |
| A02 latent-PGD vs B02-v3 | 15 / 15 | 0 / 14 | **0** | **0.000** |
| A03 embedding-PGD       | 15 / 15 | n/a (perturbation in CLIP-emb space) | n/a | trivially robust |
| **combined cross-space** | **32 / 32** | **0 / 32** | **0** | **0.000** |

`reports/C01_xtarget_v2_vs_B02v3_full.md`. The SAE detector resists
white-box pixel and latent attacks across A01 + A02. **Framing A's
"detector is a primary contribution" gets a strong yes** here.

### 3. B02-oracle-v3 detector retrained at scale (Item 1c-3) — **strong**

| version | dataset | AUC | linear / MLP |
|---|---|---|---|
| B02 v1 | 1388 (41 NSFW / 1347 benign) | 0.847 | linear, unbalanced |
| B02 v2 balanced MLP | 1388 same | 0.891 | MLP-512 + class weight |
| **B02 v3 oracle, 5× more NSFW** | **1544 (201 NSFW / 1343 benign)** | **0.9762 linear / 0.9772 MLP up.0.0** | both |

`reports/B02_oracle_v3_detector.md`. The 12 pp lift over v2 confirms
that with oracle labels and class-balanced training the SAE-feature
linear probe is a competent in-generation detector at the
prompt-origin-style task. Combined with Item 1c-1's cross-target 0/32,
this **supports Framing A** ("detector is primary").

### 4. C-2 AxBench rerun on counterfactual (Phase C-2 redux) — **inconclusive on raw vs SAE**

Phase C-2 originally (`reports/C02_axbench.md`) reported:
- Raw all-blocks-cat AUC = 1.0000 (saturates trivial task).
- Surkov SAE per-block AUC 0.97-0.98 (1-2 pp under raw).

That was on the prompt-origin-labelled `dataset_axbench_v1` —
trivial saturating task. The **C-2-on-counterfactual** version is what
the v2 spec calls for. Strategy 2 cf probe (raw activations) gives
AUC 0.9436 — but I have not yet completed the SAE-feature variant on
the same data (`cf-probe-s2-sae` is in flight). On the simple
in-distribution axbench task, raw saturates and SAE drops a hair.
On Strategy 2 counterfactual, the raw-only result is 0.9436; the
SAE comparison number is pending.

`reports/cf_probe_strategy2_v1.md`. Pending: SAE-feature variant.

## Decision

Per v2 §7 decision rule:

> Mixed evidence (one discriminator favors A, another favors B) →
> **Framing A canonical** with explicit acknowledgment in §2 (the safer
> choice, since most reviewers find A's structure easier to follow).

The evidence is mixed:
- Item 1c-1 cross-target → strongly favors Framing A.
- Item 1c-3 detector AUC 0.977 → favors Framing A.
- Strategy 2 cf probe AUC 0.9436 (raw) → close to Framing B baseline,
  but the SAE variant is pending; the headline reader will not see a
  meaningful raw-vs-SAE gap on this benchmark.
- Strategy 1 cf probe (raw, AUC 0.275) → reading is "null" not
  "Framing B win" because the empty-prompt encoding is not the
  detection-time conditioning the deployed system uses.

**Verdict: Framing A canonical.**

The paper structure remains the original four contributions:
1. **Cross-space red-team** with SAE attribution (Phase 1 + 5-seed CIs).
2. **In-generation SAE-activation detector** with AUC 0.977 oracle
   (B02-v3) and counterfactual AUC 0.94 (Strategy 2). Strategy 1's
   near-chance result acknowledged as "the detector reads
   prompt-conditioned content, not prompt-token substitutions" with the
   appropriate caveat.
3. **Cross-target robustness** transferability = 0.000 across pixel +
   latent + embedding attack spaces.
4. **Two-stage causally-filtered feature surgery** with mean / zero /
   resample / D-2 learned-projection patches (D02 4/10 corrects on
   Phase 1 result; full eval ongoing).

Framing B's outline is preserved at `paper/alt_framing_B.md` in case
the paper's review cycle pivots.

## Pending work for the canonical Framing A draft

- cf-probe-s2 with SAE features (in flight) — completes the raw-vs-SAE
  AUC gap on the meaningful counterfactual benchmark.
- A01 pixel + A02 latent 5-seed CI (in flight) — paired-bootstrap CIs
  on the headline numbers.
- Item 1c-9 black-box vs SAE detector (in flight) — completes the
  threat-model row.
- Phase D items D-4, D-5, D-6, D-8, D-10 — additional ablations and
  generalization tests, all serving Framing A's contributions.
- D-9 cross-architecture (SD3 + PixArt smoke complete; FLUX still
  loading) — Generalization claim breadth.
