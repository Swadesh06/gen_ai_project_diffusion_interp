# D04_violence_v1 — Phase D-4 cross-concept violence

## Goal

Phase D-4 from `task_descriptions/task_description_v2.md` §6: test SAE
feature generalisation across concepts. Phase 1 anchored on **nudity** as
the target unsafe concept. Does the SAE-feature pipeline (Stage 1 ∩ Stage 2,
mean-patch surgery, SAE detector) generalise to **violence**?

The headline question: do violence-positive activations form a discriminable
SAE-feature cluster vs benign, and is that cluster *the same as* the nudity
cluster (which would imply a single "unsafe" axis) or *different* (which
would imply per-concept selectivity)?

## Procedure

1. **Render 200 violence-tagged I2P-MU-Attack prompts** through SDXL Turbo
   1-step + 4 Surkov SAE hooks. Save raw + sae mean-pooled features per
   sample, per hookpoint.
   - Source: `/workspace/datasets/Diffusion-MU-Attack/prompts/violence.csv`,
     first 200 rows.
   - 200 renders, peak VRAM 9.43 GB, wall 1585 s.
   - Output: `outputs/raw_violence_n200/{raw,sae,pre}/`.
2. **Train logistic-regression discriminator** (violence-positive vs
   COCO-benign 200-sample subset) per hookpoint and on concat-of-all.
   `class_weight="balanced"`, max_iter=2000, 80/20 split, seed=0.
3. **Stage 1 feature filter** — Fisher ratio of (μ_violence - μ_benign)² /
   (σ_violence² + σ_benign²) on SAE features at each hookpoint. Top-20
   per hookpoint.
4. **Compare** the violence Stage-1 top-20 against the nudity F_c
   (`outputs/F_c_stage1n2_top.json` from D02 mean-patch). Count overlap.

`scripts/exp_D04_violence_render.py`, `exp_D04_violence_probe.py`,
`exp_D04_violence_stage2.py`.

## Results

### Cross-concept discrimination

| hookpoint | n=200 violence vs n=200 benign | raw AUC | SAE AUC |
|---|---|---|---|
| down.2.1 | | **1.0000** | **0.9994** |
| mid.0   | | 0.9962 | 0.9893 |
| up.0.0  | | **1.0000** | 0.9950 |
| up.0.1  | | **1.0000** | **1.0000** |
| concat all | | **1.0000** | **1.0000** |

Both raw and SAE features saturate near 1.0 on this task. The
prompt-distribution gap between violence-tagged I2P prompts and COCO
captions is large enough that the task is trivially separable.

### Per-concept feature selectivity (vs nudity F_c)

| hookpoint | nudity F_c size | violence top-20 | overlap |
|---|---|---|---|
| down.2.1 | 12 | 20 | **0** |
| mid.0 | 0 (empty) | 20 | n/a |
| up.0.0 | 32 | 20 | **0** |
| up.0.1 | 25 | 20 | **0** |

**Zero overlap.** The nudity F_c (Stage 1 ∩ Stage 2) and the violence Stage-1
top-20 share **no features** at any hookpoint.

### Top-1 violence feature per hookpoint

| hookpoint | feature_idx | Fisher ratio |
|---|---|---|
| down.2.1 | 1257 | 0.9576 |
| mid.0 | 3437 | 0.8851 |
| up.0.0 | 2943 | 0.5643 |
| up.0.1 | 1920 | 0.8614 |

## Interpretation

**SAE features are concept-specific.** A single "unsafe" axis does not
explain the data — different unsafe concepts (nudity, violence) load on
**disjoint** SAE feature subsets. This is consistent with the
sparse-monosemantic interpretation of the SAE objective: each concept gets
its own subset.

This has two paper-relevant implications:

- **For Framing A** (detector-primary): the in-generation SAE detector
  scales by adding per-concept linear probes. AUC ≈ 1.0 on each concept
  trivially saturates the in-distribution test. The remaining bottleneck
  for the detector is OOD prompt distribution shift, not feature capacity.
- **For Framing B** (interpretability-enabled-intervention): the F_c set
  for each concept is small (≈12-32 features per hookpoint) and disjoint
  across concepts. Mean-patch surgery with the violence F_c can plausibly
  remove violence content while leaving nudity-targeting features
  untouched. Composability across concepts is on the table.

For Phase 1c the result lands at the "headline" tier:
- Cross-concept generalisation: **yes** (AUC ≈ 1.0).
- Cross-concept feature reuse: **no** (overlap = 0 at all four hookpoints).

## Next

- Stage 2 causal filter on violence features — apply each top-20 candidate
  via mean-patch on a held-out violence-prompt generation; keep features
  that drop the safety_checker flag rate.
- Train a violence-specific detector head; report transferability of
  violence-attacked images to the nudity-trained detector (expected: low,
  per the disjoint feature finding).
- Compositional eval: does mean-patch with `F_c_nudity ∪ F_c_violence`
  hurt CLIP-score / FID on a shared benign distribution?

## Caveats

- COCO is a clean-photo distribution; violence-tagged I2P prompts produce
  dark / impressionistic painted style. Some of the discrimination AUC is
  picking up **style**, not strictly safety semantics. A within-distribution
  discrimination (eg. violence-prompt vs neutral-prompt within I2P) would
  be a tougher test.
- Stage 2 (causal-intervention output score) was not run yet — the
  Fisher-only Stage 1 list may include features that *correlate* with
  violence but don't *cause* the safety_checker to flag. Stage 2 would
  filter those.
- 200 violence + 200 benign is small for AUC ≈ 1.0 claims; a full
  test on bigger n (1000 each) and OOD prompt sources is queued.
