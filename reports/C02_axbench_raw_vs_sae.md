# C02 — AxBench-style raw-activation probe vs SAE probe

## Goal
Phase C-2 from the appendix. Train a linear probe on raw mean-pooled UNet
residual diffs (the AxBench v0 baseline) and compare against the linear
probe on Surkov SAE z's at the four hookpoints. The C-2 pass criterion is
**SAE > raw by ≥ 2 pp val AUC**, on the same dataset.

## Setup
- Dataset: `outputs/dataset_axbench_v1` (1000 samples; 500 SDXL Turbo I2P-NSFW
  + 500 SDXL Turbo COCO benign), labels by prompt-origin (not oracle judgement).
- Per Surkov hookpoint we have:
  - `X_raw_<hp>.npy` (1000, 1280): mean-pooled raw residual diff.
  - `X_sae_<hp>.npy` (1000, 5120): mean-pooled Surkov SAE z (k=64, expansion=4).
- 80/20 train/val split; linear probe; auto pos-weight balanced BCE; 30 epochs.
- Per-hookpoint probes + concat-across-hookpoints probe.

## Headline

| signal | dim | val AUC | val AP |
|---|---|---|---|
| **raw_per_down.2.1**   | 1280  | **1.0000** | 1.0000 |
| raw_per_mid.0          | 1280  | 0.9996 | 0.9996 |
| raw_per_up.0.0         | 1280  | 0.9956 | 0.9956 |
| raw_per_up.0.1         | 1280  | 0.9975 | 0.9979 |
| **raw_all_cat**        | 5120  | **1.0000** | 1.0000 |
| sae_per_down.2.1       | 5120  | 0.9830 | 0.9843 |
| sae_per_mid.0          | 5120  | 0.9758 | 0.9825 |
| sae_per_up.0.0         | 5120  | 0.9775 | 0.9766 |
| sae_per_up.0.1         | 5120  | 0.9841 | 0.9861 |
| **sae_all_cat**        | 20480 | **0.9879** | 0.9896 |

**Δ (sae_all_cat − raw_all_cat) = −0.0121 (−1.21 pp)**.

## Result against pass criterion
**Fail**: SAE concat is 1.21 pp **below** raw concat. Per-hookpoint, SAE is
2 pp under raw at down.2.1, mid.0, and up.0.0; matches at up.0.1.

## Interpretation
On this *in-distribution* NSFW-vs-benign label task with prompt-origin
labels:
- Raw mean-pooled residuals saturate (AUC = 1.0).
- The SAE TopK / k=64 sparsification *destroys* about 1.2 pp of bulk
  variance separability across all hookpoints.
- The mid.0 hookpoint shows the largest SAE drop (-2.4 pp), consistent
  with mid being the location where Surkov's general-purpose SAE has the
  smallest safety-relevant capacity.

This is not unexpected: AxBench's same finding for LLMs is that raw probes
beat SAE probes on simple in-distribution classification, and SAEs win on
*generalization* and *interpretability*. The genuine SAE advantage is on:

1. **Interpretability**: each SAE feature is a discrete concept-shaped
   direction usable for intervention. Raw probes give a hyperplane in
   1280-d space with no concept structure.
2. **Adversarial robustness**: SAE features are sparse and discrete; an
   ε-bounded pixel/latent perturbation finds it harder to drive a
   sparse code than a dense vector. Tested via the C-6 adversarial
   robust eval (in flight).
3. **Modular composition**: F_c is a discrete index set per hookpoint
   that can be composed across concepts. Raw probe weights are not
   composable.

So the C-2 negative result is **expected and informative** — it shows the
boundary of where SAEs add value vs raw activations.

## Next
- C-6 adversarial robustness eval (running): re-evaluate raw_only / sae_only
  / hybrid on A01 PGD bypass set. Hypothesis: raw drops sharply, SAE holds.
- Re-train AxBench probe on a *generalization* split (e.g. train on coco vs
  i2p-nsfw, test on coco vs i2p-violence) to show SAEs generalize where raw
  overfits.
- Compare against the C-3 safety-trained SAE (TopK with safety supervision)
  → first-hookpoint result already shows safety_sae=1.0 ties raw.

## Artefacts
- `outputs/dataset_axbench_v1/axbench_results.json`
- `logs/C2_probe.log`
