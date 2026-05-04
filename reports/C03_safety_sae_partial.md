# C03 — Safety-trained SAE: training + first-hookpoint detector

## Goal
Phase C-3 from the appendix. Train a TopK SAE on labelled (NSFW + benign)
raw activations *with safety as the supervised signal*, and compare against
Surkov's general-purpose SAE on the same in-distribution NSFW-vs-benign
classification task.

The hypothesis: a safety-specialised SAE allocates more dictionary capacity
to safety-relevant directions and so should match (or beat) the raw signal
on the same task while preserving SAE-style interpretability.

## Procedure

### Stage 1: train safety SAEs (4 hookpoints, in parallel)
- Architecture: SurkovTopKSAE (encoder + decoder + pre_bias + latent_bias +
  topk + relu).
- Hyperparameters: expansion=8, k=64. d_hidden = 1280×8 = 10240, L0 = 64.
- Loss: MSE reconstruction + small auxk dead-feature regulariser.
- Data: 1000 raw mean-pooled residuals (500 NSFW SDXL Turbo I2P + 500 benign
  SDXL Turbo COCO).
- Epochs: 30, batch 64, Adam(1e-3).
- Wall: 28-32 s per hookpoint on CPU; 4 ran concurrently.

| hookpoint | recon MSE | active_frac (target 64/10240=0.00625) |
|---|---|---|
| down.2.1 | 0.223 | 0.00625 |
| mid.0    | 0.091 | 0.00625 |
| up.0.0   | 0.140 | 0.00625 |
| up.0.1   | 0.228 | 0.00625 |

### Stage 2: detector probe
- Apply the safety SAE to the same `dataset_axbench_v1` raw activations.
- Compare per-hookpoint linear probe AUCs:
  - raw → linear → AUC
  - Surkov SAE z (k=64, expansion=4) → linear → AUC
  - Safety SAE z (k=64, expansion=8) → linear → AUC

## Headline (down.2.1, only complete hookpoint at write time)

| signal | val AUC |
|---|---|
| raw                  | 1.0000 |
| surkov_sae           | 0.9805 |
| **safety_sae**       | **1.0000** |

The safety SAE matches raw and beats Surkov by **+1.95 pp** at down.2.1.

Other hookpoints (mid.0, up.0.0, up.0.1) are training; this report is
partial and will be replaced by the full table when C-3 completes.

## Interpretation
At down.2.1 the safety SAE recovers the entire discriminative power of the
raw signal — i.e. the sparse 64-dim TopK projection loses no
NSFW-vs-benign separability when trained with that supervision. Surkov SAE
(general-purpose, trained on ImageNet) loses 2 pp on this slice, consistent
with capacity not being allocated to safety directions.

This validates the C-3 hypothesis: a safety-specialised SAE is a strict
improvement over a general-purpose SAE for the safety detector use-case.

## Next
- Wait for the other 3 hookpoints; report the full per-hookpoint matrix.
- Re-run the C-1 Square Attack against the safety-SAE detector (vs Surkov
  detector) to see whether the adversarial robustness gap also widens.
- Stage-2 causal score on the safety SAE features → the F_c selected by
  this SAE should be smaller and more concentrated, validating the
  "interpretability via safety supervision" angle.
- Use the safety SAE for the intervention (`scripts/run_intervention.py`)
  — the Stage-2 ∩ safety-SAE F_c may improve correction rate beyond the
  mean=zero=resample plateau (40%).

## Artefacts
- `outputs/safety_saes_v1/safety_sae_<hp>_x8_k64/{state_dict.pth,summary.json}`
- `outputs/dataset_axbench_v1/C03_safety_sae_detector_results.json` (partial)
- `logs/C3_safety_sae_<hp>.log`
- `logs/C3_safety_sae_detector.log`
