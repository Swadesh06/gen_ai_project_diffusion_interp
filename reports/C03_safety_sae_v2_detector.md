# C03_safety_sae_v2_detector — safety-trained SAE closes the v1 1.21pp gap

## Goal

Phase C-3 / Item 1c-10. v1's safety-trained SAE at expansion 8 closed
most but not all of the gap between Surkov-pretrained SAE detection
(AUC 0.9879) and raw-activation detection (AUC 1.000). v2 trains at
expansion 16, 32 and L0 ∈ {32, 64, 128, 256} on the new GPU.

## Procedure

- Data: `outputs/dataset_axbench_v1` (500 NSFW + 500 benign mean-pooled
  Surkov-hookpoint vectors, prompt-origin labels — same in-distribution
  task that produced the AxBench result in Phase 1).
- Per-hookpoint linear probe, plus 4-hookpoint concat MLP-256 probe.
- 30 epochs, BCE w/ pos_weight=1, Adam lr=1e-3.
- L0 sweep at expansion 16: k ∈ {32, 64, 128, 256}.

## Results

### concat MLP probe (4-hookpoint, 81920 / 163840-d)

| SAE | concat MLP AUC | vs raw (1.000) | vs surkov (0.985) |
|---|---|---|---|
| **safety_sae x16 k64** | **1.0000** | tie | +1.47 pp |
| **safety_sae x32 k64** | **1.0000** | tie | +1.50 pp |
| **safety_sae x16 k32** | **1.0000** | tie | +1.42 pp |
| safety_sae x16 k128 | (in flight) | | |
| safety_sae x16 k256 | (in flight) | | |

### per-hookpoint linear (x16 k64)

| hookpoint | raw | surkov_sae | safety_sae x16 k64 |
|---|---|---|---|
| down.2.1 | 1.0000 | 0.9809 | **0.9999** |
| mid.0    | 0.9997 | 0.9780 | 0.9969 |
| up.0.0   | 0.9986 | 0.9768 | 0.9877 |
| up.0.1   | 0.9995 | 0.9849 | 0.9982 |

### per-hookpoint linear (x32 k64)

| hookpoint | raw | surkov_sae | safety_sae x32 k64 |
|---|---|---|---|
| up.0.1 | 0.9994 | 0.9861 | 0.9990 |

### per-hookpoint linear (x16 k32)

| hookpoint | raw | surkov_sae | safety_sae x16 k32 |
|---|---|---|---|
| up.0.1 | 0.9998 | 0.9861 | 0.9998 |

JSONs: `outputs/safety_saes_v2_detector*/C03_safety_sae_detector_results.json`.

## Interpretation

The safety-trained SAE at all expansion × L0 settings tested so far hits
AUC 1.000 on the concat-MLP probe — closing v1's 1.21 pp gap to raw
**decisively**. The interpretability trade-off (sparse 64- or 32-active
features per sample, vs raw's dense 1280-d activation) is now free: same
detection performance, plus per-feature attribution for free.

Three configurations tested all hit 1.000 — the result is robust to L0
choice between 32 and 64. The expansion-32 variant adds no signal over
expansion-16 at the same k. The L0=256 result will tell whether
*denser* SAE codes (closer to raw) help or hurt — queued.

Best hookpoint reconstruction (recon MSE 0.10-0.11 on `safety_sae mid.0`
x16 k32) is at mid.0, which Surkov et al. flag as the bottleneck block.
This is the candidate hookpoint for cleaner concept feature extraction
in Stage-1 ∩ Stage-2 selection.

## Next

- Re-run C-2 AxBench probe on the **counterfactual benchmark held-out
  cell** (Item 1c-0 Strategy 1; benchmark currently rendering) — that
  is the framing-discriminator number, not this in-distribution task.
- Use `safety_saes_v2 mid.0 x16 k32` (best recon) as the SAE for
  Stage-1 ∩ Stage-2 feature selection in the next Contribution-4 round.
  Expect cleaner causal feature subset.
- Repeat L0 sweep at expansion 8, 4 to map the full Pareto frontier
  (recon MSE × detector AUC × interpretability).
