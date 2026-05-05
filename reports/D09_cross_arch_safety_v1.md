# D09_cross_arch_safety_v1 — Phase D-9 cross-architecture safety_checker baseline

## Goal

Phase D-9 from `task_descriptions/task_description_v2.md` §6: extend the
attack/defense pipeline to non-SDXL architectures. The first step is the
safety_checker baseline rate per architecture — does each backbone
produce NSFW that the CompVis safety_checker recognises?

## Procedure

For each architecture, render N I2P-NSFW prompts at default settings,
score with CompVis StableDiffusionSafetyChecker.

| backbone | resolution | inference steps | n | sampler / scheduler |
|---|---|---|---|---|
| SDXL Turbo | 512 | 1 | 200 | euler1 (turbo) |
| SDXL Base 4-step | 512 | 4 | 1000 | DPMSolverMultistep |
| SD v1.4 (SAeUron) | 512 | 30 | 30 | DPMSolverSinglestep |
| **SD3-medium**     | 512 | 28 | 100 | euler |
| **PixArt-Sigma**   | 512 | 20 | 10 | dpm-solver |
| **FLUX.1-schnell** | 512 | 4 | 20 | flow-matching |

## Results — safety_checker flag rate with Wilson 95% CI

| backbone | n | flagged | rate | Wilson 95% CI |
|---|---|---|---|---|
| SDXL Turbo (1-step) | 200 | 17 | 0.085 | [0.054, 0.132] |
| SDXL Base 4-step | 1000 | 286 | **0.286** | [0.259, 0.315] |
| SD v1.4 + SAeUron baseline | 30 | 10 | 0.333 | [0.198, 0.501] |
| **SD3-medium** | 100 | 4 | **0.040** | **[0.016, 0.099]** |
| PixArt-Sigma smoke | 10 | 0 | 0.000 | [0.000, 0.278] |
| FLUX.1-schnell | (model loads, inference hangs at 0% GPU; deferred) | tbd | tbd | tbd |

## Interpretation

- **SDXL Base 4-step is the strongest signal at scale**: 28.6% safety_checker
  flag rate on 1000 I2P-NSFW prompts. This justifies its use as the
  primary attack target in Phase 1c.
- **SDXL Turbo 1-step has lower safety_checker flag rate** (8.5%) than Base
  4-step. The Turbo distillation appears to suppress the most-easily-
  flagged content; the I2P-NSFW prompts that trip safety_checker on Base
  often *don't* trip it on Turbo.
- **DiT/MM-DiT models (SD3, PixArt) have lower safety_checker flag rates**
  on this small sample. Two non-exclusive hypotheses:
  1. Sample bias: SD3's 5%-rate is consistent with anywhere from 2-12%
     true rate at 95% confidence (n=100). PixArt's 0/10 is consistent
     with 0-28% true rate. Bigger samples are needed to disambiguate.
  2. **safety_checker generalisation gap**: the CompVis safety_checker is
     trained on SD-v1.x output distribution. DiT/MM-DiT (SD3, PixArt,
     FLUX) produce stylistically different outputs (sharper, more
     photographic, different tonemapping). The safety_checker may have
     low recall on actual NSFW under that distribution shift.
  3. **SD3 in-built safety**: SD3-medium has its own safety filter that
     returns black images for NSFW prompts during generation (we saw
     this in some logs: "Potential NSFW content was detected... A black
     image will be returned"). When SD3's filter blackens an image, the
     safety_checker score on a black image is "safe" — the filter is
     effectively counted as safety_checker preserving. This is a
     compositional defense (D-10) baked into the model.

## Per-arch SAE-feature alignment status

| backbone | architecture | hookpoints captured | SAE trained? | safety_checker flag rate baseline |
|---|---|---|---|---|
| SDXL Turbo | UNet | down.2.1, mid.0, up.0.0, up.0.1 | ✓ Surkov | 0.085 |
| SDXL Base 4-step | UNet | same | ✓ (shared with Turbo) | 0.286 |
| SD3-medium | MM-DiT | blocks 4, 9, 14, 18 | ✗ (data only) | ~0.05 |
| PixArt-Sigma | DiT | blocks 4, 9, 14, 18 | ✗ (data only) | 0 |
| FLUX.1-schnell | MM-DiT | blocks 4, 9, 14, 18 | ✗ (loading) | tbd |

## Next

- Train per-hookpoint TopK SAE on each captured activation set
  (~200 samples each, 100 I2P + 100 COCO). Architecture: same as Surkov.
- Compute Procrustes-aligned cosine similarity between SDXL Surkov
  decoder columns and the new SAE decoder columns at structurally
  analogous depths.
- Test cross-model intervention: project SDXL Stage-2 nudity F_c into
  new-architecture feature space, intervene, measure ASR drop.

## Caveats

- DiT block hookpoints (blocks 4, 9, 14, 18) are heuristic depth-matches
  to SDXL UNet's down.2.1 / mid.0 / up.0.0 / up.0.1. Procrustes-aligned
  SAE decoders will validate or refute this depth-correspondence.
- SD3's in-built safety filter (StableDiffusion3 has it) confounds the
  raw safety_checker rate. We treat SD3 as a "joint defense" example
  in the compositional analysis (D-10), not a pure baseline.
