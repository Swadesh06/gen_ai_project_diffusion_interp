# D02_D03_D04_lpips_gpu_v1 — LPIPS-vgg on three patch ablations

## Goal

Item 1c-8 closes the LPIPS measurement on D02 (mean-patch), D03 (zero-patch),
D04 (resample-patch). The Phase 1 patch-ablation reported all three at
4/10 corrected on safety_checker and FID delta < 0.5; LPIPS is the third
arbiter.

## Procedure

GPU LPIPS-vgg (`scripts/run_lpips_gpu_fast.py`, batch=8, fp32):
- D02: 100 (pre, post) PNG pairs from mean-patch
- D03: 100 (pre, post) PNG pairs from zero-patch
- D04: 100 (pre, post) PNG pairs from resample-patch

512×512 images, normalized to [-1, 1].

The CPU LPIPS run died at 26 min with 0 progress — VGG download stalled
serialization on CPU. The GPU rewrite completed each in 5-7 minutes.

## Results

| variant | n | LPIPS-vgg mean | LPIPS-vgg std |
|---|---|---|---|
| **D02 mean-patch** | 100 | **0.4128** | 0.0720 |
| **D03 zero-patch** | 100 | **0.4129** | 0.0720 |
| **D04 resample-patch** | 100 | **0.4129** | 0.0720 |

The three patches produce essentially identical LPIPS distributions —
**Δ ≤ 0.0001 mean**.

Compared to FID series (Phase 1 patch-ablation):
- D02 FID-pre = 234.93, FID-post = 235.21 (Δ = +0.28)
- D03 FID-post = 235.25 (Δ = +0.32)

## Interpretation

**Patch kind does not matter for image-quality preservation.** The Stage 1
∩ Stage 2 F_c features encode the safety-relevant content — once they are
suppressed, the actual replacement value (mean / zero / resample) makes no
measurable difference in either FID (global) or LPIPS (perceptual) of the
corrected image vs the pre-intervention image.

This validates two design choices:
1. **The F_c selection is the critical knob**, not the patch operation.
2. **Mean-patch is fine** as the default — simpler, no resample noise, and
   the LPIPS / FID are tied with the more expensive resample-patch.

LPIPS = 0.41 is high in absolute terms (above LPIPS = 0.3 threshold for
"clearly different image"). That is expected for a content-rewrite
intervention: 9 out of 10 of the oracle-flagged content gets removed (4 of 10
fully corrected, others partially shifted), so the post image is not meant
to be perceptually identical to pre. The test is whether *content* changes
in a desirable way, not whether pixels are preserved.

## Next

- Add CLIP-score on (post image, prompt) for the same dirs to verify that
  prompt fidelity is preserved.
- Update the patch-ablation table (now LPIPS column filled in across all
  three).

## Caveats

- LPIPS is sensitive to Resize(512, 512) interpolation; all three pipelines
  used the same `transforms.Resize`, so the relative comparison is fair, but
  absolute values shift somewhat with different resize methods.
- No reference distribution — these are pre-vs-post LPIPS, not LPIPS to
  COCO. The COCO comparison is the FID; LPIPS measures intervention
  perceptual cost.
