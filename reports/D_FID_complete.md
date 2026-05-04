# D-FID complete — three-way patch-kind FID ablation

## FID at n=100, vs COCO val 5K

| run | F_c | patch | FID-pre | FID-post | Δ post-pre |
|---|---|---|---|---|---|
| D02 | Stage1∩Stage2 (69) | mean      | 234.93 | **235.21** | +0.28 |
| D03 | Stage1∩Stage2 (69) | zero      | (same pre) | **235.25** | +0.32 |
| D04 | Stage1∩Stage2 (69) | resample  | (same pre) | **235.21** | +0.28 |

All three intervention primitives produce **essentially identical** post-FID
(within 0.04 of each other; within 0.4 of pre-FID). This sharpens the
patch-kind ablation conclusion from the safety-checker side:

- **Safety correction rate** (D02_D03_D04_patch_ablation): 0.40 each.
- **FID delta vs un-intervened**: 0.28 mean / 0.32 zero / 0.28 resample.

The mean-vs-zero theoretical advantage is **not present** at n=100 SDXL Turbo
1-step. F_c quality (Stage 1 ∩ Stage 2) dominates patch primitive choice on
both axes.

## Implication for paper
Mean-patching is the headline intervention because it preserves the
*per-feature mean-activation manifold* (cleanest theoretical justification,
matches the DSG / Arad mean-ablation literature) — but on this slice mean,
zero, and resample are operationally interchangeable. The intervention
ablation table for the paper:

| primitive | safety correction | FID Δ | wall (s) |
|---|---|---|---|
| mean      | 0.40 | +0.28 | 280 |
| zero      | 0.40 | +0.32 | 305 |
| resample  | 0.40 | +0.28 | 302 |

## Next
- Re-run at n=500 + scale to SDXL Base 4-step CFG=7.5 (gen-base-i2p done;
  scoring in progress) for a higher-resolution test of the FID claim.
- LPIPS-vgg paired-distance is computing for D02 (D03/D04 were killed for
  CPU contention; will restart serially).
- Combine mean patch with the safety SAE Stage 2 F_c (C-3 ongoing) — F_c
  selected by safety SAE may be smaller and improve correction rate above
  the 0.40 plateau.

## Artefacts
- `outputs/D0{2,3,4}_*/fid_post.json`
- `outputs/D02_*/fid_clipscore_pre.json` (FID only; CLIP interrupted earlier)
