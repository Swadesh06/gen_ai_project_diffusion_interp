# Patch-kind FID ablation — D02 / D03 / D04 vs un-intervened pre

## Setup
clean-fid (Inception-V3, 2048-d) vs COCO val-2017 (5000 ref images), n=100 per
post folder. SDXL Turbo 1-step. Same 100 I2P-NSFW prompts + same seeds across
the three intervention runs.

## FID

| run | F_c | patch | dir | FID ↓ |
|---|---|---|---|---|
| D02 | Stage1∩Stage2 (69) | mean      | pre  | 234.93 |
| D02 | Stage1∩Stage2 (69) | mean      | post | 235.21 |
| D03 | Stage1∩Stage2 (69) | zero      | post | 235.25 |
| D04 | Stage1∩Stage2 (69) | resample  | post | (queued) |

Δ post − pre:
- mean      : +0.28
- zero      : +0.32

## Interpretation
At n=100 the absolute FID is high (~235) because the reference is COCO val
photos and SDXL Turbo 1-step renders are stylistically different — that's
the same baseline noise floor for all three runs. The relevant signal is the
post-vs-pre delta.

Both mean and zero patches preserve image-distribution structure within
0.4 FID of the un-intervened generation. Mean wins zero by 0.04 FID
(within sampling noise at n=100). The earlier hypothesis that mean ≫ zero
on FID is **not observed at this slice** — the F_c features are sparse
enough (69 features × 5120-D = 1.3% of the latent surface) that nulling
them out doesn't visibly degrade overall image statistics.

## Implication for the paper
The "mean preserves benign distribution structure" claim from the spec is
quantitatively a FID delta of < 0.5 vs the no-defense baseline at this
scale. Larger n (say 1000) and a per-prompt LPIPS / DreamSim measurement
will sharpen the picture; LPIPS computed in parallel.

## Next
- D04 resample post FID (queued).
- LPIPS-vgg on the (pre, post) pairs for each of D02/D03/D04 (running on CPU
  in parallel; results land in `outputs/D0<x>/lpips.json`).
- Repeat at n=500 + scale to SDXL Base 4-step (gen-base-i2p running, 200
  prompts at 4-step CFG=7.5) for higher-quality renders.

## Artefacts
- `outputs/D02_stage1n2_meanpatch_n100/fid_post.json` `fid_clipscore_pre.json`
- `outputs/D03_stage1n2_zeropatch_n100/fid_post.json`
- `outputs/D04_stage1n2_resamplepatch_n100/fid_post.json` (queued)
