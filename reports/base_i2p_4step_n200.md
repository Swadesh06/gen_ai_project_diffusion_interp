# base_i2p_4step_n200 — SDXL Base 4-step CFG=7.5 pre-flag rate

## Goal
Phase 1b carry-over: SDXL Turbo 1-step pre-flag rate is only 7-12% on I2P-NSFW
(see contribution1_final.md). The Phase C extension uses SDXL Base with 4 DDIM
steps + CFG=7.5 to get a higher-quality render that should trip safety_checker
more often, providing a richer attack/intervention testbed.

## Setup
- Generator: SDXL Base 1.0 4-step DDIM CFG=7.5.
- Prompts: 200 I2P-NSFW (`category` ∈ {sexual, ...}) from the standard split.
- Seed offset: 9_000_000.
- Wall: 522 s; peak VRAM 7.6 GB.
- Scoring: CompVis safety_checker on CPU (batch=8 fp32; ran in cpu-worker
  alongside the GPU jobs).

## Headline

| metric | value |
|---|---|
| n total | 200 |
| n flagged | 37 |
| **flag_rate** | **0.185** |
| score mean | 0.185 |
| score min / max | 0.000 / 1.000 |

vs SDXL Turbo 1-step on the same prompt set:

| backbone | steps | CFG | pre-flag rate |
|---|---|---|---|
| SDXL Turbo | 1 | 0.0 | 0.10 (avg of A01/A02/A03 pre_flag rates) |
| **SDXL Base** | **4** | **7.5** | **0.185** |

~1.85× higher pre-flag rate at SDXL Base 4-step. Below the 30-50% expected
range — likely because (a) the I2P prompts are written for SD v1.5, not
SDXL, and (b) SDXL's native safety bias filters more at the diffusion
sampler stage than SD v1.5.

## Implication
Item-1 / Item-2 / Item-4 etc. were all measured with SDXL Turbo. For
ICLR-grade paper numbers we should re-do the headline cells with SDXL
Base (more representative of production deployment). The 522-s wall +
7.6 GB VRAM means a full re-do of A01 (200 prompts) + B01 (1000 prompts)
+ Item 5 (100 prompts) is on the order of an hour total; queue-able.

## Next
- Run the intervention pipeline (Item 5, mean-patch) on the 37 pre-flagged
  base images: `scripts/run_intervention.py --variant base
  --num-inference-steps 4 --guidance-scale 7.5 --n-prompts 200 ...`.
- Compare base-side post-flag rate / correction rate against turbo-side.
- Re-do the C-1 Square Attack against base-rendered seeds (higher quality
  = harder bypass).

## Artefacts
- `outputs/base_i2p_4step_n200/{i2p/, summary.json}` (200 images + safety
  sidecars).
- `logs/gen_base_i2p_4step_n200.log`, `logs/score_base_i2p.log`.
