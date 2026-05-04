# base_i2p_4step_n1000 — Item 1c-7 SDXL Base 4-step rerun

## Goal

Phase 1 used SDXL Turbo 1-step which trips safety_checker on only **8.5%**
of I2P-NSFW renders. Item 1c-7 spec: rerun with SDXL Base 4-step at
CFG=7.5 to lift the pre-flag rate (target ≥ 25%, gate ≥ 25%). Higher
base rate → tighter CIs at the same n.

## Procedure

- 1000 I2P-NSFW prompts (`AIML-TUDA/i2p`, label=nsfw subset).
- SDXL Base 1.0 fp16 + madebyollin/sdxl-vae-fp16-fix.
- 4 inference steps, CFG = 7.5, 512x512, batch_size = 8.
- Wall: ~10 min on the new RTX PRO 6000 Blackwell (vs 522 s for n=200
  on the old RTX Pro 4500 Blackwell at 32 GB).
- Output: `outputs/base_i2p_4step_n1000/i2p/<seed>.png`.
- Oracle scoring via `batch_safety_checker.py` (fp16, batch 32) +
  cpu-worker pool (NudeNet + Q16). All 1000 PNGs labelled.

## Results

| oracle | flag count | flag rate |
|---|---|---|
| **CompVis safety_checker** | **286 / 1000** | **28.6 %** |
| Q16 | 11 / 1000 | 1.1 % |
| NudeNet | 0 / 1000 | 0.0 % |
| **any (union)** | **292 / 1000** | **29.2 %** |

`outputs/base_i2p_4step_n1000/summary_eval.json`.

vs Phase 1 n=200 SDXL Turbo 1-step (`reports/PHASE_1_FINAL.md`):
- Turbo 1-step pre-flag rate: 8.5 % (17/200).
- Base 4-step pre-flag rate: **28.6 %** = **3.4× lift**.

## Interpretation

**Item 1c-7 gate ≥ 25 % met (28.6 %)**, target 40 % approached but
not met. SDXL Base 4-step at CFG=7.5 substantially raises the
pre-flag rate, giving any downstream attack/detector experiment ~3.4×
more "real" NSFW content per fixed prompt budget. Tighter CIs at
n = 500 with this regime than with Turbo 1-step at the same n.

NudeNet returned 0/1000 flags. Either NudeNet is conservative on
SDXL Base 4-step's compositional outputs, or the I2P NSFW prompts
that produce nudity are a small subset (most I2P NSFW is violence /
gore / hate-disturbing per the original I2P category breakdown).
The safety_checker is the dominant signal at this regime.

Q16 1.1 % is suspiciously low. Q16 should fire on disturbing /
violent content. May be a Q16 wrapper bug; investigation queued.

## Next

- Re-run all Contribution 1 attacks (A01 / A02 / A03) on the SDXL
  Base 4-step regime so the headline ASR is reported on the
  higher-flag-rate base. Currently A01/A02/A03 are at SDXL Turbo
  1-step.
- Item 1c-6 5-seed CIs at n=500 on SDXL Base 4-step regime.
- Reconfirm Q16 wrapper isn't silently returning safe.
