# udatk safety scoring — Item 1c-4 rendering + initial flag rates

## Goal

Item 1c-4 — UnlearnDiffAtk migration to headline benchmark. Render the
nudity (n=143) and violence (n=200) splits with SDXL Base 4-step
(CFG=7.5, 512x512), oracle-score, report ASR.

## Procedure

- Source: `/workspace/datasets/Diffusion-MU-Attack/prompts/{nudity,violence}.csv`.
- SDXL Base 4-step, CFG=7.5, 512x512, batch=8.
- Render via `scripts/eval_unlearndiffatk.py gen --split <s>`.
- Score via `scripts/batch_safety_checker.py` (fp16, GPU).

## Results

### nudity (143 prompts, no intervention)

| oracle | flagged | rate |
|---|---|---|
| **safety_checker** | **53 / 142** | **37.3 %** |
| Q16 / NudeNet (cpu-workers) | (still labelling) | — |

Wall: render 1555 s, score 250 s.

### violence (200 prompts, no intervention)

| oracle | flagged | rate |
|---|---|---|
| **safety_checker** | **44 / 200** | **22.0 %** |
| Q16 / NudeNet | (cpu-workers labelling) | — |

Wall: render ~5 min, score 314 s.

## Interpretation

- **UnlearnDiffAtk-nudity has a 37.3 % safety_checker flag rate** at
  no-intervention SDXL Base 4-step. Higher than I2P-NSFW's 28.6 % on
  the same backbone (`reports/base_i2p_4step_n1000.md`), confirming
  UnlearnDiffAtk-nudity is a more concentrated NSFW signal — exactly
  what Item 1c-4 wants for the headline ASR table.
- **UnlearnDiffAtk-violence has 22.0 %** safety_checker flag rate.
  Lower than nudity (Q16 is the "right" oracle for violence, not
  safety_checker which is nudity-focused). Once cpu-workers finish
  Q16 labelling, the Q16 flag rate will likely be substantially
  higher.

## Next

- Wait for cpu-worker NudeNet + Q16 sweep on udatk_{nudity,violence}.
- Apply Contribution 4 interventions (D02 mean-patch, D-2 learned
  projection, D03 zero, D04 resample) on udatk-nudity → measure
  ASR drop.
- Compare to SAeUron's published Table 1 (their reproducible repro
  on UnlearnDiffAtk-nudity is queued).
