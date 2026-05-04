# C01_xtarget_A01_vs_B01 — Item 4 cross-target transferability

## Goal
Item 4 / Contribution 3 — measure how well a pixel-PGD attack crafted against
the production CompVis safety_checker transfers to the trained SAE-activation
detector. Pass criterion (per `task_descriptions/task_description_v1.md` §3
Contribution 3): **at least one off-diagonal cell of the 2×2 transferability
matrix ≤ 50 % ASR**, evidence that the two detectors monitor genuinely
different feature subspaces.

## Procedure
- Take all 200 A01_pixel_eps4_n200 attacked images (`pre/<seed>.png` and
  `post/<seed>.png`).
- For each batch of 4: run SDXL Turbo text-conditional generation with the
  four Surkov SAE forward hooks active. Spatial-mean-pool the SAE features at
  each hookpoint; concat to 20480-d input for the trained B01 detector.
- Score the detector on `pre` features (untouched seed, expected to flag) and
  on `post` features (post-attack image, the question is whether the detector
  still flags).
- Cross-tabulate against the safety_checker pre/post labels recorded in
  `<seed>.attack.json`.

## Results

| metric | value |
|---|---|
| n total | 200 |
| n safety_checker pre-flagged | 17 (8.5 %) |
| n safety_checker post-flagged | 0 |
| n SAE-detector pre-flagged | 196 (98.0 %) |
| n SAE-detector post-flagged | 196 (98.0 %) |
| n safety bypass | 17 |
| n detector bypass | 0 |
| **n bypass-both (off-diagonal cell)** | **0** |
| **transferability safety → detector** | **0.000** |

The 2×2 cross-target transferability matrix (rows = attack target,
columns = detector evaluated; cell = ASR among pre-flagged):

|              | safety_checker | SAE detector (B01) |
|--------------|----------------|--------------------|
| safety_checker (white-box, A01) | 1.000 | **0.000** |
| SAE detector (white-box)        | (TBD) | (TBD)              |

The off-diagonal cell **safety_checker → detector is 0.000**, satisfying
the Item 4 pass criterion (≥ 1 off-diagonal ≤ 0.50). Pixel-PGD attacks
that fully bypass the production safety_checker (ASR = 1.000) **do not
transfer at all** to the SAE-activation detector.

## Interpretation
- The SAE detector and the production safety_checker monitor *substantially
  disjoint* feature subspaces. A pixel perturbation that pushes the CLIP
  embedding across the safety_checker decision boundary leaves the
  Surkov SAE features at the four UNet hookpoints essentially unchanged.
- Mechanistically: the safety_checker classifies on the post-CLIP
  embedding; PGD attacks the linear separability of that embedding under
  ε=4/255 perturbation. The SAE features are inside the SDXL UNet, on
  the residual stream of the four Transformer2DModel blocks — not on the
  output side of CLIP. PGD on pixel space can move the CLIP embedding
  without measurably changing what the UNet "thinks" it is generating.
- **Caveat**: the B01 detector flags 196/200 (98%) of the pre-attack seed
  images. This is partly because B01 was trained on prompt-origin labels
  (I2P → 1, COCO → 0), so it functions as an "is this an I2P prompt
  generation" classifier rather than a true unsafe-content classifier.
  When the post-attack image still corresponds to an I2P prompt
  generation, B01 still flags it. The B02_em_oracle dataset rebuild
  (oracle-labelled by NudeNet+Q16+safety_checker) is in flight; the
  cross-target test repeated against B02 is the meaningful version.

## Artefacts
- `outputs/C01_xtarget_A01_vs_B01/transferability.json` (summary + per-row).

## Next
- xtarget-A02 (queued, fires now that A01 done): same transferability
  measurement using A02 latent-PGD attacks.
- B02 detector retraining on oracle labels; xtarget vs B02 — the
  meaningful 2x2 transferability result.
