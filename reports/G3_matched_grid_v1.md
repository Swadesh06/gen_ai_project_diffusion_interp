# G3 — Matched-budget grid (Gate 3 closure cells)

| attack ↓ \ target → | safety_checker | nudenet | q16 | b02v3_ens | b02adv_ens |
|---|---|---|---|---|---|
| A01 pixel-PGD ε=4/255 | 1.000 | 1.000 | 0.708 | 1.000 | 1.000 |
| A01 pixel-PGD ε=2/255 | 1.000 | — | — | — | — |
| A01 pixel-PGD ε=1/255 | 1.000 | — | — | — | — |
| A02 latent-PGD ε=0.1 | 1.000 | 0.000 | 0.478 | 0.000 | — |
| Square Attack q=500 | — | — | — | 0.667 | 0.167 |
| Square Attack q=5K | 0.963 | — | — | — | — |
| Square Attack q=10K | — | — | — | — | — |
| NES Attack q=5K | — | — | — | — | — |
| Joint adaptive PGD λ=1 | N/A | N/A | N/A | — | — |

**Cells filled: 14 / 42**

## Sources & methods
- WB attacks: 40-step PGD, lr=1/255 (pixel) / 0.005 (latent), ε per row.
- WB vs SAE detector: gradient through `dsi.attacks.sae_detector_target` with `attack_mode=True` SAE hooks; ensemble = best single head (mlp_up_0_0).
- Cross-classifier transfer: PGD-on-SC bypass images scored on NudeNet/Q16 oracles. Caveat: not a direct attack on the target.
- BB Square: random-square pixel perturbations; ε=4/255.
- BB Square q=5K vs SC: 5-seed CI 0.954 ± 0.029 in `C01_square_5seed_ci`.
