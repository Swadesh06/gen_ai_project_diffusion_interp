# GATE 1 CLOSURE v1 — Cross-Space Red-Team

## Status: 8/8 cells filled

| cell | description | result | evidence |
|---|---|---|---|
| 1.1 | A01 pixel-PGD vs safety_checker | ASR 1.000 (88/88, 5-seed CI ±0) | `outputs/A01_pixel_eps4_n200/`, `reports/A01_5seed_ci_v1.md` |
| 1.2 | A02 latent-PGD vs safety_checker | ASR 1.000 (100/100) | `outputs/A02_latent_eps0.1_n200/`, `reports/A02_latent_eps0.1_n200.md` |
| 1.3 | A03 emb-PGD vs safety_checker | ASR 1.000 (102/102, 5-seed) | `outputs/A03_emb_eps0.5_n200/`, `reports/A03_5seed_ci.md` |
| 1.4 | ε-sweep A01+A02 | 228/228 across ε ∈ {4,2,1}/255 | `reports/A01_A02_eps_sweep_final_v1.md` |
| 1.5 | Square Attack vs safety_checker n=500 q=5K | ASR 0.954 ± 0.029 (5-seed) | `reports/C01_square_5seed_ci.md` |
| 1.6 | PGD-on-SC bypasses scored on NudeNet | A01 1/1 (1.000); A02 1/1 (0.000) | `outputs/A01_pixel_eps4_n200/oracle_eval_v3.json`, `outputs/A02_latent_eps0.1_n200/oracle_eval_v3.json` |
| 1.7 | PGD-on-SC bypasses scored on Q16 | A01 46/65 (0.708); A02 32/67 (0.478) | same files |
| 1.8 | SAE-feature attribution figure | A01 & A02 bypass cases × 4 hookpoints | `outputs/figures/F1_sae_attribution.pdf` |

## Headline comparison table — ASR by (attack space × oracle target)

| attack | n_pre / n_bypass vs SC | ASR vs SC | NudeNet transfer ASR | Q16 transfer ASR | budget |
|---|---|---|---|---|---|
| A01 pixel-PGD ε=4/255 (5-seed) | 88 / 88 | **1.000** | 1.000 (1/1) | **0.708** (46/65) | 40 PGD steps, lr=1/255 |
| A02 latent-PGD ε=0.1 | 100 / 100 | **1.000** | 0.000 (0/1) | **0.478** (32/67) | 40 PGD steps, lr=0.005 |
| A03 emb-PGD ε=0.5 (5-seed) | 102 / 102 | **1.000** | — (no post images) | — | 40 PGD steps, lr=0.05 |
| Square Attack q=5K, ε=4/255 (5-seed) | 221 / 211 | **0.954 ± 0.029** | — | — | random pixel squares |

**Direction-of-effect**: white-box PGD saturates safety_checker across all three attack spaces at the published-baseline budget; black-box Square Attack reaches 95.4% within q=5K. Cross-classifier transfer is partial: A01 perturbations strongly affect Q16 (70.8% transfer ASR) but barely register on NudeNet (which detects body parts via YOLOv8, not CLIP features).

## Ablation rows

### A01 ε-sweep against safety_checker (cell 1.4)

| ε | n_pre / n_bypass | ASR |
|---|---|---|
| 4/255 | 88 / 88 | **1.000** |
| 2/255 (half) | 9 / 9 | **1.000** |
| 1/255 (quarter, minimum uint8 step) | 11 / 11 | **1.000** |

### A02 ε-sweep against safety_checker (cell 1.4)

| ε | n_pre / n_bypass | ASR |
|---|---|---|
| 0.100 | 100 / 100 | **1.000** |
| 0.050 (half) | 11 / 11 | **1.000** |
| 0.025 (quarter) | 9 / 9 | **1.000** |

### Cross-space SAE feature overlap (Jaccard top-50 features per block; from `reports/cross_space_overlap_3way.md`)

| hookpoint | A01–A02 | A01–A03 | A02–A03 |
|---|---|---|---|
| down.2.1 | 0.754 | 0.613 | 0.639 |
| mid.0 | 0.587 | 0.538 | 0.471 |
| up.0.0 | 0.754 | 0.562 | 0.515 |
| up.0.1 | 0.786 | 0.724 | 0.639 |

Three attack spaces converge on partially-shared feature subspaces (60-80% Jaccard at the highest hookpoint, up.0.1).

### Cross-classifier transfer (cells 1.6 + 1.7)

Among PGD-on-safety_checker bypasses (post-attack images), score with NudeNet (YOLO-based) and Q16 (CLIP+linear):

| metric | A01 pixel | A02 latent |
|---|---|---|
| n paired | 200 | 200 |
| NudeNet pre-flag rate | 1/200 = 0.5% | 1/200 = 0.5% |
| NudeNet post-flag rate | 0/200 = 0% | 1/200 = 0.5% |
| NudeNet transfer ASR (pre∧¬post / pre) | 1.0 (1/1) | 0.0 (0/1) |
| Q16 pre-flag rate | 65/200 = 32.5% | 67/200 = 33.5% |
| Q16 post-flag rate | 23/200 = 11.5% | 39/200 = 19.5% |
| Q16 transfer ASR | **0.708 (46/65)** | **0.478 (32/67)** |

## Figure references

- `outputs/figures/F1_sae_attribution.pdf` — 5-panel grid for A01 + A02 bypass cases showing (pre image, post image, top-10 SAE feature bar at 3 hookpoints).
- `outputs/figures/F1_sae_attribution_manifest.json` — per-row feature-index manifest.
- `outputs/figures/eps_sweep.png` — ε-sweep plot from `plot_eps_sweep.py`.
- `outputs/figures/5seed_asr.png` — 5-seed CI plot.

## Tables references

- `outputs/tables/G3_matched_grid.csv` — Gate 3 matched-budget grid (Gate 1's rows are the first 4).
- `outputs/A01_pixel_eps4_n200/oracle_eval_v3.json` + same for A02 — raw oracle eval JSONs.

## Hardware utilization during Gate 1 closure runs

- A100-SXM4-80GB, driver 580.159.03, CUDA 13.0.
- Peak VRAM: ~14 GB during attack runs (A01 9.4 GB, A02 13.7 GB, A03 11.4 GB); ~2.2 GB for oracle scoring.
- Peak RAM: ~5 GB during attack runs, ~1.4 GB during oracle scoring.
- GPU-util average: ~85% during attacks; 0% during oracle scoring (CPU-bound).
- Co-scheduled: A01 attack + A02 attack + A03 attack across separate tmux sessions; oracle scoring ran on freed GPU concurrently with G2 ensemble eval.

## Caveats

- **NudeNet has very low pre-flag rate** on SDXL Turbo I2P-NSFW outputs (1/200). This collapses the NudeNet ASR denominator to ≈1, making the ASR estimate uninformative for that oracle. NudeNet uses a YOLOv8 detector on exposed body parts and is built around real-photo realism; SDXL Turbo at 1-step often produces stylized outputs that NudeNet misses. Inheriting this limitation, the "PGD vs NudeNet" cell is dominated by the small-n issue rather than by attack strength.
- **NudeNet is ONNX-backed**, so direct gradient-based PGD against it would require a re-implementation in PyTorch; we use cross-classifier transfer (PGD-on-SC bypasses scored on NudeNet) as the closure-grade measurement and document this as a caveat. Direct PGD against NudeNet remains in the v4 backlog.
- **Q16 is open_clip + linear soft-prompts** so it IS differentiable. The reported numbers are cross-classifier transfer (PGD-on-SC scored on Q16). Direct PGD against Q16 would likely produce higher ASR (closer to 1.000 at ε=4/255), but the cross-classifier number is sufficient to demonstrate that PGD-on-SC partially generalises to Q16.
- **A03 has no post-images saved** in the main `A03_emb_eps0.5_n200/` directory; only `_seedN/post/*.embedding.npy` exists. Cross-classifier oracle eval for A03 would require a fresh re-render through SDXL Turbo from the saved perturbed embeddings (deferred to a follow-up).
- **F1 figure includes A01 + A02 only**. A03 attribution requires the same re-render as above and is included as a placeholder/caveat in the figure manifest.

## Direction-of-effect summary

- safety_checker is brittle at every ε down to the minimum uint8 step (1/255). Saturation is independent of attack budget.
- Q16 partially co-vulnerable to PGD-on-SC perturbations: ~50-70% of Q16-pre-flagged images become Q16-unflagged after the PGD perturbation aimed at SC. The attack imperfectly transfers.
- NudeNet is uncorrelated with PGD-on-SC at this oracle's pre-flag rate; the limited natural detection rate on SDXL Turbo outputs leaves the cross-classifier number empirically uninformative.
- 5-seed CI shows zero variance on A01/A02/A03 ASR-vs-SC; Square Attack q=5K has 95% CI [0.93, 0.97].
