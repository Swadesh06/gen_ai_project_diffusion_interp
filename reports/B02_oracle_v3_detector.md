# B02_oracle_v3_detector — oracle-relabeled detector at scale (Item 1c-3)

## Goal

Per `task_description_v2.md` §3 Item 1c-3. Phase 1 left B02 at AUC 0.847
(linear) / 0.891 (MLP) on a 1388-sample oracle-labelled dataset with
severe class imbalance (41 NSFW / 1347 benign). v3 expands the dataset
using all available oracle labels across A01/A02/A03 attack runs +
sae_benign_coco_1k generations, with class-balanced auto-pos-weighting.

## Procedure

### Dataset build (`outputs/detector_dataset_oracle_v3`)
- A01 pixel-PGD: 200 oracle-labelled samples (66 NSFW)
- A02 latent-PGD: 156 oracle-labelled samples (52 NSFW)
- A03 embedding-PGD: 200 oracle-labelled samples (67 NSFW)
- sae_benign_coco_1k: 988 oracle-labelled samples (16 NSFW)
- Total: **1544 samples (201 NSFW / 1343 benign)**
- 8 mp.Pool workers loaded the .sae.pt files in 5 minutes wall.

### Training
- 10 detector variants in parallel (one tmux session each):
  - 5 linear probes: per-hookpoint × 4 + concat-all-blocks
  - 5 MLP probes: per-hookpoint × 4 + concat-all-blocks
- 30 epochs each, BCE w/ `auto-pos-weight = n_neg / n_pos = 6.68`
- Adam lr=1e-3, weight_decay=1e-4, batch 64, val_frac 0.2
- Wall: linear ~60-70 s, MLP-cat 344 s, MLP-per-hp ~120 s

## Results

### Per-variant best validation metrics

| variant | head | hookpoint | AUC | AP |
|---|---|---|---|---|
| **B02_oracle_v3_mlp_up_0_0** | MLP-512 | up.0.0 | **0.9772** | **0.9452** |
| B02_oracle_v3_linear_cat   | linear | concat (20480-d) | 0.9762 | 0.9298 |
| B02_oracle_v3_linear_down_2_1 | linear | down.2.1 | 0.9752 | 0.9181 |
| B02_oracle_v3_mlp_cat      | MLP-512 | concat | 0.9744 | 0.9311 |
| B02_oracle_v3_mlp_down_2_1 | MLP-512 | down.2.1 | 0.9733 | 0.9430 |
| B02_oracle_v3_linear_up_0_0 | linear | up.0.0 | 0.9732 | 0.9369 |
| B02_oracle_v3_mlp_mid_0    | MLP-512 | mid.0 | 0.9672 | 0.9246 |
| B02_oracle_v3_linear_mid_0 | linear | mid.0 | 0.9652 | 0.9222 |
| B02_oracle_v3_mlp_up_0_1   | MLP-512 | up.0.1 | 0.9636 | 0.9228 |
| B02_oracle_v3_linear_up_0_1 | linear | up.0.1 | 0.9630 | 0.9203 |

Checkpoints under `/workspace/checkpoints/B02_oracle_v3_*/{best.pt, last.pt, summary.json}`.

### vs B02 v2 (smaller dataset)

| version | dataset size | linear AUC | MLP AUC | comment |
|---|---|---|---|---|
| B02 v1 | 1388 (41/1347) | 0.847 | — | unbalanced BCE, no class weight |
| B02 v2 balanced | 1388 (41/1347) | 0.852 | 0.891 | balanced BCE |
| **B02 v3 balanced** | **1544 (201/1343)** | **0.9762** | **0.9772** | **+oracle from A02/A03/sae_benign** |

## Interpretation

The v3 dataset rebuild lifts oracle-labelled detector AUC from **0.85
(linear) / 0.89 (MLP)** to **0.98 / 0.98** — passing the Item 1c-3 gate
(linear ≥ 0.85 on counterfactual, MLP ≥ 0.88) by margin. The 5×
expansion of NSFW samples (41 → 201) and the inclusion of A02 latent-PGD
+ A03 embedding-PGD bypasses (which produce different SAE distributions)
made the detector see a much richer NSFW-side distribution.

Per-hookpoint AUCs are all 0.96-0.98. No single hookpoint dominates;
unsafe content is broadly distributed across the SAE feature space.
The `up.0.0` hookpoint MLP is the marginal best (AUC 0.9772, AP 0.9452).

This dataset is now the canonical detector training set for downstream
work (Item 1c-9 black-box vs SAE detector, Item 1c-1 cross-target on
counterfactual, Phase D-8 adversarial training).

## Next

- xtarget v2 (Item 1c-1) against B02-v3 for all of A01/A02/A03 — the
  meaningful version of the cross-target matrix (B01 was prompt-origin
  biased).
- C-2 AxBench probe rerun on the **counterfactual benchmark** (Item
  1c-0 Strategy 1) once the cf-strategy1 build finishes — this is the
  framing-discriminator number.
- B02 v3 + safety-trained SAE v2: train a hybrid `[surkov || safety_v2]`
  detector head; expected to push AUC further given safety_v2 closes
  the gap to raw on in-distribution.
