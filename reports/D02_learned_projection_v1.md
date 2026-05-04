# D02_learned_projection_v1 — per-hookpoint linear projection trained

## Goal

Phase D-2 from `task_description_v2.md` §6. Patch-primitive ablation
(D02 mean ≈ D03 zero ≈ D04 resample at safety_checker level)
suggests scalar mean-patching is the limit of what a single point
estimate can do. A **learned linear projection** Pi: R^d → R^d trained
to (a) preserve benign-distribution z, (b) project unsafe-distribution
z toward the benign-mean, can capture distributional structure scalar
mean-patching cannot.

## Procedure

- Data: `outputs/dataset_axbench_v1` (1000 samples, 500 NSFW + 500
  benign, mean-pooled per Surkov hookpoint). Two feature sources:
  - **raw** (d_in=1280 per hookpoint).
  - **sae** (d_in=5120 per hookpoint, Surkov SAE encoded).
- Per hookpoint: linear `Pi: R^d → R^d` initialised as identity.
- Loss: `||Pi(z_benign) - z_benign||^2 + lam * ||Pi(z_unsafe) - mu_benign||^2` with `lam = 1.0`.
- Adam lr=1e-3, 200 epochs, on cuda. Wall ≈ 7-10 s per hookpoint.

## Results — final l_pres / l_proj after 200 epochs

### SAE features (d=5120)

| hookpoint | benign_pres_mse | unsafe_proj_mse | wall (s) |
|---|---|---|---|
| down.2.1 | (logged in summary.json) | | |
| mid.0    | (logged in summary.json) | | |
| up.0.0   | 5.96e-5 | 1.29e-3 | 6.4 |
| up.0.1   | 9.28e-4 | 1.52e-2 | 8.8 |

### Raw activations (d=1280)

| hookpoint | benign_pres_mse | unsafe_proj_mse | wall (s) |
|---|---|---|---|
| up.0.0 | 0.119 | 0.231 | 7.5 |
| up.0.1 | 0.198 | 0.231 | 6.5 |

Checkpoints: `outputs/D02_learned_projection/projection_<hp>.pt` (raw),
`outputs/D02_learned_projection_sae/projection_<hp>.pt` (sae).

## Interpretation

The SAE-feature projection achieves **2-3 orders of magnitude lower
benign preservation MSE than the raw-feature projection** at the same
absolute scale (because SAE features are sparser and so each
dimension carries less variance). The trained Pi:
- preserves z_benign (post-projection mse ≈ 1e-5 to 1e-3 on SAE).
- projects z_unsafe toward mu_benign (post-projection mse ≈ 1e-3 to
  1.5e-2 on SAE; the residual = how much the projection-to-benign-mean
  cannot perfectly land on the mean given a single linear map across
  the entire unsafe distribution).

The "ranking_auc" diagnostic is near 0 for both feature sources —
expected, since post-projection both benign and unsafe map to the
benign neighbourhood. Detection is no longer the goal; **substitution
quality** is.

This produces the per-hookpoint Pi matrices needed to plug in as a
drop-in replacement for the mean-patch primitive at intervention
time. The downstream test (D02 row 11 in the v2 evaluation grid):
`Stage 1 ∩ Stage 2 + learned-projection patch` vs the existing
`Stage 1 ∩ Stage 2 + benign-mean patch`. The hypothesis: learned
projection beats mean-patch on FID or CLIP-score (preserves more
detail / structure) while matching on ASR.

## Next

- Plug Pi into `dsi.interventions.patches` with a new `LearnedProjection`
  patch kind. Apply in `run_intervention.py --patch learned` on the
  existing D02 setup (Stage 1 ∩ Stage 2 selection).
- Re-run the D02 / D03 / D04 ablation grid with row 11 added,
  on UnlearnDiffAtk-nudity once that benchmark is in flight.
- Compare SAE-feature vs raw-feature variants — the SAE-feature
  projection is the natural substrate for Stage-1 ∩ Stage-2 (it
  selects SAE features), so default to that.
- λ ablation: λ ∈ {0.1, 1.0, 10.0} to map the preservation /
  projection trade-off.
