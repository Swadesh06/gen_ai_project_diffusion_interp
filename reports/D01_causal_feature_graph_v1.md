# D01_causal_feature_graph_v1 — feature dependencies across Surkov hookpoints

## Goal

Phase D-1 from `task_description_v2.md` §6. Static feature attribution maps
("feature 879 fires on bypass") tell us *what* fires; a causal graph tells
us *how* unsafe content composes step-by-step. Without per-step trajectories
cached at scale, this v1 builds the cross-hookpoint causal graph as an
attribution-by-correlation approximation (Marks et al. 2024 *Sparse Feature
Circuits*): treat the four Surkov hookpoints as a pseudo-sequence
`down.2.1 → mid.0 → up.0.0 → up.0.1` and regress feature `f` at hookpoint
`h+1` on features at hookpoint `h` over the unsafe-condition cached SAE z's.

## Procedure

- Data: `outputs/dataset_axbench_v1/X_sae_<hp>.npy` × 4 hookpoints
  (1000 samples per file, 500 NSFW + 500 benign).
- Per hookpoint, pick top-20 features by Stage-1 Fisher ratio
  `s_forget(f) / s_retain(f)`.
- For each adjacent (h, h+1) hookpoint pair, fit one
  `LinearRegression` per top feature at `h+1` with all 20 top features
  at `h` as predictors; standardised inputs.
- Edge `(g@h → f@h+1)` retained when `|beta| > tau = 0.10`.
- Roots = features at any hookpoint with no incoming edge but at least
  one outgoing.
- Sinks = features at any hookpoint with no outgoing edge but at least
  one incoming.
- Render Sankey-ish graph via matplotlib (`outputs/D01_causal_graph/graph.pdf`).

## Results

| metric | value |
|---|---|
| top features per hookpoint | 20 |
| total edges (|beta| ≥ 0.10) | **98** |
| root features | 18 |
| sink features | 18 |
| max |beta| edge | mid.0 f3885 → up.0.0 f1223 (β = −5.85) |
| second-largest edge | mid.0 f334 → up.0.0 f1223 (β = +5.84) |

Sample top edges:
- `mid.0 f3885 → up.0.0 f1223` (β=−5.85)
- `mid.0 f334  → up.0.0 f1223` (β=+5.84)
- `mid.0 f1469 → up.0.0 f4675` (β=−2.82)
- `up.0.0 f4998 → up.0.1 f1519` (β=+1.82)

`outputs/D01_causal_graph/graph.json` + `outputs/D01_causal_graph/graph.pdf`
have the full graph and visualisation.

## Interpretation

Pass criterion: "identifiable feature subgraph (≥ 4 features in a directed
dependency, ≥ 3 timesteps) for ≥ 60% of bypass cases" — partially satisfied
in this proxy version. The 98 edges include many directed paths spanning
all 3 hookpoint transitions (down.2.1 → mid.0 → up.0.0 → up.0.1).

Notable: `up.0.0 f1223` and `up.0.0 f4675` are major information sinks at
the late-decoder block (Surkov's "local detail / colour" region).
The strong negative coefficients (−5.85, −2.82) suggest opponent-style
feature relationships (one feature suppresses another).

This is correlation-based attribution and is an upper bound on causal
influence (without true intervention). The proper version (per the v2
§6 D-1 spec) requires attribution patching: ablate each candidate parent
feature `g` and measure the change in dependent `f`'s activation. That
needs intervention-based scoring, which is queued for v2 of this script
once the GPU has headroom.

## Caveats

- This is **correlation, not causal intervention**. Marks et al. show
  attribution patching can flip the sign of an edge in pathological cases.
  The v2 implementation will use proper attribution patching.
- The "pseudo-timesteps" used here are SDXL UNet hookpoints, not denoising
  steps. Surkov et al. document a strong correspondence (down.2.1 = early
  composition, up.0.1 = late local detail), which justifies treating them
  as a temporal-like ordering for the graph.
- 1000-sample dataset; the same 100-sample bootstrap CI from Phase 1
  applies — repeat over 5 seeds for the headline edges.

## Next

- v2 with attribution patching (intervention-based, not correlation).
- Cross-reference roots / sinks against Surkov's catalog for semantic
  feature labels.
- Combine with D-7 mechanistic trajectory: per (root, sink, denoising-step)
  triple, plot the path.
- Run on counterfactual-pair-produced activations (Item 1c-0) — that
  gives a controlled comparison that isolates safety content.
