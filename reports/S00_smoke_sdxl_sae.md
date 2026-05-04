# S00 — SDXL Turbo + Surkov SAEs forward smoke

## Goal
Verify the GPU pipeline plumbs end-to-end: load SDXL Turbo + 4 Surkov TopK SAEs (`down.2.1`, `mid.0`, `up.0.0`, `up.0.1`), register Surkov-style residual hooks on the four `Transformer2DModel` blocks, generate one image, capture activations, and confirm sparsity matches the trained k=10 setting.

## Procedure
- `scripts/smoke_sdxl_sae.py --device cuda --dtype fp16`.
- SDXL Turbo loaded fp16 from local HF cache.
- Four Surkov TopK SAEs loaded via `dsi.sae.load.load_surkov_sae` (vendored `SurkovTopKSAE` matching sdxl-unbox/SAE/sae.py with `pre_bias`, `latent_bias`, k=10).
- `dsi.sae.hooks.SurkovHookManager` registers a `register_forward_hook` on each block and at hook time computes `diff = output - input`, permutes (B, C, H, W) → (B, H, W, C), feeds to `sae.encode`. No mean/std normalisation (matches Surkov's `reconstruct_sae_hook`).
- One prompt: "a photo of a yellow taxi in new york city", 1 inference step, seed=0.

## Results
- Peak VRAM: **7.86 GB** (under the 12 GB budget for this smoke).
- Wall-clock: 1.02 s per generation.
- All 4 hookpoints fired exactly once (single inference step).
- Per-hook latent shape: (1, 16, 16, 5120) — matches expected SDXL Turbo spatial 16×16 at the chosen blocks and SAE expansion factor 4× (d_in=1280 → d_hidden=5120).
- Per-hook sparsity active_frac: **0.00195** = 10/5120, exactly k=10 / d_hidden, confirming TopK gating works.
- z_max per hook: 69.3 (down.2.1), 154.8 (mid.0), 119.7 (up.0.0), 100.7 (up.0.1) — non-zero, in the expected magnitude range (Surkov reports z_max O(100) for trained SAEs).

## Artefacts
- Image: `outputs/S00_smoke_sdxl_sae/smoke.png`.
- Stats: `outputs/S00_smoke_sdxl_sae/stats.json`.

## Interpretation
Pipeline plumbed correctly. Both the SAE checkpoint loader (Surkov's `pre_bias`/`latent_bias`/topk schema) and the residual diff hook protocol are end-to-end functional on the Blackwell sm_120 / torch 2.11+cu128 stack. VRAM 7.86 GB leaves >19 GB headroom on the 27 GB cap, ample for co-locating with detector training (~4 GB) or another SAE-instrumented pipeline.

## Next
S01 — SAeUron repro on one UnlearnCanvas style; S02 — SD safety checker forward; S03 — pixel/latent/embedding PGD dry-runs.
