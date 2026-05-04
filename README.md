# DiffSafeSAE — diffusion safety with sparse autoencoders

This repository implements a research pipeline for adversarial robustness and mechanistic safety of text-to-image diffusion models. The codename is `dsi`. Four contributions:

1. Comparative cross-space red-team (pixel / VAE-latent / CLIP image-embedding) against `CompVis/stable-diffusion-safety-checker`, with per-feature SAE attribution.
2. In-generation SAE-activation detector (early-monitor and full-trajectory regimes) with per-step commit-knee diagnostic.
3. Cross-target robustness study: attacks against the SAE detector and the safety checker, with cross-target transferability matrix.
4. Detection-triggered correction via two-stage causal feature selection (Fisher ratio + Arad-style output score) and benign-mean patching, conditional on detector firing.

See `task_descriptions/task_description_v1.md` for the spec, `task_descriptions/task_description_v1_appendix.md` for the ICLR-rigor extension, `CLAUDE.md` for the agent operating manual, and `PLAN.md` for the live execution plan.

## Layout

```
dsi/                     # the package
scripts/                 # runnable entrypoints
paper/                   # ICLR-format LaTeX, agent-maintained
reports/                 # per-experiment markdown reports + INDEX.md
task_descriptions/       # binding spec + appendix
checkpoints/             # gitignored
logs/                    # gitignored (WandB is source of truth)
outputs/                 # gitignored except outputs/figures/
```

## Quick start (after env restored)

```bash
conda activate dsi
python scripts/verify_assets.py     # asserts datasets + models loaded
pytest tests/                        # CPU smoke tests
```

## Environment

Pack/unpack workflow per `/workspace/conda_setup.md`. Source of truth: `/workspace/env-archives/dsi.tar.gz`. After any package change: `bash /workspace/scripts/pack_env.sh dsi`.

## Status

Phase 1a (CPU scaffold + downloads) complete. Phase 1b (GPU experiments) starts on the GPU pod via `STARTER_PROMPT_2.md`.
