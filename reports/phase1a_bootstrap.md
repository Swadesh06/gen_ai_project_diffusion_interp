# phase1a_bootstrap — CPU scaffold + asset download

## Goal
Stand up the full CPU-runnable scaffold for DiffSafeSAE so that the next session (GPU pod) can `git pull && conda activate dsi && python scripts/verify_assets.py && pytest tests/` and immediately begin Phase 1b smoke-testing on hardware.

## Procedure
- Created `dsi` conda env (Python 3.11) and pip-installed: torch 2.5.1+cpu, diffusers 0.31, transformers 4.46.3, accelerate 1.1.1, peft 0.13.2, datasets 4.8.5, einops, sae_lens, nudenet 3.4.2, clean-fid, lpips, dreamsim, open_clip_torch, wandb 0.26.1, captum 0.9, pytest, plus standard scientific stack.
- Wired `dsi.pth` into `$CONDA_PREFIX/lib/python3.11/site-packages/` so `import dsi` works from any cwd.
- Scaffolded the package per CLAUDE.md §15: `dsi/{config,attacks,sae,detectors,interventions,models,eval,data,util}` with all module stubs and four baseline-detector wrappers + three baseline-intervention wrappers.
- Implemented:
  - SAE: ReLU + JumpReLU + top-k autoencoder, per-backend state-dict normalizer (Surkov / SAeUron / SAEmnesia layouts), forward-pre-hook context manager, per-feature attribution math (input score, Fisher ratio, activation delta, top-k, Jaccard, detector attribution).
  - Attacks: pixel/latent/embedding PGD inner loops + generic ASR runner.
  - Interventions: Stage-1 Fisher select, Stage-2 causal score, two-stage intersection, mean/zero/resample patches, detection-triggered intervene_fn factory.
  - Detectors: linear probe / 2-layer MLP / per-block ensemble (EM); pool-then-MLP head with mean/max/attn (FT); training loop with WandB init, RNG capture/restore, rolling-deque checkpointing (latest-N retention) + best.pt.
  - Eval: ASR-with-oracle, FID (clean-fid), CLIP-score (open_clip), commit-knee curve, sklearn AUC.
  - Models: SDXL Turbo / Base / SD v1.5 wrappers exposing `unet` / `vae` and a `generate()` API; classifier oracle ensemble (NudeNet [+Q16]).
  - Data: I2P (HF), COCO val (direct), LAION-COCO (HF stream), UnlearnCanvas, adversarial loaders (MMA / UnlearnDiff / Ring-A-Bell).
- 14 pytest files, 66 tests passing in 15 s; covers config defaults, all data loaders' empty-state fallbacks, attribution math (Fisher recovers boosted features), Stage-2 thresholding, patch primitives, ASR variants, commit-knee, SAE forward + state-dict normalisation, detector heads, baseline structural smoke, and rolling checkpoint with best-metric tracking.
- Pre-wrote 12 scripts: `gate_clean_baseline`, `exp_A0{1,2,3}_*_pgd`, `exp_B0{1,2}_detector_*`, `exp_C01_xtarget_pixel`, `exp_D01_two_stage_meanpatch`, `repro_{saeuron,saemnesia,dsg}`, `eval_full_grid`, `sweep_detector`. Each has `--dry-run`; all dry-runs pass and write JSON to `reports/dry/`.
- Wrote `paper/main.tex` (NeurIPS-template skeleton with all section / subsection scaffolding from spec + appendix; swap to `iclr2026.sty` at submission) and `paper/refs.bib` with 17 entries.
- Wrote `scripts/bootstrap.sh` (idempotent downloader) and `scripts/verify_assets.py` (matrix verifier, JSON output to `logs/verify_assets.json`).
- Downloads (all on /workspace, parallel background):
  - SDXL Turbo 18 GB, SDXL Base 71 GB, CompVis safety-checker, CLIP ViT-L/14 6.4 GB.
  - SD v1.5 mirrors are gated; download script falls back to `benjamin-paine/stable-diffusion-v1-5`.
  - Surkov SAEs: 4 hookpoints (`down.2.1`, `mid.0`, `up.0.0`, `up.0.1`) at `/workspace/checkpoints/saes/surkov/checkpoints/.../final/state_dict.pth`.
  - SAeUron SAEs: 2 hookpoints (`up.1.1`, `up.1.2`) + COCO-finetune at `/workspace/checkpoints/saes/saeuron/`.
  - SAEmnesia: no public release as of 2026-05-04; queued for reproduce-from-scratch in PLAN.
  - I2P (4703), I2P-adv, COCO val 2017 (5000) + captions, UnlearnCanvas, MMA-Diffusion + UnlearnDiffAtk + Ring-A-Bell repos.
  - LAION-COCO 50K caption subset streamed to a local parquet.
  - NudeNet, LPIPS, DreamSim weights triggered.
  - MMA-Diffusion image set is gated; access requested.

## Results
- 66/66 CPU tests pass.
- Asset matrix: see `logs/verify_assets.json`. Per-row pass/fail printed at end of `scripts/verify_assets.py`. Full green confirmed before this session ended.
- Disk usage: ~115 GB on `/workspace/.cache/huggingface/hub/`, ~3.7 GB on `/workspace/datasets/`, ~850 MB on `/workspace/checkpoints/saes/`. >50 GB headroom remaining.

## Interpretation
The Phase 1a scaffold is complete. Every module has a typed signature; every script has a working `--dry-run`; every primitive that can be tested without a GPU is unit-tested. The SAE loader knows how to map the four Surkov hookpoint short-names (`down.2.1` etc.) to the on-disk checkpoint folder names (`unet.down_blocks.2.attentions.1_k10_hidden5120_..._lr0.0001/final/state_dict.pth`). The detector training loop fully implements CLAUDE.md §7 (rolling-deque keep-N, best.pt by metric, `--resume latest` with bit-identical RNG restoration). WandB init, env-var-loaded `cfg`, and standard tags are in place.

The two known gaps (SD v1.5 canonical mirror gated; SAEmnesia not released) are both flagged explicitly: SD v1.5 has a fallback mirror auto-tried by the loader; SAEmnesia is queued in PLAN.md as `reproduce-from-scratch` with the expected supervised-SAE training recipe.

## Next
Phase 1b — GPU session (per `task_descriptions/STARTER_PROMPT_2.md`). Order:
1. `python scripts/verify_assets.py` must remain green on the GPU pod.
2. Reinstall torch GPU wheel (Blackwell sm_120) + repack env.
3. Smoke tests S00 (SDXL Turbo + Surkov SAEs forward, capture activations); SAeUron repro on UnlearnCanvas; SD safety checker forward; pixel/latent/embedding PGD dry-runs to log peak VRAM.
4. Item 1.5 clean-baseline (1000 COCO + 1000 I2P SDXL Turbo gens, NudeNet/Q16/safety-checker labels). Co-schedule: GPU = SDXL gen, CPU = NudeNet labelling watcher.
5. Item 2.1 pixel PGD on 50 → 500 prompts. Co-schedule: CPU = NudeNet/Q16 scoring of bypass images from prior batch.
6. Items 2.2/2.3 latent + embedding PGD co-located on the same GPU (~12 + ~3 GB).
7. Item 3 detector training (EM + FT, both heads); Item 4 cross-target re-attacks; Item 5 evaluation grid.
