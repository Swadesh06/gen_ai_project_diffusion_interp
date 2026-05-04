# PLAN.md — DiffSafeSAE live execution plan

> Maintained across sessions. Each new session appends a section; do not delete prior content.
> Operating manual: `CLAUDE.md`. Spec: `task_descriptions/task_description_v1.md`.
> Binding ICLR-rigor extension: `task_descriptions/task_description_v1_appendix.md`.

---

## Phase 1a — CPU-only session start (2026-05-04)

Goal: stand up the CPU scaffold so Phase 1b can `git pull` and immediately smoke-test on GPU.

### Done

- conda env `dsi` (Python 3.11) with torch 2.5.1 (CPU wheel), diffusers 0.31, transformers 4.46, accelerate, peft, datasets, einops, wandb, nudenet, clean-fid, lpips, dreamsim, open_clip_torch, captum, pytest. Packed to `/workspace/env-archives/dsi.tar.gz`.
- `dsi.pth` wired into `$CONDA_PREFIX/lib/python3.11/site-packages/`. `import dsi` works from any cwd.
- Repo scaffolded per CLAUDE.md §15. Every `dsi/<sub>/` has `__init__.py`. Every script in `scripts/` exists with `--dry-run`.
- Modules implemented (CPU-only parts): `dsi.config`, `dsi.data.{i2p,coco,laion_coco,unlearncanvas,adversarial}`, `dsi.attacks.{common,pixel,latent,embedding}`, `dsi.sae.{load,hooks,attribution}`, `dsi.detectors.{sae_em,sae_ft,train, baselines/*}`, `dsi.interventions.{patches,stage1_fisher,stage2_causal,pipeline, baselines/*}`, `dsi.eval.{asr,fid,clip_score,unlearncanvas,commit_knee}`, `dsi.models.{sdxl_pipeline,sd15_pipeline,classifier_oracles}`, `dsi.util.{wandb,ckpt,seed,logging}`.
- `pytest tests/`: 66/66 pass.
- `scripts/`: bootstrap, verify_assets, _common, gate_clean_baseline, exp_A0{1,2,3}, exp_B0{1,2}, exp_C01, exp_D01, repro_{saeuron,saemnesia,dsg}, eval_full_grid, sweep_detector, pack_env. All dry-runs succeed; JSON plans written to `reports/dry/`.
- Paper skeleton: `paper/main.tex` with all section / subsection scaffolding from spec + appendix. `paper/refs.bib` with 17 entries. Replace NeurIPS template with `iclr2026.sty` at submission.
- Downloads (parallel, all on `/workspace`):
  - SDXL Turbo (18 GB), SDXL Base (71 GB), CompVis safety checker, CLIP ViT-L/14 (6.4 GB).
  - Surkov SAEs (4 hookpoints, ~235 MB total).
  - SAeUron SAEs (2 hookpoints + COCO, ~611 MB total).
  - I2P + I2P-adv (HF cache); COCO val 2017 + captions; UnlearnCanvas.
  - LAION-COCO 50K caption parquet.
  - MMA-Diffusion / UnlearnDiffAtk / Ring-A-Bell repos cloned.
  - NudeNet, LPIPS, DreamSim weights triggered.

### Known gaps (carried to Phase 1b)

- **SD v1.5** canonical mirror (`runwayml/stable-diffusion-v1-5`) is gated upstream. Loader auto-falls back to `benjamin-paine/stable-diffusion-v1-5`. Document the mirror used in `reports/INDEX.md` whenever the SD v1.5 track is exercised.
- **SAEmnesia** has no public repo or HF release as of 2026-05-04. Queued for reproduce-from-scratch:
  - Train supervised SAE on labelled UnlearnCanvas concepts at the same hookpoints as SAeUron.
  - Architecture: ReLU SAE with one-to-one concept-neuron supervised loss per Cassano et al.
  - Until then, `scripts/repro_saemnesia.py` is a dry-run stub.
- **MMA-Diffusion image set** is gated. Access requested at https://huggingface.co/datasets/YijunYang280/MMA_Diffusion_adv_images_benchmark. Run text-modality + UnlearnDiffAtk + Ring-A-Bell as primary adversarial benchmarks; image set is additive.
- **Q16** concrete checkpoint not yet wired (the wrapper is a stub returning safe; flagged for fix in Phase 1b first-day work).

---

## Phase 1b — first GPU session plan

> The default state of the box is "≥ 1 GPU job + ≥ 1 CPU job + monitor", always. Saturate the hardware.

### Order of operations

| step | exp_id prefix | resource | co-scheduled with |
|---|---|---|---|
| 1 | torch GPU wheel reinstall + `pack_env.sh dsi` | GPU build | n/a |
| 2 | `scripts/verify_assets.py` (must remain green) | CPU | n/a |
| 3 | smoke S00 — SDXL Turbo + Surkov SAEs forward + capture activations | GPU 6 GB | CPU monitor |
| 4 | smoke — SAeUron repro on one UnlearnCanvas style | GPU 10 GB | NudeNet labelling watcher (CPU) |
| 5 | dry-run pixel/latent/embedding PGD (peak VRAM logging) | GPU 14/12/3 GB | n/a |
| 6 | Item 1.5 — clean-baseline (1000 COCO + 1000 I2P SDXL Turbo gens) | GPU 6 GB | CPU watcher: NudeNet + Q16 + safety-checker labels of streaming outputs |
| 7 | Item 2.1 — pixel PGD on 50 → 500 prompts | GPU 14 GB | CPU: NudeNet/Q16 scoring of prior bypass batch |
| 8 | Items 2.2 + 2.3 — latent + embedding PGD co-located | GPU 12 + 3 GB | CPU: per-block per-feature SAE attribution analysis on Item 2.1 outputs |
| 9 | Item 3 — EM + FT detector training | GPU 4 GB each | co-locate two detector jobs OR pair with cross-target attack |
| 10 | Item 4 — cross-target re-attacks (pixel/latent/embedding × SAE detector) | GPU 14/12/3 GB | CPU: detector-attribution analysis |
| 11 | Item 5 — eval grid 12 rows × 5 seeds | GPU 8 GB per row | CPU: FID/CLIP-score on cached generations |
| 12 | reports/PHASE_1_FINAL.md | n/a | n/a |

### Co-scheduling cheat-sheet

- **Pair pixel-PGD (~14 GB) with**: any CPU job. Do NOT co-locate another GPU job.
- **Pair latent-PGD (~12 GB) with**: embedding-PGD (~3 GB) or detector training (~4 GB) on the same GPU.
- **Pair embedding-PGD (~3 GB) with**: detector training, eval grid row, SAeUron repro.
- **Pair SAeUron repro (~10 GB) with**: detector training (~4 GB) or embedding-PGD.
- **Always-on CPU work**: NudeNet/Q16 scoring of any new outputs in `outputs/`. Set up a `cpu-worker` tmux watcher loop on Phase 1b first hour.

### Pass criteria (gates each item must hit)

- Item 1.5 clean baseline: I2P-NSFW prevalence ≥ 0.70; COCO ≤ 0.05.
- Item 2 (Contribution 1): ASR ≥ 0.7 on at least one of {pixel, latent, embedding} at ε=8/255 on I2P-NSFW; cross-space SAE feature overlap ≥ 60 %.
- Item 3 (Contribution 2): AUC ≥ 0.95 on I2P-naive; ≥ 0.85 on I2P-adversarial; ≥ 0.80 on MMA-text. Commit-knee plot produced. Both EM and FT regimes trained.
- Item 4 (Contribution 3): full 2×2 ASR matrix with 5-seed CIs; ≥ 1 off-diagonal cell ≤ 50 % ASR; mechanistic plot identifying divergent feature subspaces.
- Item 5 (Contribution 4): two-stage + mean-patch row dominates SAeUron and DSG-adapted on ≥ 3 of {I2P-naive ASR, I2P-adv ASR, FID, CLIP-score} at p < 0.05 over 5 seeds; Stage-1-only and zero-patch ablations regress on ≥ 1 ASR metric each.

---

## Phase C — ideas (queued; pursue after Phase 1 lands)

EV-descending order from `task_descriptions/task_description_v1_appendix.md` §G:

| C-N | idea | VRAM | co-scheduling | gate |
|---|---|---|---|---|
| C-1 | Black-box attack on SAE detector (Square + NES, 1K + 10K queries) | 6-8 GB | any GPU job ≤ 18 GB | BB ≥ 30% of WB ASR at 10K |
| C-2 | AxBench probes on raw UNet/VAE/text-emb (sanity SAEs are necessary) | 4 GB | any Contribution 1 attack | SAE detector beats all 3 by ≥ 3 pp AUC on I2P-adv |
| C-3 | Safety-trained SAE (50/50 I2P+COCO, 4 hookpoints, expansion ∈ {8,16,32}, L0 ∈ {32,64,128,256}) | 16 GB | CPU eval only | Stage-2 score ≥ 1.5× Surkov; ASR ↓ ≥ 5 pp; FID Δ ≤ 1.0 |
| C-4 | Multi-concept simultaneous defense (nudity + violence + Van Gogh) | 10 GB | C-1 or C-2 | per-concept ASR within 5 pp of single-concept; benign FID Δ ≤ 1.5 |
| C-5 | Cross-model transfer (SDXL → SD3 / FLUX) | 18 GB | CPU eval only | ≥ 40% high-output-score features align |
| C-6 | Hybrid SAE + predicted-noise detector | 5 GB | anything | hybrid AUC > max(SAE-only, noise-only) by ≥ 1 pp on I2P-adv |
| C-7 | Adversarial training of SAE detector (5 rounds) | 10 GB | C-3 SAE training | round-5 ASR drops < 50 % of round-1; benign AUC Δ ≤ 1 pp |
| C-8 | LoRA-baked safety (rank 8 / 16) | 16 GB | CPU eval | LoRA matches runtime-patched within 3 pp ASR + 1.5 FID |
| C-9 | Transcoder detector + circuit-level attribution (Dunefsky 2024) | 12 GB | CPU eval | identifiable ≥3-feature circuit on ≥ 60 % of bypasses |
| C-10 | Generation-quality preservation (LPIPS / DreamSim vs un-intervened) | 5 GB | anything | benign-FP LPIPS < 0.15, DreamSim < 0.10; true-positive LPIPS > 0.35, DreamSim > 0.30 |

Beyond these ten, ideas to prototype if compute remains:

- Per-concept causal feature graph (directed dependencies among Stage-2 survivors).
- Sequential concept removal without forgetting.
- Joint SAE + detector training with safety auxiliary loss.
- Streamlit attack-trace visualiser (per-step SAE deltas highlighted).
- Comparison against SafeGuider (arXiv:2510.05173) and GuardT2I.
- Patch-level effective-receptive-field attribution per CaFE 2025.

### Phase C → main pipeline integration rules

- Every C-X result that improves an end-to-end metric flips the default config + re-runs the §3.4 grid for affected rows + updates `paper/main.tex`.
- Negative C-X results are not discarded — documented in `paper/main.tex` Appendix as ruled-out alternatives.

---

## Repo structure status

- `dsi/` — package code; CPU smoke complete.
- `scripts/` — runnable entrypoints; all `--dry-run` working.
- `paper/main.tex` — skeleton; updated as experiments land.
- `reports/` — `INDEX.md` + `phase1a_bootstrap.md`. Each future experiment writes its own.
- `task_descriptions/` — committed, read-only spec + appendix.
- `checkpoints/` / `logs/` / `outputs/` — gitignored; created on demand.
- `tests/` — 66 tests; CI-runnable on CPU only.

---

## Operational reminders (do not delete; carry forward)

- After any pip install / uninstall / upgrade: `bash /workspace/scripts/pack_env.sh dsi` or it dies on next pod.
- Every new experiment writes `reports/<exp_id>.md` *before* launching, with goal/method written down so it's not a fishing expedition.
- Every long-running thing goes in `tmux`. Logs to `logs/<name>.log`. WandB project = `dsi-v1`.
- Commit + push after every meaningful step. SSH remote = `git@github.com:Swadesh06/gen_ai_project_diffusion_interp.git`.
- No emojis, no superlatives, no fluff in any committed text. Direct + plain.

---

## Open ops items for Phase 1b first hour

- ~~**Create the GitHub repo.**~~ DONE 2026-05-04 mid-session. `git push -u origin main` lands at commit `2427368`; remote = `git@github.com:Swadesh06/gen_ai_project_diffusion_interp.git`.
- **Repack the env after installing the GPU torch wheel.** The Phase 1a packed env at `/workspace/env-archives/dsi.tar.gz` carries the CPU torch wheel; without a repack the next pod restores the CPU build. Use `bash scripts/pack_env.sh` (which delegates to `/workspace/scripts/pack_env.sh dsi`).
- **MMA-Diffusion image-set access** is gated. Request from https://huggingface.co/datasets/YijunYang280/MMA_Diffusion_adv_images_benchmark; until granted, the text-modality + UnlearnDiffAtk + Ring-A-Bell sets are the adversarial benchmark.
- **SAEmnesia** has no public release as of 2026-05-04. PLAN flags it as reproduce-from-scratch: train supervised SAE on labelled UnlearnCanvas concepts at SAeUron's hookpoints, with one-to-one concept-neuron loss per Cassano et al. Until then, `scripts/repro_saemnesia.py` is a dry-run stub.
- **DreamSim's `./models/` cache.** DreamSim hardcodes a `./models/` cache dir relative to cwd. `.gitignore` excludes `/models/` at repo root; either chdir before calling DreamSim, or set torch hub dir explicitly via `os.environ['TORCH_HOME']`.
