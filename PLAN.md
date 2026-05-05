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

---

## Phase 1c — session start 2026-05-04 evening (v2 + RTX PRO 6000 Blackwell)

### Hardware confirmed
- 1× RTX PRO 6000 Blackwell Workstation Edition, 96 GB VRAM, sm_120, CUDA 13.0, driver 580.126.16.
- AMD EPYC 9355, 64 vCPU.
- 1133 GB system RAM (substantially exceeds the 263 GB v2 spec).
- 574 TB free disk on /workspace.
- torch 2.11.0+cu128, sm_120 in arch list. GPU import + smoke OK.
- All 24/24 verify_assets rows green. HF_TOKEN, WANDB_API_KEY, GEMINI_API_KEY all set.
- Gemini fallback chain reachable; v2-spec model ids `gemini-3.1-flash-lite`, `gemini-3-flash`, `gemini-2-flash` are 404 in v1beta API. Updated chain to use `*-preview` variants and `gemini-2.0-flash` as fallback.

### Pod-recovery actions on session start
- The 36 cpu-workers from the prior session were oversubscribed (load avg 435, 1108 running processes, ~1550 s/image NudeNet+Q16 throughput on a 64-core box). All cpu-worker-* and stale `autocode` tmux sessions killed. Process count 1108 → 73 within 90 seconds.
- Replaced with 8 sharded cpu-workers using `--shard i/8` md5-based partitioning (added to `scripts/cpu_worker.py`) so workers don't duplicate label work. NudeNet + Q16 oracle scoring continues; 2715 unlabeled PNGs in queue.
- Restarted `monitor` with `nvidia-smi dmon -s pucvmet -d 5 > logs/gpu_monitor.log`.

### Phase 1c session plan — dependency-ordered

**Wave 1 (in flight on session start — 50 GB GPU footprint):**
| tmux session | what | resource |
|---|---|---|
| `monitor` | nvidia-smi dmon | always-on |
| `cpu-worker-{0..7}` | NudeNet + Q16 oracle labelling, sharded | 8 cores, ~16 GB RAM |
| `cf-strategy1` | Item 1c-0 Strategy 1: prompt-edit pairs render (665 candidates, SDXL Base 4-step) | 14 GB GPU |
| `cf-strategy3a` | Item 1c-0 Strategy 3 Path A: Gemini paraphrase, 25 anchors × 4 paraphrases × 4 cells × 2 concepts (=800) | 0 GB GPU, network-bound |
| `c1-square-n500` | C-1 Square Attack n=500 q=5K seed 0 (replaces n=50 prior result) | 8 GB GPU |

**Wave 2 (queued — start when wave 1 has VRAM headroom):**
- C-3 safety-SAE training at expansion 16 (16 GB) and 32 (28 GB) parallel; sweep L0 ∈ {32, 64, 128, 256}.
- C-9 transcoder detector at full hookpoint coverage (12 GB).
- Item 1c-7 SDXL Base 4-step on 1000 I2P prompts (separate large run from Strategy 1 build).
- Item 1c-0 Strategy 2: same-prompt seed pairs (100 prompts × 8 seeds, 14 GB).
- Item 1c-0 Strategy 3 Path B: Llama 3.1 70B int8 paraphrase (40 GB).
- C-1 Square seeds 1-4 in parallel for 5-seed CI (Item 1c-6).

**Wave 3 (after counterfactual + B02-oracle-v3 land):**
- Item 1c-1 cross-target detector bug fix (rebuild xtarget pipeline so detector consumes post-attack image's trajectory).
- Item 1c-3 B02 oracle relabel + retrain at scale (linear, MLP-256, MLP-512, per-block, EM, FT).
- C-2 AxBench rerun on counterfactual benchmark (the meaningful-task version).
- Item 1c-4 UnlearnDiffAtk as primary headline migration.
- Item 1c-5 SAeUron + DSG + SAEmnesia repros.

**Wave 4 (Phase D — start as dependencies allow, in parallel with 1c work):**
- D-1 causal feature graphs (Stage-2 survivors → attribution patching across timesteps).
- D-2 learned-projection intervention (drop-in for mean-patch).
- D-7 mechanistic feature-firing trajectory plot (canonical paper figure).
- D-9 cross-arch FLUX (newly feasible at 96 GB).
- D-4 cross-concept transfer test.
- D-5 black-box transfer attacks across diffusion stacks.
- D-6 joint end-to-end pipeline training.
- D-8 adversarial training of two-stage selection.
- D-10 compositional / multi-concept defense.

### Both framings active
- `paper/main.tex` continues as Framing A canonical working draft.
- `paper/alt_framing_B.md` — structured outline updated as experiments land. Both stay current.
- Framing-decision moment fires when {Item 1c-0, Item 1c-1, Item 1c-3, C-2 on counterfactual} all close.

### Code added this session
- `dsi/util/img_saving.py` — already existed (CaseRecorder); marked done.
- `dsi/util/activation_cache.py` — RAM-resident LRU, 200 GB ceiling, dirty-flush, preload_dir. New.
- `dsi/data/counterfactual.py` — Strategy 1 substitution dictionary (4 clusters, ~50 token regexes). Yields 665 candidate pairs from 4703 I2P prompts (14.1% match rate). New.
- `dsi/data/paraphrase.py` — Gemini Path A with 5-model fallback. New.
- `dsi/data/paraphrase_local.py` — Llama 3.1 70B int8 Path B. New.
- `scripts/build_cf_strategy{1,2,3_gemini,3_llama}.py` — CF benchmark drivers. New.
- `scripts/cpu_worker.py` — added `--shard i/n` md5-based partitioning. Modified.

### v2 hardware utilisation rules locked in
- Default state: ≥ 3 GPU + ≥ 12 CPU + monitor. (Currently 3 GPU + 8 CPU + monitor; will scale CPUs back up after wave 1 settles.)
- VRAM target 90 % (≈ 86 GB), hard ceiling 95 % (≈ 91 GB).
- 5-seed CIs run in parallel, never serialized.
- All renders save the full bypass / corrected / false-positive set per Item 1c-2 (img_saving.CaseRecorder).

---

## Phase C — autonomous-loop session 2026-05-04 PM (running)

### Landed
- C-2 (raw-vs-SAE per-block): raw all-blocks-cat AUC=1.000, SAE per-block 0.97-0.98. SAE 1-2 pp under raw on this in-distribution NSFW-vs-benign label task.
- C-3 (safety-SAE training): 4 hookpoints, TopK x8 k=64 on 1000 mixed I2P+COCO. Trained in 30 s each.
- C-3 detector first hookpoint: safety_sae va_auc=1.000 ties raw, beats Surkov (0.9805) by 1.95 pp. Other hookpoints in flight.
- C-6 (hybrid raw||sae): hybrid AUC=1.0 (+0.05 pp vs raw, +1.34 pp vs sae). In-distribution saturates raw — adversarial test queued (C-6 addendum).
- D02/D03 FID post: 235.21/235.25 vs pre 234.93. Both intervention kinds preserve image structure within < 0.5 FID.
- gen-base 4-step: 200 SDXL Base imgs in 522 s. Safety scoring queued.

### Running
- C-2 SAE per-block + concat (CPU)
- C-3 SAE detector for hookpoints 2-4 (CPU)
- C-9 transcoder (CPU)
- C-1 Square Attack 5K-query budget vs safety_checker (GPU)
- C-6 adv-robust eval on A01 bypass set (GPU)
- LPIPS-vgg D02/D03/D04 (CPU; slow)
- FID-D04-post (CPU)
- score-base-i2p (CPU)

### Queued
- LPIPS-vgg D02/D03/D04 → reports (when each lands)
- score-base-i2p → write base-vs-turbo comparison for paper
- Patch-LoRA (C-8) — bake the F_c intervention into a LoRA adapter
- Multi-concept defense (C-4) — extend F_c to violence, Van Gogh
- Cross-model transfer (C-5) — load SD3 + extract aligned activations


---

## 2026-05-05 update — Phase 1c continuation, framing-decision moment

### Landed since last update
- REFRAMING_DECISION.md — Framing A canonical (mixed evidence; 4 inputs banked)
- cf_probe_strategy2_sae_v1 — SAE features tie raw on Strategy 2 (AUC 0.9412 vs 0.9436)
- D02_D03_D04_lpips_gpu_v1 — patch-kind ablation tied: LPIPS=0.413, FID Δ<0.06, CLIP Δ<0.001
- D04_violence_v1 — cross-concept; AUC=1.000; 0 nudity/violence feature overlap
- A01_5seed_ci_v1 — 5/5 seeds at ASR=1.000 (88/88 bypass)
- Item_1c6_5seed_ci_final — combined 5-seed CI table for all 4 attacks

### In flight
- D-5 oracle-transfer (A01-s1 post → NudeNet+Q16+SC) — 40/200 done (slow CPU NudeNet)
- D-9 SD3 I2P n=100 — 80/100 done (4 flagged so far)
- D-9 FLUX safety smoke n=20 — model loading

### Queued
- Phase D-6 joint end-to-end training (smoke done, full version skeleton)
- Phase D-8 adversarial training (script skeleton; full version uses A01-defense-static proxy)
- Phase D-10 compositional defense (script skeleton)
- B02-v3 vs MMA-Diffusion adv-gen — abandoned this session (device mismatch in head/feat); fix queued
- SAEUron correct nudity feature_idx — abandoned (no published per-concept indices for SAEUron_coco)

