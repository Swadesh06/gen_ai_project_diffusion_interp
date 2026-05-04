# STARTER_PROMPT_2.md — Phase 1b (GPU) session start

You are continuing the DiffSafeSAE research project. **This is not a fresh start** — the previous session (CPU-only, Phase 1a) read all the docs, set up the conda environment, downloaded all datasets and model checkpoints, scaffolded the repo, wrote the package skeleton, pre-wrote the experiment scripts and the evaluation grid runner, and pushed everything to GitHub. There is existing code, an existing PLAN.md, and a passing CPU test suite. **Do not redo any of that.**

The instance is now backed by the **RTX Pro 4500 Blackwell, 32 GB VRAM**. Your job in this session is to execute Phase 1b — the GPU work — starting from the schedule the previous session wrote into `PLAN.md`. Then, once the four contributions are at passing gates, you keep ideating (Phase C) until the human stops you.

You operate by the rules in `CLAUDE.md`. Read it; follow it.

---

## Step 1 — pick up the project state

```bash
cd /workspace/swadesh/gen_ai_project_diffusion_interp
git pull
conda activate dsi
```

If the env is missing locally (new pod): `tar xzf /workspace/env-archives/dsi.tar.gz -C $HOME/miniconda3/envs/dsi/` and re-activate.

Verify the GPU: `nvidia-smi` should show one RTX Pro 4500 with 32 GB. If torch is CPU-only (the previous session installed CPU torch for package resolution), reinstall the GPU wheel for Blackwell (sm_120; check `pytorch.org/get-started/locally/` or the appropriate CUDA index). After: `bash /workspace/scripts/pack_env.sh dsi` to repack.

**Run `python scripts/verify_assets.py` — every row of the verification matrix must pass green before any experiment launches.** If any fails, fix it (re-download, re-fetch, re-request access) and re-run. The CPU session was supposed to leave this green; if it's not, that's a bug to fix immediately, not work around.

Verify imports and the tests still pass: `pytest tests/`.

---

## Step 2 — read in this order, before doing anything else

1. `project_brief.md` — one-page orientation. Re-read.
2. `CLAUDE.md` — the operating manual. Full read, **especially the parallelization section (§4); it now binds.**
3. `PLAN.md` — the previous session's plan, with Phase 1b GPU schedule already laid out. Full read.
4. `task_descriptions/task_description_v1.md` — the spec. Full read. Refer back to it before each new experiment so priorities don't drift.
5. `task_descriptions/task_description_v1_appendix.md` — the ICLR-rigor binding extension. Re-read; the threat-model, three-tier generalization, statistical significance, and Phase-C concrete specs all bind every experiment you run.
6. `reports/INDEX.md` — what has been logged so far (initially: nothing or only the bootstrap entry).
7. The `dsi/` package source tree — skim, then re-read modules you're about to touch.

After reading, **append** a new "Phase 1b — GPU session start" section to `PLAN.md` (do not delete prior content) confirming or adjusting:
- The first work item to execute and which script runs it.
- The co-scheduling plan: which CPU job runs alongside the first GPU job.
- The next 3–4 experiments queued.
- The Phase C ideas you'll save for after the four contributions land.

---

## Step 3 — start the always-on infrastructure

Before launching any experiment, set up the persistent tmux infrastructure:

- `tmux new -d -s monitor 'nvidia-smi dmon -s pucvmet -d 5 > logs/gpu_monitor.log'` and a sibling `htop` watcher for CPU.
- `tmux new -d -s cpu-worker '<your CPU watcher loop, see below>'`. The watcher loop scans `outputs/` for un-evaluated bypass images and computes NudeNet / Q16 / SD-Safety-Checker labels in the background; it runs forever. Outputs from any GPU run get scored automatically — no manual launch step.

These two sessions stay up for the entire duration of the project. Verify with `tmux ls`.

---

## Step 4 — smoke tests on the GPU, then dry-runs

Before committing to multi-hour runs, verify the GPU pipeline works end-to-end and measure peak VRAM:

1. **SDXL Turbo + Surkov SAEs load test** (`scripts/smoke_sdxl_sae.py`). Generate one image from one prompt, capture all four SAE block activations. Log peak VRAM. Expected ~6 GB. Commit the smoke result to `reports/S00_smoke_sdxl_sae.md`.
2. **SAeUron reproduction smoke** (`scripts/repro_saeuron.py --dry-smoke`). Run their style-erasure on one UnlearnCanvas style. Confirm the output image looks correct (visual inspection — log to WandB as `wandb.Image`). Log peak VRAM.
3. **SD Safety Checker forward smoke** — load CompVis safety checker, run it on the generated image from step 1. Confirm a sane "safe" / "unsafe" output.
4. **NudeNet + Q16 forward smoke** (CPU is fine). Confirm both load, both score the generated image, both produce sane outputs.
5. **Pixel PGD dry-run** (`scripts/exp_A01_pixel_pgd.py --dry-run`) — load the full pipeline, run 3 PGD iterations on 4 prompts, log peak VRAM. Expected 14–16 GB. Confirm gradient-checkpointing keeps it under the 85% / 27 GB cap.
6. **VAE-latent PGD dry-run** (`scripts/exp_A02_latent_pgd.py --dry-run`) — same, expected 12–14 GB.
7. **CLIP-embedding PGD dry-run** (`scripts/exp_A03_embedding_pgd.py --dry-run`) — same, expected 3–4 GB.
8. **Detector-training smoke** (`scripts/exp_B01_detector_em.py --dry-smoke`) — 5 minibatch iterations on synthetic activations. Confirms checkpointing and resume work. Expected 2–4 GB.

Each smoke result lands a one-paragraph entry in `reports/INDEX.md`. The dry-runs feed your co-scheduling plan in §5.

---

## Step 5 — execute Phase 1b, with parallelization as a first-class concern

Follow `CLAUDE.md` §4 to the letter. **The default state of the box is "≥ 1 GPU job + ≥ 1 CPU job + monitor", always.** If you launch a GPU job and don't have a CPU companion running, that's a bug — find work to fill the CPU.

The dependency graph from `task_description_v1.md` §7:

```
Item 1 (env + baselines) ──┬──> Item 2 (red-team) ──┬──> Item 3 (detector) ──┬──> Item 5 (correction)
                            │                        │                          │
                            └────────────────────────┴────> Item 4 (cross-target) ──> Item 6 (paper)
```

Item 1 is mostly done from the CPU session; what remains GPU-side is the **clean-baseline generation** (1,000 COCO + 1,000 I2P clean SDXL Turbo images). Run that first, on GPU, with NudeNet labelling concurrent on CPU.

While Item 1 finishes:
- Begin **Item 2.1 (pixel PGD)** as the next GPU job — pixel PGD is the heaviest attack, will saturate one full GPU experiment slot.
- In parallel on CPU: dataset preprocessing, prompt deduplication, drafting `reports/A01.md` skeleton.

Once Item 2.1 finishes:
- **Item 2.2 (latent PGD)** and **Item 2.3 (embedding PGD)** can co-locate on the GPU (latent ≈ 12 GB, embedding ≈ 3 GB; combined ≈ 15 GB, comfortably under the 27 GB cap).
- In parallel: NudeNet/Q16 scoring of Item 2.1's bypass images on CPU.

Once Items 2.1–2.3 finish (collected SAE activations for all three attack spaces):
- **Item 3 (detector training)** uses ~4 GB. Co-locate with **Item 4 (cross-target re-attacks)** at 3–14 GB depending on attack space.
- In parallel on CPU: per-feature attribution analysis on the cached activations from Item 2; building the cross-space transferability matrix.

For each new experiment:
1. Refer to `task_description_v1.md` for the relevant work item's spec.
2. Plan co-scheduling **before launching** — name the partner; if you can't, find one.
3. Dry-run for 30–60 s, log peak VRAM, confirm fit.
4. Launch in tmux; log to WandB; write `reports/<exp_id>.md`; commit; push.
5. **Visually inspect output** for any change that affects rendered images (bypass examples, intervention before/after) — metrics alone are not enough.

---

## Step 6 — the evaluation grid

Once the four contributions are wired (items 2, 3, 4, 5 done), populate the §3.4 evaluation grid in `task_description_v1.md`. `scripts/eval_full_grid.py --row <name>` runs each row. Twelve rows × five seeds × multiple metrics = a non-trivial GPU schedule; **co-locate aggressively**:

- Reproductions of SAeUron / SAEmnesia / DSG-adapted are independent runs, ~10 GB each — pair them.
- The proposed two-stage + mean patch row uses our pipeline; it's ~8 GB; pair with another reproduction.
- FID / CLIP-score on cached COCO generations: pure CPU; runs alongside everything else.

Each row writes to `reports/grid_row_<name>.md`. The aggregated table goes to `reports/PHASE_1_FINAL.md` at the end of Phase 1b — that's the headline result for the paper.

---

## Step 7 — when Phase 1 lands, do not stop

Once the four contributions are passing on the gates and the headline table is populated, **do not pause**. Move to Phase C. `CLAUDE.md` §9 lists a starting set of ideas: safety-trained SAEs, transcoder detectors, multi-concept defense, cross-model transfer, black-box attacks, adversarial training of the detector, hybrid SAE+predicted-noise, patch-level attribution (CaFE), LoRA-baked safety, attack-trace visualizer, per-concept causal feature graphs, sequential concept removal.

For each Phase C idea pursued:
- Write the goal and method in `reports/<exp_id>.md` *before* launching, so it's not a fishing expedition.
- Estimate VRAM and CPU footprint; identify a co-scheduling partner.
- Prefer high-EV ideas (training a transcoder, fine-tuning with LoRA, cross-model transfer) over yet another hyperparameter sweep on already-tuned components.
- Combine ideas when sensible.

Read more papers. arXiv ICLR / ICML / NeurIPS / CVPR / ECCV 2024–2026 on diffusion safety, SAE interpretability, machine unlearning. Hugging Face papers feed. Papers With Code leaderboards. **Running out of ideas is not a stop condition.**

---

## Step 8 — the loop, until interrupted

1. Refer to `task_description_v1.md` and `PLAN.md` to anchor priorities.
2. Pick the next experiment from `PLAN.md` Phase C ideas (or the next pending row of the evaluation grid, or the next pending item dependency).
3. Plan co-scheduling.
4. Dry-run.
5. Launch in tmux; log to WandB; write the report.
6. Visually inspect any rendered output.
7. Commit; push.
8. If the result is a keep, integrate into defaults; commit again.
9. If the result is a discard, log the reasoning in the report.
10. Update `PLAN.md`.
11. Repeat.

**Do not pause to ask permission. Do not stop and wait. The human may be asleep.** The only stop condition is the human interrupting.

---

## Step 9 — the paper

The repo has a `paper/` directory. Maintain `paper/main.tex` as you go; every experiment that lands a result in the paper updates the corresponding section / table / figure. By the time the human stops you, `paper/main.tex` should be in submission shape — clear contributions, full evaluation grid, ablations, mechanistic plots, qualitative figures, related work, references. **ICLR-format LaTeX, paper-quality writing.** No fluff, no superlatives; state the result, the number, the comparison, the interpretation.

---

## Step 10 — output discipline (talking to the human)

When you produce text the human will read (commit messages, report bodies, occasional status if asked):

- **Direct. Plain. No superlatives.** No "I've successfully implemented…", no "this groundbreaking improvement…", no "remarkably, the result shows…".
- State the result, the number, the path to evidence.
- Bullet lists over prose for facts.
- One short paragraph of interpretation, max.
- No emojis. No filler. No re-stating what the human asked.

The reports are technical artifacts. Treat them like a paper appendix — every sentence earns its place.

---

That's it. Read CLAUDE.md, the v1 spec, the v1 appendix, PLAN.md. Plan with parallelization in mind — the default state of the box is "≥ 1 GPU + ≥ 1 CPU + monitor", always. Smoke-test, dry-run, launch in tmux, log to WandB, write the report, commit, push. When the four contributions land, move to Phase C. Read the literature when ideas slow down. Maintain the paper as you go.

**Do not pause to ask permission. Do not stop and wait. The human may be asleep.** The only stop condition is the human interrupting.

Execute. Improve. Don't stop.