# CLAUDE.md — DiffSafeSAE Autonomous Build & Improve (v2)

You are an autonomous research agent. You build the DiffSafeSAE pipeline end-to-end, get every spec gate to pass, then keep improving the pipeline (architectures, metrics, ablations, hyperparameter sweeps, more datasets, deeper mechanistic analysis, anything) until the human stops you. **Optimize relentlessly for the best end-to-end results, both quantitative (ASR, AUC, FID, CLIP-score, ablation tables) and qualitative (rendered example bypasses, per-feature activation maps, before/after intervention images). The target is A* AI-conference-grade (NeurIPS / ICML / CVPR / ICLR).**

You do not pause to ask permission. You do not stop and wait. The human may be asleep. The loop runs until manually interrupted.

---

## 1. What this project is — read first

`task_descriptions/task_description_v1.md` is the original spec (four contributions, six work items, evaluation grids, dependency graph). `task_descriptions/task_description_v2.md` is the **single combined v2 document**: it binds Phase 1c (structural-gap-closing), Phase D (ten new ambitious ideas), the new evaluation grid, the both-framing pursuit, and the hardware envelope. **All of v1, v1 appendix, and v2 are binding.** Read all three fully before anything else. Refer back to v2 before each new experiment.

Phase 1 has landed. Phase C is in flight (interrupted by the pod swap; resume per `STARTER_PROMPT_3.md`). v2 introduces the hardware upgrade, the image-saving discipline, the counterfactual benchmark, the both-framing pursuit, and ten new ambitious ideas (D-1 through D-10).

---

## 2. What to read, in order, before doing anything

1. `gen_ai_project_diffusion_interp/project_brief.md` — one-page orientation. Anchor the thesis.
2. `gen_ai_project_diffusion_interp/CLAUDE.md` — this file. Full read.
3. `gen_ai_project_diffusion_interp/task_descriptions/task_description_v1.md` — original spec. Full read.
4. `gen_ai_project_diffusion_interp/task_descriptions/task_description_v1_appendix.md` — v1 ICLR-rigor extension. Full read.
5. **`gen_ai_project_diffusion_interp/task_descriptions/task_description_v2.md`** — single combined v2 document. Phase 1c gates (eleven items, 1c-0 through 1c-10), the both-framing pursuit (§2, both framings active in parallel), the new evaluation grid (§4), the hardware envelope (§5), Phase D ten new ideas (§6, D-1 through D-10), the framing-decision moment (§7). **The most important new document. Full read; treat as binding.**
6. `gen_ai_project_diffusion_interp/PLAN.md` — the live plan from prior sessions. Append a new "Phase 1c — session start (v2 + 96 GB)" section; do not delete prior content.
7. `gen_ai_project_diffusion_interp/reports/INDEX.md` — chronological index of every prior experiment.
8. `gen_ai_project_diffusion_interp/reports/PHASE_1_FINAL.md` and `reports/PHASE_1B_CHECKPOINT.md` — what landed; what failed silently (the bit-identical detector logits bug, the prompt-origin label leak in B01, the SDXL-Turbo low base-flag-rate caveat).
9. `gen_ai_project_diffusion_interp/Swadesh_Swain_22116091.pdf` — original course proposal. Skim.
10. `/workspace/conda_setup.md` — pack/unpack conda workflow. Obey it.
11. The Surkov et al. paper (arXiv:2410.22366) and `surkovv/sdxl-unbox` README — for SAE hook plumbing.
12. The SAeUron paper (arXiv:2501.18052) and `cywinski/SAeUron` README — for the diffusion-native baseline reproduction (Item 1c-5).
13. The DSG paper (arXiv:2504.08192) and Arad et al. (arXiv:2505.20063) — for Contribution 4 method.

After reading, **append** a "Phase 1c — session start (v2)" section to `PLAN.md`. Cover: what's in flight (the C-experiments to resume), Phase 1c gate ordering, the hardware co-scheduling plan under the new envelope, the success criteria targeted per gate, the both-framing experiment plan.

---

## 3. Environment — pack/unpack conda

Conda env name: `dsi`. Activate with `conda activate dsi`.

The pod uses a pack/unpack workflow:
- Source of truth = `/workspace/env-archives/dsi.tar.gz`. The user has unpacked the env on this pod; activate and verify.
- After **any** package install/uninstall/upgrade you must `bash /workspace/scripts/pack_env.sh dsi` or your changes are lost on next pod.
- Caches are on the persistent volume (`PIP_CACHE_DIR`, `HF_HOME`, `TORCH_HOME` → `/workspace/.cache/...`). Don't fight them.
- pip for ML packages, conda only for Python interpreter / system libs.

The `dsi` package is wired into the env via a `.pth` file at `$CONDA_PREFIX/lib/python3.11/site-packages/dsi.pth`. `import dsi` works from any cwd.

Torch must be GPU-built (sm_120 compatible). The new pod's RTX PRO 6000 Blackwell Workstation runs sm_120; the existing wheel from the prior pod should work, but verify with `torch.cuda.get_arch_list()` on first session boot. CUDA 13.0 driver is installed.

Secrets in `gen_ai_project_diffusion_interp/.env`:
- `HF_TOKEN`, `HUGGINGFACE_TOKEN` — Hugging Face
- `WANDB_API_KEY` — Weights & Biases
- **`GEMINI_API_KEY` — Google Gemini for Strategy 3 paraphrase generation (v2 Item 1c-0)**

Load at process startup. Never echo, never commit.

**Gemini model priority order — cheapest first, fall back ONLY on rate-limit errors, NOT on individual content refusals** (refusals get logged and reported as a sampling-bias caveat; they do not trigger escalation to a more expensive model):
1. `gemini-3.1-flash-lite`
2. `gemini-3-flash`
3. `gemini-2.5-flash-lite`
4. `gemini-2.5-flash`
5. `gemini-2-flash`

In parallel, run **local Llama 3.1 70B paraphrase (int8)** as Path B for Strategy 3. The 96 GB GPU has the headroom (~40 GB resident); local runs handle whatever Gemini refuses. Both paraphrase paths run from the start — not "Path B as fallback only." See v2 §3 Item 1c-0 Strategy 3.

---

## 4. Hardware utilization — non-negotiable, this section is rewritten for the new pod

You have **1× RTX PRO 6000 Blackwell Workstation Edition** with **96 GB VRAM**, **600 W power**, **sm_120**, **CUDA 13.0**, paired with **AMD EPYC 9355** (64 vCPU) and **263 GB system RAM**. This is roughly 3× the VRAM, 4× the CPU cores, and 5× the RAM of the prior pod.

**The user has explicitly raised the cap** from the prior 85% target. New ceilings:
- **VRAM**: target **90%** (≈86 GB), hard ceiling **95%** (≈91 GB). The 5% buffer is for kernel-launch fragmentation. Do not push to 99%; OOMs cascade and kill innocent jobs.
- **GPU compute**: keep utilization ≥ **85%** during any active workload.
- **RAM**: target **90%** (≈237 GB), hard ceiling **95%** (≈250 GB). With 263 GB you can keep all four hookpoints × 100K samples × 5120 features in RAM simultaneously (~170 GB). No more disk-paging through SAE caches.
- **CPU**: keep ≥ **75%** of the 64 cores busy whenever the GPU is busy. cpu-worker pool target: **48 workers** (saturating to 64 contends with GPU launch threads).

**Be ambitious about scale.** A workload that would have been "Phase C maybe" on the old hardware is now "run today" on the new hardware. Larger SAE expansion factors, more seeds in parallel, full FLUX cross-architecture, joint end-to-end training — go.

### New default state of the box

Default is **≥ 3 GPU jobs + ≥ 12 CPU workers + monitor**, always. (Old default was 1+1+monitor; new default reflects the 3× VRAM and 4× core count.) If you launch one GPU job and the GPU is at 30 GB, the rest of the GPU is wasted — find another GPU experiment to fill it. **Typical hour: 15-25 active tmux sessions.**

### Parallelization rules — read carefully, this is critical

The work splits into two largely independent resource classes. **Plan every experiment by which class it lives in, then schedule it alongside experiments from the other class.**

**GPU-bound** (saturates VRAM and GPU compute):
- SDXL / SDXL Turbo / SDXL Base / SD v1.5 image generation
- FLUX inference (24 GB) and FLUX + SAE training (36 GB) — newly feasible on 96 GB
- SAE forward pass during generation
- Gradient-based PGD attacks on the full pipeline (14-16 GB peak per attack)
- Detector head training (4 GB; tiny relative to GPU)
- SAeUron / SAEmnesia / DSG-adapted reproductions
- Safety-specialized SAE training at expansion factor 16-32 (was 4 on the old hardware) — bigger SAEs now feasible
- LoRA-baked safety training (16-20 GB)
- Llama 3.1 70B paraphrase (40 GB int8, Strategy 3 Path B)
- Joint end-to-end pipeline training (32 GB, D-6)

**CPU-bound** (uses CPU cores, negligible GPU):
- Dataset preprocessing
- NudeNet / Q16 / safety-checker inference on cached PNGs (CPU is fine)
- FID / CLIP-score / LPIPS / DreamSim computation
- Per-feature attribution analysis on cached activation tensors
- Counterfactual prompt-edit dictionary application (Item 1c-0 Strategy 1)
- Hyperparameter sweep orchestration
- Plot rendering, table generation, side-by-side image grid construction

### Co-scheduling patterns you must use (substantially scaled up from v1)

1. **Five-seed parallelism**: any 5-seed CI requirement is *not* serialized. 5 seeds × 14 GB each = 70 GB, fits one GPU. Launch all five seeds simultaneously in five tmux sessions. (Old hardware required serialization or 5 separate cards; new hardware doesn't.)

2. **Cross-model co-location**: SDXL Turbo + 4 Surkov SAEs (~6 GB) + FLUX + 4 FLUX-SAEs (~36 GB) = 42 GB. Co-locate for cross-architecture experiments (D-9). Run two diffusion stacks in parallel.

3. **Attack + eval pipelining**: while a PGD attack runs on GPU, NudeNet/Q16/safety-checker scoring of the *previous* attack's bypass images runs on CPU. Never serialize "GPU finishes → CPU starts → next GPU".

4. **Generation + detection pipelining**: when generating 5K SDXL images for activation collection, run a CPU worker pool consuming each finished image. Don't wait for all 5K before labeling.

5. **Multi-experiment GPU co-location**: examples:
   - SDXL Turbo + Surkov SAEs (6 GB) + CLIP-embedding PGD (3 GB) + detector training (4 GB) + safety-trained SAE training expansion-32 (28 GB) + FLUX inference smoke (24 GB) → five workloads, ~65 GB total, fits comfortably.
   - When pixel-PGD is loaded (~14 GB), don't co-schedule another large GPU model — but absolutely run 4-5 other small GPU jobs + the 48-worker CPU pool in parallel.

6. **Sweep parallelism**: launch many WandB agents in separate tmux sessions. With 96 GB, that's 6 concurrent 14 GB sweep agents on one card.

7. **Dataset preprocessing in background**: kick off in a separate tmux session in parallel.

8. **Always-on CPU worker pool**: maintain `cpu-worker-{1..48}` watchers scanning for un-evaluated outputs and computing metrics.

9. **In-RAM activation cache**: 263 GB RAM is large enough to hold all four hookpoints × 100K samples × 5120 features. Build `dsi/util/activation_cache.py` with an LRU in-memory cache; flush to disk only on eviction. Query latency < 100µs per (exp, sample). Clamp max size at 200 GB.

**Process for every new experiment**: before launching, write down (in `PLAN.md` or in the experiment's report draft) which resource class it lives in, what its peak VRAM / RAM / CPU usage is, what it can be co-scheduled with. If you can't name a co-scheduling partner, find one. **The default state of the box is "≥ 3 GPU + ≥ 12 CPU + monitor" — never "1 GPU job at 20 GB, 76 GB idle, everything else asleep".**

### Hard guardrails

1. **Dry-run every new experiment** for 30–60 s with `nvidia-smi --query-gpu=memory.used,utilization.gpu,memory.total --format=csv -l 1` logged. Record peak VRAM, RAM, steady-state utilization. Use that to plan co-scheduling.

2. **Never run a single experiment idle on the GPU.** If you have spare capacity, fill it: ablations, sweeps, eval re-runs, dataset preprocessing, plotting. Default question after launching a job: "what else can run alongside this".

3. **Continuous monitoring**: keep a `monitor` tmux session running `nvidia-smi dmon -s pucvmet -d 5 > logs/gpu_monitor.log` and a sibling CPU monitor (`htop` or `pidstat`). Read the tail every ~10 minutes. **If GPU utilization drops below 60% for >5 min while jobs are queued, you're under-utilizing. If VRAM usage drops below 40 GB for >10 min while jobs are queued, you're definitely under-utilizing. Launch more.**

4. **VRAM safety**: if `nvidia-smi` reports VRAM > 95% or you see a CUDA OOM, kill the lowest-priority job and re-plan. Never let a long sweep crash because you over-packed. The 5 GB buffer is non-negotiable.

5. **RAM safety**: if `free -g` reports RAM > 95% (≈ 250 GB), kill the lowest-priority CPU job. Activation caches clamped at 200 GB max.

6. **Process isolation**: when co-scheduling on the same GPU, use separate Python processes (separate tmux sessions), not threads. Threads share the CUDA context and cause random hangs.

---

## 5. tmux — every long-running thing goes in a session

Naming convention:
- `monitor` — `nvidia-smi dmon` and CPU monitor
- `cpu-worker-{1..48}` — always-on CPU evaluation/labeling watchers (48 of them on this hardware)
- `train-<exp_id>` — one per training run
- `attack-<space>-<exp_id>-seed<N>` — one per attack run × seed
- `infer-<exp_id>` — one per inference run
- `eval-<exp_id>` — eval scripts
- `sweep-<name>-<n>` — sweep workers (one tmux per agent, numbered)
- `prep-<dataset>` — dataset preprocessing jobs
- `gen-<dataset>` — image generation runs
- `cf-strategy{1,2,3a,3b}` — counterfactual benchmark builders (Item 1c-0); 3a = Gemini, 3b = local Llama
- `paraphrase-gemini` / `paraphrase-llama70b` — paraphrase workers

```bash
tmux new -d -s <name> 'cd /workspace/swadesh/gen_ai_project_diffusion_interp && conda activate dsi && <cmd> 2>&1 | tee logs/<name>.log'
tmux ls
tmux capture-pane -t <name> -p -S -200
tmux kill-session -t <name>
```

Always redirect to a log file in `logs/`. Never let a tmux session lose its scrollback.

---

## 6. WandB logging — mandatory for every run

Project name: `dsi-v1`. One run per experiment. Required per run:

- **Config**: full hyperparameter dict, git commit short hash, dataset name + version, model name + checkpoint hash, attack space, ε, seed, hardware (record `RTX_PRO_6000_BW_WK`).
- **Scalars**: train/loss, val/loss, lr, epoch, step. For attacks: ASR, mean perturbation norm, mean SSIM, mean LPIPS. For detector: AUC, AP, F1, FPR@95%TPR, per-step AUC. For correction: ASR, FID, CLIP-score, LPIPS-vs-pre, DreamSim, latency-ms.
- **Qualitative artifacts** (mandatory, expanded from v1):
  - Attack runs: log a grid of (clean image, perturbed image, perturbation × 10) for **≥ 8 prompts per run** as `wandb.Image`.
  - Detector runs: ROC curves, commit-knee plot, top-activated SAE features as bar charts, **plus the 8 highest-confidence false positives and 8 highest-confidence false negatives from the held-out set**.
  - Correction runs: before/after image grids for **every** corrected case AND **every** false-positive collateral case (not just first-N), per-timestep activation deltas as heatmaps.
- **System**: WandB auto-logs GPU/CPU/RAM — leave on.
- **Tags**: `red-team`, `pixel`, `latent`, `embedding`, `detector`, `correction`, `baseline`, `ablation`, `sweep`, `phase-c`, `phase-1c`, `phase-d`, `repro-saeuron`, `repro-saemnesia`, `repro-dsg`, `counterfactual`, `cross-arch`, `framing-A`, `framing-B`, `improvement-<idea-name>`.

Init pattern:
```python
import wandb, os
wandb.init(project="dsi-v1", name=exp_id, config=cfg, tags=tags,
           dir="logs/wandb", reinit=False)
```

Local logs: every run also writes `logs/<exp_id>.log` and `reports/<exp_id>.md`.

---

## 7. Image-saving discipline (NEW in v2 — binding)

Phase 1's biggest weakness was missing visual evidence. Phase 1c fixes this. **Every** attack, intervention, and detector experiment now saves images per these rules:

- **Save every bypass case** as PNG. Not "first N". For 200-prompt attacks, that's ~17-50 bypass PNGs at ~400 KB each — trivial.
- **Save every corrected case** in interventions.
- **Save every false-positive case** in detectors (flagged but oracle-clean).
- **Save 50 random non-bypass / non-corrected cases per experiment** as the negative control sample.
- **Save perturbation visualization** `<seed>.perturb.png`: `(post - pre) × 10` clipped to [0, 255].
- **Save per-feature activation heatmap** at the timestep of intervention as `<seed>.heatmap.png` for each corrected case (intervention experiments only).

Build `dsi/util/img_saving.py` with `save_case(out_dir, seed, kind, pre_img, post_img, perturb_img, meta)`. Every script calls it.

For each experiment, generate `outputs/<exp_id>/figure.png`: a 4×4 grid of the most informative cases. **No experiment counts as "done" until `figure.png` exists.**

---

## 8. Checkpointing — every training run

- Save every N steps (~5-10 min cadence).
- Filename: `checkpoints/<exp_id>/step_<N>.pt`. Also a `last.pt` symlink.
- **Keep only the latest 4 checkpoints**, delete older. Rolling deque in the training loop.
- Save: model state, optimizer state, scheduler state, RNG states (torch + numpy + python), step, epoch, best metric, full config.
- **Resume**: every train script accepts `--resume <path>` and `--resume latest`. Bit-identical continuation when seeds and data order are restored.
- Final/best checkpoint: copy to `checkpoints/<exp_id>/best.pt` (does not count toward the 4-deep limit).

Apply to: detector training, LoRA / SAE retraining, learned-projection training (D-2), joint end-to-end pipeline training (D-6), adversarial training (D-8), every Phase D training experiment.

---

## 9. Reports — one markdown per experiment

Write `reports/<exp_id>.md` for every experiment. Sections, in this order, no fluff:

```
# <exp_id> — <one-line summary>

## Goal
What you tried to verify or improve. One paragraph.

## Procedure
Exact steps. Code paths touched. Hyperparameters. Dataset(s). Random seeds.
Hardware (GPU/CPU/VRAM/RAM peak). What was co-scheduled with this job.

## Results
Tables and numbers. WandB run URL. Saved-artifact paths. Compare to baseline /
previous best. Embed `outputs/<exp_id>/figure.png` reference. List bypass/
corrected/false-positive case counts.

## Interpretation
What the numbers mean. Why it worked or didn't. What this rules in/out.

## Next
What you'll try next based on this. Link to the next experiment if started.
```

Direct, plain language. No "I've successfully…", no "this groundbreaking…", no superlatives, no hedging beyond what evidence requires. State the result, the number, the path to evidence. **The human is reading these to make decisions; words that don't change a decision are wasted tokens.**

Maintain `reports/INDEX.md` listing every experiment chronologically.

---

## 10. The improvement loop

### Phase 1 — DONE. See `reports/PHASE_1_FINAL.md`.

### Phase 1c — close structural gaps (eleven items)

Per `task_descriptions/task_description_v2.md` §3, run Items 1c-0 through 1c-10. Most parallelize. **Phase 1c does not pause Phase C** — Phase 1c work runs alongside resumed C-experiments. Phase 1c does not pause Phase D either — items can run as soon as their dependencies are satisfied.

### Both framings active

Per v2 §2, the agent runs experiments for Framing A (original four contributions, Contribution 2 narrowed) **and** Framing B (causal interpretability for diffusion safety) in parallel. The canonical paper draft `paper/main.tex` is written under Framing A; `paper/alt_framing_B.md` is maintained as a structured outline updated as experiments land. Both stay current. **Do not skip Framing B's experiments because Framing A is "primary" — both are committed deliverables.**

When the four framing-discriminator experiments close (Item 1c-0 + 1c-1 + 1c-3 + Phase-C-2-on-counterfactual), write `reports/REFRAMING_DECISION.md` with the four supporting numbers and commit to Framing A or Framing B for the paper's primary structure. The other framing's notes archive to `paper/archive/` — they may inform supplementary appendices.

### Phase D — ten new ambitious ideas

Per v2 §6, all ten run. EV-descending order; combine when sensible. **Be ambitious; the new hardware is sized for this.** Don't be afraid of complex methods if simpler ones don't work. The 96 GB / 64 vCPU / 263 GB pod is paid for; under-utilizing it wastes the resource.

For every Phase D experiment: write the goal and method in `reports/<exp_id>.md` *before* launching. Save all images per §7. Run 5 seeds for any number that hits the headline table.

### Idea sourcing — read the literature

arXiv search for "diffusion safety", "sparse autoencoder concept unlearning", "machine unlearning text-to-image", "feature circuits", "attribution patching", "in-generation NSFW detection", year ≥ 2024. Skim abstracts, fetch PDFs of promising ones. **Running out of ideas is not a stop condition.** Read more, propose more, combine more.

### Loop body, after each experiment

1. **Refer to `task_description_v2.md`** every time before picking the next experiment.
2. Pick the highest-EV untried idea from `PLAN.md`.
3. Estimate VRAM: dry-run for 60 s.
4. **Plan co-scheduling**: identify a CPU-bound or independent GPU-bound experiment. Default state: ≥ 3 GPU + ≥ 12 CPU + monitor.
5. Launch in tmux; log to WandB; **save all images per §7**; write the report.
6. While it runs, design the next experiment, populate the next `reports/<exp_id>.md` skeleton, update `PLAN.md`.
7. When it finishes: write `reports/<exp_id>.md`, update `reports/INDEX.md`, generate `outputs/<exp_id>/figure.png`, commit, push.
8. If keep, integrate into defaults. **Always check qualitative output too** — a higher AUC with worse intervention images is a regression.
9. If discard, log reasoning.
10. **Repack the env** if you installed any new packages.
11. Be ambitious with the next idea. Don't ladder up cautiously when you can try something ambitious. The hardware can take it.
12. Go to 1.

**Stop conditions**: human interrupts. That's the only one. Running out of ideas is not a stop condition.

---

## 11. Resuming after the GPU pod swap

The pod swap terminated several Phase C experiments. **First-hour priority** (after standard reading + verify_assets + GPU smoke):

1. Re-launch every in-flight experiment listed in `reports/INDEX.md` last entries that's not marked `keep` or `discard`. Specifically: C-1 black-box Square Attack, C-2 AxBench (now retargeted to counterfactual benchmark per Item 1c-0), C-3 safety-trained SAE detector, C-6 hybrid raw‖SAE adversarial-robustness eval, C-9 transcoder detector, C-10 LPIPS / DreamSim quality preservation, FID-series, LPIPS-D02/D03/D04 CPU jobs, score-base-i2p CPU labeler.

2. Check `checkpoints/<exp_id>/last.pt` and `outputs/<exp_id>/` for partial state. If checkpoint is ≥ 80% complete, resume; otherwise restart cleanly.

3. **Scale up to use the new hardware.** Per Item 1c-10: bump n=50 → n=500 on C-1; train C-3 at expansion 16 and 32 (was 4); sweep L0 in parallel; run C-9 at full hookpoint coverage.

4. Document any state lost in `reports/<exp_id>.md` under a "v2 resume note".

5. Once in-flight experiments are re-launched, proceed to Phase 1c Item 1c-0 (counterfactual benchmark) as the next new work, in parallel.

**The pod swap is not a reason to wait.** It's a reason to launch all the in-flight C-experiments first, scale them up, then start Phase 1c on top.

---

## 12. Output discipline (talking to the human)

When you produce text the human will read (commit messages, report bodies, occasional status if asked):

- **Direct. Plain. No superlatives.** No "I've successfully implemented…", no "this groundbreaking improvement…", no "remarkably, the result shows…". Just say what happened.
- State the result, the number, the path to evidence. If the number is 0.847, write `0.847`, not "approximately 0.85".
- Bullet lists over prose for facts.
- One short paragraph of interpretation, max.
- No emojis. No decorative characters. No "🎉" or "✨" or "🚀" anywhere.
- No filler ("It's worth noting that…", "Interestingly, …", "As expected, …"). Cut these.
- No re-stating what the human asked.

The reports are technical artifacts. Treat them like a paper appendix — every sentence earns its place.

---

## 13. Coding rules

- **Short names.** `clf_logits` not `unsafe_image_classifier_logits`.
- **No emojis or visual characters in `print` / log statements.**
- **No narration comments.** Comments only for non-obvious intent.
- **No inline imports.** Imports at the top of each file.
- **Exhaustive switch handling** for any `Literal[...]` / enum field.
- **No bare `except:`.** Catch the specific exception.
- **Type hints** on every public function signature.
- **Determinism**: set seeds at every entry point. Log them.
- **Hooks discipline**: SAE / safety-checker / detector hooks register and unregister cleanly. No global state. Use a context manager.
- **No hardcoded paths**. Read from `dsi/config.py`.

---

## 14. Git — commit every meaningful step

GitHub SSH auth configured. Active account: **Swadesh06**. Always SSH URLs (`git@github.com:OWNER/REPO.git`).

Commit policy:
- First v2 session commit: `chore: phase 1c start, v2 docs ingested, in-flight C-experiments resumed`.
- Each Phase 1c item lands: `1c-N(<scope>): <one-line result>` with metric in body.
- Each new experiment: `exp(<exp_id>): <one-line idea>`.
- Each report: `report(<exp_id>): <result one-liner>`.
- `PLAN.md` updates: `plan: <one-line change>`.
- `paper/main.tex` updates: `paper: <section> — <change>`.
- Reframing decision: `decision: framing-A canonical` or `decision: framing-B canonical`.
- Push to `origin` after every commit.

What never gets committed: `.env`, `*token*`, `*key*`, `*secret*`, `checkpoints/`, raw `outputs/` (except `outputs/figures/`), `logs/`.

Branching: `main`. For risky architectural changes (training SAEs from scratch at expansion 32, swapping diffusion backbone, adding a transcoder, FLUX cross-arch, joint end-to-end training) branch as `exp/<exp_id>` and merge back when the report says "keep".

---

## 15. Safety, ethics, dual-use

NSFW research artifacts stay gitignored. Attack code documented with threat-model context. Curated paper figures committed only after manual review. WandB project private. MMA-Diffusion image set respected as gated.

---

## 16. Repo layout — keep it tidy

```
gen_ai_project_diffusion_interp/
├── CLAUDE.md                                     # this file (v2)
├── project_brief.md                              # one-page orientation
├── PLAN.md                                       # live plan, append-only
├── pyproject.toml
├── requirements.txt
├── README.md
├── Swadesh_Swain_22116091.pdf
├── .env                                          # gitignored — secrets (HF, WandB, GEMINI)
├── .gitignore
├── dsi/
│   ├── __init__.py
│   ├── config.py
│   ├── attacks/                                  # pixel.py, latent.py, embedding.py, common.py
│   ├── sae/                                      # hooks.py, load.py, attribution.py
│   ├── detectors/                                # sae_em.py, sae_ft.py, baselines/, train.py
│   ├── interventions/                            # stage1_fisher.py, stage2_causal.py, patches.py, projection.py (D-2), pipeline.py, baselines/
│   ├── models/                                   # sdxl_pipeline.py, sd15_pipeline.py, flux_pipeline.py (D-9), classifier_oracles.py
│   ├── eval/                                     # asr.py, fid.py, clip_score.py, lpips.py, dreamsim.py, unlearncanvas.py, unlearndiffatk.py, commit_knee.py
│   ├── data/                                     # i2p.py, coco.py, laion_coco.py, unlearncanvas.py, unlearndiffatk.py, adversarial.py, counterfactual.py (1c-0 S1+S2), paraphrase.py (1c-0 S3 Gemini), paraphrase_local.py (1c-0 S3 Llama)
│   └── util/                                     # wandb.py, ckpt.py, seed.py, logging.py, img_saving.py, activation_cache.py
├── scripts/
│   ├── bootstrap.sh
│   ├── pack_env.sh
│   ├── verify_assets.py
│   ├── gate_clean_baseline.py
│   ├── exp_A0X_<attack-space>_pgd.py
│   ├── exp_BXX_detector_<regime>.py
│   ├── exp_CXX_xtarget_<space>.py
│   ├── exp_DXX_<intervention>.py
│   ├── repro_saeuron.py / repro_saemnesia.py / repro_dsg.py
│   ├── eval_full_grid.py
│   ├── build_counterfactual_benchmark.py         # Item 1c-0 driver
│   ├── train_learned_projection.py               # D-2
│   ├── trace_causal_graph.py                     # D-1
│   ├── flux_cross_arch.py                        # D-9
│   ├── joint_e2e_train.py                        # D-6
│   ├── adv_train_two_stage.py                    # D-8
│   ├── compositional_defense.py                  # D-10
│   └── sweep_<name>.py
├── app/
│   └── streamlit_app.py
├── paper/
│   ├── main.tex                                  # canonical (Framing A working)
│   ├── alt_framing_B.md                          # parallel structured outline (Framing B)
│   ├── archive/                                  # un-chosen framing's notes after decision
│   └── refs.bib
├── reports/
│   ├── INDEX.md
│   ├── PHASE_1_FINAL.md
│   ├── PHASE_1B_CHECKPOINT.md
│   ├── PHASE_1C_PROGRESS.md
│   ├── REFRAMING_DECISION.md                     # written at the framing-decision moment
│   └── <exp_id>.md
├── task_descriptions/
│   ├── task_description_v1.md
│   ├── task_description_v1_appendix.md
│   └── task_description_v2.md                    # single combined v2 file (binding)
├── checkpoints/                                  # gitignored
├── logs/                                         # gitignored
└── outputs/                                      # gitignored except outputs/figures/
    ├── figures/                                  # committed paper figures
    ├── cf_benchmark_v1/                          # Strategy 1 prompt-edit pairs
    ├── cf_benchmark_v1_seed/                     # Strategy 2 same-prompt pairs
    ├── cf_benchmark_v1_paraphrase_gemini/        # Strategy 3 Path A
    ├── cf_benchmark_v1_paraphrase_llama/         # Strategy 3 Path B
    └── <exp_id>/{pre,post,perturb,heatmap,figure.png}/
```

---

## 17. The first thing you do, every session

1. Read this file in full.
2. Read `project_brief.md`.
3. Read `task_descriptions/task_description_v1.md` in full.
4. Read `task_descriptions/task_description_v1_appendix.md` in full.
5. **Read `task_descriptions/task_description_v2.md` in full** — single combined v2 document; the most important document.
6. Read `PLAN.md`.
7. Read the most recent 5 entries in `reports/INDEX.md`.
8. Then, and only then, plan the next move.

Don't skim. Don't pattern-match. Read.

---

## 18. The one thing that matters

Get the four contributions to passing on the v2 gates. Run both framings' experiments in parallel. Hit the framing-decision moment with the four supporting numbers. Run the ten new ambitious ideas. Iterate aggressively until the human stops you, optimizing relentlessly for A* AI conference-grade results, both quantitative and qualitative.

That's it. Read CLAUDE.md, the v1 spec, the v1 appendix, the v2 spec, PLAN.md, the recent reports. Plan with parallelization in mind — the default state of the box is "≥ 3 GPU + ≥ 12 CPU + monitor", always. **Saturate the 96 GB / 64 vCPU / 263 GB hardware; treat 50%-utilized as broken; treat 30 GB-of-VRAM-used while jobs are queued as a bug.** Save every relevant image, every time. Write reports in plain language; no superlatives, no fluff. Read the literature when you run out of ideas; running out of ideas is not a stop condition. Commit every meaningful step; push to origin. **Be ambitious — run high-scale experiments, the hardware is sized for it, the user has explicitly raised the cap.**

Execute. Improve. Don't stop.