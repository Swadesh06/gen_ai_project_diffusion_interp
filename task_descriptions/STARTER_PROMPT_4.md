# STARTER_PROMPT_4.md — v3 closure session start

You are continuing the DiffSafeSAE research project. **The framing has shifted.** Phase 1 + Phase 1c + Phase D have produced experimental material, but an honest audit by the human shows none of the four contributions is yet at publication standard. v3 binds the agent to **publication-grade closure** of the existing four contributions — no new ideas, no paper writing.

You operate by the rules in `CLAUDE.md` (v3). Read it; follow it.

---

## What changed

The human audited the v2 results and found a systematic pattern:

1. **Experiments at MVP scale, not paper scale**. n=10-200 per cell where 1000-5000 is the publication norm for some comparisons. Baselines reproduced at n=30 vs F_c at n=10 isn't a comparison.
2. **Baselines not reproduced at matched conditions**. The "we beat SAeUron / DSG / SAEmnesia" headline rests on a single SAeUron smoke test with overlapping CIs and two methods that were never reproduced at all.
3. **Cross-classifier comparisons at mismatched conditions**. The "SAE detector is more robust than safety_checker" claim is at 10× different query budgets. Untestable from current data.
4. **Ensemble across 10 trained B02-v3 heads — never tried**. Architecturally cheap, methodologically expected, missing.
5. **Positive findings not completed to publication standard.** D-1 causal graph correlation-based not attribution-patching. D-2 learned projection trained but never applied at intervention time. D-7 trajectory uses confounded baseline. UDA-nudity OOD test shows F_c net-negative — no conditional-gating fix attempted.

The pattern: the agent was rewarded for breadth and moved on at MVP gates, leaving each contribution at "experimental material exists" rather than "paper-grade evidence exists."

**v3 closes this gap. No new ideas. No new contributions. Finish the four contributions to publication standard.**

### What v3 explicitly removes from v2

- **Paper drafting**. The human will write the paper. The agent does not draft prose, abstracts, related-work text, intro, conclusion, or LaTeX sections. The agent does produce programmatic tables (CSV/JSON) and figures (PDF/PNG) as evidence artifacts. These go in `outputs/figures/` and `outputs/tables/`.
- **Mandatory 5-seed CIs everywhere.** v3 prioritizes **completed comparison grids at single seed** over multi-seed runs on partial grids. Seed-noise side-runs come *after* the main grid completes. Where existing 5-seed data exists, use it; don't downsample. Don't repeat experiments for variance characterization at the expense of finishing the grid.
- **Vague "paired-bootstrap CI" stat-rigor language.** v3 cares about concrete things: respectable n, same-prompt-list-matched baselines, ablation tables. Effect-size reporting (Δ absolute and Δ%) matters more than confidence-interval theatrics.
- **The "3-day stall" criterion.** Replaced by **continuous billed-by-the-second monitoring** — the GPU is billed per second; the agent checks the monitor at every loop iteration and acts immediately on idle hardware.

### What's changed since v2 — new hardware

The pod is now **1× NVIDIA A100 SXM4 80GB (sm_80, Ampere, HBM2e ~2 TB/s bandwidth), AMD EPYC 7763, ~233 GB system RAM, driver 580.159.03, CUDA 13.0, power limit 400 W.** (Previously v2 targeted a Blackwell PRO 6000 WK with 96 GB / sm_120 / EPYC 9355 / 263 GB.)

Key things about this hardware:
- **All v3 workloads fit at full scale**, including the heaviest (SD3 SAE training ~36 GB, SAEmnesia from-scratch ~28 GB). No staged execution, no skipping cells.
- **HBM2e memory bandwidth ~2× the consumer Blackwell** Diffusion attention and SAE forward passes will run noticeably faster. **Don't artificially batch-throttle** assuming Blackwell numbers.
- **sm_80 architecture** is the most-targeted training arch in the PyTorch ecosystem. The existing `dsi` conda env's torch wheel includes sm_80 by default. If `torch.cuda.get_arch_list()` shows sm_80 and a `tensor.cuda()` smoke works, **do not reinstall anything**. Don't fix what isn't broken.
- **vCPU count is allocated by Runpod, not the full 64 cores of EPYC 7763.** Verify `nproc` on session boot. Scale the CPU worker pool to **75 % of allocated `nproc`** — e.g. 16 vCPU → 12-worker pool, 32 vCPU → 24-worker pool. Don't hardcode 48.

The 90 % VRAM target / 95 % hard ceiling stays. The "≥ 3 GPU + ≥ N CPU + monitor as default" stays, with N scaled to allocated nproc.

### What's new in v3 operating behavior

Two new binding rules, motivated by what the new hardware enables and the deliverable demands:

**1. Throughput-discipline rule — parallelism must be faster than sequential.**

Running many GPU jobs concurrently helps *only* if aggregate throughput rises with each added job. On a single GPU, processes share CUDA streams and contend for memory bandwidth and SMs. 5 jobs each running at 20 % of solo throughput is the same wall-clock as one job running solo. After launching any new parallel GPU job, the agent:

1. Notes the per-job step-time (or sample-time, or attack-iteration-time) of the (N-1) already-running jobs from their tmux log tails. Sums to aggregate throughput.
2. Launches the Nth job. Waits until all N jobs have produced ≥ 3 progress samples in their logs.
3. Re-measures aggregate throughput across all N jobs.
4. **Aggregate throughput with N jobs must be meaningfully higher than aggregate throughput with (N-1) jobs.** Heuristic: each added GPU job should contribute at least ~60 % of its solo throughput to the aggregate. If adding the Nth job lifts aggregate by less than ~30-40 % of the new job's solo speed, you've hit contention.
5. If parallelism is hurting: kill the most recently launched GPU job; try a different mix; move CPU-bound work off the GPU (NudeNet, FID, CLIP-score, LPIPS, paraphrase rendering, oracle labelling all run perfectly on CPU); or use MIG to slice the A100. Accept current peak rather than push past it.
6. If GPU is at < 60 % utilization with N concurrent jobs, the running jobs are CPU- or I/O-bound — *add more GPU work* to push utilization higher.

Target on this A100 with 4-5 concurrent diffusion attacks: ~3-4× aggregate throughput vs solo. Less than ~2× aggregate with 3 jobs running means something is wrong; diagnose.

**CPU-bound work runs freely** — does not contend with GPU work. Run as many CPU jobs in parallel with GPU work as the vCPU pool supports. The throughput check applies only to GPU-resident jobs.

**2. Full-scale principle — no shortcuts, no fallbacks.**

The A100 80 GB fits every v3 workload at the full configuration the published baseline papers use. **The agent does not downscale architectures, reduce SAE expansion factors, drop concepts, or shorten experiments to make things fit.** Publication-grade comparisons require apples-to-apples reproduction.

If a workload appears not to fit, the response order is:
1. **Hyperparameter / runtime adjustment first**: reduce batch size, increase gradient accumulation, enable gradient checkpointing, switch fp32 → bf16, micro-batch the SAE forward pass, offload optimizer state to CPU (DeepSpeed-style). These preserve the architecture exactly.
2. **If (1) doesn't unblock**: report the specific OOM and exactly what was tried in `reports/V3_OOM_<exp_id>.md`. Pause that cell.
3. **Architecture-level downscaling is not permitted.** No reducing SAE expansion from 16 → 4 to make it fit. No reducing the number of SAEmnesia concepts. No partial reproduction at smaller scale.

Every cell runs at full scale or doesn't run.

### Don't fix what isn't broken

The conda env at `/workspace/env-archives/dsi.tar.gz` was packed on a Blackwell pod (sm_120), but PyTorch wheels since 2.0 ship with sm_80 included by default. The same env almost certainly works on this A100. The agent's job on session boot is to **verify**, not preemptively rebuild. Specifically:

- `torch.cuda.get_arch_list()` → includes `sm_80` → env works for A100, no reinstall.
- `torch.cuda.get_device_name(0)` → A100-SXM4-80GB → device is recognized.
- `torch.tensor([1.0]).cuda()` → no error → smoke test passes.

If all three pass, proceed to work. Don't reinstall torch "to be safe." Don't rebuild the env "in case of compatibility issues." Don't run `pip install -U` on anything that's working. The only acceptable trigger for reinstall is a specific failure of one of the three checks above; diagnose that specific failure and fix only it.

The volume is already 90 % full (577 / 640 GB) from the startup script having unpacked the env and project — that's expected, not a problem. Don't prune anything proactively.

---

## Step 1 — pick up project state

```bash
cd /workspace/swadesh/gen_ai_project_diffusion_interp
git pull
conda activate dsi
```

**Verify the GPU**: `nvidia-smi` → `A100-SXM4-80GB`, driver 580.159.03, CUDA 13.0. In Python: `torch.cuda.get_device_name(0)` returns the A100, `torch.cuda.get_arch_list()` includes `sm_80`, and `torch.tensor([1.0]).cuda()` succeeds. **If all three pass, do not reinstall torch or rebuild the env. Don't fix what isn't broken.** If any fails, diagnose the specific failure; fix only that.

**Verify the CPU**: `nproc` → record the allocated vCPU count (likely 16 or 32, not the full 64 of EPYC 7763). `free -g` → ~233 GB total. Set the CPU worker pool size to ⌊0.75 × nproc⌋ in your PLAN.md "v3 — session start" entry.

**Verify the env**: `pytest tests/` — all pass.

**Run `python scripts/verify_assets.py`** — must be green.

**Verify Gemini access**: `gemini-3.1-flash-lite` 1-token ping. (Strategy 3 paraphrases at 0-refusal landed under v2; this is just an env check.)

---

## Step 2 — read the docs, in this order

1. **`task_descriptions/task_description_v3.md`** — read in full. The most important document. Pay special attention to:
   - **§1 (audit)** — the honest list of what's broken per contribution.
   - **§2 (operating principles)** — completion, size-and-completeness, single-seed-now-noise-later, baselines-at-matched-conditions, ensemble-first, ablation-completeness, no-new-ideas, no-paper-writing. These are binding.
   - **§3 (the four closure gates)** — the cells that must fill. Internalize the cell count and current % per gate.
   - **§4 (experiment-size and comparison-rigor floor)** — n_pre_flagged ≥ 200 for ASR-type cells, same-prompt-list rule, single-seed-first.
   - **§6 (continuous billed-by-the-second monitoring)** — what to check at every loop iteration.
   - **§8 (framing settled)** — Framing A canonical; Framing B archive.
2. `CLAUDE.md` (v3) — the operating manual.
3. `PLAN.md` — the live plan. Append a new "v3 — session start" section.
4. `reports/V3_CLOSURE_PROGRESS.md` — create with all zeros if missing; otherwise read.
5. `task_descriptions/task_description_v2.md` — skim; binding where not superseded by v3.
6. `task_descriptions/task_description_v1.md` and `task_descriptions/task_description_v1_appendix.md` — re-anchor priorities; binding where not superseded.
7. `reports/INDEX.md` — every prior experiment.
8. `reports/SESSION_PHASE_1C_FINAL_v2.md` and `reports/REFRAMING_DECISION.md` — what landed at v2 close.
9. The 5 most recent `reports/<exp_id>.md` entries.

After reading, **append** a "v3 — session start" section to `PLAN.md`. Cover: current cell counts per gate (read from `V3_CLOSURE_PROGRESS.md`), which gate is in focus, the next 3-5 experiments planned, what's co-scheduling with what.

---

## Step 3 — bootstrap the v3 infrastructure

These are one-time setup tasks. Do them before launching any experiments.

### 3.1 Start the `monitor` tmux **first**

Before any other work. The GPU is billed per second; the monitor must be running before any real experiment so idle moments are catchable.

```bash
tmux new -d -s monitor 'nvidia-smi dmon -s pucvmet -d 5 > logs/gpu_monitor.log 2>&1'
tmux new -d -s monitor-cpu 'while true; do echo "$(date) $(free -g | grep Mem) $(uptime)" >> logs/cpu_monitor.log; sleep 5; done'
```

Confirm both are running. Glance at `tail -n 20 logs/gpu_monitor.log` to verify samples are coming in. Going forward, every loop iteration starts with a glance here.

### 3.2 Create `reports/V3_CLOSURE_PROGRESS.md` if it doesn't exist

Format (initial state reflecting the v2 audit):

```
# V3 Closure Progress — living document

Last updated: <date>

Gate 1 — Cross-Space Red-Team:        6/8 cells (75 %)
  blocking: cell 1.6 (PGD vs NudeNet), cell 1.7 (PGD vs Q16), cell 1.8 (attribution figure)
  next: nudenet_pgd_500_v3

Gate 2 — SAE Detector:                4/12 cells (33 %)
  blocking: 2.5 (ensemble), 2.7+2.8 (ensemble vs attacks), 2.9 (SD v1.4 SAE), 2.10 (SD3 SAE),
            2.11 (S3 detector eval), 2.12 (UDA + MMA AUC)
  next: b02v3_ensemble_v1

Gate 3 — Cross-Target & Adversarial Robustness:   2/38 cells (5 %)
  blocking: matched Square × NES × ε-sweep × joint adaptive grid
  next: bb_square_q5k_grid_phase1 (all 5 targets, n=500)

Gate 4 — Intervention:                1/18 cells (6 %)
  blocking: scale (4.2-4.5), learned-proj application (4.6), baselines (4.7-4.15),
            attribution v2 (4.16), trajectory v2 (4.17), compositional (4.18)
  next: d02_sdxlbase_n700_v3 (switches headline to SDXL Base 4-step)

Hardware right now: <fill in from monitor tail> GB / 80 GB, GPU-util <X> %, <N> active sessions.
Last 3 experiments: —, —, —. Cells advanced: none yet.
```

### 3.3 Add `closure_gate_cell` field to WandB config

Modify `dsi/util/wandb.py` (or wherever WandB init lives) to accept and require a `closure_gate_cell` config field — string like `"gate3_cell6"` or `"gate1_cell8"` or `"infrastructure"`. Every new run must declare which gate cell it serves.

### 3.4 Add comparison-rigor checklist template

Modify `reports/_template.md` (or create if missing) so every new report opens with:

```
## Closure-gate cell
Gate N cell M: <description from task_description_v3.md §3>. PASS / PARTIAL.

## Comparison-rigor checklist
- n_pre_flagged ≥ 200 (if applicable): yes/no (<actual_n>)
- Same prompts as comparison method: yes/no
- Same backbone as comparison method: yes/no
- Same scoring oracle as comparison method: yes/no
- Image-saving complete: yes/no (figure.png at <path>)
- Ablation rows included: yes/no (which ones)

## Hardware utilization
Peak VRAM: <GB>. Peak RAM: <GB>. GPU-util average: <%>. Co-scheduled sessions: <list>.
```

### 3.5 Commit and proceed

Commit infrastructure changes: `chore(v3): bootstrap closure-progress log, monitor running, wandb gate field, report template`.

Now you're ready to launch closure work.

---

## Step 4 — Gate 1 closure (start here)

Gate 1 is the smallest gap (2/8 cells missing + 1 figure). Close it first. It unblocks Gate 3 cell-fillers that depend on the same attack infrastructure.

### 4.1 Cell 1.6 + 1.7: A01/A02/A03 vs NudeNet + Q16

Rerun the three attack scripts against NudeNet and Q16 oracles instead of safety_checker:
- `scripts/exp_A01_pixel_pgd.py --target=nudenet --n=500 --eps=0.0157`
- `scripts/exp_A02_latent_pgd.py --target=nudenet --n=500 --eps=0.1`
- `scripts/exp_A03_emb_pgd.py --target=nudenet --n=500 --eps=0.5`
- Same three against `--target=q16`.

Total: 6 attack runs. Single-seed each (per the single-seed-now-noise-later principle). Co-schedule: the three attack spaces and two oracles are largely disjoint in VRAM; run several in parallel on the A100 80 GB **subject to the throughput-discipline check** in `CLAUDE.md` §5 / `task_description_v3.md` §6 — after launching the Nth parallel job, measure aggregate throughput and confirm it's meaningfully higher than (N-1)-job aggregate. If it isn't, kill the Nth and revert.

Image-saving discipline: every bypass case PNG, perturbation visualization, plus the `figure.png` 4×4 grid.

Comparison-rigor checklist:
- n=500 (single seed). If n_pre_flagged comes back < 200 on NudeNet (likely — NudeNet flags less aggressively), bump n upward to hit ≥ 200 pre-flagged or document why.
- Same prompts as the safety_checker baseline: yes (use cached prompt list).
- Same backbone (SDXL Turbo for now, or SDXL Base 4-step if you want to switch the whole gate; pick one and stick with it for the comparison).
- Same scoring oracle: yes (NudeNet or Q16 is the target).
- Image-saving + figure.png: yes.
- Ablation rows: this cell *is* an ablation row of cells 1.1–1.3. No further ablation needed.

Reports: `reports/G1_a01_pixel_pgd_vs_nudenet.md`, etc. — six total.

### 4.2 Cell 1.8: SAE-feature attribution figure

Pick 5 representative bypass cases (one per attack space × one per concept cluster).

For each, build a 4-panel figure: (a) pre image, (b) post image, (c) perturbation × 10, (d) SAE attribution heatmap (top 10 features at each of 4 hookpoints, before and after, with semantic labels from Surkov's catalog).

Output to `outputs/figures/F1_sae_attribution.pdf`. Script: `scripts/gen_figure_sae_attribution.py`. **This is an evidence artifact for the human's later paper-writing — not a paper figure that gets compiled into a draft.**

### 4.3 Update `V3_CLOSURE_PROGRESS.md` and write Gate 1 closure report

When all 6 attack runs + figure land, Gate 1 hits 8/8. Update the progress doc. Commit: `closure(gate1): 8/8 cells passed`.

Write `reports/GATE_1_CLOSURE_v1.md` summarizing:
- Headline table (rows: A01/A02/A03 × {safety_checker, NudeNet, Q16, Square}; columns: n_pre, n_bypass, ASR, % Δ vs safety_checker baseline).
- Ablation rows: ε-sweep (already done), cross-space Jaccard table (already done).
- F1 attribution figure embed.
- Caveats (anything that didn't reach n_pre ≥ 200, anything OOD).

Move to Gate 2.

---

## Step 5 — Gate 2 closure

Gate 2 has 8 missing cells. The biggest single piece of work is per-architecture SAE training for SD v1.4 (cell 2.9, a long-running background workload). Launch this in dedicated tmux from the start of Step 5 while other Gate 2 cells execute in parallel.

Priority order:

### 5.1 Cell 2.5: B02-v3 ensemble across 10 heads

The cheapest, highest-value cell. The 10 heads are already trained. Implement four ensemble strategies in `scripts/eval_b02v3_ensemble.py`:
- mean of logits
- max of logits
- majority vote (proportion of heads with logit > 0; threshold > 0.5)
- learned stacker (LogisticRegression on 10-d logit vector, train on val split)

Evaluate on 5 datasets: (B02-v3 val split, counterfactual Strategy 2, UDA-nudity, UDA-violence, MMA-Diffusion). 4 strategies × 5 datasets = 20 cells in `reports/G2_b02v3_ensemble_v1.md`.

Expected: ensemble matches or marginally beats single-head on in-distribution (everything saturated near 1.0); ensemble may significantly help on MMA OOD or on adaptive attacks (Gate 3).

### 5.2 Cell 2.9: SD v1.4 SAE training (long-pole, launch now in background)

In a dedicated tmux session:
- Capture SDXL-Surkov-style activations from 50K SD v1.4 generations (mixed COCO + I2P prompts).
- Train Surkov-style SAEs at 4 SD v1.4 UNet hookpoints (analogous to SDXL's `down.2.1`, `mid.0`, `up.0.0`, `up.0.1`).
- Expansion 4, k=10 (matches Surkov).
- Runs in background while all other Gate 2 cells finish.

Once SAEs trained: train per-arch B02-v3 on SD v1.4 SAE features (re-use the B02 architecture; oracle labels from NudeNet+Q16+SC on SD v1.4 outputs).

Evaluate on MMA-Diffusion. This **closes the MMA OOD failure** that has been the largest caveat to Contribution 2 across all reports.

Report: `reports/G2_sdv14_sae_detector_v1.md`.

### 5.3 Cell 2.10: SD3 SAE training

Same procedure for SD3 transformer blocks. Run concurrently with cell 2.9 in second tmux.

### 5.4 Cell 2.11: Strategy 3 detector evaluation

Render the 1200 Gemini paraphrases + 240 local-Llama paraphrases (Strategy 3 Path A + Path B from v2 Item 1c-0). Score each with safety_checker + B02-v3 ensemble. Build 4-cell consistency matrix: does the detector treat 3 paraphrases of the same anchor consistently? Report consistency rate per cell.

Already-generated paraphrases live in `outputs/cf_strategy3a_gemini_v1/` and `outputs/cf_strategy3b_gemini3flash_v1/`. Render and score is the new work.

Report: `reports/G2_cf_s3_detector_eval_v1.md`.

### 5.5 Cell 2.7 + 2.8: ensemble vs adaptive attacks

Defer to Gate 3 — these cells are the same experimental grid as Gate 3 cells.

### 5.6 Cell 2.12: full UDA-nudity + UDA-violence + MMA AUC at scale

Run B02-v3 ensemble (cell 2.5 output) and B02-adv ensemble on full UDA-nudity (n=142), UDA-violence (n=200), MMA (n=103). Report AUC + AP + recall@5%FPR per benchmark.

### 5.7 Update progress and Gate 2 closure report

When Gate 2 hits 12/12, write `GATE_2_CLOSURE_v1.md` summarizing the gate's evidence in tabular form. Move to Gate 3.

---

## Step 6 — Gate 3 closure (the biggest gap)

Gate 3 is 2/38 cells filled. This is the longest stretch of v3 work. The cells are organized as a single matched-budget matrix; close them in batches.

### 6.1 Build the driver: `scripts/eval_matched_blackbox_grid.py`

Single Python driver that takes parameters `--attack ∈ {square, nes}`, `--queries ∈ {500, 5000, 10000}`, `--target ∈ {safety_checker, nudenet, q16, b02v3_ensemble, b02adv_ensemble}`, `--n=500`. Always uses the same 500-prompt list (cached in `outputs/closure_prompt_list_v1.json` — sample once from I2P-NSFW). Saves to a structured output JSON.

### 6.2 Phase 1: Square Attack at q ∈ {500, 5K, 10K} × 5 targets

15 cells, single-seed each. Group by query budget; launch the 5 targets per budget in parallel on the A100 80 GB subject to the throughput-discipline check.

### 6.3 Phase 2: White-box ε-sweep on SAE detector targets

ε ∈ {4, 2, 1}/255 × 3 attack spaces × 2 SAE targets = 18 cells. Parallelize on the A100 80 GB subject to the throughput-discipline check.

### 6.4 Phase 3: NES at q ∈ {500, 5K, 10K} × 5 targets

15 cells. NES is query-heavier than Square; budget VRAM accordingly.

### 6.5 Phase 4: Joint adaptive PGD

`loss = safety_checker_logit + λ × sae_detector_logit`. Targets B02-v3 ensemble and B02-adv ensemble. λ sweep ∈ {0.1, 1, 10}.

### 6.6 Cross-classifier transferability matrix fill-in

Most cells already done (D-5 oracle transfer + A03 evaluation against B02-adv). Document the matrix in one place.

### 6.7 Update progress and Gate 3 closure report

Gate 3 closure → `GATE_3_CLOSURE_v1.md` with the full headline table.

---

## Step 7 — Gate 4 closure (heaviest baseline work)

### 7.1 Switch headline benchmark to SDXL Base 4-step

All Gate 4 experiments run on SDXL Base 4-step (28.6 % flag rate). n=700 prompts → ~200 pre-flagged.

### 7.2 Run D02 + D03 + D04 at n=700 × 4 benchmarks

UDA-nudity, UDA-violence, I2P-NSFW, MMA-Diffusion (OOD). Cells 4.2–4.4. Single-seed each. Parallelize seeds + benchmarks.

### 7.3 Conditional gating (cell 4.5)

Modify intervention pipeline: only patch F_c when B02-v3 ensemble fires pre-generation. Otherwise pass through unmodified. Rerun at same n × benchmarks.

### 7.4 Apply learned projection (cell 4.6)

Drop-in replacement for the mean-patch primitive. Same n × benchmarks.

### 7.5 Baseline reproductions (cells 4.7-4.15)

The slowest batch. 9 cells: SAeUron, SAEmnesia, DSG-adapted × UDA-nudity, UDA-violence, I2P-NSFW. Same n=700, same prompt list as F_c, same backbone, single-seed each. SAEmnesia is from scratch (requires training a supervised SAE on labelled UnlearnCanvas concepts). SAeUron + DSG-adapted from existing infrastructure.

### 7.6 Attribution-patching v2 (cell 4.16), trajectory v2 (cell 4.17), compositional (cell 4.18)

The three Phase-D items that have partial work. Finish per `task_description_v3.md` §3 Gate 4.

### 7.7 Update progress and Gate 4 closure report

Gate 4 closure → `GATE_4_CLOSURE_v1.md` with the full intervention-comparison table including all baselines, all ablation rows.

---

## Step 8 — Stop

When all four gates are closed:
1. Final update to `V3_CLOSURE_PROGRESS.md` showing 100 % everywhere.
2. Ensure all four `GATE_<N>_CLOSURE_v1.md` reports are complete with tables, figures, and caveats.
3. Ensure `outputs/figures/` and `outputs/tables/` contain the programmatic artifacts for the human to use later.
4. Tag the commit: `tag(v3-closure): all four gates passed`.

Stop. **Do not write the paper.** Wait for human review.

If the human starts a follow-up session, they may issue v4 with new directives (e.g. "now run seed-noise side-runs," "now train on FLUX," "now write the paper"). v3 is complete at that point.

---

## Step 9 — Seed-noise side-runs (only after Step 8)

If Step 8 lands well before any v4 directive, **and only then**, begin seed-noise characterization on representative cells:
- Pick 1-2 cells per gate that look most important to the human-facing story.
- Repeat those cells at 3-5 seeds.
- Report mean ± std in a new column added to the relevant `GATE_<N>_CLOSURE_v1.md`.

This is bonus work, not the v3 deliverable. The grids being complete is the v3 deliverable.

---

## Defaults you keep, throughout

- **Start the `monitor` tmux first**, before any experiment. Check the monitor tail at every loop iteration.
- **Default state of the box**: ≥ 3 GPU jobs + ≥ 12 CPU workers + monitor, always. GPU idle = burning money.
- **Image-saving discipline**: every bypass, every corrected case, every false positive saved. `figure.png` per experiment, no exceptions.
- **Comparison-rigor checklist**: at the top of every new report. No cell closes without the binding boxes ticked.
- **Same-prompt-list rule**: when comparing two methods, run them on the exact same prompts × same conditions. Different samples invalidate comparisons.
- **WandB**: every run has `closure_gate_cell` config field.
- **Reports**: per the v3 template with closure-gate-cell line, comparison-rigor checklist, and hardware-utilization line.
- **Commits**: per `CLAUDE.md` §13. Push to origin after every commit.
- **No new ideas.** Phase D items not in §3 closure gates are deferred.
- **No paper writing.** Evidence artifacts in `outputs/figures/` and `outputs/tables/` and `reports/GATE_<N>_CLOSURE_v1.md` are sufficient. The human will write the paper later.
- **No time estimates.** Progress is measured by cells closed, not by elapsed time.

The deliverable: four closure gates passed, with all evidence artifacts in place. That's the stop condition. Run.
