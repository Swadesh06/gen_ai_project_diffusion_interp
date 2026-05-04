# STARTER_PROMPT_3.md — Phase 1c (v2 + RTX PRO 6000 Blackwell upgrade) session start

You are continuing the DiffSafeSAE research project. **This is not a fresh start.** Phase 1 has landed. Phase C **was** in flight (5 GPU jobs + ~10 CPU jobs concurrent) and got terminated by the pod swap. The hardware has just been upgraded substantially.

You operate by the rules in `CLAUDE.md` (v2). Read it; follow it.

---

## What changed since the last session

Two things:

1. **Hardware upgraded substantially.** The pod has moved from 1× RTX Pro 4500 Blackwell (32 GB VRAM, 16 vCPU) to **1× RTX PRO 6000 Blackwell Workstation Edition (96 GB VRAM, 600 W, sm_120, CUDA 13.0)** paired with **AMD EPYC 9355 (64 vCPU)** and **263 GB system RAM**. Roughly 3× VRAM, 4× cores, 5× RAM. The user has explicitly raised the VRAM cap from 85% to **target 90% / hard 95%** and stated: *"throw everything at it, it can handle it. Don't be afraid of high-scale experiments."* **Use it.** A workload that was "Phase C maybe" on the old hardware is now "run today" on the new hardware.

2. **One new binding document**: `task_descriptions/task_description_v2.md` is the **single combined v2 file** that supersedes the v1 spec on Phase 1c, the both-framing pursuit, the new evaluation grid, the hardware envelope, and Phase D (ten new ambitious ideas D-1 through D-10). **It is binding.** v1 spec + v1 appendix remain valid where they don't conflict.

The user has unpacked the conda env on the new pod, so the env is ready. Verify it.

---

## Step 1 — pick up project state

```bash
cd /workspace/swadesh/gen_ai_project_diffusion_interp
git pull
conda activate dsi
```

**Verify the GPU**: `nvidia-smi` should show **RTX PRO 6000 Blackwell Workstation Edition** with 96 GB, driver 580.126.16, CUDA 13.0. `torch.cuda.get_device_name(0)` and `torch.cuda.get_arch_list()` — sm_120 should be present.

**Verify the CPU**: `nproc` should report 64 (AMD EPYC 9355). `free -g` should report ~263 GB total.

**Verify the env**: `pytest tests/` — all tests pass.

**Run `python scripts/verify_assets.py`** — must remain green. If anything went stale during the pod swap, re-fetch immediately.

**Verify Gemini access**: load `GEMINI_API_KEY` from `.env`, hit `gemini-3.1-flash-lite` with a one-token "ping" prompt, confirm 200 response. **If the key is missing or invalid, log it and proceed without Strategy 3 Path A — Path B (local Llama 3.1 70B) is the alternate, and Strategies 1 + 2 are sufficient on their own.**

---

## Step 2 — read the docs, in this order

1. `project_brief.md` — re-read.
2. `CLAUDE.md` — full read. The operating manual. **§4 hardware section is substantially rewritten** for the new envelope; do not skim it.
3. `PLAN.md` — the live plan from the previous session. Append a new "Phase 1c — session start (v2 + 96 GB)" section; do not delete prior content.
4. **`task_descriptions/task_description_v2.md`** — single combined v2 document. Phase 1c gates (eleven items, 1c-0 through 1c-10), the both-framing pursuit (§2 — both Framing A and Framing B run in parallel as committed deliverables, neither is "contingency"), the new evaluation grid centered on UnlearnDiffAtk (§4), the hardware envelope under the new pod (§5), Phase D ten new ambitious ideas (§6, D-1 through D-10), the framing-decision moment (§7). **The most important new document. Full read; treat as binding.**
5. `task_descriptions/task_description_v1.md` and `task_description_v1_appendix.md` — original spec + ICLR rigor extension. Re-read to anchor priorities; both remain valid where they don't conflict with v2.
6. `reports/INDEX.md` — every prior experiment.
7. `reports/PHASE_1_FINAL.md` and `reports/PHASE_1B_CHECKPOINT.md` — what landed, what failed silently. **Internalize the three honest caveats**: prompt-origin label leak in B01, SDXL-Turbo low-base-flag-rate, bit-identical detector logits bug in C01.
8. The 5 most recent `reports/<exp_id>.md` entries.

---

## Step 3 — first-hour priority: resume in-flight Phase C experiments

The pod swap terminated several Phase C experiments mid-flight. **Do not start new work until these are re-launched.** The user has explicitly stated the agent must continue these.

For each, check `checkpoints/<exp_id>/last.pt` and `outputs/<exp_id>/` for partial state. If checkpoint is ≥ 80% complete, resume; otherwise restart cleanly.

In-flight experiments to re-launch (priority order):

| tmux session | what it was | how to resume + scale up |
|---|---|---|
| `monitor` | nvidia-smi dmon | first thing to launch; always-on |
| `cpu-worker-{1..48}` | safety_checker oracle labelers | scale from 5 to **48 workers** on the 64-core CPU |
| `c1-square` | Phase C-1 black-box Square Attack vs SAE detector | resume; **bump n=50 → n=500**; eval against B02-oracle (Item 1c-3) once available |
| `C2-build` | Phase C-2 AxBench raw-vs-SAE probe | **redirect to counterfactual benchmark from Item 1c-0** — the prompt-origin-leak version is no longer the meaningful test |
| `C3-detector` | Phase C-3 safety-trained SAE detector training | resume from checkpoint; **train at expansion 16 and 32 in parallel** (was 4 on old hardware); sweep L0 ∈ {32, 64, 128, 256} |
| `C6-adv-eval` | hybrid raw‖SAE detector adversarial-robustness eval | restart cleanly; eval on counterfactual benchmark + B02-oracle |
| `C9-transcoder` | transcoder detector for circuit-level attribution | resume from cached raw activations; **run at full hookpoint coverage** (was reduced on old hardware) |
| `score-base-i2p` | scoring SDXL-Base 4-step I2P generations | resume CPU worker; serves Item 1c-7 directly |
| `fid-series` | FID for D02/D03/D04 post-intervention images | resume CPU worker |
| `lpips-D02/D03/D04` | LPIPS-VGG quality deltas | resume CPU workers |

**Document any state lost** in `reports/<exp_id>.md` under a "v2 resume note" section.

In parallel with the resume work above (most are CPU-bound or load-bound), run Step 4.

---

## Step 4 — smoke tests on the new GPU

Verify the new hardware does what the old one did, only bigger. Each is quick.

| smoke | what to check | expected |
|---|---|---|
| S00b — SDXL Turbo + Surkov SAEs forward smoke | re-run `scripts/smoke_sdxl_sae.py` | ~6 GB peak on 96 GB |
| S04 — co-located dual SDXL stack | SDXL Turbo + SDXL Base + 4 Surkov SAEs in one process; sample on each | ~30 GB peak; both work |
| S05 — FLUX inference smoke | FLUX (24 GB at fp16) + 1 prompt | ~24 GB peak; image is reasonable |
| S06 — FLUX + SDXL co-loaded | both diffusion stacks in two processes on same GPU | ~30 GB combined |
| S07 — pixel-PGD × 5 seeds parallel | five tmux processes, each running 50-prompt pixel-PGD | ~70 GB total; 5 logs growing in parallel |
| S08 — RAM-resident activation cache | load all 4 hookpoints × 100K samples × 5120 features into RAM via `dsi/util/activation_cache.py` | ~170 GB resident; query latency < 100µs |
| S09 — Gemini fallback chain | hit each of 5 Gemini models in priority order with one ping | all 200, mean latency reported |
| S10 — Llama 3.1 70B int8 paraphrase | load Llama 70B int8, generate 5 paraphrases | ~40 GB resident; output reasonable |
| S11 — 48 cpu-workers + 3 GPU jobs | launch 48 CPU + 3 GPU + monitor, observe `htop` and `nvidia-smi` | CPU 75%+ util, GPU 60%+ util sustained |

Each writes `reports/S0Xb_*.md` and updates `reports/INDEX.md`. **All must pass before launching any Phase 1c work that depends on them.**

---

## Step 5 — Phase 1c, in dependency order

Per `task_description_v2.md` §3, run Items 1c-0 through 1c-10. Most parallelize. Dependency graph:

```
1c-2 image-saving ─┐
                   ├─→ 1c-1 fix detector bug ─┐
1c-7 SDXL Base ────┤                          ├─→ 1c-9 black-box ─┐
                   │                          │                    │
1c-3 B02 oracle ───┘                          │                    │
                                              │                    │
1c-0 counterfactual ─→ AxBench-on-CF ─→ 1c-4 UnlearnDiffAtk eval ──┤
   │ (S1 + S2 + S3a + S3b all run)                                  │
   │                                                                ├─→ FRAMING-DECISION MOMENT
   │                                                                │
1c-5 SAeUron + DSG + SAEmnesia repros ──────────────────────────────┤
                                                                    │
1c-6 scale n + 5-seed CIs ──────────────────────────────────────────┘
                                                                    │
1c-8 FID/CLIP/LPIPS/DreamSim on D02/D03/D04 ────────────────────────┘
                                                                    │
                                                       Phase D — D-1 through D-10 (running in parallel from launch)
```

**Co-schedule aggressively** (the new hardware is sized for it):
- 1c-2 image-saving + 1c-7 SDXL Base + 1c-3 B02 oracle relabeling can all run in parallel on day one.
- 1c-0 counterfactual: Strategy 1 (CPU prompt-edit) + Strategy 2 (GPU multi-seed generation) + Strategy 3 Path A (Gemini paraphrase) + Strategy 3 Path B (Llama 3.1 70B local paraphrase) all run in parallel. **All four pursued; not "primary + supplementary"** — all four deliver complementary evidence.
- 1c-5: SAeUron + DSG + SAEmnesia repros — three independent baselines, three independent tmux sessions, ~10 GB each, 30 GB total.
- 1c-6 scale-n attacks: 5 seeds × 3 spaces = 15 attack jobs. 5 seeds × 14 GB pixel = 70 GB; co-locate all 5 pixel seeds in one batch. Latent + embedding alongside.

**Phase D starts as soon as its dependencies are satisfied** — not after Phase 1c fully closes. D-1 (causal feature graphs) needs Stage-2 survivors (Phase 1 already produced these), can start now. D-2 (learned projection) needs cached benign + unsafe activations — already on disk. D-9 (FLUX cross-architecture) is now feasible — start FLUX SAE training while Phase 1c runs. **All ten Phase D ideas run regardless of how Phase 1c plays out** — they strengthen the paper under either framing.

---

## Step 6 — both framings run in parallel

Per v2 §2, both Framing A and Framing B are committed deliverables. **Neither is contingency.**

- Maintain `paper/main.tex` as Framing A (working canonical draft).
- Maintain `paper/alt_framing_B.md` as a structured outline; update each time an experiment lands. Every Phase 1c result and Phase D result gets noted in both contexts ("under A this is Contribution 2; under B this is the discriminator probe").
- All Phase D experiments run regardless of framing — they serve both.

When the four framing-discriminator results land:
- Item 1c-0 (counterfactual benchmark built — all of S1, S2, S3a, S3b)
- Item 1c-1 (cross-target detector bug fixed and re-run with verified non-trivial pre/post deltas)
- Item 1c-3 (B02 oracle re-trained on counterfactual + I2P + UnlearnDiffAtk)
- Phase C-2 re-run on counterfactual benchmark

write `reports/REFRAMING_DECISION.md` with the four supporting numbers and a clear "Framing A canonical" or "Framing B canonical" verdict per the v2 §7 decision rule. Update `paper/main.tex` accordingly. Move the un-chosen framing's notes to `paper/archive/` — they may inform supplementary appendices, not undone work.

Commit with message `decision: framing-A canonical` or `decision: framing-B canonical`.

---

## Step 7 — Phase D: the ten new ambitious ideas

All ten run. Per `task_description_v2.md` §6:

| D-N | idea | VRAM | priority |
|---|---|---|---|
| D-1 | Causal feature graphs across denoising trajectory | 12-14 GB | mechanistic interp deliverable |
| D-2 | Learned-projection intervention (replaces mean-patch) | 8 GB | direct Contribution 4 upgrade |
| D-3 | UnlearnDiffAtk as primary headline | (folded into 1c-4) | benchmark migration |
| D-4 | Cross-concept transfer test | 16 GB × 4 concepts | generalization |
| D-5 | Black-box transfer attacks across diffusion models | 22 GB | threat model |
| D-6 | Joint end-to-end pipeline training | 32 GB | rigor ablation |
| D-7 | Mechanistic feature-firing trajectory plot | 4 GB | canonical paper figure |
| D-8 | Adversarial training of two-stage selection (TRADES) | 14 GB | robustness asymptote |
| D-9 | Cross-architecture generalization to FLUX / SD3 | 48 GB peak | now feasible on 96 GB |
| D-10 | Compositional / multi-concept simultaneous defense | 18 GB | production realism |

**Be ambitious. Run high-scale.** D-9 (FLUX cross-architecture) is the headline newly-feasible experiment — make it a priority. D-1 (causal feature graphs) and D-7 (mechanistic trajectory figure) produce the canonical paper figures — run them with care; the figures end up in submission. D-2 (learned projection) directly addresses the mean ≈ zero ≈ resample tie — if it beats mean-patch on FID, the headline strengthens.

For every D-N: **save all images per CLAUDE.md §7**, generate `outputs/<exp_id>/figure.png`, run 5 seeds for any number that hits the headline table, write the report before launching.

**Combine when sensible**:
- D-4 cross-concept × D-2 learned-projection → does the learned projection generalize across concepts?
- D-7 mechanistic figure × D-9 FLUX → single canonical figure showing feature firing for a cross-architecture-transferred bypass.
- D-8 adversarial training × D-5 black-box transfer → does adversarially-hardened pipeline survive cross-model attacks?
- D-10 compositional × D-2 learned-projection → compositional erasure with learned-projection intervention.

---

## Step 8 — the loop, until interrupted

1. Refer to `task_description_v2.md` and `PLAN.md` to anchor priorities.
2. Pick the next experiment from the Phase 1c queue (then Phase D, then Phase C carry-overs, then propose your own).
3. Plan co-scheduling. **Default state: ≥ 3 GPU + ≥ 12 CPU + monitor.** With 96 GB you should typically be running 15-25 active tmux sessions.
4. Dry-run for 30-60 s. Log peak VRAM, RAM, CPU.
5. Launch in tmux; log to WandB; **save all images per §7**; write the report.
6. Visually inspect any rendered output — metrics alone are not enough.
7. Commit; push.
8. If keep, integrate into defaults. If discard, log reasoning.
9. Update `PLAN.md`. Update `paper/main.tex` and `paper/alt_framing_B.md` for any result that lands in a paper section.
10. Repeat.

**Do not pause to ask permission. Do not stop and wait. The human may be asleep.** The only stop condition is the human interrupting.

---

## Step 9 — the paper

Maintain `paper/main.tex` (Framing A canonical) and `paper/alt_framing_B.md` (Framing B parallel outline) as you go. Every experiment that lands a result updates the corresponding section / table / figure in **both**. The framing-decision moment commits one as canonical and archives the other to `paper/archive/`.

Tracked figure list (commit each to `outputs/figures/` when ready):
- `fig_pipeline.pdf` — overall pipeline diagram
- `fig_cross_space_jaccard.pdf` — Contribution 1 SAE-feature overlap matrix
- `fig_commit_knee.pdf` — per-step AUC plot
- `fig_xtarget_matrix.pdf` — Contribution 3 transferability matrix
- `fig_mechanistic_trajectory.pdf` — D-7 single-prompt feature-firing plot (the money shot)
- `fig_causal_graph.pdf` — D-1 directed feature graph
- `fig_intervention_quality.pdf` — D02/D03/D04/D-2 before/after grid
- `fig_flux_alignment.pdf` — D-9 cross-architecture feature alignment
- `fig_phase1c_gates.pdf` — table of all Phase 1c gates with pass status

---

## Step 10 — output discipline

When you produce text the human will read:

- **Direct. Plain. No superlatives.** No "I've successfully implemented…", no "this groundbreaking improvement…", no "remarkably, the result shows…".
- State the result, the number, the path to evidence.
- Bullet lists over prose for facts.
- One short paragraph of interpretation, max.
- No emojis. No filler. No re-stating what the human asked.

The reports are technical artifacts. Treat them like a paper appendix — every sentence earns its place.

---

That's it. Read CLAUDE.md, the v1 spec + appendix, the **v2 spec** (single combined file), PLAN.md, the recent reports. Resume in-flight experiments first, then run smokes for the new hardware, then start Phase 1c. **Both framings run in parallel — neither is contingency.** Phase D starts as soon as dependencies allow — not after Phase 1c closes.

Plan with parallelization in mind — **the default state of the 96 GB / 64 vCPU / 263 GB box is "≥ 3 GPU + ≥ 12 CPU + monitor", always; treat 50%-utilized as broken; treat 30 GB-of-VRAM-used while jobs are queued as a bug**. Save every relevant image, every time. Write reports in plain language. Read the literature when you run out of ideas. Commit every meaningful step; push to origin. **Be ambitious — run high-scale experiments, the hardware is sized for it, the user has explicitly raised the cap and stated to throw everything at it.**

**Do not pause to ask permission. Do not stop and wait. The human may be asleep.** The only stop condition is the human interrupting.

Execute. Improve. Don't stop.