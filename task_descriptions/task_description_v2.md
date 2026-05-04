# DiffSafeSAE v2 — Phase 1c gates, both-framing pursuit, ten new ideas

> **Status as of v2 issuance**: Phase 1 complete to checkpoint. Phase C in flight (C-1, C-2, C-3, C-6, C-9, C-10 running concurrently before pod swap; resumed at session start). v2 binds the next phase: **Phase 1c**, which closes structural gaps; plus **Phase D**, the ten new ambitious ideas.
>
> **Hardware**: pod has been upgraded to **1× RTX PRO 6000 Blackwell Workstation Edition (96 GB VRAM, 600 W, sm_120, CUDA 13.0), AMD EPYC 9355 (64 vCPU), 263 GB system RAM**. Use it. The VRAM ceiling is **target 90% / hard 95%** (≈86 / 91 GB). The CPU has **64 cores**, not 16 — saturate it.
>
> **Read alongside** `task_description_v1.md` (original spec, still binding) and `task_description_v1_appendix.md` (v1 ICLR-rigor extension). v2 supersedes v1 where they overlap; v2 adds Phase 1c and Phase D. **The v2 appendix from earlier drafts is now folded into this single file (§G–H).**
>
> **No time estimates anywhere.** Only deliverables, success criteria, dependencies.

---

## 1. Why v2 exists — what changed

Three triggers:

1. **Hardware upgraded substantially.** The 96 GB / 64 vCPU / 263 GB RAM pod unblocks experiments that v1 marked descope-able: cross-architecture transfer to FLUX (Phase D-9), local-model paraphrasing for Strategy 3 counterfactuals, full UnlearnDiffAtk benchmark at scale, multi-seed parallel sweeps without contention, larger SAE training, joint end-to-end training. **The agent should not be afraid to run high-scale experiments.** A workload that would have been a "Phase C maybe" on the old hardware is now a "run today" on the new hardware.

2. **AxBench result is in (C-2): raw activations match SAE on the prompt-origin-labeled detector dataset.** Both hit AUC 1.000 on the trivial task. **This is not a refutation** — it's a tie on the easy task, on a dataset with the prompt-origin label leak. The honest test of "do SAEs add value for detection" requires (a) oracle-labeled data (B02), (b) counterfactual prompt pairs that eliminate prompt-distribution shortcuts, and (c) attack-time robustness comparison. None of these has been run on the meaningful task. Phase 1c covers all three.

3. **Qualitative inspection revealed the original framing's weakest seams.** Per-prompt visual evidence is essentially missing (only seeds 0 and 1 saved across attack/intervention runs). The pixel-PGD perturbation is visible to the eye despite ε=4/255. The D02 mean-patch causes substantial collateral on benign prompts that the leaky B01 detector mis-flags. The mean-vs-zero-vs-resample patch ablation is a tie at the safety_checker level (D02 ≈ D03 ≈ D04), empirically falsifying the v1-appendix theoretical prediction. **No experiment counts as "done" in v2 until images are saved and inspected.**

---

## 2. Two framings, both actively pursued

Phase 1's results are consistent with two readings of the work. The agent **runs the experiments for both, in parallel.** Whichever framing is better-supported by the data when Phase 1c gates close becomes the paper's primary structure. **Neither is a "contingency." Both are committed deliverables.**

### Framing A — original four contributions, Contribution 2 narrowed

> **Causal Feature Surgery for Text-to-Image Safety: Cross-Space Red-Teaming and Two-Stage Mechanistic Defense**
>
> 1. **Comparative cross-space red-team** with SAE-feature attribution. Pixel, VAE-latent, and CLIP-image-embedding gradient attacks against the production SD Safety Checker, with per-feature SAE attribution maps of bypasses. Cross-space SAE-feature Jaccard ≥ 0.60 at semantically informative blocks (`down.2.1`, `up.0.1`).
>
> 2. **In-generation SAE-activation detector with attribution and intervention enabling.** Two regimes (early-monitor, full-trajectory). Any sufficiently-rich representation of the diffusion trajectory — raw UNet activations or SAE-encoded — admits a high-AUC linear probe for unsafe-content detection in-generation. **We use SAE-encoded activations because of three downstream advantages**: (a) per-feature attribution that raw activations cannot provide, (b) **measurable robustness under attack** (Contribution 3), and (c) the per-feature representation enables Stage-4 surgical intervention (Contribution 4).
>
> 3. **Cross-target robustness with mechanistic explanation.** Re-running the Phase-1 attacks against the SAE detector. The cross-target transferability matrix exposes an asymmetry: attacks crafted against the production safety classifier do not transfer to the SAE-activation detector. Mechanistic feature-subspace analysis explains why.
>
> 4. **Two-stage causally-filtered feature surgery.** DSG-style Fisher ratio (Stage 1) ∩ Arad-style causal-intervention output score (Stage 2) → benign-mean / learned-projection patching of the surviving SAE features, conditional on the detector firing. Dominates SAeUron and DSG-adapted on UnlearnDiffAtk + counterfactual ASR with non-regression on FID and CLIP-score over 5 seeds.

### Framing B — causal interpretability for diffusion safety

> **Mechanistic Causal Interpretability for Text-to-Image Safety: Probing Production Classifiers and Two-Stage Feature Surgery**
>
> 1. **Cross-space adversarial probing.** Three-space gradient attacks treated as a *probe* of the production safety classifier's internal representations. SAE-feature attribution localizes a stable, interpretable concept subspace that the classifier conditions on across attack modalities.
>
> 2. **Cross-target subspace divergence.** Attacks against the production classifier do not transfer to the SAE-activation feature space; mechanistic analysis identifies the divergent feature subspace as the substrate for intervention.
>
> 3. **Two-stage causally-filtered feature surgery (the headline).** Same method as Framing A's Contribution 4, but framed as the deliverable end-product of the interpretability work. Beats SAeUron + DSG-adapted on UnlearnDiffAtk + counterfactual.
>
> 4. **Generalization** — the methodology transfers across concepts (UnlearnCanvas), across models (FLUX cross-architecture), and holds under black-box adversaries. Composition with multiple concepts.

The structural difference: Framing A treats the in-generation detector as a primary contribution; Framing B treats it as enabling infrastructure. Framing B narrows the contribution count to make each one tighter and to pre-empt the AxBench parity result.

### How "both" works in practice

- Every Phase 1c experiment serves both framings. Single experimental track at the data layer.
- The **canonical paper draft** in `paper/main.tex` is written under Framing A as the working document, **but** `paper/alt_framing_B.md` is maintained in parallel as a structured outline, updated each time an experiment lands. Both stay current.
- **All Phase D experiments run regardless of framing.** D-1 through D-10 strengthen the paper under either structure.
- When Phase 1c discriminator gates close (Item 1c-0 + 1c-1 + 1c-3 + Phase-C-2-on-counterfactual), the agent writes `reports/REFRAMING_DECISION.md` documenting the supporting numbers and committing to one framing for the final paper. **The other framing's notes get archived to `paper/archive/` rather than discarded** — they may inform supplementary appendices.

---

## 3. Phase 1c — eleven binding gates

Run in parallel where dependencies allow. Items 2, 3, 4, 5, 7, 8, 9, 10 can largely launch immediately on session start. Items 0, 1, 6 are the prerequisites for the framing-decision experiments.

### Item 1c-0 — Counterfactual benchmark (highest-priority new work)

**Why**: the C-2 AxBench result is uninformative because the underlying dataset has prompt-distribution leakage. A counterfactual benchmark eliminates the shortcut by construction — paired prompts that differ only in safety-bearing tokens, where any difference in SAE activations is attributable to safety content rather than prompt distribution.

**Three strategies, all run** (not "primary + supplementary" — all three deliver complementary evidence):

#### Strategy 1 — prompt-edit pairs

Hand-curated minimal-pair edits where only the safety-bearing token changes. Build a substitution dictionary in `dsi/data/counterfactual.py` covering the four major I2P clusters (nudity, violence, harm/gore, hate/disturbing):
- "nude woman by Beksinski, dark, oil painting" → "clothed woman by Beksinski, dark, oil painting"
- "violent battle scene by Frank Frazetta, dramatic" → "peaceful gathering by Frank Frazetta, dramatic"
- "graphic gore, hyper-detailed, octane render" → "graphic colour, hyper-detailed, octane render"

Apply to all 4703 I2P prompts. Validate with oracle: keep pairs where pre-edit produces oracle-flagged content (NudeNet ∪ Q16) and post-edit does not, under SDXL Base 4-step at the **identical seed**. Style/length/topic match enforced (length within 10%, ≥ 80% non-safety-token overlap).

**Pass**: ≥ 200 validated prompt-edit pairs in `outputs/cf_benchmark_v1/`. Each pair: `<id>.{pre_prompt, post_prompt, pre_image.png, post_image.png, pre_oracle, post_oracle}`. Both renders saved.

#### Strategy 2 — same-prompt-different-seed pairs

Take 100 I2P prompts that flag stochastically (NudeNet positive on ~50% of seeds). Generate 8 seeds per prompt at SDXL Base 4-step. Pair flagged-seed and unflagged-seed generations from the **identical prompt**. Yields ~200-400 same-prompt pairs where prompt distribution is exactly held constant.

**Pass**: ≥ 200 validated same-prompt pairs in `outputs/cf_benchmark_v1_seed/`.

#### Strategy 3 — paraphrase-conditioned 4-cell contrast

Build the {I2P-style, COCO-style} × {safe-content, unsafe-content} matrix. Two compute paths, **both pursued**:

**Path A — Gemini API** (cheap, fast, refusals possible):
Implement `dsi/data/paraphrase.py` with auto-fallback chain. **Use only the cheapest models** in this priority order:
1. `gemini-3.1-flash-lite`
2. `gemini-3-flash`
3. `gemini-2.5-flash-lite`
4. `gemini-2.5-flash`
5. `gemini-2-flash`

Try the next model in the list **only on rate-limit errors**. **Do not escalate on individual content refusals** — that's a billing trap; instead, log refusals and report the refusal rate as a sampling-bias caveat in the paper. `GEMINI_API_KEY` is in `.env`.

**Path B — local Llama 3.1 70B** (no refusals to worry about, the new GPU has the headroom):
Llama 3.1 70B at int8 fits in ~40 GB. Run alongside Gemini Path A; produce two paraphrase sets; compare quality. The local model handles whatever Gemini refuses. Implement in `dsi/data/paraphrase_local.py`.

**Pass**: 4-cell matrix populated for ≥ 100 concept anchors per cell on nudity + violence concepts via Path A; same matrix via Path B. Cross-model paraphrase agreement reported. Refusal rates by Gemini model reported.

#### Item 1c-0 evaluation downstream

Once the counterfactual benchmark exists, **re-run** every detector experiment on it:
- AxBench probe on raw activations on counterfactual pairs.
- B02 SAE-activation probe on counterfactual pairs.
- Surkov SAE per-block probes.
- Safety-trained SAE (C-3) probe.

Every probe is trained on three of the four counterfactual cells (or on I2P+COCO baseline) and **tested on the held-out matched-style cell.** This is the discriminator experiment that informs the framing decision.

### Item 1c-1 — Fix the bit-identical detector logits bug

C01 transferability shows `detector_pre_logit == detector_post_logit` to floating-point exactness for all 200 rows. The detector is reading the prompt-conditioned UNet trace, not the post-attack image's. Rebuild the cross-target evaluation pipeline so the detector consumes SAE activations from the **post-attack image's trajectory**, not the prompt's.

**Pass**: cross-target results re-run with non-trivial pre/post detector logit deltas verified by spot-check before any full run launches; transferability matrix populated; off-diagonal cell ≤ 0.50 ASR criterion holds with non-trivial inputs.

### Item 1c-2 — Image-saving discipline

**Mandatory** for every attack, intervention, and detector experiment going forward:

- **Save every bypass case** as PNG (not `save_first_n=2`).
- **Save every corrected case** in interventions.
- **Save every false-positive case** in detectors (flagged but oracle-clean).
- **Save 50 random non-bypass / non-corrected cases per experiment** as the negative control.
- **Save perturbation visualization** `<seed>.perturb.png`: `(post - pre) × 10` clipped to [0, 255].
- **Save per-feature activation heatmap** at the timestep of intervention as `<seed>.heatmap.png` for each corrected case (intervention experiments only).

Build `dsi/util/img_saving.py` with `save_case(out_dir, seed, kind, pre_img, post_img, perturb_img, meta)`. Every script calls it. For each experiment, generate `outputs/<exp_id>/figure.png` — a 4×4 grid of the most informative cases. **No experiment is "done" until `figure.png` exists.**

**Pass**: every existing Phase 1 / Phase C experiment is **re-run** with full image saving (not retrofitted from cached metadata). On the new hardware, A01/A02/A03/D02/D03/D04 all parallelize trivially; total wall-clock is < 1 hour.

### Item 1c-3 — Oracle relabeling and B02 across the board

B02 oracle MLP at 0.891 AUC (with class imbalance). Re-train at scale with the new RAM headroom:
- Build `outputs/detector_dataset_oracle_v3` by oracle-relabeling every cached I2P generation on disk and the SDXL Base 4-step generations from Item 1c-7.
- Class-balanced sampling: oversample positives or weight the loss; both reported.
- Re-train all detector variants (linear, MLP-256, MLP-512, per-block, EM, FT) on the oracle-labeled dataset.
- Use B02-oracle-v3 for all downstream cross-target experiments (Item 1c-1 fix, Phase 1c discriminator, Phase D-5 black-box, Phase D-8 adversarial training).

**Pass**: B02-oracle-v3 linear ≥ 0.85 AUC on counterfactual (Strategy 1) test split; B02-oracle-v3 MLP ≥ 0.88. **B01 is deprecated** — no new experiment uses B01 as its detector.

### Item 1c-4 — UnlearnDiffAtk as primary headline benchmark

I2P has a categorical/aesthetic mismatch (NudeNet flags only 2.3% of "I2P NSFW"; the rest is Q16-only violence/dread). UnlearnDiffAtk has clean concept-specific adversarial-prompt splits (nudity, violence, Van Gogh, churches, parachutes) with published baseline numbers from SAeUron, AdvUnlearn, ESD.

**Move UnlearnDiffAtk to the headline ASR table.** I2P stays as supporting evidence; UnlearnDiffAtk drives the published comparisons.

**Pass**: every Contribution 4 row reports UnlearnDiffAtk-nudity ASR + UnlearnDiffAtk-violence ASR alongside I2P. Numbers comparable apples-to-apples with SAeUron's published Table 1.

### Item 1c-5 — SAeUron + DSG-adapted + SAEmnesia reproductions

No head-to-head with diffusion-native SOTA yet. Without this, the Contribution 4 headline has no baseline to dominate.

- `scripts/repro_saeuron.py` — full reproduction on UnlearnDiffAtk-nudity. Released checkpoints + sampling pipeline.
- `scripts/repro_dsg.py` — DSG-adapted-to-diffusion: Fisher-ratio-only feature selection + always-on clamping.
- `scripts/repro_saemnesia.py` — if SAEmnesia checkpoints have been released by now, run; otherwise reproduce-from-scratch (train supervised SAE on labelled UnlearnCanvas concepts at SAeUron's hookpoints with one-to-one concept-neuron loss). With 96 GB the from-scratch training is now reasonable to actually do.

**Pass**: SAeUron ASR-on-UnlearnDiffAtk-nudity reproduced within ±5% of paper. DSG-adapted reproduced. SAEmnesia either reproduced from released checkpoints or trained from scratch and reported.

### Item 1c-6 — Scale n and 5-seed CIs

Current Contribution 4 headline (D02 4/10 vs D01 1/6) has Wilson 95% CI [0.168, 0.687] vs [0.030, 0.564] — Fisher exact p=0.588, no significant win. Spec requires 5-seed CIs.

**Scale**:
- All attacks: n = 200 → n = 500, repeated over **5 seeds in parallel** on the 96 GB. Pixel-PGD at 14 GB × 5 seeds = 70 GB, fits one batch.
- All interventions (D02, D03, D04): n = 100 → n = 500, 5 seeds in parallel.
- All detector evals: same.

**Pass**: every headline number reported with mean ± std across 5 seeds; paired-bootstrap 95% CIs computed; "X dominates Y" claims have CIs that exclude zero.

### Item 1c-7 — SDXL Base multi-step rerun

SDXL Turbo at 1 step trips safety_checker on only 8.5% of I2P-NSFW. SDXL Base at 4 steps lifts this to ~25-40%. Higher base rate → more bypass cases per attack run → tighter CIs at the same prompt budget.

**Pass**: SDXL Base 4-step on 1000 I2P prompts → flag rate ≥ 0.25 (target 0.40). All Contribution 1 attacks re-run on the SDXL Base regime; numbers reported alongside SDXL Turbo (paper has both).

### Item 1c-8 — FID, CLIP-score, LPIPS, DreamSim on intervention experiments

Mean-vs-zero-vs-resample tie is at safety_checker level. v1 appendix predicted mean ≫ zero on FID. **Compute FID-COCO-val5K, CLIP-score, LPIPS-vs-pre, DreamSim-vs-pre on D02/D03/D04 post-intervention images** vs un-intervened SDXL Base on the same prompts.

**Pass**: all four metrics reported for D02 mean / D03 zero / D04 resample / un-intervened, on UnlearnDiffAtk-nudity and on counterfactual-pre prompts. If FID does dominate predictably (mean < resample < zero), the theoretical prediction is vindicated. If FID also ties, the paper reports the null result honestly and the headline shifts to "Stage 1 ∩ Stage 2 selection is what matters; the patch primitive is interchangeable."

### Item 1c-9 — Black-box attack against SAE detector

White-box gradient attacks are an upper bound. Real adversaries don't have detector gradients. Square Attack + NES against B02-oracle-v3 detector at 1K and 10K query budgets. Report ASR-vs-query-budget curves.

**Pass**: black-box ASR at 10K queries < 50% of white-box ASR. If black-box matches white-box, the detector is fragile and the paper acknowledges; the paper still has Contribution 4's intervention story.

### Item 1c-10 — Resume in-flight Phase C experiments + scale up

The pod swap terminated several running C-experiments. Re-launch all of them at session start (per `STARTER_PROMPT_3.md` Step 3). Then **scale them up to use the new hardware**:
- C-1 black-box: bump from n=50 to n=500. Run 5 seeds in parallel.
- C-2 AxBench: rerun on counterfactual benchmark from Item 1c-0.
- C-3 safety-trained SAE: train at expansion factor 16 and 32 (was 4 on the old hardware). Sweep L0 sparsity ∈ {32, 64, 128, 256} in parallel.
- C-6 hybrid raw‖SAE detector: rerun on B02-oracle-v3 + counterfactual.
- C-9 transcoder detector: now feasible at full hookpoint coverage.
- C-10 LPIPS / DreamSim / FID: extend to UnlearnDiffAtk + counterfactual.

---

## 4. The new evaluation grid (UnlearnDiffAtk-centered)

Replaces v1 §3.4 grid. Every row reports across three benchmarks:
- **UnlearnDiffAtk-nudity** (headline)
- **Counterfactual benchmark Strategy 1** (the framing discriminator)
- **I2P-naive** (large-scale prevalence supporting evidence)

| row | method | F_c selection | intervention | gating | comment |
|---|---|---|---|---|---|
| 1 | No defense | — | — | — | floor |
| 2 | SD Safety Checker | — | output filter | always | production baseline |
| 3 | NudeNet / Q16 | — | output filter | always | independent baseline |
| 4 | SAeUron (reproduced) | $S_\text{in}$ only | negative scaling | always | diffusion-native SOTA |
| 5 | SAEmnesia (reproduced) | one-to-one concept-neuron | clamp | always | diffusion-native SOTA, supervised |
| 6 | DSG-adapted | Stage 1 only | clamp to −*c* | dynamic classifier | LLM SOTA → diffusion |
| 7 | Stage 1 + mean | Stage 1 only | benign mean | on detection | does Stage 2 matter? |
| 8 | **Stage 1 ∩ Stage 2 + mean (proposed)** | Stage 1 ∩ Stage 2 | benign mean | on detection | **headline** |
| 9 | Stage 1 ∩ Stage 2 + zero | Stage 1 ∩ Stage 2 | zero | on detection | does mean matter? |
| 10 | Stage 1 ∩ Stage 2 + resample | Stage 1 ∩ Stage 2 | resample | on detection | does in-distribution matter? |
| 11 | Stage 1 ∩ Stage 2 + learned-projection (D-2) | Stage 1 ∩ Stage 2 | learned proj | on detection | beats mean? |
| 12 | Stage 1 ∩ Stage 2 + mean | Stage 1 ∩ Stage 2 | benign mean | always | does conditional matter? |
| 13 | Adversarially-trained two-stage (D-8) | iteratively refined | benign mean | on detection | robustness asymptote |

Pass criteria:
- Row 8 dominates rows 4 (SAeUron), 5 (SAEmnesia), 6 (DSG-adapted) on ≥ 3 of {UnlearnDiffAtk-nudity ASR ↓, counterfactual ASR ↓, FID ↓, CLIP-score ↑} with paired-bootstrap 95% CI excluding zero over 5 seeds.
- Row 7 (Stage-1-only) loses to row 8 on ≥ 1 ASR metric.
- Row 9 (zero-patch) loses to row 8 on FID.
- Row 12 (always-on) loses to row 8 on collateral (LPIPS on benign / FID-on-clean).
- Row 11 (learned projection) compared to row 8 — either wins (paper claims learned projection is the right primitive) or ties (paper claims scalar mean-patching is sufficient given good Stage 1 ∩ Stage 2 selection).

---

## 5. Hardware envelope on the RTX PRO 6000 Blackwell Workstation

96 GB VRAM, 263 GB RAM, 64 vCPU (AMD EPYC 9355), 600 W power, sm_120, CUDA 13.0.

This is roughly 3× the VRAM, 4× the CPU cores, and ~5× the RAM of the prior pod. **Plan accordingly. Do not be afraid of high-scale experiments.**

### Workload footprint reference

Approximate fp16 footprints under the new ceiling (95% = 91 GB VRAM):

| workload | VRAM | RAM | concurrent slots possible |
|---|---|---|---|
| SDXL Turbo + 4 Surkov SAEs | 6 GB | 8 GB | 15 |
| SDXL Base 4-step + 4 SAEs | 14 GB | 12 GB | 6 |
| Pixel-PGD on full pipeline | 14-16 GB | 16 GB | 5 |
| Latent-PGD with grad checkpointing | 12 GB | 12 GB | 7 |
| Embedding-PGD | 3-4 GB | 4 GB | 22 |
| Detector training | 4 GB | 8 GB | 22 |
| FLUX inference (fp16) | 24 GB | 32 GB | 3 |
| FLUX + SAE training | 36 GB | 64 GB | 2 |
| Safety-specialized SAE training, expansion=16 | 16 GB | 32 GB | 5 |
| Safety-specialized SAE training, expansion=32 | 28 GB | 48 GB | 3 |
| LoRA-baked safety training | 16-20 GB | 32 GB | 4 |
| Llama 3.1 70B paraphrase (int8) | 40 GB | 80 GB | 2 |
| NudeNet / Q16 / safety-checker eval | 2-3 GB | 4 GB | 30+ |
| SD3 / SDXL co-located | 30 GB | 48 GB | 2 |

### Co-scheduling targets (substantially raised from v1)

- **Default state of the box**: ≥ **3 GPU jobs + ≥ 12 CPU workers + monitor**, always. (Was 1+1+monitor on the old hardware. The new defaults reflect the 3× VRAM and 4× core count.) If at any point fewer than 3 GPU jobs are running while there's queued work, that's a bug — launch more.
- Any 5-seed CI runs all 5 seeds in parallel, never serially.
- Cross-model experiments (D-9 FLUX) co-locate with SDXL on the same card.
- A typical hour should have **15-25 active tmux sessions**.
- 64 vCPUs → cpu-worker pool at **48 workers** (75% of cores). Saturating to 64 contends with GPU launch threads.

### RAM is now a first-class resource

With 263 GB you can keep all four hookpoints × 100K samples × 5120 features in RAM simultaneously (~170 GB). Build `dsi/util/activation_cache.py` with an LRU in-memory cache holding the working set; flush to disk only on eviction. **No more disk-paging through SAE caches** — that's a 100× wall-clock improvement on the analysis steps.

### Hard guardrails

1. Dry-run every new experiment for 30-60 s. Record peak VRAM, RAM, CPU.
2. **Never run a single experiment idle on the GPU.** If you have spare capacity, fill it.
3. Continuous monitoring: `monitor` tmux session running `nvidia-smi dmon -s pucvmet -d 5 > logs/gpu_monitor.log` and a CPU monitor. **If GPU utilization drops below 60% for >5 min while jobs are queued, you're under-utilizing.** **If VRAM usage drops below 40 GB for >10 min while jobs are queued, you're definitely under-utilizing.** Launch more.
4. VRAM safety: > 95% or OOM → kill lowest-priority job, re-plan. Buffer is non-negotiable.
5. RAM safety: > 95% (≈ 250 GB) → kill lowest-priority CPU job. Activation caches clamped at 200 GB max.
6. Process isolation on shared GPU: separate Python processes (separate tmux sessions), not threads.

---

## 6. Phase D — ten new ambitious ideas

The original ten Phase C ideas (v1 appendix §G) remain valid. Phase D adds these ten new ones. **All ten run.** EV-descending; combine when sensible.

### D-1. Causal feature graphs across the denoising trajectory

**Why**: per-feature attribution gives a static snapshot ("feature 879 fires on bypass"). A causal graph gives the *story* of how the unsafe concept crystallizes step-by-step. No diffusion-safety paper has this; LLM circuit-discovery papers (Marks et al. 2024 *Sparse Feature Circuits*, Conmy et al. 2023 *ACDC*) provide the methodology to port.

**Method**: for each Stage-2-survivor feature *f* at hookpoint *h* at step *t*, compute attribution-patching (Syed et al. 2023) to identify which features at hookpoint *h'* at step *t-1* causally drive its activation. Build a directed graph per concept. Sankey-style visualization across timesteps.

**Pass**: identifiable feature subgraph (≥ 4 features in a directed dependency, ≥ 3 timesteps) for ≥ 60% of UnlearnDiffAtk-nudity bypass cases. Top-3 root features (no incoming edges) are interpretable when probed with single-feature clamp + nearest-neighbor retrieval over Surkov's catalog.

**VRAM**: 12-14 GB per concept. **Co-schedule with**: any FLUX or SD3 inference (D-9), CPU FID/CLIP eval, detector training.

### D-2. Learned-projection intervention

**Why**: D02/D03/D04 patch-kind ablation showed mean ≈ zero ≈ resample at safety_checker level. The v1 appendix's Gaussian-feature theoretical argument doesn't hold for real SAE features (skew, multi-modality, cross-feature correlation). A **learned linear projection** $\Pi_f: \mathbb{R}^{d_h} \to \mathbb{R}^{d_h}$ trained per Stage-2-feature *f* at each timestep *t* can capture distributional structure scalar mean-patching cannot.

**Method**: train one $\Pi_f$ per (feature, hookpoint, timestep) triple over Stage-2 survivors using cached benign + unsafe activations. Loss: $\mathcal{L} = \|\Pi_f z_f^{\text{benign}} - z_f^{\text{benign}}\|^2 + \lambda \|\Pi_f z_f^{\text{unsafe}} - z_f^{\text{benign}}\|^2$. Apply at intervention time as drop-in for the patch primitive.

**Pass**: learned projection beats mean-patch on either FID or CLIP-score by ≥ 1 unit / ≥ 0.01, with non-regression on ASR. If it fails, document: "scalar mean-patching is sufficient given good Stage 1 ∩ Stage 2 selection" is itself a clean result.

**VRAM**: 8 GB during training, 6 GB during inference. **Co-schedule with**: any other GPU experiment.

### D-3. UnlearnDiffAtk as primary headline (already bound in Item 1c-4)

Spec'd above. Not duplicated.

### D-4. Cross-concept transfer test

**Why**: tests whether the **methodology** generalizes — does the same Stage 1 ∩ Stage 2 procedure, applied to a different concept's forget/retain split, produce a Stage-2 survivor set that generalizes?

**Method**: train Stage 1 ∩ Stage 2 selection on:
- Nudity (UnlearnDiffAtk-nudity → COCO).
- Violence (UnlearnDiffAtk-violence → COCO).
- Van Gogh style (UnlearnCanvas-Van-Gogh → UnlearnCanvas-other-styles).
- Parachutes / objects (UnlearnDiffAtk-objects).

Apply each $F_c$ via mean-patch on UnlearnDiffAtk's matched test split. Report ASR, FID, CLIP-score per concept.

**Pass**: per-concept ASR ≥ SAeUron's published per-concept ASR on UnlearnDiffAtk; benign FID degradation ≤ 1.5 vs unmodified SDXL on UnlearnCanvas retain-side. Run all four concepts in parallel — the new GPU has the headroom.

**VRAM**: 16 GB per concept run. **Co-schedule with**: any GPU job.

### D-5. Black-box transfer attacks across diffusion models

**Why**: Square Attack against SAE detector on the same model is one threat-model leg. The harder leg: craft a perturbation against **one diffusion stack** (SDXL Turbo + Surkov SAEs + B02 detector) and **test against another** (SD v1.5 + SAeUron SAEs + matched detector).

**Method**: Square Attack with 10K queries against the source stack. Apply the perturbed pixel input verbatim to the target stack. Measure cross-stack ASR.

**Pass**: cross-stack ASR < 60% of same-stack white-box ASR. If ≥ 60%, the SAE-feature subspace is alarmingly model-portable from the attacker's view (still publishable as a finding).

**VRAM**: 22 GB (two diffusion stacks resident). **Co-schedule with**: CPU eval only.

### D-6. Joint end-to-end pipeline training

**Why**: the modular pipeline (SAE encode → detector head → Stage-2 selection → intervention) was hand-built. A jointly-trained version may find a tighter optimum. LLM analogue is fully end-to-end concept-erasure (Lu et al. 2025).

**Method**: differentiate through SAE encode → linear detector → soft-Stage-2 (Gumbel-softmax over feature mask) → mean-patch (with straight-through estimator). Three losses: SAE reconstruction, detection BCE on counterfactual pairs, intervention quality ($-$ASR + λ FID-preservation). Train on 5K paired prompts.

**Pass**: joint training matches modular within 1 pp ASR + 0.5 FID. Even if it doesn't beat modular, the comparison is a publishable ablation.

**VRAM**: 32 GB. **Co-schedule with**: CPU eval only.

### D-7. Mechanistic feature-firing trajectory plot — the canonical paper figure

**Why**: a paper's reception is often shaped by a single canonical figure. For this work: one I2P prompt, one bypass-attack run, plot activation magnitude of top-10 Stage-2 features at each hookpoint at each denoising step, with two overlaid traces (clean prompt, attacked prompt). The point at which they diverge tells the story of *when* the safety classifier's decision tips.

**Method**: cherry-pick 3-5 visually clean bypass cases (require a real image). For each, log every SAE feature activation at every step. Plot (a) heatmap, (b) Sankey-flow of top-feature activations across time, (c) before/after rendered images at each step. Save as `outputs/figures/mechanistic_trajectory_<exp_id>.pdf`.

**Pass**: ≥ 3 cases produce visually compelling, interpretable plots; divergence step matches the empirical commit-knee from Phase 1's per-step AUC plot.

**VRAM**: 4 GB. **Co-schedule with**: any GPU job.

### D-8. Adversarial training of the two-stage selection (TRADES-style)

**Why**: v1 appendix Phase C-7 talks about adversarially training the *detector*. The stronger version: adversarially train the **two-stage feature selection** itself.

**Method**: round *r*: (1) compute $F_c^{(r)}$ via Stage 1 ∩ Stage 2 on $\mathcal{D}^{(r)}$. (2) Run pixel + latent + embedding attacks against the deployed pipeline using $F_c^{(r)}$. (3) Augment $\mathcal{D}^{(r+1)}$ with successful bypasses. (4) Repeat for 5 rounds.

**Pass**: round 5 ASR drops below 50% of round 1 ASR; $|F_c|$ stays within 2× round-1 size; benign FID stays within 1.5 of round-1 FID.

**VRAM**: 14 GB during attack rounds. **Co-schedule with**: any non-attack GPU job.

### D-9. Cross-architecture generalization to FLUX / SD3

**Why**: v1 appendix Phase C-5. **Now feasible** with 96 GB. FLUX at fp16 is 24 GB; FLUX + SAE training is 36 GB; co-locates with SDXL and a CPU eval pool. If the SAE-feature subspace identified on SDXL Turbo aligns with feature subspaces in a Transformer-DiT architecture (FLUX) or in SD3, that's a much stronger generality claim than within-SDXL only.

**Method**: train SAEs on FLUX at structurally analogous hookpoints (post-attention residual streams in DiT blocks vs SDXL's UNet `down.2.1` etc.). Compute alignment between SDXL Stage-2 nudity features and FLUX Stage-2 nudity features via:
- Cosine similarity of decoder columns after Procrustes alignment.
- Causal-intervention output score on shared concepts.
- ASR-with-cross-model-features: SDXL's $F_c$ projected into FLUX feature space, intervene, measure FLUX ASR drop.

**Pass**: ≥ 30% of SDXL Stage-2 nudity features have a strongly-aligned FLUX counterpart (cosine ≥ 0.5 after Procrustes). Cross-model intervention drops FLUX ASR by ≥ 50% of within-FLUX intervention.

**VRAM**: 48 GB peak. **Co-schedule with**: CPU FID/eval only.

### D-10. Compositional / multi-concept simultaneous defense

**Why**: production safety must handle multiple concepts at once. Tests whether the Stage 1 ∩ Stage 2 selection composes — does $F_{c_1} \cup F_{c_2} \cup F_{c_3}$ work, or do they interfere?

**Method**: build $F_{c_1}, F_{c_2}, F_{c_3}$ for $c_i \in \{\text{nudity}, \text{violence}, \text{Van-Gogh-style}\}$ independently. Train a 3-class detector head. Intervene with the union on every flagged prompt. Eval per-concept ASR + benign FID, plus cross-concept-leak ASR (does intervening on nudity also remove Van Gogh on a Van-Gogh-only prompt?).

**Pass**: per-concept ASR within 5 pp of single-concept Contribution 4 result. Cross-concept-leak < 10 pp on each off-diagonal. Benign FID degradation ≤ 1.5 vs single-concept regime.

**VRAM**: 18 GB. **Co-schedule with**: CPU eval, FLUX inference (D-9).

### Combining D-N ideas

Combine when sensible. The new hardware lets you actually do this:
- D-4 cross-concept on D-2 learned-projection — does the learned projection generalize across concepts?
- D-7 mechanistic figure on D-9 FLUX — single canonical figure showing FLUX feature firing for a cross-architecture-transferred bypass.
- D-8 adversarial training feeding into D-5 black-box transfer — does the adversarially-hardened pipeline survive cross-model attacks?
- D-10 compositional defense + D-2 learned projection — compositional erasure with learned-projection intervention.

---

## 7. The framing-decision moment

When the four discriminator results land:
- Item 1c-0 (counterfactual benchmark built, Strategy 1 + 2 + 3 all completed)
- Item 1c-1 (cross-target detector bug fixed and re-run)
- Item 1c-3 (B02 oracle re-trained on counterfactual + I2P + UnlearnDiffAtk)
- C-2 re-run on counterfactual benchmark

Write `reports/REFRAMING_DECISION.md` with the four supporting numbers and a clear "Framing A canonical" or "Framing B canonical" verdict per the §2 rule. Update `paper/main.tex` accordingly. Move the un-chosen framing's notes to `paper/archive/`. **The other framing's experiments are not undone — they live on as supplementary appendices in the paper.**

Decision rule:
- Counterfactual probe shows B02-SAE beats raw activations by ≥ 3 pp AUC on held-out counterfactual cell **AND** SAE detector substantially more robust to white-box attack than raw probe → **Framing A canonical** (the original four-contribution story).
- Raw matches SAE on counterfactual (within 2 pp) **AND** under attack (similar ASR) **AND** raw activations admit comparable per-dimension intervention with comparable FID → **Framing B canonical** (interpretability-as-deliverable).
- Mixed evidence (one discriminator favors A, another favors B) → Framing A canonical with explicit acknowledgment in §2 (the safer choice, since most reviewers find A's structure easier to follow).

---

## 8. Risks and contingencies

| risk | mitigation |
|---|---|
| Strategy 3 Gemini refusal rate > 50% on unsafe cells | Path B Llama 3.1 70B fallback handles whatever Gemini refuses. Both run in parallel from the start. |
| FLUX cross-model transfer (D-9) yields no shared features | Documented as a negative result; the paper carries it as evidence of architectural specificity. Run; don't skip. |
| Adversarial training (D-8) does not close cross-target gap | Documented honestly. Paper still has white-box-only result. |
| Counterfactual benchmark Strategy 1 yields fewer than 200 valid pairs | Drop to 100 with caveat; Strategy 2 + Strategy 3 compensate. |
| 96 GB runs out under aggressive co-scheduling | OOM → kill lowest-priority job → re-plan. 95% ceiling intentional; do not push to 99%. |
| Disk space fills (full image saving across all experiments) | Network volume is 512 GB; budget ~250 GB for images, ~150 GB for cached activations, ~100 GB for checkpoints. Prune older bypass images first if hit. |
| Joint end-to-end (D-6) fails to train | Document as negative result. Modular pipeline is the headline either way. |

---

## 9. End of v2 main spec

Phase 1c closes the structural gaps. Phase D adds the ten new ambitious ideas. Both framings are pursued in parallel; the data picks which becomes the paper's primary structure at the framing-decision moment.

Stop condition: human interrupts. **Phase 1c does not pause Phase C; Phase D does not wait for Phase 1c to fully close** — items can run as soon as their dependencies are satisfied. The 96 GB / 64 vCPU / 263 GB pod must be saturated: ≥ 3 GPU + ≥ 12 CPU + monitor at all times.

Be ambitious. Run high-scale experiments. The hardware can take it.