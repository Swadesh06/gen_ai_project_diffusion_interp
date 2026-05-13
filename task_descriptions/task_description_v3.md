# DiffSafeSAE v3 — Publication-grade closure of every contribution

> **Status as of v3 issuance**: Phase 1c + Phase D have produced experimental material but **none of the four contributions is yet at publication standard.** Honest audit (below, §1) shows every contribution has at least one binding gap — undersized experiments, missing baselines, missing ablations, or missing matched-condition head-to-head comparisons.
>
> **v3 supersedes v2** on what counts as "done." v3 is a closure spec — its job is to take the contributions from "experimental material exists" to "paper-grade evidence exists." It does *not* add new ideas. It binds the agent to **finish** what v1 and v2 started.
>
> **Read alongside** `task_description_v1.md`, `task_description_v1_appendix.md`, and `task_description_v2.md`. All four documents are now binding. v3 supersedes v2 on completion criteria and adds the publication-grade gates.
>
> **No new ideas are introduced in v3.** Phase D ideas not yet executed (D-1 attribution patching, D-2 learned projection applied at intervention time, D-6 joint end-to-end, D-8 TRADES, D-9 cross-architecture SAE training, D-10 compositional defense) remain on the queue but rank below closing the existing contributions. The agent does **not** start new D-items until the four contributions have closure gates met.
>
> **Paper writing is out of scope for v3.** The human will write the paper later. The agent's job is to produce the experimental evidence that will populate the paper — full ablation tables, head-to-head baseline comparisons, qualitative figures, properly-sized comparison grids. The agent does not draft prose, abstracts, or LaTeX sections. The agent does produce programmatic tables (CSV/JSON) and figures (PDF/PNG) as natural artifacts of experiments, saved to `outputs/figures/` and `outputs/tables/` for later use.
>
> **The default operating posture changes**: previously the agent was rewarded for breadth (start many experiments, mark partial progress). v3 rewards depth. Open experiments are debt; the bar to opening a new one is high.

---

## 1. Honest audit — why v3 exists

After three reviewing passes, the existing reports show **a systematic pattern of under-execution**:

### Contribution 1 (red-team) — closest to ready, still has gaps

What's solid: A01/A02/A03 against safety_checker at ASR=1.000 across multiple seeds. ε-sweep showing saturation at quarter-ε. Black-box Square Attack at 95.4 % at q=5K on n=500 against safety_checker.

What's missing for publication:
- **Per-bypass image saving was retrofit unevenly.** Some experiments have figure.png, many do not. A paper needs canonical qualitative figures for every attack space.
- **SAE-feature attribution maps for bypasses are not in any report.** The cross-space Jaccard table tells you features overlap; the paper needs to *show* which features and what they mean (visualization of top features firing on bypasses).
- **No comparison of cross-space SAE Jaccard across attack budgets.** Does the overlap pattern hold at ε=1/255 the way it does at ε=4/255?
- **NudeNet / Q16 / safety_checker ensemble baseline for ASR is not measured at matched conditions.** The "all three classifiers are brittle to PGD" claim needs the NudeNet-PGD and Q16-PGD results at the same n as safety_checker. Currently only safety_checker has the full grid.

### Contribution 2 (SAE detector) — methodologically reframed, baselines incomplete

What's solid: B02-v3 at AUC 0.977 on oracle-balanced 1544-image dataset. Safety-trained SAE v2 closes the 1.21 pp gap to raw. C-9 transcoder at AUC 0.991. Counterfactual Strategy 2 establishes raw ≈ SAE on the meaningful discriminator.

What's missing for publication:
- **No ensemble of B02-v3 variants.** Ten heads were trained; one was picked. The ensemble across heads (mean, max, vote, learned-stacker) was never tested. This is a trivial extension that could move the AUC from 0.977 to 0.99+, and may meaningfully change the adaptive-attack robustness picture.
- **Counterfactual Strategy 3 (paraphrase-conditioned 4-cell) was paused after 0-refusal validation.** No downstream rendering, no detector evaluation on it. Strategy 1 + Strategy 2 are not enough by themselves.
- **No per-architecture SAE detector training.** B02-v3 on MMA = AUC 0.388 (chance, OOD failure on SD v1.4 images). The whole "SAE detector generalizes" claim is unsupported until per-arch SAE detectors are trained and evaluated. SD v1.4 SAE + detector is the most important.
- **Hybrid raw‖SAE detector (C-6) was tested in-distribution only.** Hybrid vs raw vs SAE under attack at matched ε — not tested.

### Contribution 3 (cross-target robustness) — strongest contribution, but framing is overclaimed

What's solid: Bit-identical detector-logits bug fixed and verified. Cross-target transferability across multiple seeds on A01+A02 = 1/173 (0.58 %). D-5 oracle-transfer: 87.5 % of A01 bypasses escape NudeNet+Q16+SC ensemble; SAE detector catches 100 %. D-6 B02-adv on held-out A03 = 99.0 % catch (vs B02-v3's 32.4 %), a +66.6 pp lift.

What's missing for publication (this is the largest gap in the whole project):
- **Matched-budget black-box comparison was never run.** safety_checker Square attack at q=5K = 95.4 % ASR. SAE detector Square attack at q=500 = 16.7 % ASR (n=12 against B02-adv only). These are not comparable. The whole "SAE detector is more robust" claim is at a 10× budget mismatch.
- **The required matched experiment**: Square Attack at q ∈ {500, 5K, 10K} against {safety_checker, NudeNet, Q16, B02-v3, B02-adv} on the **same** 500-prompt list. **15 cells. Currently 1 filled.** Without this, Contribution 3 has no quantitative robustness claim.
- **ε-sweep against the SAE detector white-box was never run.** Only one ε value (4/255) tested against B02-adv white-box. The hypothesis "sparse-topk features are harder to attack at small ε" was never measured. The SAE detector could saturate at ε=1/255 (no advantage) or hold at ε=1/255 (real mechanistic finding). Unknown.
- **A02 latent-PGD and A03 embedding-PGD adaptive against SAE detector were never run.** Only A01 pixel-PGD adaptive. Three spaces should all be tested.
- **NES (gradient-free zeroth-order) was never run.** Square Attack is one black-box family; NES is the standard complement.
- **Joint adaptive PGD with sum-of-logits across (safety_checker + SAE detector) was never run.** This is the realistic "attacker knows both defenses" threat model.

### Contribution 4 (intervention) — weakest contribution, multiple binding gaps

What's solid: F_c selection (Stage 1 ∩ Stage 2) produces 69 features, effective rank ~24. D-4 cross-concept: 0 nudity-violence feature overlap (clean monosemantic result). Mean/zero/resample patch primitives empirically tied (falsifies v1 prediction, but is itself a clean methodological finding).

What's missing for publication (the contribution cannot be submitted at this state):
- **n_pre_flagged = 10 on the headline D02 number.** With only 10 pre-flagged prompts, any "dominates baselines" claim is on too narrow a base. Must scale to a respectable size — equivalent to roughly 700 prompts at SDXL Base 4-step's 28.6 % flag rate to get n_pre ≈ 200.
- **SAeUron baseline is at n=30, single seed.** Not a baseline. Must rerun at matched scale on the same prompt list as F_c.
- **SAEmnesia baseline does not exist.** Must be reproduced from scratch (no public release) at matched scale.
- **DSG-adapted baseline does not exist.** Must be reproduced (Fisher-only feature selection, always-on clamping) at matched scale.
- **UDA-nudity OOD test shows F_c is net-negative (+12 pp flag rate).** Must add **conditional intervention** — only intervene when B02-v3 fires pre-generation — to fix the FP cascade. Conditional intervention is the actual deployable version anyway.
- **D-2 learned projection Pi matrices were trained but never applied at intervention time.** The downstream comparison (D02 + Pi vs D02 + mean) does not exist.
- **D-1 causal feature graph is correlation-based, not attribution patching.** Real intervention-based causal scoring is required.
- **D-7 mechanistic trajectory uses fresh-noise vs post-attack-noise as "clean" baseline.** Must rerun with same-noise pre/post comparison.

### Cross-cutting gaps

- **Experiment sizes inconsistent.** Some experiments at n=500; others at n=10. Baselines reproduced at n=30. Same-method runs across different gates use different denominators. Within a single comparison table, every method should be run on the same prompt list and same scale.
- **Baselines never reproduced at matched conditions.** SAeUron at n=30 vs F_c at n=10 isn't a comparison. Each baseline must run on the same prompts × same conditions as our method.
- **No experiment at meaningfully large scale has been run.** The hardware is sized for this. The agent has been running 100-200-prompt experiments and moving on. Publication-grade comparisons on UDA-nudity, UDA-violence, I2P-NSFW, MMA-Diffusion at sizes appropriate to each benchmark, for every method × every benchmark, is the missing scale.

### What this audit means operationally

The previous task descriptions implied that "passing a gate" was sufficient to move on. **It was not.** Gates were set at MVP scale (n=100, n=200, single seed). Publication scale is 10-50× larger in many cases. The agent moved on from contributions whose gates passed at MVP scale but whose results would not survive peer review.

v3 binds the agent to **publication-scale completion**. Until each contribution has its closure gate met, no new contribution or new idea can begin.

---

## 2. Operating principle for v3

### The completion principle

Each of the four contributions has a **closure gate** in §3 below. A contribution is "done" only when its closure gate is fully met. Until then, the contribution remains the agent's primary focus, regardless of how interesting Phase D or other new ideas appear.

### The size-and-completeness principle

For each closure-gate cell, the size of the experiment is set by what's appropriate for *that comparison* — generally large enough to be respectable rather than indicative. Concrete defaults:

- **For attack-success comparisons** (ASR-type metrics): aim for n_pre_flagged ≥ 200 in the denominator. On SDXL Base 4-step (28.6 % flag rate) this is ~700 prompts. Below 200 pre-flagged the comparison is too narrow.
- **For detection / AUC comparisons**: use the full available test split (e.g. UDA-nudity = 142 prompts, UDA-violence = 200 prompts, MMA = 103 images). When benchmarks have an established protocol, use that protocol's full scale.
- **For baseline reproductions**: run on the same prompt list × same conditions as our method. Same n, same prompts.

If a cell cannot reach n_pre_flagged ≥ 200 within reasonable execution (e.g. an OOD benchmark with low natural flag rate), document why and report the achieved n. Don't fudge by reporting from a smaller denominator without context.

### The single-seed-now, seed-noise-later principle

For v3, the priority is **completed experiments** — fully-populated comparison tables across methods and benchmarks. Repeat the same experiment over multiple seeds **only after** the full grid is filled at a single seed. The seed-noise study is a separate side-track that runs after the main grid completes, picking a few representative cells to repeat over (say) 3-5 seeds and quantifying within-cell variance.

This means:
- A single-seed run that compares F_c, SAeUron, SAEmnesia, DSG-adapted on UDA-nudity, UDA-violence, I2P-NSFW, MMA at the right n is **more valuable** than a 5-seed F_c-only run at the same scale.
- Headline tables in `reports/GATE_<N>_CLOSURE_v1.md` report single-seed numbers when that's what's available; seed-noise side-runs add a "±" column later.
- Where data already exists at multiple seeds (e.g. some Phase 1 attacks already have 5-seed CIs), use it. Don't downsample.

### The baselines-at-matched-conditions principle

When claiming "our method beats X" or "our method is more robust than X," the baseline X must be run on:
- The same prompt list (same input set).
- The same backbone (same SDXL Base / Turbo / SDv1.4 etc.).
- The same scoring oracle.
- The same n.

Different prompt samples, different backbones, different n → not a comparison. The published numbers from the original baseline paper do **not** substitute for in-pipeline reproduction.

### The ensemble-first principle

When multiple variants of the same component exist (B02-v3 has 10 trained heads), the **default reported result is the ensemble**, with single-head results as ablations. Ensembles are cheap (the agent has trained the heads already) and reviewers will ask why they weren't tried. Pre-empt the question.

### The ablation-completeness principle

For every claim about a method, the corresponding ablation is reported. Specifically:
- F_c performance → ablate: Stage 1 only, Stage 2 only, Stage 1 ∩ Stage 2 (proposed).
- F_c patch primitive → ablate: mean, zero, resample, learned-projection.
- F_c intervention timing → ablate: always-on vs conditional-on-detector-firing.
- Detector ensembling → ablate: each single head, each ensemble strategy (mean / max / vote / learned).
- Detection signal → ablate: raw activations, SAE features, hybrid, transcoder, safety-trained SAE.

Any headline number must have its ablation table in the same gate's closure report.

### The "no new ideas" principle

v3 does **not** introduce new methods. The agent does not invent new attacks, new detectors, new feature-selection methods. Every experiment in v3 either:
- Closes a publication-scale gap in an existing contribution, **or**
- Reproduces a published baseline at matched conditions, **or**
- Adds a matched-condition cell to an existing comparison table.

Curiosity-driven exploration of new directions is paused.

### Resource budget transparency

The agent keeps a running **closure-gate progress log** in `reports/V3_CLOSURE_PROGRESS.md` updated after each experiment lands. Format: per closure gate, list cells filled / cells required / blocking experiments / next experiment. This makes under-execution visible early.

### How to handle conflict with v1/v2

When v1 or v2 specifies a smaller-scale experiment than v3 requires, **v3 wins**. v1/v2 reproductions at small n that already exist are kept on file but the headline numbers come from the v3-scale reruns.

---

## 3. The four closure gates

Each gate lists: cells required, current status, what to run.

### Gate 1 — Contribution 1 closure: Cross-Space Red-Team

**Cells required (8)**:

| cell | size guidance | status |
|---|---|---|
| 1.1 A01 pixel-PGD vs safety_checker | n=500 | ✅ done (88/88) |
| 1.2 A02 latent-PGD vs safety_checker | n=500 | ✅ done (100/100) |
| 1.3 A03 emb-PGD vs safety_checker | n=500 | ✅ done (102/102) |
| 1.4 ε-sweep on A01 + A02 | sweep | ✅ done (228/228 at ε ∈ {4, 2, 1}/255) |
| 1.5 Square Attack vs safety_checker | n=500, q=5K | ✅ done (95.4 % at q=5K) |
| 1.6 PGD vs NudeNet at matched ε | n=500, 3 spaces | ❌ **missing** |
| 1.7 PGD vs Q16 at matched ε | n=500, 3 spaces | ❌ **missing** |
| 1.8 SAE-feature attribution visualization | 5-10 bypass cases | ❌ **missing** |

**Closure work**:
- **1.6 + 1.7**: rerun A01/A02/A03 against NudeNet and Q16 oracles instead of safety_checker. Same protocol: 500 prompts × 3 attack spaces × 2 new oracles = 6 runs. Reuse infrastructure. This establishes the multi-classifier-brittleness claim — currently only safety_checker has the data.
- **1.8**: pick 5-10 representative A01/A02/A03 bypass cases (one per attack space × concept). For each, generate the SAE attribution heatmap (top 10 features firing on pre vs post image at each of the 4 hookpoints) as a figure. Output `outputs/figures/F1_sae_attribution.pdf`. Programmatic — no manual matplotlib tweaks.

**Pass**: 8/8 cells filled with results JSON and figures saved.

### Gate 2 — Contribution 2 closure: SAE Detector

**Cells required (12)**:

| cell | description | status |
|---|---|---|
| 2.1 B02-v3 AUC on oracle-balanced dataset | 1544 samples | ✅ done (0.977) |
| 2.2 B02-v3 AUC on counterfactual Strategy 2 | 246 pairs | ✅ done (SAE 0.941 ≈ raw 0.944) |
| 2.3 Safety SAE v2 closes raw gap | expansion 16/32 | ✅ done (1.000) |
| 2.4 C-9 transcoder AUC | 3 pairs | ✅ done (best 0.991) |
| 2.5 **B02-v3 ensemble across 10 heads** | 4 strategies × 5 datasets | ❌ **missing** |
| 2.6 B02-v3 hybrid raw‖SAE under attack | matched ε | ❌ in-dist only |
| 2.7 B02-v3 ensemble vs adaptive white-box PGD | ε-sweep | ❌ **missing** |
| 2.8 B02-v3 ensemble vs adaptive black-box Square | q-sweep | ❌ **missing** |
| 2.9 Per-architecture SAE detector on SD v1.4 | train + eval on MMA | ❌ **missing** |
| 2.10 Per-architecture SAE detector on SD3 | train + eval on UDA | ❌ **missing** |
| 2.11 Counterfactual Strategy 3 detector eval | 4-cell matrix | ❌ **missing** — paraphrases generated but not rendered or scored |
| 2.12 Detection AUC on full UDA-nudity + UDA-violence + MMA | full benchmarks | ❌ **missing** |

**Closure work**:
- **2.5**: take the 10 already-trained B02-v3 heads (linear + MLP variants × 4 hookpoints + 2 concat). Implement 4 ensemble strategies: mean-logit, max-logit, majority-vote, learned-stacker (10-d → 1 logistic regression on val split). Evaluate AUC on (a) B02-v3 val split, (b) counterfactual Strategy 2, (c) UDA-nudity, (d) UDA-violence, (e) MMA-Diffusion. Report all 4 strategies × 5 datasets = 20 cells in `reports/B02_v3_ensemble_v1.md`.
- **2.7 + 2.8**: with the ensemble in hand, run adaptive white-box PGD and black-box Square Attack against it. **Combine with Gate 3 cells below** — these are the same experimental grid.
- **2.9**: train Surkov-style SAEs on SD v1.4 activations (4 hookpoints, expansion 4, k=10). Captures ~50K images of SD v1.4 outputs. Train a new B02-style oracle detector on top. Evaluate on MMA-Diffusion. **This closes the MMA OOD failure.**
- **2.10**: train SAE on SD3 transformer block activations (4 blocks). Train detector. Evaluate on UDA-nudity rendered through SD3.
- **2.11**: render the existing 1200 Gemini paraphrases and 240 local-Llama paraphrases (Strategy 3 Path A + Path B). Score each with safety_checker + B02-v3 ensemble. Build 4-cell consistency matrix: does the detector treat 3 paraphrases of the same anchor consistently? Report consistency rate per cell.
- **2.12**: run B02-v3 ensemble (cell 2.5 output) and B02-adv ensemble on full UDA-nudity (n=142), UDA-violence (n=200), MMA (n=103). Report AUC + AP + recall@5%FPR for each.

**Pass**: 12/12 cells filled. Ensemble result included as default.

### Gate 3 — Contribution 3 closure: Cross-Target & Adversarial Robustness

This is the largest gap. Contribution 3 currently has one strong cell (cross-target unaware-adversary), and the binding adaptive-adversary claim is one isolated cell. Publication grade requires the full matrix.

**Cells required (38)** — the matched-budget grid:

For each cell: 500 prompts, paired comparison via same-prompt-list, image-saving on every bypass.

| target → \ attack ↓ | safety_checker | NudeNet | Q16 | B02-v3 ensemble | B02-adv ensemble |
|---|---|---|---|---|---|
| White-box A01 pixel-PGD ε=4/255 | ✅ 1.000 | ❌ | ❌ | ❌ (have small-n) | ❌ |
| White-box A01 ε=2/255 | ✅ 1.000 | ❌ | ❌ | ❌ | ❌ |
| White-box A01 ε=1/255 | ✅ 1.000 | ❌ | ❌ | ❌ | ❌ |
| White-box A02 latent-PGD | ✅ 1.000 | ❌ | ❌ | ❌ | ❌ |
| Black-box Square q=500 | ❌ | ❌ | ❌ | ❌ | partial 16.7 % n=12 |
| Black-box Square q=5K | ✅ 0.954 | ❌ | ❌ | ❌ | ❌ |
| Black-box Square q=10K | ❌ | ❌ | ❌ | ❌ | ❌ |
| Black-box NES q=5K | ❌ | ❌ | ❌ | ❌ | ❌ |
| Joint adaptive PGD (sum-of-logits SC + SAE) | N/A | N/A | N/A | ❌ | ❌ |

Roughly **38 unique cells, 2 filled, 36 missing**.

**Closure work** (in priority order):

1. **Black-box Square at q ∈ {500, 5K, 10K} against {safety_checker, NudeNet, Q16, B02-v3 ensemble, B02-adv ensemble}**. 15 cells. Same 500-prompt list. Parallelize on the 80 GB A100 within the throughput-discipline rule (§6).

2. **White-box ε-sweep against B02-v3 ensemble and B02-adv ensemble**. ε ∈ {4, 2, 1}/255 × 3 attack spaces × 2 targets = 18 cells. This tests the "SAE features are mechanistically harder to attack at small ε" hypothesis.

3. **NES at q ∈ {500, 5K, 10K}** against the same 5 targets. 15 cells. NES is query-heavier than Square; budget VRAM accordingly.

4. **Joint adaptive PGD** with `loss = safety_checker_logit + λ × sae_detector_logit` against B02-v3 ensemble and B02-adv ensemble. 2 cells × λ sweep ∈ {0.1, 1, 10}. The realistic "attacker knows everything" threat model.

5. **Cross-classifier transferability**: PGD-on-X bypass images scored on Y for every (X, Y) pair. Already partially done (D-5 oracle transfer covered some); fill the matrix.

**Pass**: All cells filled. Headline table reads as ASR(target, attack, budget) with no missing entries.

### Gate 4 — Contribution 4 closure: Intervention

**Cells required (18)**:

| cell | description | status |
|---|---|---|
| 4.1 F_c structure (size, effective rank, concept-specificity) | — | ✅ done (69, ~24 ER, 0 violence overlap) |
| 4.2 D02 mean-patch on I2P at SDXL Base 4-step | n_pre ≥ 200 | ❌ **n=10 only** |
| 4.3 D03 zero-patch matched | n_pre ≥ 200 | ❌ **n=10 only** |
| 4.4 D04 resample-patch matched | n_pre ≥ 200 | ❌ **n=10 only** |
| 4.5 D02 + B02-v3 conditional gating | n_pre ≥ 200, UDA-nudity | ❌ **missing — would fix the +12 pp FP cascade** |
| 4.6 D-2 learned projection applied at intervention | n_pre ≥ 200 | ❌ **trained but never applied** |
| 4.7 SAeUron repro on UDA-nudity | n ≥ 200, matched | ❌ **n=30 only** |
| 4.8 SAeUron repro on UDA-violence | n ≥ 200, matched | ❌ missing |
| 4.9 SAeUron repro on I2P-NSFW | n ≥ 200, matched | ❌ missing |
| 4.10 SAEmnesia reproduced from scratch on UDA-nudity | n ≥ 200, matched | ❌ missing |
| 4.11 SAEmnesia on UDA-violence | n ≥ 200, matched | ❌ missing |
| 4.12 SAEmnesia on I2P-NSFW | n ≥ 200, matched | ❌ missing |
| 4.13 DSG-adapted on UDA-nudity | n ≥ 200, matched | ❌ missing |
| 4.14 DSG-adapted on UDA-violence | n ≥ 200, matched | ❌ missing |
| 4.15 DSG-adapted on I2P-NSFW | n ≥ 200, matched | ❌ missing |
| 4.16 D-1 attribution-patching causal graph (not correlation) | top-69 F_c features | ❌ correlation only |
| 4.17 D-7 trajectory v2 (same-noise pre/post comparison) | 5-10 cases | ❌ confounded baseline |
| 4.18 D-10 compositional defense (F_c_nudity ∪ F_c_violence) | n ≥ 200, both concepts | ❌ scripts only |

**Closure work** (priority order — this is the heaviest gate):

1. **Switch headline benchmark to SDXL Base 4-step** (28.6 % flag rate). At n=700 prompts you get ~200 pre-flagged, enough denominator. **Item 1c-7 already established this works.**

2. **Run D02 + D03 + D04 at n=700 on UDA-nudity + UDA-violence + I2P-NSFW + MMA-Diffusion as OOD**. 4 benchmarks × 3 patches × 700 prompts. Cells 4.2–4.4. Parallelize on the 80 GB A100 within the throughput-discipline rule (§6).

3. **Implement conditional gating (cell 4.5)**: intervene only when B02-v3 ensemble fires pre-generation, otherwise pass through unmodified. This fixes the UDA-nudity +12 pp regression. Same n=700 × 4 benchmarks.

4. **Apply D-2 learned projection (cell 4.6)**: drop-in replacement for mean-patch primitive. Same n × benchmarks. Direct comparison to D02.

5. **Baseline reproductions (cells 4.7–4.15) — 9 cells**: SAeUron, SAEmnesia, DSG-adapted on UDA-nudity, UDA-violence, I2P-NSFW. Same n=700 per benchmark, same prompt list as F_c, same backbone. SAEmnesia is the hardest — requires training a supervised SAE from scratch on labelled UnlearnCanvas concepts since no public release. With the A100 80 GB this fits at full scale.

6. **Attribution-patching causal graph (cell 4.16)**: replace the correlation-based D-1 with proper intervention-based scoring. For each candidate edge (f_A at hookpoint h_A, f_B at hookpoint h_B), clamp f_A to a baseline value, measure Δf_B. Use Syed et al. 2023 attribution-patching approximation.

7. **D-7 trajectory v2 (cell 4.17)**: rerun mechanistic trajectory plots with **same initial noise** for pre and post, varying only whether the attack's perturbation is applied. 5-10 representative cases.

8. **D-10 compositional (cell 4.18)**: F_c_nudity ∪ F_c_violence applied jointly. Measure per-concept ASR + cross-concept leak (does nudity intervention damage Van Gogh style on UnlearnCanvas?).

**Pass**: 18/18 cells filled. Headline table compares Stage 1 ∩ Stage 2 + mean (proposed), + zero, + resample, + learned-projection, + conditional gating vs SAeUron, SAEmnesia, DSG-adapted across 4 benchmarks at the appropriate n with same-prompt-list matching.

---

## 4. Experiment-size and comparison-rigor floor for v3

Every cell in every gate must meet these:

- **Respectable denominator**: n_pre_flagged ≥ 200 for any cell whose metric is "ASR among pre-flagged" or "correction rate among pre-flagged." Below 200, do not report — run more prompts. Use SDXL Base 4-step (28.6 % flag rate) as the default backbone for this reason. For detection AUC, use the full benchmark split when one exists.
- **Same-prompt-list rule**: when comparing two methods, they must be evaluated on the **exact same prompts** at the **same scale** through the **same backbone** with the **same scoring oracle**. Different prompt samples invalidate the comparison. **This is the single most binding rule.**
- **Single-seed grid first, seed-noise side-runs later**: complete the full method × benchmark grid at a single seed before repeating any cell. After the grid is filled, pick representative cells and re-run at 3-5 seeds to characterize within-cell variance. Don't block grid completion for variance characterization.
- **Effect-size reporting**: report absolute difference and relative improvement (Δ% = 100 × (new − old) / old) alongside any direction-of-effect claim. Reviewers care about effect size, not just "this is bigger than that."
- **Ablations alongside headlines**: every headline number is accompanied by the ablation rows defined in §2 (ablation-completeness principle). The closure report's table is the headline plus its ablations as a single, populated grid.
- **Full-scale principle — no shortcuts, no fallbacks**: every experiment runs at full scale with the same architecture, batch size, expansion factor, training schedule, and feature dimensionality that the published baseline uses. **The hardware (A100 SXM4 80GB, see §6) fits every v3 workload at full scale.** If a workload appears not to fit, the agent's first move is **hyperparameter and runtime adjustment** (batch size, gradient accumulation, gradient checkpointing, fp16/bf16, micro-batching, activation offload) — not architecture downscaling, not reduced SAE expansion, not smaller n. Architecture-level downscaling (smaller SAE, fewer concepts, partial reproduction) is **not permitted** unless the agent first documents that every hyperparameter and runtime mitigation has been tried and failed. The goal is publication-grade results, which means full-scale experiments comparable apples-to-apples with the baseline papers.

---

## 5. What's frozen and what's in scope

### Frozen (no further work)

- Phase D-9 cross-architecture *exploration* (the smoke tests and baselines). Replaced by the targeted per-arch SAE detector training in Gate 2 cells 2.9 + 2.10.
- New methodological ideas not already in v1/v2.
- Mechanistic exploration beyond what's needed for the four contributions (single-feature deep-dives, novel SAE architectures, attention visualizations).
- **Paper writing.** The human will handle this. The agent produces the experimental evidence; it does not write prose, abstracts, related-work sections, intro, conclusion, or LaTeX. Programmatic tables (CSV, JSON) and figures (PDF, PNG) produced as natural outputs of experiments are kept in `outputs/figures/` and `outputs/tables/` for later use.

### In scope (binding work)

- Everything in §3 closure gates above.
- Bug fixes and infrastructure improvements that unblock gate work.
- One-time reproductions of published baselines.
- Programmatic generation of result tables (CSV/JSON) and figures (PDF/PNG) as part of each experiment.

### Out of scope (deferred to a hypothetical v4)

- D-6 joint end-to-end pipeline training (not on the critical path).
- D-8 TRADES adversarial training (5-round loop — not on the critical path).
- FLUX cross-architecture transfer (the activations are captured; revisit only if Gates 1–4 close early).
- LoRA-baked safety training (not in v1 spec).
- Seed-noise side-runs (begin once the main grid completes).

---

## 6. Hardware utilization under v3

The current pod is **1× NVIDIA A100 SXM4 80GB (sm_80, Ampere) with HBM2e memory at ~2 TB/s bandwidth, paired with AMD EPYC 7763 (verify `nproc` for allocated vCPU count) and ~233 GB system RAM. Driver 580.159.03, CUDA 13.0, power limit 400 W.**

This is different from the v2 spec (which was written for a Blackwell PRO 6000 WK). Key implications:

- **VRAM ceiling: 80 GB** (was 96 GB on PRO 6000 WK). Hard target stays **90 %** (~72 GB), hard ceiling **95 %** (~76 GB). Per-card slot for ~14 GB heavy workloads: roughly 5 concurrent (was 6 on 96 GB). All v3 workloads — including the heaviest (SD3 SAE training ~28-36 GB, SAEmnesia from-scratch ~20-28 GB) — fit comfortably with headroom.
- **Memory bandwidth ~2 TB/s** (was ~960 GB/s on Blackwell GDDR7). The A100's HBM2e gives roughly **2× the bandwidth** for diffusion attention and SAE forward passes, which are memory-bandwidth-bound. Per-cell wall-clock should be substantially faster than what v2 measured on PRO 6000 WK for the same operations. **Use this — don't artificially batch-throttle assuming Blackwell-class bandwidth.**
- **Architecture: sm_80 (Ampere)**, not sm_120 (Blackwell). The agent verifies on first session boot via `torch.cuda.get_arch_list()` — sm_80 should be present in the existing conda env's torch wheel (it almost always is, since sm_80 is the most-targeted training arch). **If verification passes, do not reinstall anything.** Don't fix what isn't broken.
- **RAM ~233 GB** (was 263 GB). The 200 GB activation-cache target from v2 stays feasible but tighter — clamp at ~180 GB max if needed.
- **CPU cores: verify `nproc`** on session boot. EPYC 7763 has 64 cores natively, but Runpod typically allocates a subset (often 16-32 vCPUs per A100 SXM). The CPU worker pool size from v2 should be **scaled to 75 % of the allocated `nproc`** — e.g. 16 vCPU → 12-worker pool, 32 vCPU → 24-worker pool. Don't hardcode 48.

### Closure-gate parallelism on the A100

| gate | parallelism profile | what to co-schedule |
|---|---|---|
| Gate 1 closure (2 missing attack cells + figure) | A01/A02/A03 × {NudeNet, Q16} — disjoint VRAM, ~5 concurrent attacks on 80 GB | CPU oracle scoring + figure generation |
| Gate 2 closure (ensemble + per-arch SAEs + S3 eval + UDA/MMA AUC) | SD v1.4 SAE training (~16 GB) + SD3 SAE training (~36 GB) as long-running background workloads; ~30 GB free for eval | run alongside any attack work |
| Gate 3 closure (matched Square × NES × ε-sweep × joint adaptive) | 38 cells, embarrassingly parallel; ~5 concurrent on 80 GB | NES is query-heavy — pair with CPU eval |
| Gate 4 closure (D02 scale + 9 baselines + projection + causal v2) | SAEmnesia training (~28 GB) + ~3 baseline inferences (~42 GB) = 70 GB co-located; rest queued | scaled CPU pool for FID / CLIP / LPIPS |

### Continuous billed-by-the-second monitoring

**The GPU is billed per-second. Idle GPU = burning money.** The agent maintains a `monitor` tmux session running `nvidia-smi dmon -s pucvmet -d 5 > logs/gpu_monitor.log` (5-second sample interval), plus a CPU monitor.

The agent **checks the monitor at every loop iteration** — every time the agent finishes one experiment and is about to launch the next, it first checks the monitor's most recent samples (last ~60 s of `nvidia-smi dmon` output). Checks include:

- VRAM in use right now (GB).
- GPU utilization average over the last minute (%).
- Number of active tmux sessions running real work (excluding `monitor`, `cpu-worker-*`).
- Time since the last sample where GPU utilization was > 80 %.

**Triggers — act immediately, no waiting**:
- VRAM in use < 35 GB while there are unfilled gate cells → launch more work to fill VRAM up to the 90 % target (~72 GB).
- GPU utilization < 60 % in the most recent minute → launch more work, or diagnose whether a stalled / failed / hung process is blocking.
- GPU utilization 0 % at any sampled moment while gate cells are unfilled → emergency; diagnose and relaunch within the current loop iteration.

### Throughput-discipline rule — parallelism must actually be faster than sequential

Co-scheduling helps **only if** the per-job throughput stays close to its solo throughput. On a single GPU with multiple concurrent processes sharing CUDA streams, this can break: 4 jobs each at 25 % of their solo throughput is the same wall-clock as running them sequentially. The A100's HBM2e bandwidth and large L2 cache mean concurrent jobs *should* scale well, but the agent must **verify**, not assume.

The rule: **after launching any new parallel GPU job, the agent measures whether parallel throughput is meaningfully higher than sequential**. Specifically:

1. Before launching the Nth parallel job: note the per-job step-time (or sample-time, or attack-iteration-time) for the (N-1) already-running jobs from their tmux log tails. Sum to get aggregate throughput.
2. Launch the Nth job. Wait until all N jobs have produced at least 3 progress samples in their logs.
3. Re-measure the aggregate throughput across all N jobs.
4. **Aggregate throughput with N jobs running concurrently must be meaningfully higher than aggregate throughput with (N-1) jobs running.** A good heuristic: each added job should contribute at least ~60 % of its solo throughput to the aggregate. If adding the Nth job lifts aggregate throughput by less than ~30-40 % of the new job's solo speed, you've hit GPU contention.
5. If the parallelism is hurting rather than helping, **kill the most recently launched GPU job** and try a different co-scheduling shape:
   - Move CPU-bound work off the GPU (NudeNet, FID, CLIP, LPIPS, paraphrase rendering, labelling all run perfectly on CPU; do not run them on the GPU).
   - Use MIG to slice the A100 into isolated instances (the A100 supports MIG; useful when two jobs are memory-bandwidth contending).
   - Accept that you're at peak concurrent capacity for the current workload mix and don't try to push past it.
6. **If the GPU is at < 60 % utilization with N concurrent jobs**, the limit is *not* contention — the limit is that the running jobs are CPU-bound or I/O-bound. Add more GPU work to push utilization higher.

On the A100's HBM2e you typically see ~0.85-0.9× per-job scaling up to 4-5 concurrent diffusion jobs, which gives ~3.5-4× aggregate vs solo. If you see less than 2× aggregate vs solo with 3 jobs running, something is wrong — diagnose.

**For CPU-bound work, parallelism is freer.** CPU work (NudeNet inference, FID, CLIP-score, LPIPS, paraphrase rendering, labelling) does **not** contend with GPU work. Run as many CPU jobs in parallel with GPU work as the vCPU pool supports. The throughput check above applies only to GPU-resident jobs.

### Full-scale principle — no shortcuts, no fallbacks

The A100 SXM4 80GB fits every v3 workload at full published-baseline scale. **If a workload appears not to fit**, the agent's response order is:

1. **First, hyperparameter / runtime adjustment**: reduce batch size, increase gradient accumulation, enable gradient checkpointing, switch to bf16 if fp32, micro-batch the SAE forward pass, offload optimizer state to CPU (DeepSpeed-style), shard activations.
2. **Second, if (1) doesn't unblock**: report the specific OOM and what was tried in a report `reports/V3_OOM_<exp_id>.md`. Pause that cell.
3. **Architecture-level downscaling is not permitted.** No reducing SAE expansion from 16 → 4. No reducing the number of concepts in SAEmnesia. No partial reproduction at smaller scale. The published baselines were run at specific configurations; reproducing them at smaller scale doesn't constitute a comparison.

The goal is publication-grade results, which means full-scale experiments comparable apples-to-apples with the baseline papers.

### Hard guardrails

1. Dry-run every new experiment 30-60 s. Record peak VRAM, RAM, utilization.
2. Never run one job idle. If you have spare capacity, fill it.
3. VRAM > 95 % or OOM → adjust **hyperparameters first** (batch, accumulation, checkpointing, bf16, offload) before considering anything else.
4. RAM > 95 % → kill lowest-priority CPU job.
5. Separate processes (separate tmux sessions), not threads.

### Verification on session boot — only act if something actually fails

The agent verifies the env on session boot. **Do not preemptively reinstall packages or rebuild the env.** Specifically:

```python
import torch
print(torch.cuda.get_arch_list())     # expect sm_80 present
print(torch.cuda.get_device_name(0))  # expect "NVIDIA A100-SXM4-80GB" or similar
torch.tensor([1.0]).cuda()             # smoke test
```

If all three pass → proceed directly to work. If any fail → diagnose specifically what failed and fix only that. The conda env was packed on a Blackwell pod but PyTorch builds since 2.0 include sm_80 wheels by default, so the same wheel works on A100. **Don't fix what isn't broken.**

### No time estimates anywhere

The agent does not project how long any gate or cell will take. Progress is measured exclusively by cells closed in `V3_CLOSURE_PROGRESS.md`, not by elapsed time. The agent responds to actual hardware conditions in real time; it does not pace against any expected duration. The only "stall" signal is hardware idleness or repeated consecutive experiments that don't advance any closure-gate cell — both observable directly from the monitor and the progress log, not predicted ahead of time.

---

## 7. Communication and reporting under v3

### Per-experiment reports

Each report still follows the v2 template (Goal / Procedure / Results / Interpretation / Next), but with three additions:

- **"Closure-gate cell"**: which of the §3 cells this experiment is filling (e.g. "Gate 3 cell: Black-box Square q=5K vs NudeNet, n=500").
- **"Comparison-rigor checklist"**: yes/no on each of (n_pre ≥ 200 if applicable, same prompts as comparison method, same backbone, same oracle, image-saving complete, ablation rows included).
- **"Hardware utilization during this experiment"**: peak VRAM, peak RAM, GPU-util average. Reported from `monitor` log tail.

### Per-gate closure reports

When a gate hits 100 % cell completion: write `reports/GATE_<N>_CLOSURE_v1.md` with the headline table, all numbers, all ablation rows, all figures, all caveats. This is the document that captures the gate's evidence for the human to read when writing the paper.

### V3 progress log

`reports/V3_CLOSURE_PROGRESS.md` is a single living document updated after every experiment. Format:

```
Gate 1: 6/8 cells (75 %)  blocking: 1.6 + 1.7  next exp: nudenet_pgd_500_v3
Gate 2: 4/12 cells (33 %) blocking: ensemble + per-arch SAEs  next exp: b02v3_ensemble_v1
Gate 3: 2/38 cells (5 %)  blocking: ALL  next exp: bb_square_grid_phase1
Gate 4: 1/18 cells (6 %)  blocking: scale + baselines  next exp: d02_sdxlbase_n700_v3

Hardware right now: <VRAM_used> GB / 80 GB, GPU-util <X> %, <N> active sessions.
Last 3 experiments: <exp_id_1>, <exp_id_2>, <exp_id_3>. Cells advanced: <list>.
```

This is the at-a-glance status check. If multiple consecutive experiments land without advancing any closure-gate cell, the agent has stalled — self-diagnose immediately.

---

## 8. The framing question, settled

v2 specified "both framings active." That was correct when the data was incomplete. Now that the audit shows the discriminator experiments came back as **mixed evidence with raw ≈ SAE on counterfactual Strategy 2**, the framing-decision rule from v2 §7 fires: mixed evidence → Framing A canonical (the safer choice for reviewers).

**Framing A is the working frame. Framing B is the archive.** v3 stops maintaining Framing B in parallel. `paper/alt_framing_B.md` and `paper/archive/` stay as historical record, but no new experiments target Framing B.

The narrowed Framing A claim — for the agent's internal reference about what claims its evidence is being collected to support, **not** for the agent to write down in any paper section:

> Sparse autoencoder features on UNet activations match raw activations on the in-distribution NSFW-vs-benign detection task (counterfactual benchmark Strategy 2: SAE 0.941, raw 0.944), while providing per-feature attribution that raw activations cannot. The SAE-feature representation enables two complementary capabilities: (1) per-feature concept localization for mechanistic analysis (Contribution 1 SAE attribution maps), and (2) two-stage feature surgery for conditional inference-time intervention (Contribution 4). The SAE-feature detector is defense-in-depth against unaware adversaries, with adaptive-adversary robustness quantified in Gate 3.

---

## 9. End of v3 spec

v3 closes the four contributions to publication standard. The agent does not start new ideas, does not add new methods, does not declare anything "done" until its closure gate is met. **The agent does not write the paper.**

**Stop condition**: human interrupts, or all four closure gates pass. The latter is the deliverable.

The honest reading of the project at v3 issuance: the experiments to date were sufficient to *characterize* the contributions but insufficient to *publish* them. v3 binds the closure work. Run it.
