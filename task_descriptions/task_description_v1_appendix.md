# task_description_v1_appendix.md — ICLR rigor and Phase C experiment specs

> **Purpose**: tighten v1 to ICLR-submission rigor and pre-spec the highest-EV Phase C experiments so the agent has concrete pass criteria, not just titles.
>
> **Apply on top of** `task_description_v1.md`. The four contributions in §3 are unchanged. This appendix adds: (A) threat-model formalism, (B) theoretical motivation, (C) baselines that v1 missed, (D) a three-tier generalization protocol, (E) statistical significance protocol, (F) compute transparency, (G) ten Phase-C experiments with concrete specs.

---

## A. Threat model — make this explicit in §3 and the paper

A reviewer will reject a paper that doesn't pin this down. State it once, formally, in `paper/main.tex` §2 and reference from every result.

For each defender (the safety checker, the SAE detector, the corrected pipeline), specify:

| Axis | Choices | Default for this work |
|---|---|---|
| **Attacker knowledge of defender** | white-box (gradients, weights), grey-box (architecture only), black-box (input/output queries only) | white-box for §3 Contribution 1; both white-box and grey-box (transfer) for Contribution 3; **black-box added in Phase C-1** |
| **Attacker knowledge of generator** | white-box, black-box | white-box (we own the SDXL pipeline) |
| **Perturbation budget** | ℓ∞, ℓ2, ℓ0 ; ε grid | ℓ∞ ε ∈ {2/255, 4/255, 8/255} for pixel; matched-norm in latent / embedding spaces |
| **Query budget** | unlimited / N queries | unlimited for white-box; capped at 1K and 10K for the black-box variant |
| **Goal** | targeted (specific concept) / untargeted (any unsafe) | both reported |
| **Adversary's compute** | report GPU-hours per attack | required field in every attack `reports/<exp_id>.md` |

**Honest framing**: white-box attacks are an upper bound on attacker capability. Real systems face grey-box / black-box adversaries; reporting both (Phase C-1) keeps the conclusions production-relevant.

---

## B. Theoretical motivation for two-stage + mean patch (§3 Contribution 4)

A short derivation belongs in `paper/main.tex` §4. Sketch:

1. **DSG Theorem 3.1** (Muhamed et al. 2025): for an SAE with small reconstruction error, the expected squared activation $\mathbb{E}[z_f(h)^2]$ is proportional to the Fisher Information of the corresponding decoder weights, and Fisher Information approximates causal influence. This justifies $s_\text{forget}/s_\text{retain}$ as a *causal* (not merely correlational) selection signal. The proof transfers to diffusion essentially unchanged because the SAE is a feed-forward block over UNet activations; the timestep index is a free variable, so the per-timestep variant just averages the bound across $t$.

2. **Mean patching is the minimum-divergence intervention under a Gaussian feature model.** If $z_f \mid \text{benign} \sim \mathcal{N}(\mu_f^{(t)}, \sigma_f^{(t)2})$, then replacing $z_f$ by $\mu_f^{(t)}$ (mean-patch) is the maximum-likelihood projection onto the benign distribution conditional on a single-feature intervention. Zero-patch is the minimum-likelihood projection (zero is far from the benign mean for most non-rare features). Resample-patch with a nearest-benign activation is a sample-based estimator of the same quantity with higher variance. Predictions: mean ≥ resample > zero on FID; mean ≈ zero on ASR (both remove the unsafe activation); resample sometimes > mean on CLIP-score (preserves prompt-conditional structure).

3. **Conditional firing reduces collateral.** If the detector has FPR $\alpha$ on benign prompts, then the unconditional intervention damages benign generations at rate $\geq \alpha \cdot \text{KL}(\mathcal{F}_c \text{-patched} \,\Vert\, \text{unpatched})$, while the conditional version damages at rate $\leq \alpha \cdot \mathbb{E}_\text{benign}[\text{detector}(x) > \tau]$. Conditional firing is strictly better when the detector AUC is non-trivial.

These three observations together are the methodological justification for the proposed Contribution 4 design.

---

## C. Critical baselines v1 missed — add them to §3 Contribution 2 and §3 Contribution 4 evaluation grid

### C.1. AxBench-style direct probing on UNet activations

`AxBench` (Wu et al., 2025) found that simple linear probes on raw transformer activations often match or beat SAE-based steering in LLMs. **The diffusion analogue must be tested** before claiming SAEs are necessary for detection.

Add to Contribution 2 evaluation:
- **Linear probe on raw UNet activations** at the same hookpoints (`down.2.1`, `mid.0`, `up.0.0`, `up.0.1`), bypassing the SAE entirely.
- **Linear probe on raw VAE latents** at each step.
- **Linear probe on text embedding alone** (no UNet, no diffusion — pure prompt-classification).

If any of these match the SAE-activation detector AUC within 2 pp, the SAE story for Contribution 2 collapses; the contribution then narrows to *interpretability* (per-feature attribution) and *Contribution 4 enablement*, which is still defensible but must be reframed honestly.

### C.2. Random-feature ablation control for Contribution 4

For every row in the §3 Contribution 4 grid that uses Stage-1 ∩ Stage-2 selection, also run:
- **Random-feature ablation**: |$\mathcal{F}_c$| randomly chosen features from each block, mean-patched conditionally.

Required signal: random-feature patching either does nothing (selection matters) or hurts FID badly (selection is necessary to avoid collateral). Either outcome confirms the selection method is doing real work; both null = selection is unnecessary, paper has a hole.

### C.3. Static-vs-dynamic gating ablation

DSG's contribution is partly the *dynamic* (input-conditional) gating. Compare:
- **Static-gated** (always intervene with $\mathcal{F}_c$): SAeUron's regime, but with our two-stage selection.
- **Dynamic-gated** (intervene only when detector flags): the proposed regime.

If static dominates dynamic on ASR with comparable FID, the conditional-firing component is unnecessary; the paper's framing must shift.

### C.4. Cross-classifier validation of Stage 2 output score

The Stage 2 output score depends on the choice of independent classifier ($\Pr_\text{clsf}$). Run Stage 2 with each of {NudeNet, Q16, CLIP-zero-shot} and report the Jaccard overlap of the resulting feature sets. Low overlap (< 0.3) = the score is classifier-specific and the paper must caveat. High overlap (> 0.7) = the causal feature subspace is classifier-invariant, which is a stronger result.

---

## D. Three-tier generalization protocol

Every Contribution 4 row must report on three tiers:

1. **In-distribution** (T1): I2P nudity → trained on, evaluated on. Floor.
2. **Cross-category** (T2): I2P violence, hate, harm splits → not trained on; tests whether the SAE / detector / Stage-2 selection generalizes within the I2P umbrella.
3. **Cross-domain** (T3): UnlearnCanvas styles + objects → trained on nothing related; tests whether the methodology generalizes to non-NSFW concepts. UnlearnCanvas accuracy on retain-side objects/styles is the FID-like utility metric for this tier.

Headline claim is allowed only if the proposed method dominates baselines on **all three tiers** simultaneously, or is dominant on T1+T2 with non-regression on T3.

---

## E. Statistical significance protocol

- 5 seeds per cell, every metric, every row of every grid.
- Report mean ± std. For ASR comparisons claimed as "wins", run a paired bootstrap (n=10000) for the difference and report the 95 % CI; the win must have a CI that excludes zero.
- For FID, report the std over 5 seeds; differences within 0.5 FID with overlapping stds are not claimed as wins.
- For multi-row grids, use Bonferroni or Holm correction when claiming any single cell as "significantly best".

---

## F. Compute transparency

ICLR's checklist requires this. Every `reports/<exp_id>.md` records:
- GPU type (RTX Pro 4500), driver version.
- Wall-clock time of the run.
- Peak VRAM.
- Total GPU-hours (sum across this run + concurrent co-scheduled runs sharing the device).

End-of-Phase-1 aggregate: total GPU-hours, total CPU-hours, total wall-clock days. Goes in `paper/main.tex` Appendix.

---

## G. Phase C — ten experiments with concrete specs

These are the highest-EV ideas from CLAUDE.md §9, expanded with pass criteria, VRAM estimates, and co-scheduling partners. Run them in roughly EV-descending order; combine when sensible.

### Phase C-1. Black-box attack against the SAE detector

**Why**: white-box gradient attacks are an upper bound; production adversaries are query-only. ICLR reviewers expect both.

**Method**: implement Square Attack (Andriushchenko et al. 2020) and NES (Ilyas et al. 2018) targeting the SAE detector with a 1K-query and 10K-query budget. Pixel space only (latent/embedding don't have natural query interfaces). Report ASR vs. query budget as a curve.

**Pass criterion**: at 10K queries, black-box ASR ≥ 30 % of white-box ASR (i.e., the SAE detector is meaningfully more robust to black-box than to white-box, but not bulletproof). If black-box ASR ≈ white-box ASR, the detector is fragile and the paper acknowledges. If black-box ASR ≪ white-box ASR (< 10 %), the white-box result is over-stated as a threat.

**VRAM**: 6–8 GB (just SDXL Turbo + safety checker + detector forward pass). **Co-schedule with**: another GPU experiment using ≤ 18 GB.

### Phase C-2. AxBench sanity (the C.1 baseline above)

**Method**: linear probe on raw UNet activations, raw VAE latents, raw text embedding. Same training data as the SAE-activation detector.

**Pass criterion**: the SAE-activation detector beats all three non-SAE probes by ≥ 3 pp AUC on I2P-adversarial. If it does not, the paper reframes the SAE story to interpretability + Contribution 4 enablement only.

**VRAM**: 4 GB. **Co-schedule with**: any Contribution 1 attack.

### Phase C-3. Safety-specialized SAE training

**Why**: Surkov SAEs are general-purpose (LAION-COCO). A SAE trained on a balanced safety distribution may surface concept features more cleanly and improve Stage-2 selection.

**Method**: collect activations from 50K SDXL Turbo generations on a 50/50 mix of I2P + COCO. Train a sparse autoencoder on each of the four hookpoints (`down.2.1`, `mid.0`, `up.0.0`, `up.0.1`) at expansion factors $\{8, 16, 32\}$ and L0 sparsity targets $\{32, 64, 128, 256\}$. Use the EleutherAI / Surkov training recipe.

**Pass criterion**: the safety-trained SAE produces a Stage-2-survivor feature set whose causal output score is ≥ 1.5× the Surkov-out-of-the-box set, *and* the resulting Contribution 4 row has ASR-on-I2P-adversarial ≥ 5 pp lower than the Surkov-SAE row, *and* FID does not regress more than 1.0.

**VRAM**: 16 GB during SAE training (large activation buffers). **Co-schedule with**: CPU FID/CLIP-score evaluation only — this saturates the GPU.

### Phase C-4. Multi-concept simultaneous defense

**Why**: production safety needs to handle nudity + violence + a chosen artist style at once. Tests that the two-stage filter scales without interference.

**Method**: build $\mathcal{F}_{c_1} \cup \mathcal{F}_{c_2} \cup \mathcal{F}_{c_3}$ for $c_i \in$ {nudity, violence, "Van Gogh style"}; train a multi-class detector head; intervene with the union. Eval: per-concept ASR + benign FID.

**Pass criterion**: per-concept ASR is within 5 pp of the single-concept Contribution 4 result (no interference); benign FID degrades by ≤ 1.5 vs. the single-concept regime; cross-concept feature overlap reported (low overlap = clean concept localization, high overlap = polysemy that complicates interpretation).

**VRAM**: 10 GB. **Co-schedule with**: black-box attacks (Phase C-1) or AxBench (Phase C-2).

### Phase C-5. Cross-model transfer

**Why**: do the SAE-feature subspaces identified on SDXL Turbo transfer to SD v3 or FLUX? If yes, the methodology generalizes; if no, the paper's claim narrows.

**Method**: take the Stage-2 surviving feature set $\mathcal{F}_c$ from SDXL Turbo. Train a *new* SAE on SD3 (or FLUX) at structurally analogous hookpoints. Compute feature-to-feature alignment via cosine similarity of decoder columns and via causal-intervention output score on shared concepts (nudity). Report alignment matrix; report whether the Stage-2 feature set in the new model overlaps significantly with the SDXL-Turbo set.

**Pass criterion**: ≥ 40 % of the high-output-score features in SDXL Turbo have a corresponding high-output-score feature in SD3 by alignment metric. Below 40 % = the safety-feature subspace is model-specific and the paper adjusts.

**VRAM**: SD3 inference + SAE training ~ 18 GB. **Co-schedule with**: CPU evaluation only.

### Phase C-6. Hybrid SAE + predicted-noise detector

**Why**: IGD's predicted-noise signal and our SAE-activation signal may be complementary. Concatenating them as inputs to the same head is a cheap potential AUC win and a cleaner ablation than either alone.

**Method**: Detector head takes [SAE activations || predicted noise summary statistics] as input. Train on the same data as Contribution 2.

**Pass criterion**: hybrid AUC > max(SAE-only AUC, predicted-noise-only AUC) by ≥ 1 pp on I2P-adversarial. If not, the two signals are redundant.

**VRAM**: 5 GB. **Co-schedule with**: anything.

### Phase C-7. Adversarial training of the SAE detector

**Why**: Contribution 3 will surface attacks that succeed against the detector. Feeding those bypasses back as training data is the standard adversarial-training recipe; the question is whether it closes the cross-target ASR gap and at what cost.

**Method**: alternate (a) train detector on current data, (b) run Contribution 1 attacks against the new detector, (c) add successful bypasses to training data, (d) repeat. 5 rounds.

**Pass criterion**: at round 5, ASR-against-detector on freshly crafted attacks drops below 50 % of round-1 ASR, and benign AUC does not regress > 1 pp. If benign AUC regresses, log it; the trade-off is a result.

**VRAM**: 10 GB during attack loop. **Co-schedule with**: Phase C-3 SAE training (different hookpoint).

### Phase C-8. LoRA-baked safety

**Why**: tests whether the safety property can be made *parametric* — internalized into the UNet — rather than requiring runtime SAE forward passes and patching. If yes, deployment cost drops.

**Method**: generate (clean prompt, intervened generation) pairs using the proposed Contribution 4 pipeline as a teacher. Train a small LoRA adapter (rank 8 or 16) on the SDXL Turbo UNet to produce intervened-equivalent outputs without runtime SAE patching. Use `audiocraft/musicgen-dreamboothing`-style LoRA recipe (not music; just the LoRA training pattern).

**Pass criterion**: the LoRA-baked SDXL on I2P-naive matches the runtime-patched pipeline within 3 pp ASR and 1.5 FID, *without* any runtime intervention. If yes, this is a strong deployment story.

**VRAM**: LoRA training ≤ 16 GB; inference < 6 GB. **Co-schedule with**: CPU evaluation.

### Phase C-9. Transcoder detector for circuit-level attribution

**Why**: SAEs explain *what* features fire; transcoders (Dunefsky et al., NeurIPS 2024) explain *how features compose through MLP layers*. Substituting transcoder for SAE in the detector enables tracing the circuit from text embedding → composition → bypass.

**Method**: train a transcoder on each of the same four hookpoints. Build the detector on transcoder features. Apply Dunefsky-style attribution patching to identify the feature-circuit (subgraph) responsible for each successful bypass.

**Pass criterion**: identifiable feature circuit (≥ 3 features in a directed dependency) for ≥ 60 % of bypass cases on I2P-adversarial. The circuit visualization is a paper figure even if quantitative gains are modest.

**VRAM**: 12 GB during transcoder training. **Co-schedule with**: CPU FID/CLIP eval only.

### Phase C-10. Generation-quality preservation under intervention

**Why**: FID is reference-based (against COCO statistics) and does not directly measure preservation of *the user's* prompt intent. LPIPS / DreamSim against the un-intervened generation does. ICLR reviewers will ask.

**Method**: for every benign prompt that the detector flags (false positive) and every adversarial prompt that's intervened-on (true positive), compute LPIPS and DreamSim between the intervened image and the un-intervened image. Report mean ± std per row.

**Pass criterion**: on benign false-positive flags, LPIPS < 0.15 and DreamSim < 0.10 (close to original — minor change). On true-positive flags, LPIPS > 0.35 and DreamSim > 0.30 (substantial change — concept actually removed). The gap between these is itself the headline metric; large gap = surgical intervention; small gap = collateral damage proportional to detection.

**VRAM**: 5 GB (LPIPS / DreamSim are small models). **Co-schedule with**: anything.

---

## H. Cross-cutting Phase-C → main results integration

When a Phase-C experiment lands a result that improves an end-to-end metric, integrate into the main pipeline (default config flip, branch merge), then re-run the §3.4 evaluation grid for the affected rows. Update `paper/main.tex` to reflect.

When a Phase-C experiment lands a null or negative result, **document the negative** in `paper/main.tex` Appendix. Negative results are signal — ICLR reviewers value them when they rule out plausible alternatives.

---

## I. The "what does an ICLR reviewer ask, what's our answer?" checklist

Before paper submission, the agent self-audits against these:

| Reviewer question | Answer |
|---|---|
| Why SAEs and not raw activations? | Phase C-2 AxBench result |
| Why this attack space and not another? | §3 Contribution 1 cross-space matrix |
| Is the detector robust to black-box adversaries? | Phase C-1 |
| Does Stage 2 actually do anything beyond Stage 1? | §3 Contribution 4 ablation grid Stage-1-only row |
| Does mean-patching matter, or is zero-patch fine? | §3 Contribution 4 zero-patch row |
| Does it generalize beyond NSFW? | §D three-tier protocol, T3 = UnlearnCanvas |
| Is the conditional gating necessary? | §C.3 static-vs-dynamic ablation |
| Does the result hold across seeds? | §E significance protocol |
| What's the compute cost? | §F transparency |
| Is the framework specific to SDXL Turbo? | Phase C-5 cross-model transfer |
| Does the intervention preserve prompt intent? | Phase C-10 LPIPS/DreamSim |
| Is the safety property deployable without runtime SAE? | Phase C-8 LoRA-baked |

If any answer is "we didn't test", the paper is not ready. If any answer is "tested, negative result", the paper documents it honestly.

---

## End of appendix

This appendix moves the project from "course deliverable with novel composition" to "ICLR-submission-grade research". The agent treats it as a binding extension of v1; every contribution and every Phase-C experiment is held to the rigor specified here.
