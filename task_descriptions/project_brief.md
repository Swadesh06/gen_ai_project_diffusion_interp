# project_brief.md — DiffSafeSAE orientation

> **For agent intake.** This is the one-page summary. Full spec is in `task_descriptions/task_description_v1.md`. Operating manual is in `CLAUDE.md`. Read all three before doing anything.

---

## What this is

A research project on **adversarial robustness and mechanistic safety in text-to-image diffusion**, framed for ICLR submission. The work uses sparse autoencoders (SAEs) on the SDXL UNet as a substrate for both (i) red-teaming post-hoc safety classifiers and (ii) building a more robust replacement.

## Thesis

SAE features learned on diffusion UNets provide a mechanistically interpretable substrate for both attacking and defending post-hoc T2I safety classifiers; adversarial trajectories in pixel, VAE-latent, and CLIP image-embedding spaces activate a small, stable, identifiable set of SAE features; monitoring those features yields a detector complementary to existing baselines; and conditioning a two-stage causally-filtered mean-patch intervention on that detector produces stronger forget-utility trade-offs than current diffusion-native concept erasure.

## The four contributions

1. **Comparative cross-space red-team**: gradient attacks in pixel, VAE-latent, and CLIP image-embedding spaces against `CompVis/stable-diffusion-safety-checker`, with per-feature SAE attribution maps of successful bypasses. **Novel**: no published comparative study across these three spaces against the same post-hoc classifier with feature-level attribution.

2. **In-generation SAE-activation detector**: classifier on SAE activations across the denoising trajectory; two regimes (early-monitor, full-trajectory) with a per-step commit-knee diagnostic. **Novel**: existing in-generation detectors (IGD, FlowGuard, NDM) use raw predicted-noise; SAE features add interpretability and per-feature attribution.

3. **Cross-target robustness**: re-running the Phase-1 attacks against the SAE detector itself; cross-target transferability matrix between safety-checker target and SAE-detector target. **Novel**: zero prior work has measured ASR migration between two T2I detectors with mechanistic explanation of the off-diagonal asymmetry.

4. **Detection-triggered correction via two-stage causal feature selection + mean patching**: Stage 1 = DSG-style Fisher-ratio filter on forget/retain datasets; Stage 2 = Arad-style causal-intervention output score on Stage-1 survivors; intervention = per-feature, per-timestep mean patching from a benign reference; conditional on detector firing. **Novel**: composing DSG (LLM unlearning) with Arad's two-stage filter in a diffusion safety setting, with a benign-mean intervention conditional on an SAE-activation detector — none of these elements has been combined this way before, in any modality.

## What already exists in the literature (so the contribution boundary is honest)

- **SAeUron** (Cywiński & Deja, ICML 2025): activation-contrast SAE feature selection + always-on negative scaling, on SD v1.5. Diffusion-native unlearning SOTA.
- **SAEmnesia** (Cassano et al., 2025): supervised SAEs with one-to-one concept-neuron mapping; +9.22 % on UnlearnCanvas vs. SAeUron.
- **IGD** (Yang et al., 2025), **FlowGuard** (2026), **NDM** (2025): in-generation NSFW detection on predicted noise / linear latent decoding.
- **DSG** (Muhamed et al., COLM 2025): Fisher-ratio SAE feature selection + dynamic classifier-gated clamping, in LLM unlearning. Establishes Theorem 3.1 (squared SAE activation ≈ Fisher Information ≈ causal influence).
- **Arad et al.** (EMNLP 2025): two-stage filter (input-score ∩ output-score) for SAE feature selection in LLM steering; output-score is the causal-intervention component.
- **MMA-Diffusion** (Yang et al., CVPR 2024): pixel-space and text-modality attacks on safety checker. Pixel-space attacks exist; latent and CLIP-embedding attacks against post-hoc safety checkers do not.
- **UnlearnDiffAtk** / **AdvUnlearn** (Zhang et al., ECCV 2024 / NeurIPS 2024): standard adversarial-prompt benchmark for unlearned diffusion models.

## What's novel about this composition

- Cross-space (pixel + latent + embedding) attack comparison on the same target — none exist.
- SAE-activation in-generation detector — not raw noise (IGD), not linear latent decoding (FlowGuard).
- DSG transferred to diffusion + Arad's Stage 2 added — neither is in the diffusion-safety literature.
- Conditional, attribution-aware mean patching from a benign reference — none of SAeUron / SAEmnesia / DSG does this combination.
- Cross-target transferability between safety-checker and SAE detector with mechanistic feature-subspace explanation — not in any prior work.

## Hardware and constraints

- 1× RTX Pro 4500 Blackwell, 32 GB VRAM, always-on, no session caps.
- Target ≤ 85 % VRAM (≈ 27 GB) with 15 % buffer.
- Saturate the box: ≥ 1 GPU job + ≥ 1 CPU job + monitor at all times.
- Public datasets and checkpoints only (I2P, COCO, LAION-COCO, UnlearnCanvas, MMA-Diffusion, UnlearnDiffAtk; SDXL Turbo, SDXL Base, SD v1.5; Surkov SAEs, SAeUron SAEs; CompVis safety checker, NudeNet, Q16, CLIP ViT-L/14).

## What success looks like

- Headline table (`reports/PHASE_1_FINAL.md`) showing the proposed two-stage + mean-patch row dominating SAeUron and DSG-adapted on ≥ 3 of {I2P-naive ASR, I2P-adv ASR, FID, CLIP-score} with statistical significance over 5 seeds.
- Cross-target transferability matrix with ≥ 1 off-diagonal cell ≤ 50 % ASR (i.e., the SAE detector and the safety checker monitor genuinely different feature subspaces).
- Mechanistic plots: per-feature activation deltas on bypass, per-timestep commit-knee, per-feature collateral KL on benign data.
- ICLR-format paper draft (`paper/main.tex`) in submission shape.

## What this is not

- Not a reproduction. SAeUron / SAEmnesia / DSG-adapted are reproduced **only** as baselines in the evaluation grid.
- Not a new SAE architecture. The Surkov / SAeUron checkpoints are used as-is in the primary track (Phase C may train a safety-specialized SAE as an ablation).
- Not a new diffusion model. SDXL Turbo / SD v1.5 are used as-is.
- Not a course-grade deliverable. The course-project version is the floor; ICLR is the ceiling, and the agent is told to optimize for the ceiling.

---

## Read next

`task_descriptions/task_description_v1.md` and `task_descriptions/task_description_v1_appendix.md`.
