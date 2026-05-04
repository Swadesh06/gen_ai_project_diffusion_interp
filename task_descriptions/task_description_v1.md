# DiffSafeSAE v1 — research spec

> **Project codename**: `dsi` (diffusion-safety-interpretability), repo: `gen_ai_project_diffusion_interp/`.
>
> **Hardware**: 1× RTX Pro 4500 Blackwell, 32 GB VRAM, always-on.
>
> **Target**: ICLR-grade quantitative + qualitative results. Optimize aggressively for both. Be ambitious; the hardware is sized for it.
>
> **No time estimates anywhere in this document** — only deliverables, success criteria, and dependencies.
>
> **Read alongside `task_description_v1_appendix.md`** — the appendix is binding. It adds the threat-model formalism, theoretical motivation, critical baselines this v1 missed (AxBench, random-feature control, static-vs-dynamic gating), three-tier generalization, statistical significance protocol, compute transparency, and ten Phase-C experiment specs with concrete pass criteria. Every experiment described below is held to that rigor.

---

## 1. Problem

Text-to-image diffusion pipelines (SDXL, SD v1.5) ship with post-hoc CLIP-based safety classifiers (`CompVis/stable-diffusion-safety-checker`) and external moderation services (NudeNet, Q16). Two gaps motivate this project:

1. **Adversarial robustness of post-hoc safety classifiers in non-text spaces is under-studied.** Existing work on T2I attacks targets the prompt encoder (SneakyPrompt, Ring-A-Bell, UnlearnDiffAtk, MMA-Diffusion text-modality). Image-modality work (MMA-Diffusion image-modality) attacks pixel space only. **Comparative attack studies in pixel, VAE-latent, and CLIP image-embedding space against the same post-hoc classifier do not exist.** Without that, we cannot say where the safety classifier is weakest or whether the same internal features mediate bypasses across spaces.

2. **Existing SAE-based diffusion-safety methods are erasure-shaped, not detection-shaped.** SAeUron (Cywiński & Deja, ICML 2025) and SAEmnesia (Cassano et al., 2025) both ablate SAE features unconditionally. No published method (a) trains an in-generation classifier on SAE activations, (b) uses red-team-derived feature attributions to inform feature selection, or (c) applies a conditional, attribution-aware mean-patching intervention. The closest LLM analogue is DSG (Muhamed et al., COLM 2025) — Fisher-ratio feature selection plus dynamic classifier — which has not been adapted to diffusion.

**Thesis**: SAE features learned on diffusion UNets provide a mechanistically interpretable substrate for both attacking and defending post-hoc T2I safety classifiers. Adversarial trajectories in pixel, VAE-latent, and CLIP image-embedding spaces activate a small, stable, identifiable set of SAE features. Monitoring those features yields a detector complementary to existing baselines, and conditioning a two-stage causally-filtered mean-patch intervention on that detector yields stronger forget-utility trade-offs than current diffusion-native concept erasure.

---

## 2. Related work — what already exists, what's the gap

### 2.1 Attacks on T2I safety

| Reference | Attack space | Target | Note |
|---|---|---|---|
| SneakyPrompt (Yang et al., IEEE S&P 2024) | Text tokens | Prompt filter | RL token perturbation |
| SurrogatePrompt (Ba et al., CCS 2024) | Text | Prompt filter | LLM-guided substitution |
| MMA-Diffusion (Yang et al., CVPR 2024) | Text + pixel | Prompt filter + post-hoc safety checker | Pixel attacks exist, latent does not |
| Ring-A-Bell (Tsai et al., ICLR 2024) | Text embedding | Concept-erased models | Audits filter blind spots |
| UnlearnDiffAtk (Zhang et al., ECCV 2024) | Text | Concept-erased models | Standard benchmark for unlearned-DM robustness |
| AdvUnlearn (Zhang et al., NeurIPS 2024) | Text | Robust concept erasure | Adversarial-training defense |
| SC-Pro (Jan 2025, arXiv:2501.05359) | Latent + embedding | Probing as defense | Closest precedent; defense-side |
| Latent Guard (Liu et al. 2024b) | Text-embedding | Adversarial-prompt detection | Detector side |
| GuardT2I (Yang et al. NeurIPS 2024) | Prompt embedding | Defense | Detector side |

**Gap**: no head-to-head comparison of pixel vs. VAE-latent vs. CLIP-image-embedding gradient attacks against the same post-hoc safety classifier, with the same evaluation set, with feature-level attribution.

### 2.2 SAE interpretability for diffusion

| Reference | Method | Note |
|---|---|---|
| Surkov et al. (NeurIPS 2025, arXiv:2410.22366) | Train SAEs on SDXL Turbo UNet `down.2.1`, `mid.0`, `up.0.0`, `up.0.1` | Found block specialization: `down.2.1` = composition (early), `up.0.1` = local detail / colour |
| SAeUron (Cywiński & Deja, ICML 2025, arXiv:2501.18052) | Activation-contrast feature selection + always-on negative scaling | Diffusion-native unlearning SOTA |
| SAEmnesia (Cassano et al., 2025, arXiv:2509.21379) | Supervised SAE training enforcing one-to-one concept-neuron mapping | +9.22% on UnlearnCanvas vs. SAeUron |

### 2.3 In-generation safety detection

| Reference | Signal | Note |
|---|---|---|
| IGD (Yang et al., 2025, arXiv:2508.03006) | Predicted noise ε_t | 91.3% over 7 NSFW categories incl. adversarial |
| FlowGuard (2026, arXiv:2604.07879) | Linear latent decoding | Lightweight in-generation |
| NDM (Huang et al., 2025, arXiv:2510.15752) | Early-timestep predicted noise | Detection + adaptive negative guidance |

### 2.4 SAE-based unlearning in LLMs (transferable methodology)

| Reference | Method | Note |
|---|---|---|
| DSG (Muhamed et al., COLM 2025, arXiv:2504.08192) | Fisher-ratio SAE feature selection + dynamic classifier-gated clamping | Establishes Theorem 3.1: squared SAE activations ≈ Fisher Information ≈ causal influence |
| Arad et al. (EMNLP 2025, arXiv:2505.20063) | Two-stage filter: input-score (correlational) ∩ output-score (causal intervention) | "SAEs Are Good for Steering — If You Select the Right Features" |
| Farrell et al. (2024, arXiv:2410.19278) | Static SAE feature ablation for LLM unlearning | DSG's predecessor |
| CAST (Lee et al. 2024, arXiv:2409.05907) | Conditional activation steering, classifier-gated | LLM analogue of conditional intervention |
| O'Brien et al. (2024) | Conditional refusal steering with SAEs | LLM analogue |

**Gap**: DSG's two-stage / classifier-gated framework has not been adapted to diffusion safety. Arad et al.'s Stage 2 (causal-intervention output score) has never been applied alongside Stage 1 (Fisher ratio) in any setting, including LLMs. Both filters in series, with mean-patching from a benign reference, with conditional firing keyed off an SAE-activation in-generation detector, in a diffusion model — this composition is unpublished.

---

## 3. Contributions

### Contribution 1 — comparative cross-space adversarial attack study with SAE attribution

Gradient-based attacks in three spaces against `CompVis/stable-diffusion-safety-checker`:

- **Pixel**: PGD on 3×512×512 image input to the safety checker's CLIP image encoder. Constraints: ℓ∞ ε ∈ {2/255, 4/255, 8/255}. Method: standard PGD (Madry et al.).
- **VAE-latent**: PGD on 4×64×64 SDXL VAE latent; decode with VAE; pass to safety checker. The decode is differentiable. ℓ∞ on the latent.
- **CLIP image-embedding**: PGD directly on the 768-dim CLIP image embedding the safety checker classifies on. Most permissive attack space (no decoding) — establishes an attack ceiling.

Each successful bypass is paired with the SAE activation trace (per-block, per-feature, per-timestep) recorded during the SDXL generation that produced the seed image. Attribution: detector-weight × activation for a linear probe; input-gradient × activation for a non-linear head.

**Pass criteria**:
- ASR (attack success rate, fraction of attempts where safety checker flips to "safe") reported for each attack space at each ε level on the I2P-NSFW split (≥500 prompts) and on a clean COCO-derived control (no I2P; verifies attacks are producing real bypasses, not random).
- Cross-space transferability matrix (3×3): does a perturbation that bypasses in pixel-space also bypass when the same image is reprojected into VAE-latent and re-attacked in that space?
- Per-block, per-feature SAE activation delta on bypass vs. baseline. Top-50 features ranked. **Required**: at least 60% feature-set overlap between pixel and CLIP-embedding bypasses on the same prompt (sanity that SAEs identify a stable "unsafe-feature subspace" rather than space-specific noise).

### Contribution 2 — in-generation SAE-activation detector

Two detector regimes, both trained on the SAE activations collected from generations on a labelled mix of I2P + COCO captions:

- **Early-monitor (EM)**: classifier consumes activations from the first k steps only (k ∈ {1, 2, 3} for SDXL Turbo). Drives single-pass early-trigger correction (Contribution 4).
- **Full-trajectory (FT)**: classifier consumes pooled (mean / max / attention-pooled) activations across all denoising steps. Drives two-pass regenerate correction (Contribution 4).

Architectures to compare: linear probe; 2-layer MLP head; per-block linear probes ensembled. Feature input options: raw SAE activations (sparse); top-k by importance score (compressed); per-block CLS-pool.

**Baselines** (every result reported against all of these):
- `CompVis/stable-diffusion-safety-checker` (production baseline)
- NudeNet (independent ground-truth detector)
- Q16 (Schramowski et al., independent)
- IGD (Yang et al. 2025) — predicted-noise classifier
- FlowGuard (linear latent decoding)
- SAeUron's blocking (re-purposed as a flag: any SAE feature in the unlearned set above a threshold = unsafe)
- SAEmnesia's classification head (re-purposed similarly)

**Pass criteria**:
- AUC on I2P-naive split ≥ 0.95 (production baseline ≈ 0.85, IGD reports 0.91).
- AUC on I2P-adversarial split (Ring-A-Bell + UnlearnDiffAtk-crafted) ≥ 0.85.
- AUC on MMA-Diffusion text-modality adversarial set ≥ 0.80.
- **Per-timestep commit-knee diagnostic**: plot detector AUC as a function of how many denoising steps are observed (1, 2, ..., T). Identify the step at which AUC saturates. This is itself a publishable diagnostic — Surkov et al.'s finding that `down.2.1` does composition early predicts an early knee.
- Compare: does the EM regime (1-3 steps only) match FT performance? If yes, single-pass early-trigger correction is viable; if no, two-pass regenerate is necessary.

### Contribution 3 — cross-target robustness study

Re-run all three attacks from Contribution 1, but now targeted at **the SAE detector** instead of the safety checker:

- Optimize the perturbation to drive the SAE detector's logit toward "safe" while keeping the underlying image NSFW (verified by NudeNet + Q16, used as ground-truth oracles excluded from the attack loop).
- Report ASR-on-detector for the three attack spaces.
- **Cross-target transferability matrix** (2×2): perturbations crafted against safety-checker → tested on SAE detector, and vice versa. Quantifies whether the SAE detector is exploiting features the safety checker misses (off-diagonal low) or merely redundant (off-diagonal high).
- Mechanistic explanation: for the off-diagonal cells, identify which SAE features are activated under safety-checker-targeted attacks but not flagged by the SAE detector (and vice versa). This is the core diagnostic of the project.

**Pass criteria**:
- Cross-target ASR matrix reported with confidence intervals over ≥ 5 seeds.
- At least one off-diagonal cell with ASR ≤ 50% (i.e., at least one direction where attacks do not transfer well — evidence the two detectors monitor genuinely different feature subspaces).

### Contribution 4 — detection-triggered correction via two-stage causal feature selection

When the Contribution 2 detector flags a prompt, patch the activations of SAE features pre-identified as causally responsible for the unsafe concept *c*. Intervention: replace each feature's activation with its per-timestep mean computed over a benign reference (5,000 COCO captions, fixed seeds).

**Two-stage filter** (adapted from Arad et al. 2025 + DSG / Muhamed et al. 2025):

- *Stage 1 — Fisher ratio* (DSG-style; cheap):

$$
s_{\text{forget}}(f) = \mathbb{E}_{x \sim \mathcal{D}_c,\, t}[z_f(x,t)^2], \quad
s_{\text{retain}}(f) = \mathbb{E}_{x \sim \mathcal{D}_{\text{benign}},\, t}[z_f(x,t)^2]
$$

Keep features with $s_{\text{forget}}(f)/s_{\text{retain}}(f) > \tau_{\text{ratio}}$ (default: 95th percentile on retain).

- *Stage 2 — causal-intervention output score* (Arad-style; expensive, run only on Stage-1 survivors):

$$
S_{\text{out}}(f, c) = \Pr_\text{clsf}\!\left[c \mid \text{gen}(\mathbf{p}_\text{neutral}; z_f \leftarrow +\lambda)\right] - \Pr_\text{clsf}\!\left[c \mid \text{gen}(\mathbf{p}_\text{neutral})\right]
$$

where $\Pr_\text{clsf}$ is an independent classifier (NudeNet for nudity; UnlearnCanvas ViT-Large for objects/styles), $\mathbf{p}_\text{neutral}$ is a neutral prompt set, and $\lambda$ is a high clamp value. Keep features with $S_{\text{out}}(f,c) > \tau_{\text{out}}$.

- **Final feature set** for concept *c*: $\mathcal{F}_c = \{f : \text{Stage 1} \land \text{Stage 2}\}$.

**Concept dataset $\mathcal{D}_c$** sources:
- Nudity → I2P nudity-category prompts.
- Object/style → UnlearnCanvas concept splits (60 styles × 20 objects).
- Adversarial-augmented → successful bypass prompts from Contribution 1 (red-team-derived).

**Intervention regimes** (matched to detector regime):
- EM detector → single-pass early-trigger: monitor SAE activations every step; on detector firing at step *t\**, patch $\mathcal{F}_c$ from *t\**+1 → T.
- FT detector → two-pass regenerate: full generation, classify; if flagged, regenerate from *t* = 0 with $\mathcal{F}_c$ patched throughout.

**Evaluation grid** (each variant answers a specific scientific question):

| Method | Selection | Intervention | Gating | Question |
|---|---|---|---|---|
| No defense | — | — | — | floor |
| SD Safety Checker | — | output filter | always | production baseline |
| NudeNet / Q16 | — | output filter | always | independent baseline |
| SAeUron (reproduced) | $S_\text{in}$ only | negative scaling | always | diffusion-native SOTA |
| SAEmnesia (reproduced) | one-to-one concept-neuron | clamp | always | diffusion-native SOTA, supervised |
| DSG-adapted | Stage 1 only | clamp to −*c* | dynamic classifier | LLM SOTA transferred to diffusion |
| Attribution-only | detector attribution | benign mean | on detection | does per-prompt attribution alone work? |
| Stage-1 + mean patch | Stage 1 only | benign mean | on detection | does Stage 2 add value over DSG? |
| **Two-stage + mean patch (proposed)** | Stage 1 ∩ Stage 2 | benign mean | on detection | **primary** |
| Two-stage ∩ attribution | (Stage 1 ∩ Stage 2) ∩ detector attribution | benign mean | on detection | does per-prompt adaptation of a causal pre-filter improve specificity? |
| Two-stage ∪ attribution | (Stage 1 ∩ Stage 2) ∪ detector attribution | benign mean | on detection | does red-team signal add recall? |
| Zero-patch on two-stage | Stage 1 ∩ Stage 2 | zero | on detection | does mean vs. zero matter once selection is causal? |
| Resample-patch on two-stage | Stage 1 ∩ Stage 2 | nearest-benign activation | on detection | in-distribution vs. dataset-mean patching |

**Metrics (every row, every concept)**:
- ASR on I2P-naive (NudeNet + Q16 ground truth, ensemble label).
- ASR on I2P-adversarial.
- ASR on MMA-Diffusion adversarial benchmark (text-modality, image-modality).
- ASR on UnlearnDiffAtk crafted prompts.
- FID on COCO-clean (5K).
- CLIP-score on COCO-clean.
- UnlearnCanvas accuracy retain-side (object/style generalization).
- Latency overhead per generation (ms).
- Per-feature collateral impact: KL-divergence between with-patch and without-patch SAE activation distributions on $\mathcal{D}_\text{benign}$.

**Pass criteria**:
- The proposed two-stage + mean-patch row dominates SAeUron and DSG-adapted on at least three of {I2P-naive ASR, I2P-adversarial ASR, FID, CLIP-score} simultaneously, with statistical significance over 5 seeds.
- The Stage-1-only ablation does not match the two-stage variant on at least one ASR metric — confirming Stage 2 contributes.
- Zero-patch underperforms mean-patch on FID — confirming benign-mean is an in-distribution choice, not equivalent to zeroing.

---

## 4. Models and datasets — all public, verified accessible

### 4.1 Models

| Model | Hosting | Gated | Use |
|---|---|---|---|
| **SDXL Turbo** | `stabilityai/sdxl-turbo` (HF) | No | primary diffusion backbone for SAE alignment |
| **SDXL Base 1.0** | `stabilityai/stable-diffusion-xl-base-1.0` (HF) | No | multi-step generation for cross-step generalization tests |
| **SD v1.5** | `runwayml/stable-diffusion-v1-5` (HF) | No | for SD Safety Checker alignment (the checker was trained for v1.x) |
| **Surkov SAEs** | `surkovv/sdxl-unbox` (GitHub); demo HF Space `surokpro2/Unboxing_SDXL_with_SAEs` | No | pretrained SAEs for `down.2.1`, `mid.0`, `up.0.0`, `up.0.1` |
| **SAeUron** | `cywinski/SAeUron` (GitHub); `bcywinski/SAeUron` (HF) | No | pretrained SAEs trained on SD v1.5; primary baseline |
| **SD Safety Checker** | `CompVis/stable-diffusion-safety-checker` (HF) | No | red-team target |
| **NudeNet** | `notAI-tech/NudeNet` (GitHub, PyPI `nudenet`) | No | ground-truth detector |
| **Q16** | Schramowski et al. checkpoint | No | ground-truth detector |
| **CLIP ViT-L/14** | `openai/clip-vit-large-patch14` (HF) | No | embedding space for image-embedding attacks |

### 4.2 Datasets

| Dataset | Hosting | Gated | Use |
|---|---|---|---|
| **I2P** | `AIML-TUDA/i2p` (HF) | No | 4,703 prompts; primary attack/eval set |
| **I2P-adversarial-split** | `AIML-TUDA/i2p-adversarial-split` (HF) | No | harder adversarial NSFW split |
| **COCO 2017** | direct download (118K train, 5K val) | No | benign reference; FID + CLIP-score eval |
| **LAION-COCO subset** | HF / direct | No | SAE activation collection (subset of ~50K prompts) |
| **UnlearnCanvas** | `OPTML-Group/UnlearnCanvas` (GitHub + HF) | No | 60 styles × 20 objects = 24K images; concept generalization eval |
| **MMA-Diffusion adv set** | `cure-lab/MMA-Diffusion` (GitHub); HF `YijunYang280/MMA_Diffusion_adv_images_benchmark` (request approval) | gated (image set; text set open) | adversarial benchmark |
| **UnlearnDiffAtk benchmark** | `OPTML-Group/Diffusion-MU-Attack` (GitHub); HF | No | adversarial prompt evaluation |
| **Ring-A-Bell prompts** | from paper repo | No | embedding-space attack prompts |

The MMA-Diffusion image set requires an access request; if approval is delayed, evaluate against the text-modality set first and treat the image set as a follow-up evaluation.

### 4.3 VRAM budget on the Blackwell

| Component | Peak VRAM (fp16) | Concurrent? |
|---|---|---|
| SDXL Turbo + Surkov SAEs (4 blocks, frozen) | ~6 GB | yes |
| SDXL Base + SAeUron SAEs (frozen) | ~10 GB | yes |
| Pixel-space PGD on full pipeline | ~14 GB | yes |
| VAE-latent PGD with grad-checkpointing | ~12 GB | yes |
| CLIP-embedding PGD (encoder + head only) | ~3 GB | yes |
| MLP detector training | ~4 GB | yes |
| NudeNet + Q16 + SD Safety Checker eval | ~3 GB total | yes |
| **Headroom** | ~14 GB at 85% cap | for parallelism |

The 32 GB budget at the 85% cap (≈27 GB) easily fits two simultaneous experiments from the table above. **Treat the GPU as a resource to be saturated — single-job execution wastes the box.**

---

## 5. Initial work items (the agent's checklist before iterating)

These are the concrete first-pass deliverables. After these land and the four-contribution evaluation grid is populated, the agent enters Phase C (own ideas, see §8 of `CLAUDE.md`).

### Item 1 — environment, data, baselines stand up

1.1. Conda env `dsi` with torch 2.4+, diffusers, transformers, peft, sae_lens, einops, captum, accelerate, wandb, mir_eval (no, not relevant here — this is the diffusion project), nudenet, clean-fid, etc. Pack-and-unpack workflow per `/workspace/conda_setup.md` (same as music project's pod).

1.2. Download all models and datasets in §4.1 / §4.2 to a persistent cache. Verify each loads.

1.3. Repro of the `surokpro2/Unboxing_SDXL_with_SAEs` HF Spaces demo, locally, on one prompt. Confirm all four Surkov SAEs hook correctly into SDXL Turbo's UNet.

1.4. Repro of the SAeUron sampling pipeline on UnlearnCanvas: `accelerate launch scripts/sample_unlearning_distr.py` per their README. Confirm we can erase a style and verify with their classifier. This gives us a working SAeUron baseline to compare against.

1.5. Generate clean baseline images (1,000 from COCO, 1,000 from I2P) with SDXL Turbo at fp16; cache to disk. Run NudeNet + Q16 + SD Safety Checker over all 2,000; confirm NSFW prevalence on I2P side ≥ 70%, on COCO side ≤ 5%. Validates the eval pipeline before any attack.

**Pass**: clean-image NSFW prevalence rates above + working SAeUron erasure repro.

### Item 2 — Contribution 1 (red-team)

2.1. Implement pixel-space PGD attack in `dsi/attacks/pixel.py`. Test on 50 I2P-NSFW prompts at ε=4/255. Report ASR.

2.2. Implement VAE-latent PGD attack in `dsi/attacks/latent.py` with gradient checkpointing through the VAE decoder. Test on the same 50 prompts. Report ASR.

2.3. Implement CLIP-embedding PGD attack in `dsi/attacks/embedding.py`. Test on the same 50 prompts. Report ASR.

2.4. Scale all three to ≥ 500 prompts at three ε levels. Build the cross-space transferability matrix.

2.5. Hook SAE collection during attack runs. Build the SAE activation delta map (per-block, per-feature) for successful bypasses. Persist to disk as a structured dataset for Contributions 2 and 4 to consume.

**Pass**:
- ASR ≥ 0.7 on at least one of {pixel, latent, embedding} at ε=8/255 on I2P-NSFW.
- Cross-space attack overlap ≥ 60% at the SAE-feature level (verifies feature-level structure exists).

### Item 3 — Contribution 2 (detector)

3.1. Build the labelled training set: 5,000 SDXL Turbo generations split 50/50 NSFW / benign, with SAE activations cached per step per block.

3.2. Train both detector regimes (EM, FT). Architectures: linear probe, 2-layer MLP, per-block ensemble.

3.3. Evaluate against all baselines from §3.2. Build the per-timestep commit-knee plot.

3.4. Train on adversarial-augmented data (mix in successful bypasses from Item 2). Re-evaluate; this is the version used in Contributions 3 and 4.

**Pass**: AUC ≥ 0.95 on I2P-naive; AUC ≥ 0.85 on I2P-adversarial; commit-knee diagnostic plot produced; both regimes (EM, FT) trained and reported.

### Item 4 — Contribution 3 (cross-target robustness)

4.1. Re-run all three Phase 1 attacks against the SAE detector. Report ASR-on-detector at the same ε levels.

4.2. Build the 2×2 cross-target transferability matrix.

4.3. Mechanistic analysis: identify the SAE features that mediate the off-diagonal asymmetry.

**Pass**: full 2×2 matrix with 5-seed CIs; at least one off-diagonal cell ≤ 50% ASR; mechanistic plot identifying the divergent feature subspaces.

### Item 5 — Contribution 4 (detection-triggered correction)

5.1. Implement Stage 1 (Fisher ratio) feature selection. Verify on the I2P nudity split → matches DSG's reported behavior on a comparable LLM benchmark in the limit.

5.2. Implement Stage 2 (causal-intervention output score). Run on Stage-1 survivors at λ ∈ {100, 250, 500} per Arad et al.

5.3. Implement mean / zero / resample patching. Compute per-feature, per-timestep benign means over 5K COCO captions.

5.4. Hook the patch into both detector regimes (EM single-pass, FT two-pass).

5.5. Run the full evaluation grid (table in §3.4). Five seeds per cell.

**Pass**:
- Two-stage + mean-patch row dominates SAeUron and DSG-adapted on ≥ 3 of {I2P-naive ASR ↓, I2P-adv ASR ↓, FID ↓, CLIP-score ↑}.
- Stage-1-only and zero-patch ablations lose vs. two-stage on ≥ 1 ASR metric each.

### Item 6 — final reporting

6.1. Re-run all gates with final defaults; produce the headline table.

6.2. Generate qualitative figures: example bypass images (red-team output), per-feature activation maps, before/after intervention images.

6.3. Write a short paper draft (~ICLR-format LaTeX; the agent maintains it under `paper/main.tex`); include all results, ablations, mechanistic plots.

6.4. README, license matrix, citation list.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Attacking SD v1.5 safety checker while monitoring SDXL SAEs | Run two parallel tracks — (a) SDXL pipeline + Surkov SAEs vs. an SDXL-ported safety checker (community impl exists); (b) SD v1.5 + SAeUron SAEs vs. the original v1.5 safety checker. Report both. |
| MMA-Diffusion image-set access delayed | Use text-modality set + UnlearnDiffAtk-crafted prompts as primary adversarial benchmark; image set is additive. |
| Stage-2 causal-intervention scoring is expensive (per-feature gen with classifier) | Run only on Stage-1 survivors (≤ 5% of features). Batch generations. Gradient checkpointing where applicable. |
| Detector overfits to bypass-derived features (shortcut-learning) | Ablation: train detector on benign + plain-NSFW only, no adversarial; evaluate on adversarial. Report degradation. |
| SAE features at SDXL multi-step diverge from SDXL Turbo single-step | Surkov et al. show generalization. Verify empirically on 100 prompts; report cross-step feature stability. |
| Mean-patching drives activations off-manifold | Resample-patch ablation in the grid checks this directly. If resample > mean on FID, switch primary to resample. |
| FID-on-COCO is noisy at small N | Use 5K COCO; 5 seeds per generation; report mean ± std. CLIP-score is lower-variance and reported alongside. |
| Reproducing SAeUron / SAEmnesia exactly takes time | Use their released checkpoints + sampling scripts unmodified; treat reproduction as a baseline lock-in step before any new method runs. |
| SD Safety Checker is hard-coded to 17 concepts (limited attack surface) | Frame as a feature: this makes "bypass" a clean binary. For object/style generalization, use UnlearnCanvas. |
| Adversarial robustness claims need significance | 5 seeds per cell; report mean ± std for every ASR / FID; bootstrap confidence intervals. |

---

## 7. Dependency graph

```
Item 1 (env + baselines) ──┬──> Item 2 (red-team) ──┬──> Item 3 (detector) ──┬──> Item 5 (correction)
                            │                        │                          │
                            └────────────────────────┴────> Item 4 (cross-target) ──> Item 6 (paper)
```

Items 2, 3, 4 are largely parallelizable on a 32 GB card once Item 1 is done. Item 5 depends on outputs from 2 and 3.

---

## 8. End of v1

This is the first description of the project. The agent's job is to:
1. Get all four contributions to passing on the gates above.
2. Then iterate further (Phase C, see `CLAUDE.md`) — additional ablations, more advanced architectures, more datasets, deeper mechanistic analysis, etc., with the explicit goal of producing ICLR-grade results.

The story to be told in the final paper, if everything lands: *We red-team post-hoc T2I safety classifiers across three representation spaces and identify a stable SAE-feature subspace that mediates bypasses; we train an in-generation SAE-activation detector that complements existing safety baselines; we adapt DSG's two-stage causal feature selection from LLM unlearning to diffusion safety, with mean-patching from a benign reference, conditional on the detector firing; the resulting pipeline improves the forget-utility trade-off over SAeUron, SAEmnesia, and DSG-direct simultaneously, with mechanistic interpretability throughout.*
