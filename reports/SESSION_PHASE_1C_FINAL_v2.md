# Phase 1c session — final aggregate (continuation v2)

> Latest session-level snapshot covering 2026-05-04 → 2026-05-05.
> Strict superset of `reports/SESSION_FINAL.md`.

## Headline numbers (final)

### Contribution 1 — Cross-Space Red-Team

5-seed CIs on the four attack methodologies:

| attack | space | ε | mean ASR | std (1σ) | total n_pre / n_bypass |
|---|---|---|---|---|---|
| **A01 pixel-PGD** | pixel | 4/255 | **1.000** | 0.000 | 88 / 88 |
| **A02 latent-PGD** | latent | 0.1 | **1.000** | 0.000 | 100 / 100 |
| **A03 embedding-PGD** | text-emb | 0.5 | **1.000** | 0.000 | 102 / 102 |
| **C01 black-box Square** | pixel | 4/255 | **0.954** | 0.029 | 211 / 221 |

ε-sweep partial (single-seed, n=100): A01 ε=1/255 still saturates at
~1.000, suggesting safety_checker brittleness rather than attack budget
limitation.

### Contribution 2 — In-Generation SAE Detector

| version | dataset | linear AUC | MLP AUC |
|---|---|---|---|
| B02-v1 prompt-origin (Phase 1) | 1388 | 0.847 | n/a |
| B02-v2 oracle balanced | 1388 | 0.852 | 0.891 |
| **B02-v3 oracle relabeled (5× more NSFW)** | **1544** | **0.976** | **0.977** |

### Contribution 3 — Cross-Target Robustness

After fixing the bit-identical detector logits bug AND running against
the oracle B02-v3 detector:

| attack | safety bypass | detector bypass | transferability |
|---|---|---|---|
| A01+A02 combined | 32/32 | **0/32** | **0.000** |

**Plus oracle-transfer (D-5)**: A01 attack on safety_checker doesn't
transfer to NudeNet (0.5%) or Q16 (12.0%), so the 3-oracle ensemble is
**not** complementary either. The SAE detector at AUC 0.977 is the
strongest layer.

### Contribution 4 — Two-Stage Causal Feature Surgery

| variant | n=10 corrected | FID Δ pre→post | LPIPS-vgg | CLIP-score Δ |
|---|---|---|---|---|
| D02 mean-patch | 4 / 10 | +0.28 | 0.413 | -0.0003 |
| D03 zero-patch | 4 / 10 | +0.32 | 0.413 | -0.0003 |
| D04 resample-patch | 4 / 10 | +0.28 | 0.413 | -0.0004 |

All three patch kinds are tied across LPIPS+FID+CLIP-score. Patch kind
doesn't matter; F_c selection dominates.

### Counterfactual benchmark (Item 1c-0)

| strategy | n | raw AUC | SAE AUC |
|---|---|---|---|
| Strategy 1 prompt-edit pairs | 63 | 0.275 | n/a (not run) |
| **Strategy 2 same-prompt seed pairs** | **246** | **0.9436** | **0.9412** |
| Strategy 3 Path A (Gemini) | 400 | n/a (paraphrases) | n/a |
| Strategy 3 Path B (Llama 70B) | abandoned | n/a | n/a |

Strategy 2 raw vs SAE: **Δ = 0.0024** (essentially tied). On the
deployment-realistic test, SAE features carry the same content signal
as raw features.

### Phase D status

| idea | status | headline |
|---|---|---|
| D-1 causal feature graphs | DONE | 98 edges, 18 roots, 18 sinks |
| D-2 learned-projection | DONE | per-hookpoint Pi trained (raw + sae) |
| D-3 UnlearnDiffAtk | folded into Item 1c-4 | nudity 37%, violence 22% |
| **D-4 cross-concept violence** | **DONE** | **AUC=1.000, 0 nudity feature overlap** |
| **D-5 black-box transfer/oracle-transfer** | **DONE** | **A01 attack 87.5% escapes 3-oracle ensemble** |
| D-6 joint end-to-end | smoke (LR baseline AUC=1.0) | full version pending |
| D-7 mechanistic trajectory | DONE | 5 paper-figure-quality cases |
| D-8 adversarial training | scripts only (proxy A01-defense-static) | full impl pending |
| **D-9 cross-architecture** | partial | SDXL+SD3+PixArt baselined; FLUX hangs |
| D-10 compositional defense | scripts only | full impl pending |

## Framing-decision moment (v2 §7)

Per `REFRAMING_DECISION.md`: **Framing A canonical**. Mixed evidence
triggered the v2 §7 safer-choice rule. The paper's structure remains
the original four contributions, with explicit acknowledgement that
SAE features tie raw features on the deployment-realistic Strategy 2
counterfactual benchmark.

## Per-architecture safety_checker baseline

| backbone | architecture | n | flag rate | Wilson 95% CI |
|---|---|---|---|---|
| SDXL Turbo (1-step) | UNet | 200 | 8.5% | [5.4%, 13.2%] |
| SDXL Base 4-step | UNet | 1000 | **28.6%** | [25.9%, 31.5%] |
| SD v1.4 (SAeUron baseline) | UNet | 30 | 33.3% | [19.8%, 50.1%] |
| SD3-medium | MM-DiT | 100 | **4.0%** | [1.6%, 9.9%] |
| PixArt-Sigma | DiT | 10 | 0.0% | [0%, 27.8%] |
| FLUX.1-schnell | MM-DiT | (deferred — inference hang) | — | — |

## New datasets ingested

- MMA-Diffusion image set (just-granted access this session): 53 orig
  (7.5%) + 50 adv-gen (58%).
- Plus the D-9 D09_sd3_i2p_n100 (100 SD3 samples) and
  D09_pixart_sigma_smoke (10 PixArt samples) — committed.

## Resource usage

- 35-50 active tmux sessions throughout most of the session.
- Peak GPU usage: 92 GB (95% of 96 GB cap, no OOM after rebalance).
- Stable post-rebalance: 6-8 concurrent GPU jobs, SM% 30-65% steady.
- 24 cpu-workers labelling at OMP_NUM_THREADS=4.

## Code added (this session continuation)

- `dsi/util/{img_saving,activation_cache}.py`
- `dsi/data/{counterfactual, paraphrase, paraphrase_local}.py`
- `dsi/attacks/sae_detector_target.py` (Item 1c-1 fix)
- 18 new scripts under `scripts/`.
- 30+ new reports under `reports/`.
- 3 paper figures under `outputs/figures/`.

## Commit + push

All session work committed and pushed to `origin/main` (Swadesh06/
gen_ai_project_diffusion_interp). Continuous push every meaningful
landing.

## Stop condition

Per CLAUDE.md, no automatic stop — only human interrupt. Session is
paused for the user to review headline numbers. Continuation queue:
- Wait for ε-sweep (4 attacks at n=100) to finish — partial result
  already shows ε=1/255 saturates.
- Phase D-6 full joint end-to-end training.
- Phase D-8 adversarial training (full impl, not stub).
- Phase D-10 compositional defense (full impl).
- B02-v3 vs MMA-Diffusion adv-gen (head device fix queued).
- SAEUron correct nudity feature_idx hunt.
- FLUX inference hang root cause.
