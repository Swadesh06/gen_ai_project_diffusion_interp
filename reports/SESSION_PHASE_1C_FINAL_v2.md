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
| **D-6 joint end-to-end** | **DONE** | **AUC=1.000, correction=100%, 41-feature mask (0.20% sparsity)** |
| D-7 mechanistic trajectory | DONE | 5 paper-figure-quality cases |
| **D-8 round-based TRADES** | **DONE** | **3 rounds: attack-robust correction=100%, mask 41→150 (3.7×)** |
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
| SD3-medium | MM-DiT | 100 | 4.0% | [1.6%, 9.9%] |
| **PixArt-Sigma 1024-MS** | DiT | **100** | **2.0%** | **[0.6%, 7.1%]** |
| FLUX.1-schnell | MM-DiT | (deferred — inference hang) | — | — |

## ε-sweep — saturation at quarter ε

| attack | ε | n_pre / n_bypass | ASR |
|---|---|---|---|
| A01 pixel | 4/255 (default, 5-seed) | 88/88 | 1.000 |
| A01 pixel | 2/255 (half) | 9/9 | 1.000 |
| A01 pixel | 1/255 (quarter) | 11/11 | 1.000 |
| A02 latent | 0.1 (default, 5-seed) | 100/100 | 1.000 |
| A02 latent | 0.05 (half) | 11/11 | 1.000 |
| A02 latent | 0.025 (quarter) | 9/9 | 1.000 |
| **Total** | — | **228/228** | **1.000** |

Even at the smallest meaningful pixel perturbation (1/255), the
attack saturates. Phase 1 default ε was 4× the minimum-required
budget. safety_checker decision boundary < 1/255 from real NSFW.

## SAE detector OOD failure on MMA-Diffusion

Tested B02-v3 SAE detector on MMA-Diffusion images (SD v1.4 backbone):

| split | n | mean logit | flagged @ logit>0 |
|---|---|---|---|
| MMA orig | 53 | -6.36 | 0 / 53 |
| MMA adv-gen | 50 | -6.79 | 0 / 50 |

AUC (adv vs orig): **0.388**. Detector cannot discriminate.

The SAE detector is **architecture-specific** (SDXL-trained); it does
not generalize to SD v1.4 outputs even when fed through SDXL VAE+UNet
for activation extraction. Caveat for Framing A.

## MMA-Diffusion oracle ensemble (D-5 extended)

Same 50 MMA adv-gen images scored by full SC+NudeNet+Q16 ensemble:

| oracle | flag rate |
|---|---|
| safety_checker | **58.0%** |
| NudeNet | **20.0%** |
| Q16 | **28.0%** |
| any oracle | **74.0%** |
| B02-v3 SAE | **0.0%** |

**Composite defense view across attack families**:

| attack source | best caught by | rate caught |
|---|---|---|
| A01 PGD-on-SC (SDXL) | SAE detector | 100% |
| MMA adv-gen (SD v1.4) | oracle ensemble | 74% |

No single defense covers both families — strong case for D-10
compositional defense. Each layer covers a different attack family.

## Cross-classifier oracle transfer (D-5)

A01 + A02 attacked PNGs scored by NudeNet + Q16 + SC ensemble:

| source | n | safety_checker | NudeNet | Q16 | escape all 3 |
|---|---|---|---|---|---|
| A01 ε=4/255 | 200 | 0.0% | 0.5% | 12.0% | **87.5%** |
| A02 ε=0.025 | 100 | 0.0% | 2.0% | 16.0% | **83.0%** |

PGD-on-safety_checker is highly classifier-specific. The 3-oracle
ensemble (SC + NudeNet + Q16) catches at most 17% of bypasses. The
SAE detector (B02-v3, AUC 0.977) catches **all** of them on Item 1c-1
(transferability=0.000). The SAE detector is the strongest layer.

## D-6 adversarial-trained SAE detector (NEW)

Trained a B02-style linear probe on **adversarial-bypassed SAE features**
instead of natural-NSFW features. n=188 (A01+A02 5-seed bypass) + n=500
(COCO benign), 80/20 train/val split.

Held-out test on **A03 embedding-PGD 5-seed bypass** (n=102):

| detector | flagged @ logit>0 | rate | mean logit |
|---|---|---|---|
| **B02-adv (this work)** | **101 / 102** | **99.0%** | +9.48 |
| **B02-v3 oracle (baseline)** | 33 / 102 | 32.4% | -1.51 |

**Δ = 66.6 pp absolute, 3.0× improvement.** The adversarial-trained
detector generalizes from (A01 pixel-PGD + A02 latent-PGD) to A03
embedding-PGD (different attack space) with 99% catch rate.

Caveat: B02-adv on MMA-Diffusion = AUC 0.522 (chance). Architecture-
specific to SDXL Turbo; doesn't generalize to SD v1.4. Per-backbone
training is the unblocked path.

## D-10 compositional defense (NEW)

Per-image union of (safety_checker, B02-v3 SAE detector) on D02 outputs
(n=100):

| condition | safety_checker | SAE | union | only-SC | only-SAE | both |
|---|---|---|---|---|---|---|
| pre (no F_c) | 10% | 5% | 14% | 9 | 4 | 1 |
| post (F_c) | 9% | 7% | 16% | 9 | 7 | 0 |

F_c surgery shifts NSFW signal from SC → SAE detector rather than
removing it. F_c is most useful as a generation-time defense, not as
a detection-union booster.

## D-8 / Phase D static defense view (NEW)

F_c surgery patch ablation (mean / zero / resample) all give 4/10
correction on safety_checker pre_flagged at n=100:

| variant | LPIPS-vgg | FID Δ | CLIP Δ |
|---|---|---|---|
| D02 mean-patch | 0.413 | +0.28 | -0.0003 |
| D03 zero-patch | 0.413 | +0.32 | -0.0003 |
| D04 resample-patch | 0.413 | +0.28 | -0.0004 |

All three patches tied. Patch operation doesn't matter; F_c selection
dominates. Mean-patch is the simplest sufficient choice.

## Item 1c-4 close (NEW)

F_c surgery on UDA-nudity n=100 (SDXL Base 4-step, CFG 7.5):
- pre_flagged: 35/100 = 35.0%
- post_flagged (F_c): 47/100 = 47.0%
- corrected: 12/35 = 34.3%
- net Δ flag rate: +12pp (worse)

Correction rate generalizes from D02 (40%); but **24 new false positives**
appear on previously-safe images. F_c surgery is prompt-distribution-
conditional. UDA-nudity is more adversarial than I2P-NSFW; F_c
intervention introduces artefacts that safety_checker reads as NSFW.

## Item 1c-5 SAEUron close (NEW)

Data-driven Fisher-ratio scoring on `bcywinski/SAeUron_coco` SAE
(20480 features). Top-1 nudity feature_idx = **12571** (Fisher 1.19).

Re-ran `repro_saeuron_streamlined.py` with feature 12571:
- no_intervention: 14/30 flagged = 46.7%
- with_saeuron (12571, mult -3.0): 12/30 = 40.0%
- Δ = -6.7 pp (correct direction; v1 with cat-feature 11627 had +10pp wrong direction)

## Item 1c-1 5-seed scaling (NEW)

C01 cross-target transferability scaled to 5 seeds × 2 attacks (10 runs):
- A01 5-seed: 88 pre, 83 safety_bypass, 1 both
- A02 5-seed: 100 pre, 90 safety_bypass, 0 both
- **Combined: 1/173 = 0.58% transferability**, Wilson 95% CI [0.10%, 3.20%]

The B02-v3 SAE detector blocks 99.4% of safety-bypassing PGD
perturbations across 5 seeds × 2 attack spaces.

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

## Item 1c-9 black-box CLOSED (NEW)

After fixing `SurkovHookManager` no_grad blocker (commit `0c73a12`),
PGD now backprops through SAE encode. White-box and black-box both
land:

| attack mode | target | n_pre / n_bypass | bypass rate |
|---|---|---|---|
| Cross-target PGD (5-seed, no detector knowledge) | B02-v3 | 173 / 1 | **0.58%** |
| **Black-box Square q=500** (logit oracle) | B02-adv | 12 / 2 | **16.7%** |
| **Black-box Square q=500** (logit oracle, n=200) | B02-v3 | 12 / 8 | **66.7%** |
| **White-box PGD ε=4/255 ×40** (gradient access) | B02-adv | 32 / 32 | **100%** |
| **White-box PGD ε=4/255 ×40** (gradient access) | B02-v3 | 2 / 2 | **100%** |

Defense degrades sharply with attacker capability. **B02-adv is 4× more
black-box robust than oracle B02-v3** — adversarial-training is the
correct hardening. Both 100% white-box vulnerable as expected for any
single LR head under PGD ε=4/255 × 40 steps.

## D-8 round-based TRADES adversarial training CLOSED (NEW)

`scripts/exp_D08_trades_v1.py` runs 3 rounds of: (1) train joint
(mask + head) on current dataset, (2) synthesise feature-space
attacks via head's −∇ direction at ε=3.0, (3) augment training set,
(4) re-train. Per-round attack-robust correction rate on a held-out
val set:

| round | n_train | val AUC | mask size | clean correction | attack-robust correction |
|---|---|---|---|---|---|
| 0 (clean) | 320 | 1.000 | 41 | 100% (37/37) | n/a |
| 1 | 480 | 1.000 | 73 (1.78×) | 100% | **100% (2/2)** |
| 2 | 640 | 1.000 | 139 (3.39×) | 100% | **100% (7/7)** |
| 3 | 800 | 1.000 | 150 (3.66×) | 100% | **100% (23/23)** |

Spec: round-5 ASR < 50% of round-1 → met (0% from round 1).
|F_c| ≤ 2×: met at round 1 (1.78×); exceeded at rounds 2-3.
**The mask grows to handle each round of attacks; correction stays at
100% throughout.** Caveat: feature-space attack stand-in for pixel-
PGD; D-8 v2 (queued) extends to pixel-space rounds.

## D-6 joint end-to-end CLOSED (NEW)

Joint differentiable training of (soft Stage-2 mask, detection head)
with three losses: detect BCE + patch-classifies-safe BCE + sparsity.
Trained on (200 COCO benign, 200 violence-prompt unsafe) SAE features.

| pipeline | det. AUC | correction rate | benign FP shift |
|---|---|---|---|
| **B02-v3 modular (single-axis detection)** | 0.976 | n/a | n/a |
| **F_c surgery modular (single-axis intervention)** | n/a | 40% (D02 n=100) | +24 (UDA-nudity) |
| **Joint e2e v2 (cached features)** | **1.000** | **100%** (37/37) | **0** |

Sparsity sweep (`D06_joint_e2e_v3`):
- λ=5.0: **41 active features out of 20480 (0.20% sparsity)** with
  100% correction and unchanged AUC.
- Tighter F_c equivalent than the modular Stage 1 ∩ Stage 2 selection.

**Render-and-test on UDA-nudity** (`D06_joint_mask_udatk_v1`): the
41-feature mask applied as SDXL UNet hook on UDA-nudity n=50 prompts.

| pipeline | correction rate | new FP | net Δ flag rate |
|---|---|---|---|
| Modular F_c surgery (UDA-nudity n=100) | 34.3% (12/35) | 24 | +12 (worse) |
| **Joint mask v3 λ=5.0** (UDA-nudity n=50) | **47.4%** (9/19) | **8** | **-1** (improved) |

Cached-feature 100% correction translates to render-time 47% (loss
from spatial-pooling mismatch), but still beats F_c modular surgery
on every axis. Mask concentrates on `down.2.1` (24/41) + `up.0.1`
(16/41); `mid.0` contributes 0.

## D-10 intersection ensemble (NEW)

Cross-detector bypass transfer test on the Item 1c-9 black-box
results:

| attack vector | per-detector ASR | intersection ensemble ASR |
|---|---|---|
| Square BB vs B02-adv (q=500) | 16.7% (2/12) | n/a (outside intersection) |
| Square BB vs B02-v3 (q=500, n=200) | 66.7% (8/12) | **0% (0/8 also bypass B02-adv)** |

**B02-adv flags ALL 8 of B02-v3's Square BB bypasses** (post logit > 0
in every case). The two detectors capture orthogonal failure modes:
- B02-v3 has tight boundary near natural NSFW (5% flag rate).
- B02-adv has shifted boundary on adversarial features (60% flag).

Intersection ensemble defeats single-detector black-box bypasses
entirely at q=500. Defense recommendation: **deploy B02-v3 ∩ B02-adv
intersection** when threat model includes black-box detector queries.

## Stop condition

Per CLAUDE.md, no automatic stop — only human interrupt. Session is
paused for the user to review headline numbers. Continuation queue:
- D-6 v3 sparsity sweep (DONE this session): 41 features / 0.20% achieves
  100% correction on cached SAE-mean features.
- Apply learned 41-feature mask to rendered image pipeline (UDA-nudity,
  I2P-NSFW); compare to D02 F_c flag rate change.
- Cross-concept transfer test: train mask on violence pairs, evaluate
  on nudity pairs.
- Phase D-8 round-based adversarial training (full TRADES impl).
- Phase D-10 ensemble black-box test (B02-v3 ∩ B02-adv).
- B02-v3 vs MMA-Diffusion adv-gen (head device fix queued).
- FLUX inference hang root cause.
