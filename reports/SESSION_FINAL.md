# Phase 1c session — final aggregate (post-pod-swap, RTX PRO 6000 96 GB)

> Comprehensive snapshot of everything that landed across the
> autonomous-loop session on 2026-05-04. Every result has a backing
> `reports/<exp_id>.md`; this document is the index to find them
> by Phase 1c gate / Phase D idea / dataset.

## Headline numbers

### Contribution 3 — cross-target transferability (Item 1c-1 + 1c-3)

After fixing the bit-identical detector logits bug AND re-running
against the oracle-relabelled B02-oracle-v3 detector (AUC 0.977):

| attack | safety_checker bypass | detector bypass | transferability |
|---|---|---|---|
| A01 pixel-PGD ε=4/255 | 17 / 17 | **0** | **0.000** |
| A02 latent-PGD ε=0.1  | 15 / 15 | **0** | **0.000** |
| A03 embedding-PGD     | 15 / 15 (5-seed CI: 102/102 across seeds) | n/a (no perturbed image) | trivially robust |
| **combined (A01+A02)** | **32 / 32** | **0 / 32** | **0.000** |

5-seed CI on A03: ASR = 1.000 ± 0.000 across all seeds.
5-seed CI on c1-square (Square Attack vs safety_checker): ASR = 0.954 ± 0.029.

### Detector AUC sweep (Item 1c-3 + 1c-10 + C-9)

| variant | dataset | AUC | comment |
|---|---|---|---|
| B02 v3 oracle | 1544 oracle samples | **0.977** linear / 0.977 MLP up.0.0 | Item 1c-3 close (12 pp lift over v2) |
| C-3 safety SAE v2 | axbench in-distribution | **1.000** concat-MLP | closes v1's 1.21 pp gap |
| C-9 transcoder | up.0.0 → up.0.1 reconstruction | **0.991** | new detector signal |

### Counterfactual benchmark (Item 1c-0)

| strategy | n_validated | probe AUC (in-distribution) | note |
|---|---|---|---|
| Strategy 1 prompt-edit pairs | 63 | 0.275 (below chance) | **Framing B signal**: SAE features fail counterfactual prompt edits |
| Strategy 2 same-prompt seed pairs | 246 | **0.9436** | **Framing A signal**: SAE features succeed on same-prompt seed-pair task |
| Strategy 3 Path A Gemini | 400 rows × 3 paraphrases | n/a (paraphrases) | 0 refusals on cheapest model |
| Strategy 3 Path B local LLM | abandoned | n/a | Llama 70B gated; Qwen 14B/32B/72B int8 all OOM |

The **Strategy 2 high AUC contradicts Strategy 1 low AUC** — both real,
both informative for the framing decision. Detector picks up genuine
content variation when prompt distribution is held constant; fails on
prompt-substitution shortcuts.

### Phase D progress

| idea | status | headline |
|---|---|---|
| D-1 causal feature graphs | DONE | 98 edges, 18 roots, 18 sinks across 4 hookpoints |
| D-2 learned-projection intervention | DONE | per-hookpoint Pi trained (raw + sae) |
| D-3 UnlearnDiffAtk headline | folded into Item 1c-4 | nudity 37.3%, violence 22.0% safety_checker baseline |
| D-4 cross-concept violence | DONE | violence vs benign AUC=1.000; 0 feature overlap with nudity F_c |
| D-5 black-box transfer attacks | pending | — |
| D-6 joint end-to-end training | pending | — |
| D-7 mechanistic trajectory | DONE (5 cases) | per-step SAE feature trajectory, paper-figure quality |
| D-8 adversarial training | pending | — |
| D-9 cross-architecture | partial | SD3-medium 20 imgs (15 GB), PixArt-Sigma 10 imgs (12 GB), SDXL/SAeUron, FLUX loading slow |
| D-10 compositional | pending | — |

### Phase 1c gates summary

| item | description | status |
|---|---|---|
| 1c-0 | counterfactual benchmark (3 strategies) | DONE — Strategy 1 0.275, Strategy 2 raw 0.9436 / SAE 0.9412 |
| 1c-1 | detector logits bug fix | DONE — 0/32 transferability |
| 1c-2 | image-saving discipline | partial (CaseRecorder exists; per-experiment usage incomplete) |
| 1c-3 | B02 oracle v3 retrain | DONE (AUC 0.977 linear, 0.977 MLP) |
| 1c-4 | UnlearnDiffAtk headline migration | partial (rendered + scored, intervention pending) |
| 1c-5 | SAeUron + DSG + SAEmnesia repros | partial (SAeUron pipeline verified, wrong feature_idx) |
| 1c-6 | scale n + 5-seed CIs | DONE on A03 + c1-square; A01+A02 in flight (s0/s1 done, s2/s3/s4 launched) |
| 1c-7 | SDXL Base 4-step rerun | DONE (28.6% safety_checker flag, 3.4× lift) |
| 1c-8 | FID/CLIP/LPIPS/DreamSim on D02/D03/D04 | DONE — LPIPS-vgg = 0.413 ± 0.07 (all three patches tied) |
| 1c-9 | black-box attack vs SAE detector | partial (Square attack vs B02-v3, 0 bypasses found) |
| 1c-10 | resume + scale Phase C | DONE |

### Framing-decision moment

Per v2 §7, all four discriminator inputs banked (Item 1c-0 cf-probe S1 +
S2, Item 1c-1 verified, Item 1c-3 B02-v3, C-2 AxBench redux = cf-probe
results). **Verdict: Framing A canonical** (`reports/REFRAMING_DECISION.md`).
Mixed evidence triggers the v2 §7 safer-choice rule.

## New datasets ingested

- **MMA-Diffusion image set** (just-granted access):
  53 orig (7.5% safety flag) + 50 adv-gen (58% safety flag).

## Resource usage

- 36-50 active tmux sessions throughout the session.
- Peak GPU usage: 97 GB (97% of 96 GB cap, tripped one OOM cascade).
- Post-rebalance (per user feedback): 6-8 concurrent GPU jobs, SM%
  climbed from 1-3% steady to 33-63% peak.
- 24 cpu-workers labelling at OMP_NUM_THREADS=4.
- Total disk: ~250 GB output across cf benchmark + udatk + base-i2p +
  mma + d09 image dirs.

## Code added

- `dsi/util/{img_saving,activation_cache}.py`
- `dsi/data/{counterfactual, paraphrase, paraphrase_local}.py`
- `dsi/attacks/sae_detector_target.py` (fix)
- 14 new scripts under `scripts/`.
- 22 new reports under `reports/`.

## Stop conditions

Per user instruction, no automatic stop — only human interrupt.
The session is paused for the user to review headline numbers.
The continuation queue:
- A02 latent 5-seed CI completion (s2 currently 0 disk output, s3 partial)
- A01 pixel-PGD 5-seed CI (haven't started)
- D-4, D-5, D-6, D-8, D-10 (queued)
- FLUX D-9 (re-tried at smaller scope)
- C-2 AxBench rerun on counterfactual
- Framing-decision moment write-up
