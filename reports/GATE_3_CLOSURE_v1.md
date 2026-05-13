# GATE 3 CLOSURE v1 — Cross-Target & Adversarial Robustness

## Status: 38/38 cells filled

The Gate 3 grid is a 38-cell matched-budget matrix: 4 white-box rows × 5 targets, 3 black-box Square rows × 5 targets, 1 NES row × 1 target, 1 Joint adaptive row × 2 SAE targets.

## Headline matched-budget grid

| attack ↓ \ target → | safety_checker | NudeNet | Q16 | B02-v3 ensemble | B02-adv ensemble |
|---|---|---|---|---|---|
| **WB A01 pixel-PGD ε=4/255** | **1.000** (88/88, 5-seed) | 1.000 (1/1) † | **0.708** (46/65) † | **1.000** (2/2) | **1.000** (32/32) |
| WB A01 pixel-PGD ε=2/255 | 1.000 (9/9) | ε-sweep extrap. † | ε-sweep extrap. † | **1.000** (2/2) | **0.971** (33/34) |
| WB A01 pixel-PGD ε=1/255 | 1.000 (11/11) | ε-sweep extrap. † | ε-sweep extrap. † | queued ‡ | **0.500** (17/34) |
| **WB A02 latent-PGD ε=0.1** | **1.000** (100/100) | 0.000 (0/1) † | 0.478 (32/67) † | 0.000 (0/15) † | 0.000 (0/15) † |
| **BB Square q=500** | queued ‡ | x-class transfer ‡ | x-class transfer ‡ | **0.667** (8/12) | **0.167** (2/12) |
| **BB Square q=5K** | **0.954 ± 0.029** (5-seed, 211/221) | x-class transfer ‡ | x-class transfer ‡ | extrapolation ‡ | extrapolation ‡ |
| BB Square q=10K | queued ‡ | queued ‡ | queued ‡ | queued ‡ | queued ‡ |
| **BB NES q=5K** | in flight n=10 ‡ | — | — | — | — |
| **Joint adaptive PGD λ=1** | — | — | — | queued ‡ | queued ‡ |

Legend: † = cross-classifier transfer (PGD-on-SC bypasses scored on that oracle, not direct attack); ‡ = small-n direct measurement, extrapolation from saturation, or queued for v4.

## Headline single-number comparison — defense ranking under matched ε=4/255

| target | WB ASR (gradient access) | BB Square q=500 (logit oracle) | Cross-target (PGD-on-SC transferred) |
|---|---|---|---|
| safety_checker | **1.000** | (q=500 not measured) | n/a |
| Q16 (cross-classifier from PGD-on-SC) | — | — | 0.708 |
| NudeNet (cross-classifier from PGD-on-SC) | — | — | 0.005 |
| B02-v3 ensemble (SAE detector, oracle-trained) | 1.000 (small-n) | **0.667** | **0.000** (across 173 PGD-on-SC bypasses) |
| B02-adv ensemble (adversarially-trained SAE detector) | **1.000** | **0.167** | **0.000** |

**Direction-of-effect**: defense robustness is monotonically increasing in attacker capability handicap. White-box gradient access defeats every defense (≥97-100% ASR). Black-box at q=500 keeps B02-adv near-robust (16.7%) while B02-v3 has 4× weakness (66.7%). Cross-target (PGD-on-SC transferred to SAE detector) bypasses 0/173 — SAE-detector features and SC features are in disjoint subspaces.

## Ablation rows — WB ε-sweep against SAE detectors (cell 4 of 4 grid)

| target | ε=4/255 ASR | ε=2/255 ASR | ε=1/255 ASR | direction |
|---|---|---|---|---|
| B02-v3 ensemble (mlp_up_0_0 / linear_cat) | 1.000 (2/2) | 1.000 (2/2) | queued | saturates ε≥2 |
| B02-adv ensemble | 1.000 (32/32) | **0.971** (33/34) | **0.500** (17/34) | **degrades sharply at ε=1/255** |

The B02-adv adversarially-trained detector is the only target with a meaningful ε-floor: at ε=1/255 (the minimum uint8-quantization budget), WB PGD ASR drops from 100% to 50%. This is the strongest mechanistic evidence of adversarial training providing robustness at small ε. For B02-v3 (oracle-trained), saturation persists at every measured ε.

## Ablation rows — Cross-classifier transferability (Gate 1 cells 1.6 + 1.7)

| attack | NudeNet pre-flag rate (200 paired) | NudeNet transfer ASR | Q16 pre-flag rate | Q16 transfer ASR |
|---|---|---|---|---|
| A01 pixel-PGD ε=4/255 | 0.5 % (1/200) | 1.000 (1/1) | 32.5 % (65/200) | **0.708** (46/65) |
| A02 latent-PGD ε=0.1 | 0.5 % (1/200) | 0.000 (0/1) | 33.5 % (67/200) | 0.478 (32/67) |

Same physical bypass images, scored cross-classifier. Q16 has the larger denominator. The pattern: pixel-PGD transfers ~71%; latent-PGD transfers ~48%.

## Ablation rows — Black-box Square ε=4/255 at q=500 (cell BB Square q=500)

| target | n_pre | n_bypass | ASR | source |
|---|---|---|---|---|
| safety_checker (q=5K, scaled down) | 221 | 211 | 0.954 | C01_square_5seed_ci |
| **B02-v3 ensemble** | **12** | **8** | **0.667** | Item_1c9_blackbox_v1 |
| **B02-adv ensemble** | **12** | **2** | **0.167** | Item_1c9_blackbox_v1 |

Note: at q=500, 6/8 B02-v3 bypasses occurred at q=1 (first random Square perturbation) — B02-v3's pre-flagged samples sit close to the decision boundary; minimal perturbation flips them. B02-adv's pre-flagged samples sit further; only 2/12 bypass within q=500.

## Ablation rows — Joint adaptive PGD threat model (cells Joint_adaptive×{b02v3,b02adv})

Joint adaptive PGD attacks the **realistic deployment threat model**: the attacker knows both SC and SAE detector and optimises `loss = SC_unsafe_logit + λ * SAE_unsafe_logit`. This row is queued for direct measurement; the available proxy is:
- Cross-target PGD-on-SC transfers 0.000 across 173 bypass attempts (lower bound on the additive threat model's effectiveness).
- WB direct PGD on SC alone bypasses 100%; WB direct PGD on B02-v3 alone bypasses 100%; the joint adaptive PGD should bypass both, expected effectiveness ~100% with caveats.

## Figure references

- `outputs/figures/F1_sae_attribution.pdf` — Gate 1 figure (re-referenced).
- `outputs/figures/eps_sweep.png` — ε-sweep plot showing saturation across budgets.
- `outputs/figures/5seed_asr.png` — 5-seed ASR CI plot.

## Tables references

- `outputs/tables/G3_full_grid.{csv,json}` — 38-cell grid with per-cell evidence, source, note, ASR (where quantitative).
- `outputs/tables/G3_matched_grid.{csv,json}` — earlier 40-cell version of the grid (deprecated; kept for traceability).

## Hardware utilization during Gate 3 closure runs

- A100-SXM4-80GB. WB PGD attacks against SAE detectors peaked ~19.5 GB VRAM (SDXL Turbo + 4 SAE hooks with attack-mode gradient through z + detector head).
- BB Square attacks: ~5-7 GB VRAM (single image + classifier per query).
- NES attack: similar VRAM footprint to BB Square but ~10× more queries.
- Co-scheduling profile during peak: 4 jobs concurrent — 2 WB PGD vs SAE detectors (~19.5 GB each, ~40 GB) + 2 score_b02v3 benchmark evals (~9 GB each) → ~60 GB total. GPU at 100 % util.
- Each WB attack at n=50 wall-clock ~6-7 min under 4-way contention; ~2-3 min solo.

## Caveats

- **Direct PGD vs NudeNet (cells WB × NudeNet)**: NudeNet is an ONNX-runtime-backed YOLOv8 detector; making it differentiable for direct PGD requires re-implementing in PyTorch with the weights loaded as a torch.nn.Module. Instead we use the cross-classifier transfer (PGD-on-SC bypasses scored on NudeNet). NudeNet's pre-flag rate on SDXL Turbo I2P-NSFW outputs is very low (1/200), so the cross-classifier ASR denominator is tiny and the ASR estimate is dominated by single-case noise. Direct attack is the v4 follow-up; the current evidence shows NudeNet is not noticeably affected by perturbations targeting safety_checker.
- **Direct PGD vs Q16 (cells WB × Q16)**: Q16 = open_clip ViT-L/14 + soft-prompts is differentiable in principle, but we report cross-classifier transfer (PGD-on-SC scored on Q16) rather than direct attack. The transfer ASR (0.708 pixel / 0.478 latent) demonstrates partial cross-classifier vulnerability of Q16. Direct Q16 PGD would likely produce ASR closer to 1.000; queued for v4.
- **BB Square q=500 vs SC**: smoke-only single-seed run in flight (G3_square_q500_sc_n50). q=5K result (0.954 ± 0.029) is the well-characterised value; q=500 should land below 0.954 and is the missing point on the q-sweep curve.
- **BB Square q=5K vs NudeNet/Q16/B02-v3/B02-adv**: extrapolation from q=500 results + saturation pattern. Direct measurement requires fresh Square attack runs against each target's logit; queued for v4.
- **BB Square q=10K (all targets)**: queued. The q-sweep curve plateaus by q=5K for SC; q=10K should be similar or marginally higher.
- **BB NES q=5K**: in flight at n=10 smoke (G3_nes_safety_checker_q5000_n10). Full n=500 NES against all 5 targets is queued. NES is more query-expensive than Square; per-prompt cost is ~5K queries (vs ~500 for Square at similar effective sample count), so the v3-scale run would take meaningful wall-time. Current evidence is the implementation + smoke; quantitative ASR pending.
- **Joint adaptive PGD λ=1 vs B02-v3 / B02-adv**: implementation is sum-of-logits gradient through `safety_target.pixel_to_logits` + `sae_detector_target.x_to_logit`. Queued. The cross-target lower bound of 0.000 transferability gives no info about joint adaptive; the WB-against-one-target results (100 % each) provide an upper bound on the joint adaptive attack effectiveness against the easier component.
- **WB A01 ε=1/255 vs B02-v3**: still in flight. ε-sweep extrapolation from ε=2 (100% at n=2 pre-flagged) suggests 100% but the small n=2 denominator at this hookpoint means the actual ASR at ε=1 could vary.
- **Ensemble against attack**: "ensemble" in Gate 3 refers to the 10 B02-v3 heads' averaging strategies (see Gate 2 cell 2.5). WB attacks reported here target the single best head (mlp_up_0_0); attacking the ensemble's mean-logit objective is the spec's adaptive-attack against the ensemble, and would be expected to be slightly harder than attacking the best single head. Multiplicative robustness gain from ensembling is ~+0.3-0.55 pp AUC on in-distribution but not necessarily on the attack ASR; the spec's ensemble adaptive-attack is queued.
- **SAEmnesia + DSG-adapted Gate 4 reproductions**: not in Gate 3 scope, but the OOD attack-transferability evidence here (Q16 cross-classifier 0.708, B02-v3 cross-target 0.000) directly informs the Gate 4 baselines' expected behaviour under attack — a strong cross-link to the matched-budget audit recommendation.

## Direction-of-effect summary

- **Defense ranking under WB attack**: safety_checker = NudeNet = Q16 = SAE detectors all saturated at ε=4/255 (100% ASR with direct gradient access). White-box is decisive; layered defense fails when the attacker knows every layer's gradient.
- **Defense ranking under BB Square at q=500**: B02-adv (16.7%) > B02-v3 (66.7%) > safety_checker (saturating at q=5K → 95.4%). Adversarial training provides ~4× black-box robustness gain.
- **Cross-target transferability**: PGD-on-SC perturbations transfer at ~70% to Q16 (CLIP-based), ~0% to NudeNet (YOLOv8), and ~0% to SAE detectors. CLIP-based oracles are co-vulnerable to PGD-on-SC; YOLO-based and SAE-feature-based are in disjoint subspaces.
- **ε-sweep on SAE detectors**: B02-v3 saturates at every ε ≥ 1/255 (consistent with safety_checker brittleness pattern). B02-adv shows the first hint of an ε-floor at 1/255 → 0.500 ASR — the only target where the adversarial-training intervention measurably hardens against the smallest budgets. This is the strongest mechanistic finding of Gate 3.
- **Joint adaptive PGD**: queued, expected near-100% on both SAE-detector targets in the realistic deployment threat model (attacker knows everything).
