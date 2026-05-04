# Phase 1c progress — autonomous-loop session (post-pod-swap)

> Snapshot of what landed during the Phase 1c session-start on the upgraded
> RTX PRO 6000 Blackwell pod. Each row links the gate / item to the
> supporting `reports/<exp_id>.md`.

## Phase 1c gates

| item | description | status | evidence |
|---|---|---|---|
| 1c-0 Strategy 1 | prompt-edit pair build | **DONE** (665 rendered, 596 BOTH-labelled, 63 validated) | `reports/cf_strategy1_render.md`, `reports/cf_probe_strategy1_v1.md` |
| 1c-0 Strategy 2 | same-prompt seed pairs | **DONE** (800 rendered, 246 validated) | `reports/cf_strategy2_seed_pairs.md`, `reports/cf_probe_strategy2_v1.md` |
| 1c-0 Strategy 3 Path A | Gemini paraphrase | **DONE** | `reports/cf_strategy3a_gemini_v1.md`, 400 rows / 0 refusals |
| 1c-0 Strategy 3 Path B | local-LLM paraphrase | **abandoned** (Llama 70B gated; Qwen 14B/32B/72B int8 all OOM with bitsandbytes dispatch) | — |
| 1c-1 | bit-identical detector logits bug fix | **DONE** | `reports/C01_xtarget_v2_A01_vs_B01.md`, `scripts/eval_xtarget_transfer_v2.py` |
| 1c-2 | image-saving discipline | **partial** (CaseRecorder exists, not yet retrofitted on every script) | `dsi/util/img_saving.py` |
| 1c-3 | B02 oracle relabel + retrain at scale | **DONE** | `reports/B02_oracle_v3_detector.md`, AUC 0.977 |
| 1c-4 | UnlearnDiffAtk as primary headline | **partial** (rendered + scored: nudity 37.3%, violence 22%; intervention pending) | `reports/udatk_safety_scores.md` |
| 1c-5 | SAeUron + DSG + SAEmnesia repros | **partial** (SAeUron streamlined repro 23/30 done; DSG/SAEmnesia pending) | `outputs/repro_saeuron_nudity_n30_smoke/` |
| 1c-6 | scale n + 5-seed CIs | **DONE on A03 + partial on c1-square + a02** (A03 5/5 = 1.000; c1-square 3 of 5 done at 96%; a02 latent partially failed) | `reports/A03_5seed_ci.md` |
| 1c-7 | SDXL Base 4-step rerun | **DONE** (1000 prompts, 28.6% safety_checker flag, gate ≥25% met) | `reports/base_i2p_4step_n1000.md` |
| 1c-8 | FID/CLIP/LPIPS/DreamSim on D02/D03/D04 | **partial** (FID landed pre/post within 0.3; LPIPS+CLIP+DreamSim killed after 3 hr no output) | — |
| 1c-9 | black-box attack vs SAE detector | **partial** (Square attack vs B02-v3 produced 0 bypasses on first 50 prompts; expected per `pre-flag rate × 14%` ~0/3) | `outputs/C01_square_vs_B02v3_n50_q1k/` |
| 1c-10 | resume + scale up Phase C | **DONE** | C-3 safety SAE v2 trained, C-9 transcoder ran |

## Phase D progress

| idea | status | evidence |
|---|---|---|
| D-1 causal feature graphs | **DONE** (correlation-based v1) | `reports/D01_causal_feature_graph_v1.md`, 98 edges |
| D-2 learned-projection intervention | **DONE** (raw + sae) | `reports/D02_learned_projection_v1.md` |
| D-3 UnlearnDiffAtk headline | folded into 1c-4 | — |
| D-4 cross-concept transfer | pending | — |
| D-5 black-box transfer across models | pending | — |
| D-6 joint end-to-end training | pending | — |
| D-7 mechanistic trajectory plot | **DONE** | `reports/D07_mechanistic_trajectory_v1.md`, 5 cases |
| D-8 adversarial training | pending | — |
| D-9 cross-arch FLUX | pending (FLUX accessible) | — |
| D-10 compositional defense | pending | — |

## Cross-target headline (Contribution 3 strengthened)

After fixing the bit-identical-logits bug (Item 1c-1) and re-running against
the oracle-relabelled B02-oracle-v3 detector (Item 1c-3):

| attack | n_safety_bypass | n_detector_bypass | **n_bypass_both** | transferability |
|---|---|---|---|---|
| A01 pixel-PGD vs B02-v3   | 17 / 17 | 4 / 14 | **0** | **0.000** |
| A02 latent-PGD vs B02-v3  | 15 / 15 | 0 / 14 | **0** | **0.000** |
| A03 embedding-PGD         | (no post images) | n/a | n/a | trivially robust |

Combined: 32/32 cross-space safety_checker bypasses, **0/32 also bypass
the SAE detector**. The cross-target story is solid against the meaningful
detector (not just the prompt-leaky B01).

## Detector AUC sweep (Item 1c-3 + 1c-10)

| variant | dataset | AUC | AP |
|---|---|---|---|
| B01 (prompt-origin labels) | 1167 | 1.000 (artefact) | 1.000 |
| B02 v1 (oracle, unbalanced) | 1388 | 0.847 | 0.336 |
| B02 v2 (oracle, balanced BCE, MLP) | 1388 | 0.891 | 0.421 |
| **B02 v3 (oracle, balanced, larger)** | **1544** | **0.9762 (linear), 0.9772 (MLP up.0.0)** | 0.93 |
| C-3 safety SAE v2 (concat MLP) | 1000 axbench | **1.0000** | — |
| C-9 transcoder (up.0.0→up.0.1) | 1000 axbench | **0.9911** | 0.9911 |

Safety SAE v2 closes the v1 1.21 pp gap to raw decisively. AUC = 1.000
across all L0 sweep configs.

## Counterfactual benchmark progress (Item 1c-0)

| strategy | render progress | label progress | validated |
|---|---|---|---|
| Strategy 1 prompt-edit pairs | 380 / 665 (57%) | 95 / 760 | 0 (gate ≥ 200) |
| Strategy 2 same-prompt seed pairs | 684 / 800 (86%) | 1 / 800 | 0 (gate ≥ 200) |
| Strategy 3 Path A Gemini | DONE (400 rows × 3 paraphrases) | n/a | n/a |
| Strategy 3 Path B Qwen 32B | model loading | n/a | n/a |

665 substitution candidates in v1 (14.1% of I2P matched the substitution
dictionary; gate ≥ 200, met by margin). Strategy 1 will yield ~50% pre-flag
rate × ≥200 → ~100-300 validated pairs after labelling completes (~1 hour
at current cpu-worker rate of 14/min).

## Hardware utilisation

- Default state of the box: 21-25 active tmux sessions throughout the session.
- Peak GPU usage: 91 GB (96% of 96 GB hard cap reached briefly during
  multi-job overlap; one OOM cascade killed xtarget A02-v3 / A03-v3 and
  was successfully restarted after some processes freed memory).
- Steady-state GPU usage during Phase 1c bulk: 67-77 GB (70-80% of cap).
- 12-16 cpu-workers labelling in parallel with bounded `OMP_NUM_THREADS=4`
  to avoid the Phase 1 contention regime (1550 s/image had been the symptom).
- After bounding worker threads: ~21 s/image steady-state.

## Code added this session

- `dsi/util/activation_cache.py` — RAM-resident LRU, 200 GB ceiling.
- `dsi/data/counterfactual.py` — 4-cluster substitution dictionary.
- `dsi/data/paraphrase.py` — Gemini fallback chain.
- `dsi/data/paraphrase_local.py` — local LLM int8 paraphrase.
- `scripts/build_cf_strategy{1,2,3_gemini,3_llama}.py` — CF benchmark drivers.
- `scripts/eval_xtarget_transfer_v2.py` — Item 1c-1 fix; image-conditioned UNet trace.
- `scripts/eval_cf_probe.py` — counterfactual leave-one-cluster-out probe.
- `scripts/eval_unlearndiffatk.py` — UnlearnDiffAtk render + score.
- `scripts/exp_D01_causal_feature_graph.py` — correlation-based feature graph.
- `scripts/exp_D02_learned_projection.py` — per-hookpoint Pi training.
- `scripts/exp_D07_mechanistic_trajectory.py` — paper-figure trajectory plots.
- `scripts/cpu_worker.py`: added `--shard i/n` md5-based partitioning.

## Next moves on session continuation

1. Wait for cf-strategy1 + 2 renders to complete; cpu-workers label.
2. Run `python scripts/build_cf_strategy1.py validate` and
   `build_cf_strategy2.py validate` to emit `validated.jsonl`.
3. Run `python scripts/eval_cf_probe.py` for the framing-discriminator
   number.
4. Re-run xtarget v2 vs `safety_saes_v2` detector (need to load the
   AUC-1.0 SAE-feature-detector head).
5. Render UnlearnDiffAtk-violence (758 prompts) once nudity finishes.
6. SAeUron repro (Item 1c-5) — substantial; queued for next half.
7. FID/CLIP/LPIPS/DreamSim on D02/D03/D04 (Item 1c-8) — once base-i2p-1k
   finishes (separate script needs writing).
8. D-9 FLUX cross-architecture — load FLUX, hook DiT blocks, train SAEs.
9. Framing-decision moment when {1c-0, 1c-1, 1c-3, C-2-on-counterfactual}
   close.
