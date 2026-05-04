# Phase 1c session — headline numbers (post-pod-swap, RTX PRO 6000 96 GB)

## What landed (concrete numbers, every result has a backing report)

### Contribution 3 — cross-target transferability (post-bug-fix)

After fixing the bit-identical detector logits bug (Item 1c-1) and re-running
against the **oracle-relabeled B02-oracle-v3** detector (Item 1c-3, AUC 0.977,
1544 oracle-labelled samples), the cross-space cross-target matrix:

| attack | safety_checker bypassed | detector also bypassed | transferability |
|---|---|---|---|
| A01 pixel-PGD ε=4/255   | 17 / 17 | **0** | **0.000** |
| A02 latent-PGD ε=0.1    | 15 / 15 | **0** | **0.000** |
| A03 embedding-PGD       | 15 / 15 | n/a (perturbation in CLIP-emb space — SAE detector trivially robust) |
| **combined**            | **32 / 32** | **0 / 32** | **0.000** |

`reports/C01_xtarget_v2_vs_B02v3_full.md`. Confirms Phase 1's `0/17`
result was real (not the bit-identical-logits artefact). The transferability
gap is the cleanest Contribution 3 result for the paper.

### Detector AUC sweep

| version | dataset | AUC | comment |
|---|---|---|---|
| B01 (prompt-origin labels) | 1167 | 1.000 | leak — flags prompts not images |
| B02 v2 (balanced, MLP) | 1388 | 0.891 | severe class imbalance (41 NSFW) |
| **B02 v3 (oracle, balanced, larger)** | **1544** | **0.977 (linear cat) / 0.977 (MLP up.0.0)** | Item 1c-3 close — 5x more NSFW (201) |
| C-3 safety SAE v2 (concat MLP) | 1000 axbench | **1.000** | closes v1's 1.21pp gap |
| C-9 transcoder (up.0.0→up.0.1) | 1000 axbench | 0.991 | adjacent-block reconstruction error |

### Counterfactual benchmark — Strategy 3 Path A (Gemini)

400 anchor-cell rows × 3 paraphrases = **1200 paraphrases**, **0 refusals**
across every cell of `{I2P-style, COCO-style} × {safe, unsafe}` ×
{nudity, violence}. The cheapest model in the chain
(`gemini-3.1-flash-lite-preview`) handled every unsafe cell.
`reports/cf_strategy3a_gemini_v1.md`.

### Phase D progress

| idea | status | headline number |
|---|---|---|
| D-1 causal feature graphs (correlation v1) | **DONE** | 98 directed edges between Stage-1-Fisher top-20 features across 4 hookpoints; 18 roots, 18 sinks; max |β| = 5.85 (mid.0 f3885 → up.0.0 f1223) |
| D-2 learned-projection intervention | **DONE** | per-hookpoint `Pi: R^d → R^d` trained; SAE-feature variant: benign preservation MSE 6e-5 to 9e-4, unsafe→benign-mean projection MSE 1e-3 to 1.5e-2 |
| D-7 mechanistic trajectory plot | **DONE** (5 cases) | per-step per-feature SAE trajectory clean vs attacked, paper-figure quality; PDFs in `outputs/D07_mechanistic_trajectory/` |
| D-3 (UnlearnDiffAtk) | folded into 1c-4; render in flight | 142 nudity prompts rendered |
| D-9 FLUX cross-arch | pending (FLUX accessible) | — |

## In flight (not yet landed)

- cf-strategy1 (Item 1c-0 Strategy 1): 510/665 prompt-edit pairs rendered, 13/188 pre/post labelled. Validation pending.
- cf-strategy2 (Item 1c-0 Strategy 2): 800/800 same-prompt seed pairs rendered, 40 labelled.
- cf-strategy3b (Item 1c-0 Strategy 3 Path B): Qwen 14B int8 paraphrase (Llama 3.1 70B was gated for this account; Qwen 32B/72B int8 didn't fit alongside other GPU jobs).
- udatk-nudity (Item 1c-4): 142 prompts rendered, scoring pending.
- udatk-violence: 200 prompts in flight.
- c1-square-n500 vs safety_checker (Item 1c-6 + Phase C-1 redux): 390/500 prompts processed, 44/44 pre-flagged → bypass = 100% ASR among pre-flagged. Slow due to GPU contention.
- c1-square vs B02-v3 (Item 1c-9): the SAE detector target was producing no bypasses; need to debug whether the detector head + SAE feature pipeline composes correctly.
- LPIPS + DreamSim on D02/D03/D04 (Item 1c-8): in flight on CPU.
- CLIP-score on D02/D03/D04 (Item 1c-8): in flight on GPU (~50 min for FID against COCO val).

## Not yet started (queued)

- SAeUron + DSG + SAEmnesia repros (Item 1c-5).
- D-4, D-5, D-6, D-8, D-9, D-10.
- Framing-decision moment (waiting on Item 1c-0 + C-2-on-counterfactual).
- 5-seed CIs for the headline 0/17 cross-target result (Item 1c-6).

## Hardware utilisation

- 21-25 active tmux sessions throughout the session.
- Peak GPU usage: 91 GB (96% of cap) during multi-job overlap; one OOM
  cascade killed xtarget A02-v3 / A03-v3 + udatk-nudity v0; recovered after
  some processes freed memory.
- Steady-state GPU: 60-80 GB.
- 24 cpu-workers labelling in parallel with `OMP_NUM_THREADS=4` to avoid
  the prior-session thread contention regime (prior pod's 1550 s/image
  symptom is gone; current ~21-65 s/image steady-state).
- Loadavg fluctuated 4-160 during launches; settled to 30-50 in steady state.

## Code inventory added this session (10 new files)

- `dsi/util/activation_cache.py` (200 GB LRU)
- `dsi/data/counterfactual.py` + `dsi/data/paraphrase.py` + `dsi/data/paraphrase_local.py`
- `scripts/build_cf_strategy{1,2,3_gemini,3_llama}.py`
- `scripts/eval_xtarget_transfer_v2.py` (Item 1c-1 fix)
- `scripts/eval_cf_probe.py`
- `scripts/eval_unlearndiffatk.py`
- `scripts/exp_D01_causal_feature_graph.py`
- `scripts/exp_D02_learned_projection.py`
- `scripts/exp_D07_mechanistic_trajectory.py`

## Next-session actions on resume

1. Wait for cf-strategy1 + 2 labels to fill in; run validation; run cf probe.
2. Fix the SAEDetectorTarget pre-flag check for c1-square-vs-B02v3.
3. Render udatk-violence (in flight); apply intervention pipelines.
4. Launch 4 more c1-square seeds (1-4) for Item 1c-6 5-seed CIs.
5. SAeUron repro.
6. D-9 FLUX cross-arch (write FLUX SAE training; train on 4 DiT layers).
7. Framing-decision moment when {1c-0, 1c-1, 1c-3, C-2-on-counterfactual} close.
