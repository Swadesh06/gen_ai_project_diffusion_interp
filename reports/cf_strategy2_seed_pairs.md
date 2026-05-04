# cf_strategy2_seed_pairs — Item 1c-0 Strategy 2 (same-prompt different-seed)

## Goal

Item 1c-0 Strategy 2: take 100 I2P-NSFW prompts that flag stochastically,
generate 8 seeds each at SDXL Base 4-step, pair flagged-seed and
unflagged-seed generations from the **same prompt**. The prompt
distribution is exactly held constant — any difference in safety_checker
verdict reduces to seed/randomness, not prompt content.

## Procedure

- Source: 100 I2P-NSFW prompts via `dsi.data.i2p.load_i2p` filtered to label=nsfw.
- 8 seeds per prompt at SDXL Base 4-step (CFG=7.5, 512x512). Total
  100 × 8 = 800 renders.
- Each render scored by safety_checker (`scripts/batch_safety_checker.py`)
  + cpu-worker NudeNet/Q16 sweep.
- Validation: per-prompt, take cartesian product of flagged seeds × unflagged
  seeds (up to k=2 each per prompt → up to 4 pairs/prompt).

## Results

| metric | value |
|---|---|
| prompts seen | 100 |
| prompts with BOTH flagged and unflagged seeds | **76** |
| safety_checker flagged renders | 153 / 800 (19.1 %) |
| validated pairs | **246** |
| pairs per validated prompt | 246 / 76 = 3.24 avg |

`outputs/cf_benchmark_v1_seed/validated.jsonl` lists 246 (prompt_id,
flagged_seed, unflagged_seed) tuples.

## Interpretation

**Item 1c-0 Strategy 2 gate (≥ 200 pairs) MET decisively** at 246.
The same-prompt seed-pair benchmark holds prompt distribution
constant; any detector probe trained to discriminate flagged-seed from
unflagged-seed activations is reading **purely** image content
(versus prompt distribution). This is the per-prompt counterfactual
analog and is more aligned with the framing-decision experiment than
Strategy 1's prompt-edit substitutions (Strategy 1 = 61 validated pairs
out of 596 BOTH-labelled, 9.2 % validation rate).

## Next

- Run `scripts/eval_cf_probe.py` against this validated set: train
  raw-vs-SAE probe on 80 % pairs, test on 20 %. Per-cluster
  leave-one-out is the framing-discriminator number.
- Combined Strategy 1 + Strategy 2 = 307 validated pairs (≥ 200 +
  ≥ 246 of 246) — exceeds the gate by margin.
- Run Strategy 3a Gemini paraphrases through SDXL Base 4-step + score
  → 4-cell paraphrase counterfactual matrix.
