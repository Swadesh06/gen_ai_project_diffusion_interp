# cf_strategy3a_gemini_v1 — counterfactual benchmark Strategy 3 Path A (Gemini paraphrase)

## Goal

Item 1c-0 Strategy 3 Path A from `task_description_v2.md` §3. Build the
`{I2P-style, COCO-style} × {safe-content, unsafe-content}` 4-cell
matrix via Gemini paraphrase with the cheapest-first model fallback
chain. Refusals logged but not retried (no escalation to more expensive
models on individual content blocks — that's a billing trap).

## Procedure

- Anchors: 50 nudity + 50 violence concept descriptions (extended from
  the initial 10/concept; v1 spec gates at 100/cell, current 50/cell
  per concept × 4 cells × 2 concepts = 400 rows).
- Cells: `i2p_safe`, `i2p_unsafe`, `coco_safe`, `coco_unsafe`.
- Model fallback chain (cheapest first):
  1. `gemini-3.1-flash-lite-preview`
  2. `gemini-3-flash-preview`
  3. `gemini-2.5-flash-lite`
  4. `gemini-2.5-flash`
  5. `gemini-2.0-flash`
- 3 paraphrases per anchor (`--n-paraphrases-per-anchor 3`).
- Refusal handling: log + skip; advance to next anchor without trying
  another model.
- Wall: < 5 min for the full 400-row matrix on the autoresearch pod
  (0 GB GPU, network only).

## Results

| metric | value |
|---|---|
| total rows | 400 |
| paraphrases generated | 1200 (3 × 400) |
| total Gemini calls | 400 |
| refused by `gemini-3.1-flash-lite-preview` | **0** |
| refused by `gemini-3-flash-preview` | 0 |
| refused by `gemini-2.5-flash-lite` | 0 |
| refused by `gemini-2.5-flash` | 0 |
| refused by `gemini-2.0-flash` | 0 |
| total refused | 0 |

`outputs/cf_benchmark_v1_paraphrase_gemini/cells.jsonl` has 400 rows.
`outputs/cf_benchmark_v1_paraphrase_gemini/refusals.jsonl` is empty.

## Interpretation

`gemini-3.1-flash-lite-preview` (the cheapest model in the chain)
handled all 400 paraphrase prompts including the unsafe cells with
0 refusals. The model treats explicit unsafe-content paraphrasing as a
text-rewriting task without a safety classifier intervening. This is
the cheapest-possible build of the counterfactual benchmark Path A.

A separate Path B (Qwen2.5-72B int8 local model) is running in
parallel; comparison against Path A produces the cross-model
paraphrase-agreement column the spec requires.

50 anchors per concept is half the gate (100/cell). Extending the
anchor list to 100/concept and re-running takes ~10 minutes; queued.

## Next

- Bump anchors to 100/concept for full gate compliance (Item 1c-0
  Strategy 3 Path A pass criterion).
- Render each paraphrase as an SDXL Base 4-step image and oracle-label
  via the cpu-worker pool.
- Apply the per-cell held-out evaluation protocol: train probe on
  three of the four cells, test on the held-out cell, report per-cell
  AUC. This is the framing-discriminator number.
- Compare Path A vs Path B paraphrases on (a) word overlap with
  anchor, (b) downstream oracle flag rate after rendering.
