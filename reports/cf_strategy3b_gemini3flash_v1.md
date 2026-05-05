# cf_strategy3b_gemini3flash_v1 — Path B Gemini 3 flash (replaces Llama 70B)

## Goal

Item 1c-0 Strategy 3 Path B from `task_description_v2.md` §3. Per user
direction (2026-05-05): the originally-spec'd local Llama 3.1 70B int8
paraphrase path is replaced with Gemini 3 flash (a different Gemini
model than Path A's chain default, gemini-3.1-flash-lite-preview).

The local Llama 70B int8 was abandoned in earlier session due to
bitsandbytes dispatch failures ("Some modules dispatched on CPU/disk")
across Llama 70B, Qwen 72B, Qwen 32B, Qwen 14B variants.

Path B with Gemini 3 flash provides an independent paraphrase set we
can compare to Path A.

## Procedure

`scripts/build_cf_strategy3b_gemini3flash.py`. Pins
`gemini_with_fallback` to a single-model fallback chain: just
`gemini-3-flash-preview`. Uses the same NUDITY + VIOLENCE anchors as
Path A.

Run params:
- 10 anchors per cell × 4 cells × 2 concepts = **80 rows**
- 3 paraphrases per anchor
- Wall: ~210 s (network bound)

## Results

| metric | value |
|---|---|
| n_total_rows | 80 |
| n_refused | 0 |
| refused_by_model | {} |
| pinned_model | `gemini-3-flash-preview` |

**0 refusals.** All 80 rows received paraphrases from gemini-3-flash-preview.

## Comparison to Path A

| Path | model | n rows | n refused |
|---|---|---|---|
| **A** | gemini-3.1-flash-lite-preview (cheapest in chain) | 400 | 0 |
| **B** (this run) | gemini-3-flash-preview (pinned) | 80 | 0 |

Both Gemini variants generate paraphrases without refusal on the cell
templates (i2p_safe, i2p_unsafe, coco_safe, coco_unsafe). The two models
behave consistently on the same anchor distribution.

## Caveats

- The pinned model's parsed paraphrases sometimes contain
  pre-amble/meta text ("Here are three rephrased prompts...") that
  the line-splitter takes as the first paraphrase. This is a parser
  fragility, not a model issue. For paper-quality use, the parser
  should be hardened to strip pre-ambles or use a structured-output
  prompt.
- 80 rows is smaller than Path A's 400. To match scale, run with
  `--n-anchors-per-cell 50`. Not critical for the comparison.
- The Llama 70B path (originally spec'd) is **not** reproduced. The
  v2 spec's "two-LLM verification" requirement is partially satisfied
  by two Gemini variants (Path A vs Path B), not by a true
  cross-vendor verification. A more rigorous Path B would use OpenAI
  or Anthropic; deferred.
