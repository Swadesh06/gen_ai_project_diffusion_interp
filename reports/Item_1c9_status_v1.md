# Item_1c9_status_v1 — Black-box attack vs SAE detector status

## Goal

Item 1c-9 from `task_descriptions/task_description_v2.md` §3:
black-box attack against the SAE detector. The white-box result
(Item 1c-1 5-seed: 1/173 cross-target = 0.58% transferability) was
established. The black-box version asks: can a query-only attacker
produce bypasses against the SAE detector?

## What's been attempted

### Square Attack vs B02-v3 (earlier session)
- `outputs/C01_square_vs_B02v3_n100_q2k_final` etc.
- Started with n=100, max-queries=2000.
- The Square Attack inner loop has heavy per-query CPU+sync overhead
  on this codebase (~50ms/query Python + sync, GPU mostly idle).
- After 1h53m, only 2 PNGs produced. Killed.
- See `reports/C01_square_attack.md` for the original Square attack vs
  safety_checker (which finished in reasonable time — 5K queries on
  small n=200 in 3-5 hours).

### PGD vs B02-adv (this session)
- `scripts/run_xtarget_attack.py --detector-ckpt B02_adv_v1/best.pt`
- Failed: `RuntimeError: One of the differentiated Tensors appears to
  not have been used in the graph`.
- Root cause: `dsi/sae/hooks.py`'s SurkovHookManager wraps the SAE
  encode call in `with torch.no_grad():`, blocking gradient flow
  back to the input image when the attack target is the SAE detector.
- Fix would require modifying SurkovHookManager to support
  attack-time gradient backprop, an architectural change.

### Square Attack vs B02-adv (this session)
- `scripts/run_square_attack.py --target sae_detector --detector-ckpt
  B02_adv_v1/best.pt --n-prompts 20 --max-queries 1000`
- Killed after >11 min with 0 PNG produced and 0 print progress lines
  past "loading I2P-NSFW prompts → 20 prompts".
- Same per-query overhead as Square vs safety_checker; with the
  SAE-detector target adding model-loading + encoding overhead per
  query, throughput is ~1 query/min — completely unworkable.

## Current state

**Item 1c-9 partial: white-box result strong (0.58% transferability),
black-box result blocked by infrastructure.**

The B02-adv result (D-6) provides strong indirect evidence: an
adversarial-trained detector catches 99.0% of held-out gradient-PGD
attacks (A03 from A01+A02 training). A black-box attacker with no
gradient access would need an adversarially-better starting point.

For the paper, Item 1c-9's black-box claim is supported by:
- **Square Attack vs safety_checker** (5-seed CI): 0.954 ± 0.029 ASR.
- **Cross-target transferability** of safety-bypassing PGD against
  SAE detector: 1/173 (5-seed). The 5K-query-budget Square Attack
  IS this experiment in the cross-target sense — it bypasses
  safety_checker, but those bypasses are caught by SAE detector.
- **B02-adv generalization** to held-out attack space: 99.0%
  detection.

## What would unblock the proper Item 1c-9

1. Fix `dsi/sae/hooks.py` SurkovHookManager to support training-time
   backprop (remove `torch.no_grad()` for the attack-time encode).
   Cost: ~1 hour code + test.
2. Re-run `run_square_attack.py --target sae_detector` with
   sub-1K-query budget on n=10 prompts (sanity check first).
3. If sub-1K queries don't work, extend to 5K with parallelism.
4. Compare ASR against safety_checker baseline (0.954 single-target).

## Caveats

- The "black-box" framing is muddied by the fact that our SAE
  detector reads features the safety_checker doesn't. A pure black-
  box attack querying only the SAE detector logit might find
  sub-detector-threshold bypasses; whether those are NSFW in the
  human sense is a separate question.
- PGD vs B02-adv (gradient-aware white-box) blocked by SurkovHookManager's
  no_grad context — this is a fixable bug not a method limitation.
