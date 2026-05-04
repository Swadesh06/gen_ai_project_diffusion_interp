# C01_xtarget_v2_vs_B02v3 — cross-target transferability against oracle-relabeled detector

## Goal

The original Phase 1 C01 result `transferability_safety_to_detector = 0.000`
against the B01 detector had two confounds:
1. The **bit-identical detector logits bug** (fixed in `eval_xtarget_transfer_v2.py`,
   verified in `reports/C01_xtarget_v2_A01_vs_B01.md`).
2. **Prompt-origin label leak in B01**: B01 was trained on (prompt-origin)
   labels, not oracle-judgement-on-rendered-image. It flags 78% of any I2P
   prompt regardless of image content.

This report runs the v2 (image-conditioned UNet trace) cross-target evaluation
against **B02-oracle-v3** — the 1544-sample, oracle-labeled, balanced-BCE
detector that hits AUC 0.9762 (linear concat).

## Procedure

- Source attack records: `outputs/A0{1,2,3}_pixel/latent/embedding_*.attack.json`
  + `pre/<seed>.png` + `post/<seed>.png`.
- Detector: `/workspace/checkpoints/B02_oracle_v3_linear_cat/best.pt`
  (20480-d concat linear probe, AUC 0.9762).
- For each row: VAE-encode pre and post images separately → single UNet forward
  pass at timestep 50 + empty prompt embeds → SurkovHookManager captures z's
  → spatial-mean-pool to (B, 20480) → detector logits.
- Match `n_safety_pre_flagged` to the original attack record's `pre_flagged`
  field; `n_detector_pre_flagged` is computed live from this run.

## Results

### A01 pixel-PGD vs B02-oracle-v3

| metric | value |
|---|---|
| n_total | 200 |
| n_logit_identical_pre_post | 0 (bug fixed) |
| n_safety_pre_flagged | 17 |
| n_safety_post_flagged | 0 |
| n_detector_pre_flagged | 14 |
| n_detector_post_flagged | 10 |
| n_safety_bypass | 17 |
| n_detector_bypass | 4 |
| **n_bypass_both** | **0** |
| asr_safety_among_pre_flagged | **1.000** |
| **transferability_safety_to_detector** | **0.000** |

### A02 latent-PGD vs B02-oracle-v3

In flight (re-launched after a CUDA OOM on the first attempt; the GPU was
at the cap when 3 SDXL Turbo + 4 SAE forwards tried to load
simultaneously alongside 3 SDXL Base renders).

### A03 embedding-PGD

A03 perturbs the CLIP image embedding fed to the safety_checker; it does
not produce a perturbed image. The SAE detector traces UNet on the actual
image, which is identical for pre and post in A03 — the SAE detector is
**robust to A03 by construction**. Documented as a non-result; not a row
in the cross-target ASR matrix.

## Interpretation

The cross-target transferability result holds **for the meaningful
detector** (B02-oracle-v3), not just the prompt-leaky B01:

- 17/17 pixel-PGD attacks crafted against the production safety_checker
  successfully bypass safety_checker (ASR 1.000).
- **None** of those 17 attacks bypasses the SAE-activation detector
  (n_bypass_both = 0; transferability = 0.000).
- The detector flags 14/200 pre images (7%) — orders of magnitude lower
  than B01's 157/200 (78%), confirming B02-v3 reads actual image content
  rather than prompt-origin shortcut.
- `detector_post_flagged = 10` (from `pre = 14`) means the attacks
  *do* perturb the SAE detector's logits non-trivially (4 detector
  bypasses out of 14 pre-flagged → 28.6% detector-only ASR), but those
  4 are disjoint from the 17 safety bypasses.

This is the crisp Contribution-3 result for the paper headline.

## Next

- A02 + A03 vs B02-v3 (A02 in flight after OOM; A03 is a non-result).
- Same matrix vs `safety_saes_v2` detector (AUC 1.000 on in-distribution
  axbench; needs the saved checkpoint loadable through the standard probe
  head).
- Black-box C-1 result vs B02-v3: ongoing in
  `outputs/C01_square_vs_B02v3_n100_q2k`.
- 5-seed CIs for the headline 0/17 ratio (Item 1c-6).
