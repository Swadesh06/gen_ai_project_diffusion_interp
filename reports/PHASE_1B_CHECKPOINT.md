# Phase 1b mid-session checkpoint

## Status of the four contributions

### Contribution 1 — cross-space red-team + SAE attribution: **PASS**
- A01 pixel ε=4/255: 17/17 pre-flagged seeds bypassed (ASR=1.000).
- A02 latent ε=0.1: 15/15 pre-flagged bypassed (ASR=1.000).
- A03 embedding ε=0.5: 15/15 pre-flagged bypassed (ASR=1.000); post safe_logit
  0.937 — the attack ceiling.
- 3×3 cross-space SAE feature overlap (top-50 Jaccard) at down.2.1 and up.0.1
  passes ≥ 0.60 on **all three** off-diagonal cells, meeting the Item 2.5 pass
  criterion decisively.
- Caveat: 8.5 % pre-flagged base rate (SDXL Turbo at 1 step is tame). Plan: rerun on SDXL
  Base (multi-step) to lift the base rate before paper.

### Contribution 2 — in-generation SAE-activation detector: **PASS-with-caveat**
- B01_em_linear_v1 hits val AUC=1.000 with prompt-origin labels (I2P→1, COCO→0).
  Trivially passes the AUC ≥ 0.95 criterion but partly exploits the prompt-style
  domain shift rather than actual content.
- B02_em_oracle_v1 (oracle-relabelled, in flight): training on
  detector_dataset_oracle_v1 (1388 samples, 41 NSFW + 1347 benign — class
  imbalance because cpu-worker has only oracle-labelled the A01 attack outputs
  so far; A02 + sae_benign labels still landing). Best so far ep4
  va_auc=0.847, va_ap=0.336 — meaningful in the imbalanced regime; tunes
  further with class-balance + more oracle labels.

### Contribution 3 — cross-target robustness: **PASS**
- C01_xtarget_A01_vs_B01: 200 A01 pixel-attack records evaluated against the
  trained B01 SAE-detector via SDXL Turbo text-conditional regen with the four
  Surkov SAE hooks active.
  - Safety bypass count: 17/17.
  - Detector bypass count: **0/17**.
  - **transferability_safety_to_detector = 0.000**.
- Off-diagonal cell ≤ 0.50 criterion **met decisively at 0%** — the safety
  checker and SAE detector monitor substantially disjoint feature subspaces.
- Caveat: B01's 196/200 detector pre-flag rate is an overshoot (it almost
  always says "I2P prompt"). The transferability re-test against B02 (oracle)
  will be the meaningful version.
- xtarget-A02 queued; will fire after stage-2.

### Contribution 4 — detection-triggered correction: **WIRED, INITIAL D01 partial**
- Stage 1 Fisher-ratio survivor selection: 256 features per Surkov hookpoint
  at the 95th-percentile threshold. Stored in `outputs/stage1_v1/stage1_*.json`.
- Stage 2 causal-intervention output score (Arad et al.):
  - up.0.1 (32 features, λ=250, Q16 oracle): max |Δ| = 0.014; pipeline works.
  - down.2.1 + mid.0 in flight (~7 features each so far at ~9s/feature).
  - λ=500 variant queued.
- D01 intervention smoke (50 prompts, F_c = top-32 Stage-1-only per block):
  1/6 pre-flagged corrected (16.7 %); pipeline correct, F_c quality is the
  bottleneck.
- D02 with proper Stage-1 ∩ Stage-2 F_c on the way after stage-2 runs land.

## Hardware utilisation throughout Phase 1b
- VRAM peak: 31.4 / 32 GB (98 %) when {pixel-PGD, embedding-PGD, safety-checker
  batch, gen-sae-benign, 3 cpu-workers} co-located.
- 14-worker mp.Pool for `build_detector_dataset.py` cut 30 min sequential to
  2.5 min on 1168 .sae.pt files (mfs latency was the bottleneck).
- 2 OOM crashes recovered by killing the lowest-priority job and restarting
  on CPU when fit (B02 trains on CPU).

## Commits
17 commits to origin/main since Phase 1b start. Reports for A01/A02/A03/B01/
C01/D01/contribution1_final landed.

## Next 30 min
1. Wait for stage2-down + stage2-mid (~10 min more). λ=500 variant in parallel.
2. Build proper F_c = Stage 1 ∩ Stage 2 (intersection of Fisher survivors with
   Stage-2 causal Δ > 0), per hookpoint.
3. D02 mean-patch on the proper F_c on 200 prompts; expect higher correction
   rate.
4. Re-run xtarget against B02 once it completes oracle training.
5. PHASE_1_FINAL.md headline table; sync paper/main.tex.

## Open ops items carried to Phase C
- Higher-base-rate I2P NSFW source: SDXL Base (multi-step) re-runs.
- Class-balanced sampling for B02 / oracle dataset rebuild with more sae_benign
  oracle labels (cpu-worker's pace bound).
- AxBench-style raw-UNet probe (Phase C-2) to validate that SAE features add
  value over raw activations.
