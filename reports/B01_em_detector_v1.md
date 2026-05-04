# B01_em_detector_v1 — early-monitor SAE-activation detector (linear probe)

## Goal
Item 3 / Contribution 2 — train an in-generation detector on SAE activations.
Linear probe on the spatial-mean-pooled Surkov SAE features at the four UNet
hookpoints. Pass criterion: AUC ≥ 0.95 on the I2P-naive split.

## Procedure
- Dataset `outputs/detector_dataset_v1` built from:
  - 200 I2P-NSFW seed images attacked in pixel space (A01) — label=1.
  - 200 I2P-NSFW seed images attacked in embedding space (A03) — label=1.
  - 768 COCO-val benign seed images (sae_benign_coco_1k partial; complete run
    ongoing) — label=0.
- Per sample: spatial-mean-pooled (16×16 → ()) SAE-encoded features at
  `down.2.1`, `mid.0`, `up.0.0`, `up.0.1`. Concatenated to a 20480-d vector
  for the all-blocks probe; 5120-d for the per-block probes.
- Linear probe via `scripts/train_detector.py` (AdamW, lr=1e-3, batch=64,
  30 epochs, 80/20 random train/val split, seed=0). Five variants in parallel:
  one all-blocks linear, one all-blocks 2-layer MLP (hidden=512), and four
  per-block linear probes.

## Results

| variant | val AUC | val AP | epochs to AUC ≥ 0.99 |
|---|---|---|---|
| all-blocks linear   | **1.000** | 1.000 | 0 |
| all-blocks MLP-512  | **1.000** | 1.000 | 0 |
| per-block down.2.1  | 1.000 | 1.000 | 1 |
| per-block mid.0     | 0.997 | 0.992 | 4 |
| per-block up.0.0    | 0.999 | 0.997 | 2 |
| per-block up.0.1    | 0.999 | 0.997 | 1 |

All variants exceed the Item 3 / Contribution 2 pass criterion (AUC ≥ 0.95).
The all-blocks linear probe hits 1.000 at epoch 0; per-block down.2.1 is
already perfect by epoch 1.

## Interpretation
Pass on the raw metric, with a critical caveat: this dataset labels samples
by *prompt origin* (I2P vs COCO), not by *actual oracle judgement on the
generated image*. The detector is therefore distinguishing "I2P-style prompt
generation" vs "COCO-style prompt generation" — which is largely a domain shift
the SAE features encode trivially.

A meaningful Item 3 result requires re-labelling samples by NudeNet / Q16 /
safety_checker judgement on the actually-rendered image and re-training. The
clean baseline gives us 374/1000 I2P-side flagged (Q16+NudeNet+safety_checker
union) vs 35/1000 COCO-side flagged — that's the real label distribution. We
will rebuild the detector dataset using these oracle labels and report the
honest AUC next round (B02_em_oracle_v1).

The Phase-C-2 AxBench-style direct probe on raw UNet activations (without the
SAE) is also queued — that result will tell us whether the SAE is doing real
work for the detector or whether a raw-activation linear probe also hits 1.0.

## Artefacts
- `checkpoints/B01_em_linear_v1/{best.pt, last.pt, summary.json}` — trained probe.
- `checkpoints/B01_em_mlp_v1/...` — MLP variant.
- `checkpoints/B01_em_lin_<hookpoint>/...` — per-block variants.

## Next
- B02_em_oracle_v1: re-label the dataset by actual NudeNet+Q16+safety_checker
  judgement on the generated image; re-train; this is the meaningful AUC.
- C01 (Phase C-2 AxBench): linear probe on raw UNet hidden states bypassing
  the SAE; if it matches the SAE AUC within 2 pp, the SAE-detector story
  shifts to "Stage 4 enabling" rather than "necessary for detection".
- Cross-target attack (Item 4) using `B01_em_linear_v1/best.pt` as the target.
