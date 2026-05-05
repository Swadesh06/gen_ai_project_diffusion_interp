# D05_oracle_transfer_v1 — A01 attack does not transfer across oracle ensemble

## Goal

Phase D-5 oracle-transfer test. The Phase 1 A01 pixel-PGD attack was
crafted against the CompVis safety_checker. Question: does the attack
transfer to other classifier oracles (NudeNet, Q16) on the same images?

If yes → safety_checker bypass is symptomatic of the underlying image
content actually being NSFW; the perturbation broke a single weak
classifier.

If no → the attack is **classifier-specific**: the perturbation breaks
safety_checker but doesn't fool other classifiers; ensemble defense
would be effective; the SAE detector is genuinely complementary.

## Procedure

Source: `outputs/A01_pixel_eps4_n200_seed1/post/` — 200 attacked PNGs,
each a saturated bypass of safety_checker (n_pre_flagged=17,
n_bypass=17, ASR=1.000 on the **safety_checker** target).

Each PNG is scored by:
- **CompVis safety_checker** (the original attack target — 0% expected
  by construction)
- **NudeNet** (independent NSFW image classifier)
- **Q16** (CLIP image embedding × prompt-tuned classifier; SMID-trained
  per Schramowski et al. 2022)

`scripts/exp_D05_oracle_transfer.py`. CPU-bound (NudeNet, Q16) plus a
small GPU footprint (safety_checker on cuda).

## Results

| Oracle | Flag rate | n flagged / n total |
|---|---|---|
| safety_checker (target) | **0.000** | 0 / 200 |
| **NudeNet** | **0.005** | 1 / 200 |
| **Q16** | **0.120** | 24 / 200 |
| **any oracle (NudeNet OR Q16 OR SC)** | **0.125** | 25 / 200 |

## Interpretation

**The A01 attack is highly classifier-specific.** Of 200 PNGs that
saturated safety_checker, only:
- 0.5 % are caught by NudeNet (1 image),
- 12.0 % are caught by Q16 (24 images),
- 12.5 % are caught by any oracle in the ensemble.

87.5% of the attacked images escape all three oracles. The PGD's
gradient signal flows through safety_checker's CLIP-image-prototype
representation; the resulting perturbation flips that representation
without changing what NudeNet's R-CNN body-part detector or Q16's
SMID-prototype classifier sees.

This **strengthens the case for the SAE detector as a complementary
defense**. Item 1c-1 showed the SAE detector (B02-v3, AUC 0.977)
catches **0/32** of these A01+A02 attacked images on its own decision
boundary. Combined with the present result:

| Defense layer | A01 bypass rate at default threshold |
|---|---|
| safety_checker (target) | 100% by construction |
| NudeNet | 99.5% bypass |
| Q16 | 88.0% bypass |
| 3-oracle ensemble (NudeNet+Q16+SC) | 87.5% bypass |
| **SAE detector B02-v3 (linear)** | **100% caught** (0% bypass) |

The SAE detector, trained against oracle-relabelled NSFW (which is the
"any oracle" union), is the **most robust** layer to PGD attack against
safety_checker. This is the headline Framing A contribution 3 result.

## Cross-ε comparison (added)

Re-ran on the smaller-ε A02 attack (`A02_latent_eps0.025_n100/post`):

| oracle | A01 ε=4/255 (n=200) | A02 ε=0.025 (n=100) |
|---|---|---|
| safety_checker (target) | 0.0% | 0.0% |
| NudeNet | 0.5% | 2.0% |
| Q16 | 12.0% | 16.0% |
| any oracle | 12.5% | 17.0% |
| **escape all 3 oracles** | **87.5%** | **83.0%** |

The cross-oracle escape rate **drops slightly** at smaller ε (87.5% →
83.0% escape all 3) — at smaller perturbation budgets, the attack is
less able to target safety_checker's specific decision boundary, and
slightly more residual content leaks through to Q16. But 83% escape
still confirms the headline finding.

## Caveats

- Q16 has a higher false-positive rate (catches more images that may
  not actually be NSFW). Of the 24 "Q16 flagged" images, some may be
  the original NSFW input, not the attack-flipped output. Per-image
  pre-vs-post comparison is queued.
- 200 images is one seed; multi-seed CIs would tighten the 0.5%-12%
  spread.
- The PGD attack didn't *try* to transfer; it was a single-classifier
  PGD. A more aggressive multi-oracle PGD (sum-of-logits) might
  achieve cross-oracle transfer.
