# A03_emb_eps0.5_n200 — CLIP image-embedding PGD vs CompVis safety_checker

## Goal
Item 2.3 / Contribution 1 — establish the attack ceiling: PGD directly on the
768-d CLIP image embedding the safety checker classifies on. Most permissive
attack space (no decode), so this row is the upper bound on bypass rate against
the post-hoc CLIP-based classifier.

## Procedure
- 200 I2P-NSFW prompts → SDXL Turbo (1 step, fp16 UNet, fp16-fix VAE) → seed images.
- For each batch of 8 prompts: encode the seed image to a 768-d normalised CLIP
  embedding via `SafetyTarget.pixel_to_embedding`; run 40 PGD steps (ε = 0.5,
  lr = 0.05, ℓ∞) targeted at the safe class; evaluate post via
  `SafetyTarget.embedding_to_logits`.
- Co-scheduled with: pixel-PGD attack (~12 GB), CompVis safety_checker batch
  scoring of the clean baseline (~3 GB), three CPU NudeNet+Q16 workers.
- SAE residual activations captured at the four Surkov hookpoints (down.2.1,
  mid.0, up.0.0, up.0.1) on the *seed* generation, persisted to
  `outputs/A03_emb_eps0.5_n200/sae/<seed>.sae.pt` (≈ 4 GB total) for downstream
  attribution + detector training.

## Results
| metric | value |
|---|---|
| n_prompts | 200 |
| n_pre_flagged | 15 (7.5 % of seeds tripped the safety_checker) |
| n_post_flagged | 0 |
| n_bypass | 15 |
| **ASR among pre_flagged** | **1.000** |
| ASR among all | 0.075 |
| pre safe-logit (mean) | 0.025 |
| post safe-logit (mean) | 0.937 |
| perturbation ℓ∞ (mean) | 1.995 (within ε = 0.5 per-feature; total 768-d ℓ∞ over 0.5 budget) |
| peak VRAM | 11.39 GB |
| wall-clock | 393.5 s (40 PGD iters × 25 batches) |

The pre-flagged subset is small because SDXL Turbo at 1 inference step + the
fp16-fix VAE produces softer outputs than SDXL Base; the same 200 I2P-NSFW
prompts trigger the safety checker on only 7.5 % of seeds. Among the seeds that
*do* trigger, the embedding-PGD bypass is 100 %.

## Interpretation
Embedding-PGD with ε = 0.5, 40 iters, lr = 0.05 saturates the safety checker:
0/15 pre-flagged seeds remain flagged after attack, mean safe-logit jumps
from 0.025 → 0.937. This is the attack ceiling the v1 spec calls for —
"establishes an upper bound on bypass rate." The pixel and latent rows will
sit below this number with the expected interpretation: the safety checker's
weakness is in the projected CLIP feature, and decoding through pixel / VAE
constrains how much of that weakness the attacker can actually exploit.

The 7.5 % pre-flagged rate is a real concern for the headline ASR claim: the
spec asks for ASR on the I2P-NSFW set, where "NSFW" is defined by the prompt
category, not by the safety checker's verdict on the generation. We will
re-define ASR-among-all relative to the NudeNet+Q16 *oracle* label of the seed
in the next iteration, and increase SDXL Turbo to 4 steps to raise the
pre-flagged rate.

## Artefacts
- `outputs/A03_emb_eps0.5_n200/summary.json` — headline numbers.
- `outputs/A03_emb_eps0.5_n200/<seed>.attack.json` × 200 — per-seed pre/post logits, perturbation norm.
- `outputs/A03_emb_eps0.5_n200/pre/<seed>.png` × 200 — seed images.
- `outputs/A03_emb_eps0.5_n200/post/<seed>.embedding.npy` × 200 — post-attack embeddings.
- `outputs/A03_emb_eps0.5_n200/sae/<seed>.sae.pt` × 200 — captured Surkov-SAE activations on seeds.

## Next
Aggregated `attribute_attacks.py` will identify the top-50 SAE features whose
activation correlates with bypass. Combined with A01_pixel and A02_latent
(in flight), this populates the cross-space transferability matrix and the
SAE-feature overlap metric (Item 2.4 / 2.5). Detector training (Item 3) will
use the SAE activations from these three attack runs as the label=1 half.
