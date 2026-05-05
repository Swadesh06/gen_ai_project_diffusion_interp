# B02v3_on_mma_v1 — SAE detector OOD generalization to MMA-Diffusion

## Goal

Test the B02-v3 SAE detector (linear probe, AUC 0.977 in-distribution
on SDXL Turbo prompt-origin NSFW vs benign) on the MMA-Diffusion
adversarial image dataset. MMA's adv-gen images are produced by
adversarial **prompt** engineering against the CLIP text encoder; the
underlying generation pipeline is SD v1.4. So this is a hard
distribution shift: different backbone (SD v1.4 vs SDXL Turbo) and
different threat model (text-attack adv-gen vs image-attack PGD).

The MMA-Diffusion baseline (`reports/mma_diffusion_safety_baseline.md`):
- 53 orig images: safety_checker flagged 4 (7.5%)
- 50 adv-gen images: safety_checker flagged 29 (58%)

Question: does the SAE detector also flag MMA's adv-gen images at
elevated rate vs orig? If yes → strong cross-arch + cross-attack
robustness. If no → SAE detector is architecture-specific.

## Procedure

For each MMA image:
1. PIL → PIL.Resize(512×512) → SDXL VAE encode → SDXL UNet at t=50 with
   empty prompt conditioning → 4 Surkov SAE encoder forward.
2. Mean-pool z over spatial axis per hookpoint, concat to 20480-dim.
3. Apply B02-v3 linear probe head; output logit = log P(NSFW | features).

`scripts/eval_b02v3_on_mma.py` (with the device fix on hook captures).
~10 sec per image (sequential VAE+UNet+SAE forward), ~20 min total.

## Results

| split | n | mean logit | std | min | max | flagged @ logit>0 |
|---|---|---|---|---|---|---|
| **adv-gen** | 50 | **-6.786** | 0.807 | -8.547 | -4.796 | **0/50 (0%)** |
| **orig** | 53 | **-6.363** | 1.263 | -9.199 | -3.892 | **0/53 (0%)** |

**AUC (adv vs orig under detector logit) = 0.388** — essentially
indistinguishable. The detector flags **none** of the MMA images, both
splits at very negative logit (high confidence "safe").

Compare to safety_checker on the same images:
| split | safety_checker rate | B02-v3 SAE rate |
|---|---|---|
| orig | 7.5 % | 0.0 % |
| adv-gen | 58.0 % | 0.0 % |

## Interpretation

**The B02-v3 SAE detector does not generalize to MMA-Diffusion adv-gen
images.** Both adv and orig score at a confident "safe" logit (-6 to -7).

Two non-exclusive explanations:

1. **Architecture-specificity**: B02-v3 was trained on SDXL Turbo's
   UNet residual activations at the 4 Surkov hookpoints. MMA's
   underlying generation is SD v1.4, and the rendered images are then
   passed through SDXL Turbo's VAE+UNet for SAE encoding — but the
   resulting activation distribution is OOD from the training data.
   The SAE features that B02-v3's linear probe reads as "NSFW" don't
   activate on SD v1.4-style content fed through SDXL Turbo.

2. **Adv-gen images carry safety_checker-specific artifacts**: MMA's
   adv prompts are crafted to defeat the CLIP-based safety_checker
   image classifier. The resulting images may have features that trip
   safety_checker (high flag rate 58%) without being "NSFW content"
   in a typical sense. The SAE detector, trained on SDXL-rendered
   I2P-NSFW prompts, doesn't see this kind of adversarial pattern in
   training.

**Implication for the paper**:

- This is a **caveat for Framing A's contribution 2**: the SAE
  detector's robustness story is conditional on the generation
  backbone and the attack family.
- It is **not a failure of the SAE-feature approach in general**:
  the detector trained on SDXL produces correct logits on SDXL output,
  including A01+A02 white-box bypasses (transferability=0.000 per
  Item 1c-1). The OOD-failure is a model-specificity property, not a
  feature-extraction property.
- **Future work**: train per-backbone detectors (one for SDXL family,
  one for SD v1.4 family). Or train a joint detector on both
  distributions. D-9 cross-architecture is the broader umbrella.

## Caveats

- B02-v3 was trained at (input pre-)t=50 with empty prompt
  conditioning; the inference here uses the same encoding. So the
  encoding is matched.
- The `flagged @ logit>0` threshold is the natural decision boundary
  of logistic regression (centered at p=0.5). A lower threshold could
  recover some true positives but at high false-positive cost. ROC
  AUC (0.388) is the threshold-free version.
- 50 + 53 is a small sample; the AUC near 0.388 has wide CI.
