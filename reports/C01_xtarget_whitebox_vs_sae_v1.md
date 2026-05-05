# C01_xtarget_whitebox_vs_sae_v1 — Item 1c-9 close: white-box PGD vs SAE detectors

## Goal

Item 1c-9 from `task_descriptions/task_description_v2.md` §3:
attack against the SAE detector. After fixing the SurkovHookManager
no_grad context (commit `0c73a12`), a gradient-aware PGD attack can
now backprop through the SAE encode and target the detector logit
directly.

The cross-target version (Item 1c-1: attacker targets safety_checker,
then we score the bypassed images with the detector) gave 1/173 =
0.58% transferability. This experiment tests the white-box version
(attacker targets the detector directly).

## Procedure

`scripts/run_xtarget_attack.py` with the fixed SurkovHookManager
(attack_mode=True). Pixel-PGD ε=4/255, 40 steps, lr=1/255.
Loss = detector logit (driven toward "safe" = logit < 0).

Two runs, n=50 prompts each (I2P-NSFW first 50, seed-offset
5500000/5600000):

1. **PGD vs B02-adv** (the adversarially-trained linear probe from D-6).
2. **PGD vs B02-v3 oracle** (the standard detector).

## Results

| attack target | n_prompts | n_pre_unsafe (logit>0) | n_post_unsafe | mean Δlogit | white-box ASR |
|---|---|---|---|---|---|
| **B02-adv** | 50 | **32** | **0** | -8.85 | **32/32 = 100%** |
| **B02-v3 oracle** | 50 | 2 | 0 | -6.74 | 2/2 = 100% |

(B02-v3's 2/50 pre_unsafe rate is consistent with the n=100 D-10
result: B02-v3 flags ~5% of natural SDXL Turbo I2P-NSFW outputs at
logit>0. B02-adv flags 64% because it was trained on adversarial-
bypass features that occur frequently in fresh I2P-NSFW SDXL Turbo
output.)

## Combined threat-model comparison

| Threat model | Detector | n_pre / n_post bypass | Bypass rate |
|---|---|---|---|
| Cross-target (Item 1c-1 5-seed): attacker hits SC, we test detector | B02-v3 | 1 / 173 | **0.58%** |
| Cross-target (D-6 hold-out A03): attacker hits SC, we test detector | B02-adv | 1 / 102* | **1.0%** |
| **White-box (this report): attacker hits detector directly** | **B02-adv** | **32 / 32** | **100%** |
| **White-box (this report): attacker hits detector directly** | **B02-v3** | **2 / 2** | **100%** |

*A03 had 102 pre-flagged on safety_checker; B02-adv caught 101/102 = 99.0% pre-attack.
Inverse rate on the n_pre subset = 1/102 = 1.0%.

## Interpretation

**The SAE detectors (B02-v3, B02-adv) are robust to cross-target
attacks but fully white-box vulnerable.** This is consistent with the
behavior of any gradient-trained linear probe under PGD: ε=4/255 with
40 steps is sufficient budget to drive the logit anywhere.

**Defense implication**: the SAE detector adds genuine security only
when the attacker doesn't know about it. If deployed as a hidden /
ensembled defense layer, attackers attacking safety_checker
(the public API) will not transfer.

If the attacker has detector-API access and gradient signal, the
detector is bypassed in 100 / 100 / 32 / 2 cases. The natural
mitigations are:

1. **Don't expose the detector logit directly** (return only "safe/
   unsafe" boolean, increase query cost via rate-limiting).
2. **Adversarial training on detector-targeting attacks** — train
   B02-adv-v2 with positives = (B02-adv-bypassed images), iterate.
3. **Ensemble** of SAE detectors with different SAE training seeds /
   different attention layers, attacker faces mass-target PGD which
   is harder.

## Item 1c-9 state

| sub-claim | status |
|---|---|
| Black-box vs SAE detector (Square attack) | infrastructure-blocked: too slow per-query (~1 query/min); queued |
| **White-box gradient PGD vs B02-v3** | **DONE: 100% bypass (2/2)** |
| **White-box gradient PGD vs B02-adv** | **DONE: 100% bypass (32/32)** |
| Cross-target transferability (attacker hits SC, we test detector) | DONE: 0.58% (Item 1c-1 5-seed) |

Item 1c-9 white-box leg is covered. Black-box Square attack against
the SAE detector remains unblocked work (would need optimization of
the per-query overhead in `scripts/run_square_attack.py`).

## Caveats

- The 32 pre_unsafe in B02-adv vs 2 in B02-v3 is because B02-adv was
  trained on adversarial-bypass features and so reads natural NSFW
  generations as more "adversarial" than B02-v3 does. In the
  deployment threat model where the attacker has post-attack image
  inputs, B02-adv is intended; but the "false positive" rate on
  natural NSFW (32/50 = 64%) is worth noting as a B02-adv usability
  caveat. In practice, B02-adv would be paired with B02-v3 (intersection,
  not union, to reduce false positives).
- ε=4/255 is the default; smaller ε might not fully bypass.
  ε-sweep against B02-adv specifically queued.
- 40 steps PGD is the default; more steps might converge on lower
  post_logit but the 100% bypass rate is already saturated.
