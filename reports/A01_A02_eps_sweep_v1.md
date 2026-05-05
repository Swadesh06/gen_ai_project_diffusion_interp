# A01_A02_eps_sweep_v1 — Attack ASR vs perturbation budget

## Goal

Single-seed at ε=full saturates (A01 ε=4/255 ASR=1.000, A02 ε=0.1
ASR=1.000). The interesting question is **what's the saturation
boundary**? Does ASR drop sharply at ε/2 or ε/4, or does the attack
remain effective at lower budgets?

## Procedure

A01 pixel-PGD at:
- ε=4/255 (Phase 1 default, saturates) — done
- ε=2/255 (half) — `A01_pixel_eps2_n100`, n=100, in flight
- ε=1/255 (quarter) — `A01_pixel_eps1_n100`, n=100, in flight

A02 latent-PGD at:
- ε=0.1 (Phase 1 default, saturates) — done
- ε=0.05 (half) — `A02_latent_eps0.05_n100`, n=100, in flight
- ε=0.025 (quarter) — `A02_latent_eps0.025_n100`, n=100, in flight

40 PGD steps each, n=100 prompts (smaller than 200 to be quick).

## Results — pending

| attack | ε | n_pre / n_bypass | ASR among pre-flagged |
|---|---|---|---|
| A01 pixel | 4/255 | 17 / 17 (Phase 1 single seed) | 1.000 |
| A01 pixel | 2/255 | (in flight) | (in flight) |
| A01 pixel | 1/255 | (in flight) | (in flight) |
| A02 latent | 0.1 | 24 / 24 (typical seed) | 1.000 |
| A02 latent | 0.05 | (in flight) | (in flight) |
| A02 latent | 0.025 | (in flight) | (in flight) |

## Expected interpretation

If ASR drops sharply at ε/2: the attack is at its budget boundary;
slightly tighter ε defeats the attack; defenders have a budget-based
defense.

If ASR remains saturating at ε/2 / ε/4: the attack budget is overkill;
the saturation is a property of safety_checker brittleness rather than
the attack power; defenders need a fundamentally different defense (the
B02-v3 SAE detector is one such; transferability=0.000).

## Caveats

- Single-seed at smaller-ε; 5-seed CIs would be needed for headline
  numbers but n=100 is the smaller-scale ε-sweep.
- ε=1/255 is the smallest meaningful pixel perturbation (1 step in
  uint8). Below that the attack has no expressivity.
