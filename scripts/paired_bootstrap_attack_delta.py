#!/usr/bin/env python
"""Paired bootstrap on per-seed ASR delta between two attacks.

For two attacks A and B, where both have 5-seed CIs, compute the
paired-bootstrap 95% CI on (ASR_A - ASR_B). 'Paired' means the same
seed indexes are used together (since each seed shifts the prompt
sample).
"""
import json, sys, math
from pathlib import Path

def load_seeds(prefix):
    rows = []
    for s in range(5):
        cands = [Path(f"outputs/{prefix}_seed{s}/summary.json")] if s > 0 else [Path(f"outputs/{prefix}_seed{s}/summary.json"), Path(f"outputs/{prefix}/summary.json")]
        for c in cands:
            if c.exists():
                d = json.loads(c.read_text())
                rows.append((s, float(d.get("asr_among_pre_flagged", 0))))
                break
    return rows

def main():
    if len(sys.argv) < 3:
        print("usage: paired_bootstrap_attack_delta.py <prefix-A> <prefix-B> [--n-boot 10000]")
        return 1
    prefA, prefB = sys.argv[1], sys.argv[2]
    seedsA = load_seeds(prefA)
    seedsB = load_seeds(prefB)
    common = sorted(set([s for s,_ in seedsA]) & set([s for s,_ in seedsB]))
    if not common:
        print(f"no common seeds for {prefA} vs {prefB}")
        return 1
    aA = [a for s,a in seedsA if s in common]
    aB = [a for s,a in seedsB if s in common]
    deltas = [aA[i] - aB[i] for i in range(len(common))]
    mean = sum(deltas) / len(deltas)
    n = len(deltas)
    print(f"=== {prefA} vs {prefB} ===")
    print(f"  n_seeds = {n}")
    print(f"  per-seed deltas: {[f'{d:.4f}' for d in deltas]}")
    print(f"  mean delta = {mean:.4f}")
    if all(d == deltas[0] for d in deltas):
        print(f"  zero variance — CI [{mean:.4f}, {mean:.4f}]")
        return 0
    # bootstrap
    import random
    random.seed(0)
    boots = []
    n_boot = int(sys.argv[sys.argv.index("--n-boot")+1]) if "--n-boot" in sys.argv else 10000
    for _ in range(n_boot):
        sample = [random.choice(deltas) for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    print(f"  paired-bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
