#!/usr/bin/env python
"""WandB sweep config for the EM/FT detector (lr, batch, head, em_steps, pool)."""

from __future__ import annotations

import sys

import yaml

SWEEP = {
    "method": "bayes",
    "metric": {"name": "val/auc", "goal": "maximize"},
    "parameters": {
        "lr":        {"min": 1e-5, "max": 5e-3, "distribution": "log_uniform_values"},
        "batch":     {"values": [32, 64, 128, 256]},
        "head":      {"values": ["linear", "mlp", "ensemble"]},
        "em_steps":  {"values": [1, 2, 3]},
        "pool":      {"values": ["mean", "max", "attn"]},
        "weight_decay": {"min": 1e-6, "max": 1e-2, "distribution": "log_uniform_values"},
    },
}


def main() -> int:
    print(yaml.safe_dump(SWEEP, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
