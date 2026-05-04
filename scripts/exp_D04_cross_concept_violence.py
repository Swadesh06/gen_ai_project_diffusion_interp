#!/usr/bin/env python
"""Phase D-4 — cross-concept transfer test on violence.

Builds Stage 1 ∩ Stage 2 selection F_violence using cached
raw_violence_300 + raw_coco_500 activations, then evaluates whether
applying mean-patch on F_violence reduces ASR on UnlearnDiffAtk-violence
prompts (rendered + scored separately).

Stage 1 (Fisher ratio): per-feature s_violence(f) = E[z_f^2 | violence],
                        s_benign(f)   = E[z_f^2 | coco],
                        ratio s_violence/s_benign > tau_ratio.

Stage 2 (causal intervention): for each Stage-1 survivor f, compute
ΔP_q16(f) = P_Q16(c=violent | gen with z_f := +λ) - P_Q16(c=violent | gen).

Output: outputs/D04_F_violence/{stage1_survivors.json, stage2_scores.json,
        F_violence.json}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import cfg  # noqa: E402

HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-id", default="D04_F_violence")
    ap.add_argument("--violence-raw-dir", default="outputs/raw_violence_300/raw")
    ap.add_argument("--benign-raw-dir", default="outputs/raw_coco_500/raw")
    ap.add_argument("--surkov-saes", action="store_true",
                    help="encode raw activations through Surkov SAE first (vs raw)")
    ap.add_argument("--tau-ratio-percentile", type=float, default=95.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import numpy as np
    import torch

    out_dir = cfg.paths.output_root / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def _iter_raw(d: Path):
        files = sorted(d.glob("*.raw.pt"))
        for f in files:
            try:
                yield torch.load(f, map_location="cpu", weights_only=False)
            except Exception:
                continue

    print("loading violence + benign cached raw activations", flush=True)
    violence_X: dict[str, list] = {hp: [] for hp in HOOKPOINTS}
    benign_X: dict[str, list] = {hp: [] for hp in HOOKPOINTS}
    for payload in _iter_raw(Path(args.violence_raw_dir)):
        for hp in HOOKPOINTS:
            if hp in payload:
                violence_X[hp].append(payload[hp].float().numpy())
    for payload in _iter_raw(Path(args.benign_raw_dir)):
        for hp in HOOKPOINTS:
            if hp in payload:
                benign_X[hp].append(payload[hp].float().numpy())
    for hp in HOOKPOINTS:
        violence_X[hp] = np.stack(violence_X[hp]) if violence_X[hp] else np.zeros((0, 1280))
        benign_X[hp] = np.stack(benign_X[hp]) if benign_X[hp] else np.zeros((0, 1280))
        print(f"  {hp}: violence={violence_X[hp].shape}, benign={benign_X[hp].shape}", flush=True)

    if args.surkov_saes:
        print("encoding via Surkov SAEs", flush=True)
        from dsi.sae.load import load_surkov_sae
        sae_dict = {hp: load_surkov_sae(hp).to(args.device).eval() for hp in HOOKPOINTS}
        for hp in HOOKPOINTS:
            v_t = torch.from_numpy(violence_X[hp]).to(args.device).to(next(sae_dict[hp].parameters()).dtype)
            b_t = torch.from_numpy(benign_X[hp]).to(args.device).to(next(sae_dict[hp].parameters()).dtype)
            with torch.no_grad():
                v_z = sae_dict[hp].encode(v_t).cpu().numpy()
                b_z = sae_dict[hp].encode(b_t).cpu().numpy()
            violence_X[hp] = v_z
            benign_X[hp] = b_z
            print(f"  {hp}: encoded violence={v_z.shape}, benign={b_z.shape}", flush=True)

    # Stage 1 — Fisher ratio
    print("Stage 1 — Fisher ratio per feature per hookpoint", flush=True)
    stage1: dict[str, list[int]] = {}
    stage1_full: dict[str, dict] = {}
    for hp in HOOKPOINTS:
        Xv = violence_X[hp]; Xb = benign_X[hp]
        if Xv.shape[0] == 0 or Xb.shape[0] == 0:
            stage1[hp] = []
            continue
        s_for = (Xv ** 2).mean(axis=0)
        s_ret = (Xb ** 2).mean(axis=0).clip(min=1e-12)
        ratio = s_for / s_ret
        tau = float(np.percentile(s_ret, args.tau_ratio_percentile))
        survivors = [int(i) for i, r in enumerate(ratio) if s_for[i] / max(s_ret[i], 1e-12) > 1.0 and s_for[i] > tau]
        # Top-N rank by ratio
        ranked = np.argsort(-ratio)
        # Take top by percentile
        n_keep = max(1, int(len(ratio) * (100 - args.tau_ratio_percentile) / 100.0))
        survivors = ranked[:n_keep].tolist()
        stage1[hp] = survivors
        stage1_full[hp] = {
            "n_features_total": len(ratio),
            "n_survivors": len(survivors),
            "tau": tau,
            "top_5_ratios": [(int(i), float(ratio[i])) for i in ranked[:5]],
        }
        print(f"  {hp}: {len(survivors)} survivors of {len(ratio)} features", flush=True)

    (out_dir / "stage1_survivors.json").write_text(json.dumps(stage1, indent=2))
    (out_dir / "stage1_summary.json").write_text(json.dumps(stage1_full, indent=2))

    # Stage 2 — placeholder note (full Stage 2 needs causal generation, slower)
    note = {
        "stage2_note": "Stage 2 causal-intervention scoring needs SDXL Turbo gen with feature clamps + Q16 oracle. Run via scripts/stage2_causal_score.py with --concept violence + --stage1-json {out_dir}/stage1_survivors.json",
        "stage1_json": str(out_dir / "stage1_survivors.json"),
    }
    (out_dir / "stage2_pending.json").write_text(json.dumps(note, indent=2))

    summary = {
        "exp_id": args.exp_id,
        "n_violence_samples": int(violence_X[HOOKPOINTS[0]].shape[0]),
        "n_benign_samples": int(benign_X[HOOKPOINTS[0]].shape[0]),
        "feature_source": "surkov_sae" if args.surkov_saes else "raw",
        "stage1_per_hp": {hp: len(s) for hp, s in stage1.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
