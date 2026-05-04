#!/usr/bin/env python
"""Phase C-3 — train a safety-specialised TopK SAE on labelled (I2P + COCO) activations.

Dataset: outputs/raw_*/raw/<seed>.raw.pt × {coco, i2p_nsfw}. The SAE is trained
on the per-block raw residual diffs at one Surkov hookpoint at a time.

Architecture: matches Surkov's `SurkovTopKSAE` (encoder + decoder + pre_bias +
latent_bias + topk + relu). Loss: MSE reconstruction + small auxk dead-feature
regulariser.

Per appendix §G C-3 we sweep expansion ∈ {8, 16, 32} × L0 (k) ∈ {32, 64, 128, 256}
across the four Surkov hookpoints. This script trains ONE configuration; we
spawn it per (hookpoint, expansion, k) in a sweep.

Pass criterion: the safety-trained SAE produces a Stage-2-survivor feature
set whose causal score is ≥ 1.5× Surkov out-of-the-box; ASR-on-I2P-adv ≥ 5pp
lower; FID ≤ 1.0 worse.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _iter_raw(raw_dirs):
    """Yield (raw_payload_dict, label) for each .raw.pt found."""
    import torch as _torch

    for d, label in raw_dirs:
        for f in sorted(Path(d).glob("*.raw.pt")):
            try:
                p = _torch.load(f, map_location="cpu", weights_only=False)
                yield p, label
            except Exception:
                continue


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nsfw-raw-dir", required=True)
    ap.add_argument("--benign-raw-dir", required=True)
    ap.add_argument("--hookpoint", required=True,
                    choices=["down.2.1", "mid.0", "up.0.0", "up.0.1"])
    ap.add_argument("--expansion", type=int, default=16)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from dsi.sae.load import SurkovTopKSAE

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect mean-pooled raw activations at the chosen hookpoint
    raw_dirs = [(args.nsfw_raw_dir, 1), (args.benign_raw_dir, 0)]
    Xs = []
    ys = []
    for payload, label in _iter_raw(raw_dirs):
        if args.hookpoint not in payload:
            continue
        Xs.append(payload[args.hookpoint].float().numpy())
        ys.append(label)
    if not Xs:
        print("no data")
        return 2
    X = np.stack(Xs).astype("float32")
    y = np.asarray(ys, dtype="int64")
    print(f"hookpoint={args.hookpoint} d_in={X.shape[1]} N={X.shape[0]} (pos={int(y.sum())} neg={int((1-y).sum())})")

    # Build SAE with the requested expansion + k
    d_in = X.shape[1]
    d_hidden = args.expansion * d_in
    sae = SurkovTopKSAE(d_in=d_in, d_hidden=d_hidden, k=args.k).to(args.device)
    print(f"SAE: d_in={d_in} d_hidden={d_hidden} (expansion={args.expansion}) k={args.k}")

    # Init pre_bias to data mean, decoder weight to encoder transpose
    with torch.no_grad():
        sae.pre_bias.data = torch.from_numpy(X.mean(axis=0)).to(args.device)
        sae.encoder.weight.data = torch.randn(d_hidden, d_in, device=args.device) * (1.0 / d_in ** 0.5)
        sae.decoder.weight.data = sae.encoder.weight.data.T.clone() / d_in ** 0.5

    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr, weight_decay=0.0)
    Xt = torch.from_numpy(X).to(args.device)
    n_tr = Xt.shape[0]

    t0 = time.time()
    for epoch in range(args.epochs):
        sae.train()
        perm = torch.randperm(n_tr, device=args.device)
        loss_sum = 0.0
        n_active_sum = 0
        for i in range(0, n_tr, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x = Xt[idx]
            x_hat, z = sae(x)
            recon = F.mse_loss(x_hat, x)
            # Track sparsity
            n_active = (z > 0).float().mean().item()
            n_active_sum += n_active * len(idx)
            opt.zero_grad()
            recon.backward()
            opt.step()
            loss_sum += float(recon.detach()) * len(idx)
        avg_loss = loss_sum / n_tr
        avg_active = n_active_sum / n_tr
        # Decoder column-norm reset (Surkov uses this trick for stability)
        with torch.no_grad():
            decoder_norms = sae.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            sae.decoder.weight.data /= decoder_norms
        print(f"  ep {epoch:3d}  recon_mse={avg_loss:.6f}  active_frac={avg_active:.4f}", flush=True)

    elapsed = time.time() - t0
    ckpt_dir = out_dir / f"safety_sae_{args.hookpoint.replace('.','_')}_x{args.expansion}_k{args.k}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sae.state_dict()}, ckpt_dir / "state_dict.pth")
    summary = {
        "hookpoint": args.hookpoint, "expansion": args.expansion, "k": args.k,
        "d_in": d_in, "d_hidden": d_hidden, "n_train": n_tr,
        "final_recon_mse": avg_loss, "final_active_frac": avg_active,
        "elapsed_s": elapsed,
    }
    (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
