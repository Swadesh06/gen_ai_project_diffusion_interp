#!/usr/bin/env python
"""Phase C-3 part 2 — apply the safety-trained SAEs to existing raw activations
and train a safety-SAE detector. Compares vs the Surkov-SAE detector trained
on the same underlying data.

Pipeline:
  1. Load X_raw_<hp> for each Surkov hookpoint from dataset_axbench_v1.
  2. Forward through the corresponding safety SAE to get z_safety[<hp>].
  3. Concatenate across hookpoints → X_safety (1000, sum(d_hidden_safety)).
  4. Train linear + MLP probe; report AUC.
  5. Compare to X_sae_<hp> (Surkov SAE) probe AUC.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from torch import nn

HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")


def load_safety_sae(path: Path):
    sd = torch.load(path / "state_dict.pth", map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    summary = json.loads((path / "summary.json").read_text())
    d_in = summary["d_in"]; d_hidden = summary["d_hidden"]; k = summary["k"]
    return sd, d_in, d_hidden, k


def encode_topk(x_np: np.ndarray, sd: dict, d_in: int, d_hidden: int, k: int) -> np.ndarray:
    """Run x through TopK SAE: pre_bias, encoder.weight, latent_bias, topk, relu."""
    x = torch.as_tensor(x_np, dtype=torch.float32)  # (N, D)
    pre_bias = sd["pre_bias"].float()              # (D,)
    Wenc = sd["encoder.weight"].float()            # (H, D)
    latent_bias = sd["latent_bias"].float()        # (H,)
    z = (x - pre_bias) @ Wenc.t() + latent_bias    # (N, H)
    # TopK: keep top-k per row, zero others; then relu
    vals, idx = z.topk(k, dim=-1)
    out = torch.zeros_like(z)
    out.scatter_(-1, idx, vals.relu())
    return out.numpy()


def train_probe(X: np.ndarray, y: np.ndarray, head: str, epochs: int, device: str) -> dict:
    n = len(X); rng = np.random.default_rng(42)
    perm = rng.permutation(n); cut = int(0.8 * n)
    tr = perm[:cut]; va = perm[cut:]
    Xt = torch.as_tensor(X[tr], dtype=torch.float32, device=device)
    yt = torch.as_tensor(y[tr], dtype=torch.float32, device=device)
    Xv = torch.as_tensor(X[va], dtype=torch.float32, device=device)
    yv = torch.as_tensor(y[va], dtype=torch.float32, device=device)

    D = X.shape[1]
    if head == "linear":
        net = nn.Linear(D, 1).to(device)
    elif head == "mlp":
        net = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)).to(device)
    else:
        raise ValueError(head)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_w = torch.tensor((yt == 0).sum().item() / max(1, (yt == 1).sum().item()), device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best_auc = 0.0
    for ep in range(epochs):
        net.train()
        ix = torch.randperm(len(tr), device=device)
        for i in range(0, len(tr), 64):
            sl = ix[i:i+64]
            opt.zero_grad()
            out = net(Xt[sl]).squeeze(-1)
            loss = crit(out, yt[sl])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv).squeeze(-1)).cpu().numpy()
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = float(roc_auc_score(y[va], pv))
        if auc > best_auc:
            best_auc = auc
            best_ap = float(average_precision_score(y[va], pv))
    return {"head": head, "best_va_auc": best_auc, "best_va_ap": best_ap, "n_tr": len(tr), "n_val": len(va)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="outputs/dataset_axbench_v1")
    ap.add_argument("--safety-sae-base", default="outputs/safety_saes_v1")
    ap.add_argument("--sae-suffix", default="x8_k64",
                    help="suffix used in safety_sae_<hp>_<suffix>/ directory naming")
    ap.add_argument("--out-dir", default="outputs/dataset_axbench_v1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    sae_base = Path(args.safety_sae_base)
    y = np.load(data_dir / "y.npy")
    print(f"y: {len(y)} samples ({(y==1).sum()} pos / {(y==0).sum()} neg)")

    results = {"per_hp": {}, "concat": {}}
    z_safe_by_hp = {}
    z_surkov_by_hp = {}
    for hp in HOOKPOINTS:
        hp_us = hp.replace(".", "_")
        Xraw_p = data_dir / f"X_raw_{hp_us}.npy"
        Xsae_p = data_dir / f"X_sae_{hp_us}.npy"
        if not Xraw_p.exists():
            continue
        Xraw = np.load(Xraw_p); Xsae_surkov = np.load(Xsae_p)
        sae_dir = sae_base / f"safety_sae_{hp_us}_{args.sae_suffix}"
        sd, d_in, d_hidden, k = load_safety_sae(sae_dir)
        print(f"\n=== {hp} ===  raw ({Xraw.shape}) -> safety_sae (D={d_hidden}, k={k})")
        z_safe = encode_topk(Xraw, sd, d_in, d_hidden, k)
        z_safe_by_hp[hp] = z_safe
        z_surkov_by_hp[hp] = Xsae_surkov

        r_raw = train_probe(Xraw, y, "linear", args.epochs, args.device)
        r_safe_lin = train_probe(z_safe, y, "linear", args.epochs, args.device)
        r_surk_lin = train_probe(Xsae_surkov, y, "linear", args.epochs, args.device)
        results["per_hp"][hp] = {
            "raw": r_raw, "safety_sae": r_safe_lin, "surkov_sae": r_surk_lin,
        }
        print(f"  raw          va_auc={r_raw['best_va_auc']:.4f}")
        print(f"  surkov_sae   va_auc={r_surk_lin['best_va_auc']:.4f}")
        print(f"  safety_sae   va_auc={r_safe_lin['best_va_auc']:.4f}")

    # Concat across hookpoints
    if len(z_safe_by_hp) > 0:
        X_safe_concat = np.concatenate([z_safe_by_hp[h] for h in HOOKPOINTS if h in z_safe_by_hp], axis=1)
        X_surk_concat = np.concatenate([z_surkov_by_hp[h] for h in HOOKPOINTS if h in z_surkov_by_hp], axis=1)
        print(f"\n=== concat (safety: D={X_safe_concat.shape[1]}, surkov: D={X_surk_concat.shape[1]}) ===")
        r_safe_concat = train_probe(X_safe_concat, y, "mlp", args.epochs, args.device)
        r_surk_concat = train_probe(X_surk_concat, y, "mlp", args.epochs, args.device)
        results["concat"]["safety_sae_mlp"] = r_safe_concat
        results["concat"]["surkov_sae_mlp"] = r_surk_concat
        print(f"  safety_sae mlp  va_auc={r_safe_concat['best_va_auc']:.4f}")
        print(f"  surkov_sae mlp  va_auc={r_surk_concat['best_va_auc']:.4f}")

    out = Path(args.out_dir) / "C03_safety_sae_detector_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
