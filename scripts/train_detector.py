#!/usr/bin/env python
"""Train an EM detector head on cached SAE activations (Item 3).

Loads `<data_dir>/X_<hookpoint>.npy` + `<data_dir>/y.npy`, splits 80/20 train/val,
trains a linear probe (or 2-layer MLP), reports AUC + AP + commit-knee statistics
(only meaningful with multi-step T data; for SDXL Turbo at 1 step this is just
overall AUC). Writes checkpoint via RollingCheckpointer + best.pt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import DetectorCfg, cfg  # noqa: E402


def _load_dataset(data_dir: Path, hookpoint: str | None):
    import numpy as np

    y = np.load(data_dir / "y.npy").astype("int64")
    if hookpoint:
        Xs = {hookpoint: np.load(data_dir / f"X_{hookpoint.replace('.', '_')}.npy").astype("float32")}
    else:
        Xs = {}
        for p in sorted(data_dir.glob("X_*.npy")):
            hp = p.stem.removeprefix("X_").replace("_", ".")
            Xs[hp] = np.load(p).astype("float32")
    return Xs, y


def _split(X, y, val_frac: float, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_val = max(1, int(len(y) * val_frac))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--exp-id", default="B01_em_linear")
    ap.add_argument("--hookpoint", default=None,
                    help="if omitted, concatenate all hookpoints")
    ap.add_argument("--head", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, roc_auc_score

    from dsi.util.ckpt import RollingCheckpointer
    from dsi.util.seed import set_seed

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    Xs, y = _load_dataset(data_dir, args.hookpoint)
    print(f"loaded {len(y)} samples ({sum(y)} pos, {len(y)-sum(y)} neg) from {data_dir}")
    if args.hookpoint:
        X = Xs[args.hookpoint]
    else:
        X = np.concatenate([v for v in Xs.values()], axis=1)
    print(f"X shape: {X.shape}")

    Xtr, ytr, Xva, yva = _split(X, y, args.val_frac, seed=args.seed)
    Xtr_t = torch.from_numpy(Xtr).to(args.device)
    ytr_t = torch.from_numpy(ytr).float().to(args.device)
    Xva_t = torch.from_numpy(Xva).to(args.device)
    yva_t = torch.from_numpy(yva).float().to(args.device)

    if args.head == "linear":
        head = nn.Linear(X.shape[1], 1).to(args.device)
    else:
        head = nn.Sequential(
            nn.Linear(X.shape[1], args.hidden), nn.ReLU(), nn.Linear(args.hidden, 1)
        ).to(args.device)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ckptr = RollingCheckpointer(cfg.paths.checkpoint_root / args.exp_id, keep_n=4)

    n_tr = len(Xtr_t)
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n_tr, device=args.device)
        loss_sum = 0.0
        for i in range(0, n_tr, args.batch_size):
            idx = perm[i : i + args.batch_size]
            logits = head(Xtr_t[idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(idx)
        head.eval()
        with torch.no_grad():
            tr_logits = head(Xtr_t).squeeze(-1).float().cpu().numpy()
            va_logits = head(Xva_t).squeeze(-1).float().cpu().numpy()
        tr_auc = roc_auc_score(ytr, tr_logits)
        va_auc = roc_auc_score(yva, va_logits)
        va_ap = average_precision_score(yva, va_logits)
        h = {"epoch": epoch, "loss": loss_sum / n_tr, "tr_auc": float(tr_auc),
             "va_auc": float(va_auc), "va_ap": float(va_ap)}
        history.append(h)
        print(f"ep {epoch:3d}  loss {h['loss']:.4f}  tr_auc {tr_auc:.4f}  va_auc {va_auc:.4f}  va_ap {va_ap:.4f}",
              flush=True)
        ckptr.save(epoch + 1, epoch + 1, head.state_dict(), opt.state_dict(), None,
                   config={"args": vars(args)}, metric=-va_auc, better="lower")

    elapsed = time.time() - t0
    summary = {
        "exp_id": args.exp_id, "data_dir": str(data_dir), "hookpoint": args.hookpoint or "all-cat",
        "head": args.head, "epochs": args.epochs, "lr": args.lr, "batch": args.batch_size,
        "n_train": int(n_tr), "n_val": int(len(yva_t)),
        "history": history,
        "best_va_auc": max(h["va_auc"] for h in history),
        "best_va_ap": max(h["va_ap"] for h in history),
        "elapsed_s": elapsed,
    }
    out_dir = cfg.paths.report_root
    out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.checkpoint_root / args.exp_id / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
