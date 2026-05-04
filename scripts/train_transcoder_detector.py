#!/usr/bin/env python
"""Phase C-9 — transcoder detector (Dunefsky et al. 2024).

A transcoder predicts the next-block residual diff from the previous-block
residual diff via a small MLP / linear map. The reconstruction error is
the detector signal.

Hypothesis: NSFW vs benign generations have different cross-layer linear
relationships; the transcoder trained on benign predicts benign residuals
well and NSFW residuals poorly → reconstruction error is a detector.

Comparison: SAE detector (D01) is at-the-block reconstruction; transcoder
is between-block reconstruction. Different signal.

Pairs we train:
  down.2.1 -> mid.0
  mid.0    -> up.0.0
  up.0.0   -> up.0.1

Uses dataset_axbench_v1 raw activations.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="outputs/dataset_axbench_v1")
    ap.add_argument("--out-dir", default="outputs/C09_transcoder")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import numpy as np, torch
    from torch import nn
    from sklearn.metrics import roc_auc_score, average_precision_score

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    y = np.load(data_dir / "y.npy")
    HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")
    raws = {hp: np.load(data_dir / f"X_raw_{hp.replace('.','_')}.npy") for hp in HOOKPOINTS}
    print(f"y: {(y==1).sum()}/{(y==0).sum()}")
    for hp, x in raws.items():
        print(f"  raw {hp}: {x.shape}")

    pairs = [("down.2.1", "mid.0"), ("mid.0", "up.0.0"), ("up.0.0", "up.0.1")]
    benign = (y == 0)

    def run_pair(src, dst):
        Xs = raws[src]; Xd = raws[dst]
        # transcoder = small MLP src -> dst, trained on benign only
        tx_train = Xs[benign]; ty_train = Xd[benign]
        # 80/20 within-benign split
        n_b = len(tx_train)
        rng = np.random.default_rng(42)
        perm = rng.permutation(n_b); cut = int(0.8 * n_b)
        tr_x = torch.as_tensor(tx_train[perm[:cut]], dtype=torch.float32, device=args.device)
        tr_y = torch.as_tensor(ty_train[perm[:cut]], dtype=torch.float32, device=args.device)
        va_x = torch.as_tensor(tx_train[perm[cut:]], dtype=torch.float32, device=args.device)
        va_y = torch.as_tensor(ty_train[perm[cut:]], dtype=torch.float32, device=args.device)

        net = nn.Sequential(
            nn.Linear(Xs.shape[1], 512), nn.ReLU(),
            nn.Linear(512, Xd.shape[1])
        ).to(args.device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.MSELoss()
        for ep in range(args.epochs):
            net.train()
            ix = torch.randperm(len(tr_x), device=args.device)
            ep_loss = 0.0
            for i in range(0, len(tr_x), 64):
                sl = ix[i:i+64]
                opt.zero_grad()
                loss = crit(net(tr_x[sl]), tr_y[sl])
                loss.backward(); opt.step()
                ep_loss += loss.item()
            if ep == args.epochs - 1:
                net.eval()
                with torch.no_grad():
                    va_pred = net(va_x)
                    val_mse = ((va_pred - va_y) ** 2).mean().item()
                print(f"  {src}->{dst}: ep{ep} train_loss={ep_loss/len(tr_x):.4f} val_mse={val_mse:.4f}")

        # detector signal: per-sample reconstruction error
        net.eval()
        all_x = torch.as_tensor(Xs, dtype=torch.float32, device=args.device)
        all_y = torch.as_tensor(Xd, dtype=torch.float32, device=args.device)
        with torch.no_grad():
            pred = net(all_x)
            err = ((pred - all_y) ** 2).mean(dim=-1).cpu().numpy()  # (N,)
        # AUC of err vs label
        auc = float(roc_auc_score(y, err))
        ap = float(average_precision_score(y, err))
        return {"src": src, "dst": dst, "auc": auc, "ap": ap, "n_train": cut, "n_val": n_b - cut,
                "val_mse": val_mse}

    results = {"pairs": [run_pair(s, d) for s, d in pairs]}

    # combine all pairs by averaging recon errors
    combined_err = None
    for s, d in pairs:
        Xs = torch.as_tensor(raws[s], dtype=torch.float32)
        Xd = torch.as_tensor(raws[d], dtype=torch.float32)
        # quickly retrain so we have the net (would be cleaner to cache, but small)
        # Instead, just record per-pair AUC; combined is the sum-of-squares MSE
    print("\n=== per-pair AUCs ===")
    for r in results["pairs"]:
        print(f"  {r['src']}->{r['dst']}: AUC={r['auc']:.4f} AP={r['ap']:.4f}")

    (out_dir / "transcoder_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir / 'transcoder_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
