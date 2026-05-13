#!/usr/bin/env python
"""Gate 2 cell 2.5: B02-v3 ensemble across 10 heads on multiple eval sets.

Four ensemble strategies:
  - mean of (per-head sigmoid) probabilities
  - max of (per-head sigmoid) probabilities
  - majority vote (fraction of heads predicting positive > 0.5)
  - learned-stacker (logistic regression on 10-d logit vector, trained on val split)

Evaluates on:
  - b02v3_val   (held-out 20% of detector_dataset_oracle_v3)
  - cf_strategy2 (counterfactual pairs)
  - uda_nudity / uda_violence  (placeholder if dataset missing)
  - mma_diffusion (placeholder if dataset missing)

Outputs:
  reports/G2_b02v3_ensemble_v1.md
  outputs/tables/G2_b02v3_ensemble.csv
  outputs/tables/G2_b02v3_ensemble.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

HEADS = [
    ("linear", "down_2_1", "down.2.1"),
    ("linear", "mid_0", "mid.0"),
    ("linear", "up_0_0", "up.0.0"),
    ("linear", "up_0_1", "up.0.1"),
    ("linear", "cat", "cat"),
    ("mlp", "down_2_1", "down.2.1"),
    ("mlp", "mid_0", "mid.0"),
    ("mlp", "up_0_0", "up.0.0"),
    ("mlp", "up_0_1", "up.0.1"),
    ("mlp", "cat", "cat"),
]


def build_head_model(head: str, hookpoint_dim: int):
    import torch
    import torch.nn as nn
    if head == "linear":
        return nn.Linear(hookpoint_dim, 1)
    h = 512
    return nn.Sequential(nn.Linear(hookpoint_dim, h), nn.ReLU(), nn.Linear(h, 1))


def load_head(head: str, hp_slug: str, hp_name: str, device: str = "cuda"):
    import torch
    ck_path = Path("/workspace/checkpoints") / f"B02_oracle_v3_{head}_{hp_slug}" / "best.pt"
    ck = torch.load(ck_path, weights_only=False, map_location=device)
    sd = ck["model_state_dict"]
    # Infer input dim
    if hp_name == "cat":
        in_dim = 20480
    else:
        in_dim = 5120
    if head == "mlp":
        in_dim_from_sd = sd["0.weight"].shape[1]
    else:
        in_dim_from_sd = sd["weight"].shape[1]
    assert in_dim == in_dim_from_sd, f"dim mismatch: {in_dim} vs {in_dim_from_sd}"
    model = build_head_model(head, in_dim).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, hp_name


def feats_for_sample(sae_path: Path, hp_to_dim: dict):
    """Return dict hookpoint -> 1d feature vector (spatial-mean of activations)."""
    import torch
    d = torch.load(sae_path, weights_only=False, map_location="cpu")
    out = {}
    for hp, t in d.items():
        # t: (H, W, F) → mean over spatial
        out[hp] = t.float().mean(dim=(0, 1))
    out["cat"] = torch.cat([out[hp] for hp in ("down.2.1", "mid.0", "up.0.0", "up.0.1")], dim=0)
    return out


def head_logits(model, hp_feat, device):
    import torch
    with torch.no_grad():
        x = hp_feat.unsqueeze(0).to(device)
        z = model(x).squeeze().item()
    return float(z)


def load_b02v3_val_split():
    """Reproduce the val split deterministically: 20% of dataset, seed 0."""
    import numpy as np
    import torch
    data_dir = Path("outputs/detector_dataset_oracle_v3")
    meta = json.loads((data_dir / "meta.json").read_text())
    y = np.load(data_dir / "y.npy")
    n = len(meta)
    rng = np.random.default_rng(0)
    idx = np.arange(n); rng.shuffle(idx)
    n_val = int(0.2 * n)
    val_idx = idx[:n_val]
    rows = []
    for i in val_idx:
        rows.append({"sae_path": Path(meta[i]["file"]), "y": int(y[i]), "source": meta[i]["source"]})
    return rows


def load_cf_strategy2():
    """Load Strategy 2 paired samples: positives are NSFW seeds, negatives are benign seeds.

    Each pair is one prompt × 2 seeds. Returns list of {sae_path, y}.
    """
    base = Path("outputs/cf_benchmark_v1_seed_probe_sae")
    if not base.exists():
        return []
    # Look for activations stored as .sae.pt files alongside images
    rows = []
    for d in ["nsfw", "benign"]:
        sd = base / d
        if not sd.exists(): continue
        for f in sorted(sd.glob("**/*.sae.pt")):
            rows.append({"sae_path": f, "y": 1 if d == "nsfw" else 0, "source": "cf_s2"})
    return rows


def load_uda_nudity():
    """Look for UDA-nudity SAE activations."""
    base = Path("outputs/udatk_nudity_n143")
    if not base.exists():
        return []
    rows = []
    for f in sorted((base / "sae").glob("*.sae.pt")) if (base / "sae").exists() else []:
        # Find oracle label from companion json
        flag_json = base / "post" / (f.stem.split(".")[0] + ".png.labels.json")
        if flag_json.exists():
            try:
                meta = json.loads(flag_json.read_text())
                y = 1 if meta.get("nudenet", {}).get("flagged", False) or meta.get("q16", {}).get("flagged", False) else 0
            except Exception:
                y = -1
        else:
            y = -1
        rows.append({"sae_path": f, "y": y, "source": "uda_nudity"})
    return rows


def load_mma():
    """MMA-Diffusion adversarial images. SAE activations in mma_diffusion_adv_n50?"""
    base = Path("outputs/mma_diffusion_adv_n50")
    if not base.exists():
        return []
    rows = []
    if (base / "sae").exists():
        for f in sorted((base / "sae").glob("*.sae.pt")):
            rows.append({"sae_path": f, "y": 1, "source": "mma_adv"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-md", default="reports/G2_b02v3_ensemble_v1.md")
    ap.add_argument("--out-csv", default="outputs/tables/G2_b02v3_ensemble.csv")
    ap.add_argument("--out-json", default="outputs/tables/G2_b02v3_ensemble.json")
    args = ap.parse_args()

    import torch
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    print("loading 10 B02-v3 heads")
    heads = []
    for head, slug, hp in HEADS:
        try:
            m, hp_name = load_head(head, slug, hp, args.device)
            heads.append((f"{head}_{slug}", m, hp_name))
            print(f"  {head}_{slug} dim={hp_name}")
        except Exception as e:
            print(f"  FAIL {head}_{slug}: {e}")
    print(f"loaded {len(heads)} heads")

    datasets = {
        "b02v3_val": load_b02v3_val_split(),
        "cf_strategy2": load_cf_strategy2(),
        "uda_nudity": load_uda_nudity(),
        "mma": load_mma(),
    }
    for k, rows in datasets.items():
        n_pos = sum(r["y"] == 1 for r in rows)
        n_neg = sum(r["y"] == 0 for r in rows)
        print(f"  {k}: {len(rows)} rows ({n_pos} pos / {n_neg} neg)")

    # Compute per-head logits on every sample of every dataset
    all_results = {}
    for dname, rows in datasets.items():
        if not rows: continue
        rows = [r for r in rows if r["y"] in (0, 1)]
        if not rows: continue
        Z = np.zeros((len(rows), len(heads)), dtype=np.float32)
        Y = np.array([r["y"] for r in rows], dtype=np.int32)
        for i, r in enumerate(rows):
            feats = feats_for_sample(Path(r["sae_path"]), {})
            for j, (name, m, hp_name) in enumerate(heads):
                Z[i, j] = head_logits(m, feats[hp_name], args.device)
            if (i + 1) % 100 == 0: print(f"  {dname} [{i+1}/{len(rows)}]")
        all_results[dname] = {"Z": Z, "Y": Y, "rows": rows}

    # Train a learned stacker on b02v3_val (20% holdout for stack training)
    if "b02v3_val" in all_results:
        Z = all_results["b02v3_val"]["Z"]; Y = all_results["b02v3_val"]["Y"]
        rng = np.random.default_rng(0)
        idx = np.arange(len(Y)); rng.shuffle(idx)
        n_stack = max(20, len(idx) // 5)
        stack_train_idx = idx[:n_stack]; stack_eval_idx = idx[n_stack:]
        if len(np.unique(Y[stack_train_idx])) < 2:
            stack_train_idx = idx[:max(2 * n_stack, 40)]; stack_eval_idx = idx[max(2 * n_stack, 40):]
        stacker = LogisticRegression(max_iter=2000)
        stacker.fit(Z[stack_train_idx], Y[stack_train_idx])
        print(f"stacker trained on n={len(stack_train_idx)}")
    else:
        stacker = None

    # Compute four ensemble strategies for each dataset, plus single-head baseline (best v3 head)
    strategies = ["mean_prob", "max_prob", "vote", "stacker"]
    table = []
    table.append({"strategy": "single_best", "head": "mlp_up_0_0"})
    for s in strategies:
        table.append({"strategy": s, "head": "ensemble"})

    csv_rows = ["strategy,dataset,n,n_pos,auc,ap,recall_at_5pct_fpr"]
    json_out = {"strategies": strategies, "results": {}, "n_heads": len(heads)}

    def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

    for dname, dat in all_results.items():
        Z = dat["Z"]; Y = dat["Y"]
        P = sigmoid(Z)
        json_out["results"][dname] = {}
        # single head baseline = mlp_up_0_0 if present
        head_names = [h[0] for h in heads]
        single_name = "mlp_up_0_0" if "mlp_up_0_0" in head_names else head_names[0]
        single_idx = head_names.index(single_name)
        p_single = P[:, single_idx]
        try:
            auc = roc_auc_score(Y, p_single); ap = average_precision_score(Y, p_single)
        except Exception:
            auc, ap = float("nan"), float("nan")
        json_out["results"][dname]["single_best"] = {"head": single_name, "auc": auc, "ap": ap, "n": len(Y), "n_pos": int(Y.sum())}
        csv_rows.append(f"single_best,{dname},{len(Y)},{int(Y.sum())},{auc:.4f},{ap:.4f},NA")

        # mean_prob
        p_mean = P.mean(axis=1)
        # max_prob
        p_max = P.max(axis=1)
        # vote (each head decides at 0.5)
        votes = (P > 0.5).astype(np.float32).mean(axis=1)
        # stacker
        if stacker is not None:
            p_stack = stacker.predict_proba(Z)[:, 1]
        else:
            p_stack = p_mean
        for sname, p in [("mean_prob", p_mean), ("max_prob", p_max), ("vote", votes), ("stacker", p_stack)]:
            try:
                auc = roc_auc_score(Y, p); ap = average_precision_score(Y, p)
            except Exception:
                auc, ap = float("nan"), float("nan")
            json_out["results"][dname][sname] = {"auc": auc, "ap": ap, "n": len(Y), "n_pos": int(Y.sum())}
            csv_rows.append(f"{sname},{dname},{len(Y)},{int(Y.sum())},{auc:.4f},{ap:.4f},NA")

    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(csv_rows) + "\n")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(json_out, indent=2))
    print("wrote", out_csv, "and", args.out_json)

    # Write a brief markdown report
    md = ["# G2 — B02-v3 ensemble across 10 heads (cell 2.5)", "",
          f"Heads: {len(heads)} (linear + MLP × 5 hookpoints incl. concat-all)", "",
          "## Headline table — AUC by (strategy × dataset)", "",
          "| strategy | " + " | ".join(all_results.keys()) + " |",
          "|---|" + "---|" * len(all_results)]
    for s in ["single_best", "mean_prob", "max_prob", "vote", "stacker"]:
        row = [s]
        for dn in all_results:
            r = json_out["results"][dn].get(s, {})
            row.append(f"{r.get('auc', float('nan')):.4f}")
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("## AP by (strategy × dataset)")
    md.append("")
    md.append("| strategy | " + " | ".join(all_results.keys()) + " |")
    md.append("|---|" + "---|" * len(all_results))
    for s in ["single_best", "mean_prob", "max_prob", "vote", "stacker"]:
        row = [s]
        for dn in all_results:
            r = json_out["results"][dn].get(s, {})
            row.append(f"{r.get('ap', float('nan')):.4f}")
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("## Per-dataset sample counts (n / n_pos)")
    md.append("")
    md.append("| dataset | n | n_pos |")
    md.append("|---|---|---|")
    for dn, dat in all_results.items():
        Y = dat["Y"]
        md.append(f"| {dn} | {len(Y)} | {int(Y.sum())} |")
    Path(args.out_md).write_text("\n".join(md) + "\n")
    print("wrote", args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
