#!/usr/bin/env python
"""Counterfactual benchmark probe — the framing-discriminator number.

Item 1c-0 evaluation downstream. Once `outputs/cf_benchmark_v1/` has
oracle-validated pairs and SAE/raw features captured for both pre and
post images, train a probe on three of the four (cluster × side) cells
and test on the held-out cell. Per-cell AUC is the framing-decision
input.

This script:
  1. Reads `outputs/cf_benchmark_v1/validated.jsonl` (cluster, side).
  2. For each pair, encodes pre and post images via VAE → UNet single
     forward → SurkovHookManager z capture → mean-pool to per-hookpoint
     vectors.
  3. Per-cell-out: train probe on 3 cells, test on 4th. Report AUC.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

HOOKPOINTS = ("down.2.1", "mid.0", "up.0.0", "up.0.1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cf-dir", default="outputs/cf_benchmark_v1")
    ap.add_argument("--out-dir", default="outputs/cf_benchmark_v1_probe")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--feature-source", choices=["sae", "raw"], default="raw")
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    cf_dir = Path(args.cf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_path = cf_dir / "validated.jsonl"
    if not val_path.exists():
        print(f"no validated.jsonl at {val_path} — run build_cf_strategy1.py validate first")
        return 2
    pairs = [json.loads(l) for l in val_path.read_text().splitlines()]
    pairs = [p for p in pairs if p.get("validated")]
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"loaded {len(pairs)} validated pairs")

    if not pairs:
        print("no validated pairs — cf-strategy1 build needs to finish + cpu-workers need to label")
        return 2

    print("loading SDXL Turbo + 4 Surkov SAEs", flush=True)
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.hooks import SurkovHookManager
    from dsi.sae.load import load_surkov_sae

    pipe_w = SDXLPipelineWrapper(variant="turbo", device=args.device, dtype="fp16").load()
    sae_dict = {hp: load_surkov_sae(hp).to(args.device).eval() for hp in HOOKPOINTS}

    @torch.no_grad()
    def encode_features(pil_imgs):
        import numpy as np
        # VAE encode at 512
        arrs = []
        for pil in pil_imgs:
            a = np.asarray(pil.convert("RGB").resize((512, 512)), dtype=np.float32) / 255.0
            arrs.append((a * 2.0 - 1.0).transpose(2, 0, 1))
        x = torch.as_tensor(np.stack(arrs, 0), device=args.device, dtype=pipe_w.vae.dtype)
        latent = pipe_w.vae.encode(x).latent_dist.sample()
        latent = latent * pipe_w.vae.config.scaling_factor
        latent = latent.to(next(pipe_w.unet.parameters()).dtype)

        B = latent.shape[0]
        # Empty prompt SDXL conditioning
        pe, neg_pe, pooled, neg_pooled = pipe_w.pipe.encode_prompt(
            prompt=[""] * B, prompt_2=[""] * B, device=args.device,
            num_images_per_prompt=1, do_classifier_free_guidance=False,
        )
        time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]] * B,
                                device=args.device, dtype=latent.dtype)

        with SurkovHookManager(pipe_w.unet, sae_dict, capture=True, keep_inputs=False) as mgr:
            t = torch.full((B,), 50, device=args.device, dtype=torch.long)
            pipe_w.unet(latent, t, encoder_hidden_states=pe,
                        added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
        feats_per_hp = {}
        for hp in HOOKPOINTS:
            zs = mgr.captured[hp].z
            v = zs[-1].float()
            spatial = tuple(range(1, v.ndim - 1))
            feats_per_hp[hp] = v.mean(dim=spatial).cpu().numpy()
        return feats_per_hp

    # Encode all pairs
    rows = []
    t0 = time.time()
    BATCH = 4
    pre_paths = [cf_dir / "pre" / f"{p['pair_id']}.png" for p in pairs]
    post_paths = [cf_dir / "post" / f"{p['pair_id']}.png" for p in pairs]
    pair_keys = []
    feats_pre: dict = {hp: [] for hp in HOOKPOINTS}
    feats_post: dict = {hp: [] for hp in HOOKPOINTS}
    pair_meta: list = []

    for start in range(0, len(pairs), BATCH):
        batch = pairs[start:start + BATCH]
        pre = []
        post = []
        keep_idx = []
        for k, p in enumerate(batch):
            pre_p = pre_paths[start + k]
            post_p = post_paths[start + k]
            if not (pre_p.exists() and post_p.exists()):
                continue
            try:
                pre.append(Image.open(pre_p).convert("RGB"))
                post.append(Image.open(post_p).convert("RGB"))
                keep_idx.append(k)
            except Exception:
                continue
        if not pre:
            continue
        pre_feats = encode_features(pre)
        post_feats = encode_features(post)
        for hp in HOOKPOINTS:
            for v in pre_feats[hp]:
                feats_pre[hp].append(v)
            for v in post_feats[hp]:
                feats_post[hp].append(v)
        for k_local, k_orig in enumerate(keep_idx):
            pair_meta.append({"pair_id": batch[k_orig]["pair_id"],
                              "cluster": batch[k_orig]["cluster"]})
        if (start // BATCH) % 10 == 0:
            print(f"  encoded {len(pair_meta)} / {len(pairs)} ({time.time()-t0:.0f}s)", flush=True)

    n = len(pair_meta)
    print(f"encoded {n} pairs total")
    if n == 0:
        return 2

    # Build feature matrix per hookpoint, label vector by side (pre=1 unsafe, post=0 safe), cluster vector
    X_per_hp = {hp: np.stack(feats_pre[hp] + feats_post[hp], axis=0) for hp in HOOKPOINTS}
    y = np.array([1] * n + [0] * n, dtype="int64")
    cluster = np.array([m["cluster"] for m in pair_meta] + [m["cluster"] for m in pair_meta])

    # Concat across hookpoints
    X_cat = np.concatenate([X_per_hp[hp] for hp in HOOKPOINTS], axis=1)

    # Per-cluster leave-one-out: train on 3 clusters, test on 4th
    clusters_present = sorted(set(cluster))
    print(f"clusters: {clusters_present}")
    results = {"per_cluster_loo": {}, "in_distribution": {}}
    for held_out in clusters_present:
        train_mask = cluster != held_out
        test_mask = cluster == held_out
        if test_mask.sum() < 5 or train_mask.sum() < 5:
            print(f"  skipping {held_out} (too few samples train={train_mask.sum()} test={test_mask.sum()})")
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_cat[train_mask], y[train_mask])
        scores = clf.predict_proba(X_cat[test_mask])[:, 1]
        try:
            auc = float(roc_auc_score(y[test_mask], scores))
            ap = float(average_precision_score(y[test_mask], scores))
        except Exception:
            auc = ap = float("nan")
        print(f"  held-out cluster='{held_out}'  n_test={int(test_mask.sum())}  AUC={auc:.4f}  AP={ap:.4f}")
        results["per_cluster_loo"][held_out] = {
            "auc": auc, "ap": ap, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        }

    # In-distribution 80/20 split for reference
    rng = np.random.default_rng(0)
    perm = rng.permutation(2 * n)
    cut = int(0.8 * 2 * n)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_cat[perm[:cut]], y[perm[:cut]])
    scores = clf.predict_proba(X_cat[perm[cut:]])[:, 1]
    auc = float(roc_auc_score(y[perm[cut:]], scores))
    ap = float(average_precision_score(y[perm[cut:]], scores))
    results["in_distribution"]["auc"] = auc
    results["in_distribution"]["ap"] = ap
    print(f"  in-distribution 80/20 AUC={auc:.4f} AP={ap:.4f}")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
