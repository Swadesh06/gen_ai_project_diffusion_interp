#!/usr/bin/env python
"""Counterfactual probe variant for Strategy 2 (same-prompt seed pairs).

The Strategy 2 benchmark has format `outputs/cf_benchmark_v1_seed/validated.jsonl`
with rows {prompt_id, prompt, flagged_seed, unflagged_seed, flagged_path,
unflagged_path}. For each pair, encode flagged + unflagged via the actual
prompt's UNet trace (not empty), then probe.

Per-prompt leave-one-out (LOPO): train on n-1 prompts, test on held-out
prompt. AUC mean ± std over LOPO is the framing-discriminator number.

Output: outputs/cf_benchmark_v1_seed_probe/results.json
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
    ap.add_argument("--cf-dir", default="outputs/cf_benchmark_v1_seed")
    ap.add_argument("--out-dir", default="outputs/cf_benchmark_v1_seed_probe")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--feature-source", choices=["raw", "sae"], default="raw")
    ap.add_argument("--use-actual-prompt", action="store_true",
                    help="encode with the actual prompt vs empty conditioning")
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
        print(f"no validated.jsonl at {val_path}")
        return 2
    pairs = [json.loads(l) for l in val_path.read_text().splitlines()][: args.max_pairs]
    print(f"loaded {len(pairs)} validated pairs")

    if not pairs:
        return 2

    print("loading SDXL Turbo + 4 Surkov SAEs", flush=True)
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.hooks import SurkovHookManager
    from dsi.sae.load import load_surkov_sae

    pipe_w = SDXLPipelineWrapper(variant="turbo", device=args.device, dtype="fp16").load()
    sae_dict = {hp: load_surkov_sae(hp).to(args.device).eval() for hp in HOOKPOINTS}

    @torch.no_grad()
    def encode_features(pil_imgs, prompt_text: str = ""):
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

        prompts = [prompt_text or ""] * B
        pe, neg_pe, pooled, neg_pooled = pipe_w.pipe.encode_prompt(
            prompt=prompts, prompt_2=prompts,
            device=args.device, num_images_per_prompt=1,
            do_classifier_free_guidance=False,
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

    # Encode ALL flagged + unflagged
    flag_feats: dict[str, list] = {hp: [] for hp in HOOKPOINTS}
    unflag_feats: dict[str, list] = {hp: [] for hp in HOOKPOINTS}
    pair_meta: list = []

    BATCH = 4
    t0 = time.time()
    for start in range(0, len(pairs), BATCH):
        batch = pairs[start:start + BATCH]
        flag_pils = []
        unflag_pils = []
        prompt_for_batch = ""
        ok_idxs = []
        for k, p in enumerate(batch):
            fp = Path(p["flagged_path"])
            up = Path(p["unflagged_path"])
            if fp.exists() and up.exists():
                try:
                    flag_pils.append(Image.open(fp).convert("RGB"))
                    unflag_pils.append(Image.open(up).convert("RGB"))
                    prompt_for_batch = p["prompt"]  # last in batch
                    ok_idxs.append(k)
                except Exception:
                    continue
        if not flag_pils:
            continue
        try:
            f_feats = encode_features(flag_pils, prompt_for_batch if args.use_actual_prompt else "")
            u_feats = encode_features(unflag_pils, prompt_for_batch if args.use_actual_prompt else "")
        except Exception as e:
            print(f"  encode error: {e}", flush=True)
            continue
        for hp in HOOKPOINTS:
            for v in f_feats[hp]:
                flag_feats[hp].append(v)
            for v in u_feats[hp]:
                unflag_feats[hp].append(v)
        for k in ok_idxs:
            pair_meta.append({"prompt_id": batch[k]["prompt_id"],
                              "prompt": batch[k]["prompt"]})
        if (start // BATCH) % 10 == 0:
            print(f"  encoded {len(pair_meta)} / {len(pairs)} pairs ({time.time()-t0:.0f}s)",
                  flush=True)

    n = len(pair_meta)
    print(f"encoded {n} pairs total")
    if n < 5:
        return 2

    # Build feature matrix per hookpoint, label = 1 for flagged side, 0 for unflagged
    X_per_hp = {hp: np.stack(flag_feats[hp] + unflag_feats[hp], axis=0) for hp in HOOKPOINTS}
    y = np.array([1] * n + [0] * n, dtype="int64")
    prompt_ids = np.array([m["prompt_id"] for m in pair_meta] + [m["prompt_id"] for m in pair_meta])

    X_cat = np.concatenate([X_per_hp[hp] for hp in HOOKPOINTS], axis=1)

    # In-distribution 80/20
    rng = np.random.default_rng(0)
    perm = rng.permutation(2 * n)
    cut = int(0.8 * 2 * n)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_cat[perm[:cut]], y[perm[:cut]])
    scores = clf.predict_proba(X_cat[perm[cut:]])[:, 1]
    in_auc = float(roc_auc_score(y[perm[cut:]], scores))
    in_ap = float(average_precision_score(y[perm[cut:]], scores))
    print(f"  in-distribution 80/20 AUC={in_auc:.4f} AP={in_ap:.4f}")

    # Per-prompt leave-one-out (LOPO)
    unique_prompts = np.unique(prompt_ids)
    if len(unique_prompts) >= 5:
        per_prompt_aucs = []
        for held in unique_prompts:
            train_mask = prompt_ids != held
            test_mask = prompt_ids == held
            if test_mask.sum() < 2 or train_mask.sum() < 5:
                continue
            try:
                clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                clf.fit(X_cat[train_mask], y[train_mask])
                scores = clf.predict_proba(X_cat[test_mask])[:, 1]
                if len(np.unique(y[test_mask])) >= 2:
                    auc = float(roc_auc_score(y[test_mask], scores))
                    per_prompt_aucs.append({"prompt_id": str(held), "auc": auc,
                                             "n_test": int(test_mask.sum())})
            except Exception:
                continue
        if per_prompt_aucs:
            aucs = [r["auc"] for r in per_prompt_aucs]
            mean_lopo = float(np.mean(aucs))
            std_lopo = float(np.std(aucs))
            print(f"  LOPO mean AUC = {mean_lopo:.4f} ± {std_lopo:.4f} (n_prompts = {len(aucs)})")
        else:
            mean_lopo = std_lopo = float("nan")
    else:
        per_prompt_aucs = []
        mean_lopo = std_lopo = float("nan")

    results = {
        "n_pairs": n,
        "n_unique_prompts": int(len(unique_prompts)),
        "in_distribution": {"auc": in_auc, "ap": in_ap},
        "lopo_mean_auc": mean_lopo,
        "lopo_std_auc": std_lopo,
        "per_prompt_aucs": per_prompt_aucs[:50],
        "feature_source": args.feature_source,
        "use_actual_prompt": args.use_actual_prompt,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
