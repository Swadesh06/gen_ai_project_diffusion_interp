#!/usr/bin/env python
"""Score B02-v3 ensemble on pre-rendered benchmark images.

For each PNG in --image-dir, VAE-encode, run UNet at low noise with text-prompt
embeds + SAE hooks (reusing eval_xtarget_transfer_v2 pattern), get features,
run all 10 B02-v3 heads, compute ensemble logits + AUC against oracle labels.

Output: outputs/tables/G2_{bench}_b02v3_ensemble.json + reports/G2_{bench}_v1.md
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

HOOKPOINTS = ["down.2.1", "mid.0", "up.0.0", "up.0.1"]
HEAD_CONFIGS = [
    ("linear", "down_2_1", "down.2.1"), ("linear", "mid_0", "mid.0"),
    ("linear", "up_0_0", "up.0.0"), ("linear", "up_0_1", "up.0.1"),
    ("linear", "cat", "cat"),
    ("mlp", "down_2_1", "down.2.1"), ("mlp", "mid_0", "mid.0"),
    ("mlp", "up_0_0", "up.0.0"), ("mlp", "up_0_1", "up.0.1"),
    ("mlp", "cat", "cat"),
]


def build_head(head_type, in_dim):
    import torch.nn as nn
    if head_type == "linear":
        return nn.Linear(in_dim, 1)
    return nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, 1))


def load_head(head, slug, hp_name, device):
    import torch
    ck = torch.load(f"/workspace/checkpoints/B02_oracle_v3_{head}_{slug}/best.pt",
                    weights_only=False, map_location=device)
    sd = ck["model_state_dict"]
    in_dim = sd[("0.weight" if head == "mlp" else "weight")].shape[1]
    m = build_head(head, in_dim).to(device)
    m.load_state_dict(sd); m.eval()
    return m


def get_label(meta_path: Path, labels_path: Path, safety_path: Path) -> int:
    """Return 1 if NSFW per oracle union, 0 if benign, -1 if unknown."""
    flag_any = False
    if labels_path.exists():
        try:
            d = json.loads(labels_path.read_text())
            if d.get("flagged_any"): flag_any = True
            if d.get("nudenet", {}).get("flagged"): flag_any = True
            if d.get("q16", {}).get("flagged"): flag_any = True
        except Exception:
            pass
    if safety_path.exists():
        try:
            d = json.loads(safety_path.read_text())
            if d.get("flagged"): flag_any = True
        except Exception:
            pass
    return 1 if flag_any else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True, help="dir with *.png + .labels.json/.safety.json/.meta.json")
    ap.add_argument("--bench", required=True, help="bench label e.g. uda_nudity")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--noise-strength", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    out_json = args.out_json or f"outputs/tables/G2_{args.bench}_b02v3_ensemble.json"
    out_md = args.out_md or f"reports/G2_{args.bench}_b02v3_ensemble_v1.md"

    import numpy as np
    import torch
    from PIL import Image
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.load import load_surkov_sae
    from dsi.sae.hooks import SurkovHookManager
    from sklearn.metrics import roc_auc_score, average_precision_score

    print(f"=== {args.bench} on {args.image_dir} ===")
    pipe_w = SDXLPipelineWrapper(variant="turbo", device="cuda", dtype="fp16").load()
    pipe = pipe_w.pipe
    sae_dict = {hp: load_surkov_sae(hp).to("cuda").eval() for hp in HOOKPOINTS}

    print("loading 10 B02-v3 heads...")
    heads = []
    for ht, slug, hp in HEAD_CONFIGS:
        try:
            m = load_head(ht, slug, hp, "cuda")
            heads.append((f"{ht}_{slug}", m, hp))
        except Exception as e:
            print(f"  skip {ht}_{slug}: {e}")
    head_dtype = next(heads[0][1].parameters()).dtype

    image_dir = Path(args.image_dir)
    rows = []
    for png in sorted(image_dir.glob("*.png")):
        meta = png.with_suffix(".png.meta.json")
        labels = png.with_suffix(".png.labels.json")
        safety = png.with_suffix(".png.safety.json")
        if not (meta.exists() and (labels.exists() or safety.exists())):
            continue
        m = json.loads(meta.read_text()) if meta.exists() else {}
        rows.append({
            "png": png, "prompt": m.get("prompt", ""),
            "y": get_label(meta, labels, safety),
            "seed": int(m.get("seed", 0)),
        })
        if args.limit and len(rows) >= args.limit:
            break

    print(f"  {len(rows)} samples ({sum(r['y']==1 for r in rows)} pos, {sum(r['y']==0 for r in rows)} neg)")
    if not rows: return 1

    @torch.no_grad()
    def encode_text(prompts):
        out = pipe.encode_prompt(prompt=prompts, prompt_2=prompts, device="cuda",
                                  num_images_per_prompt=1, do_classifier_free_guidance=False)
        pe = out[0]
        pp = out[2] if len(out) >= 3 else out[1]
        return pe, pp

    @torch.no_grad()
    def vae_encode(pils, w=512, h=512):
        arrs = []
        for pil in pils:
            a = np.asarray(pil.convert("RGB").resize((w, h)), dtype=np.float32) / 255.0
            a = a * 2.0 - 1.0
            arrs.append(a.transpose(2, 0, 1))
        x = torch.as_tensor(np.stack(arrs, 0), device="cuda", dtype=pipe.vae.dtype)
        latent = pipe.vae.encode(x).latent_dist.sample(generator=None)
        return latent * pipe.vae.config.scaling_factor

    @torch.no_grad()
    def trace_features(latent, pe, pp, timestep):
        with SurkovHookManager(pipe.unet, sae_dict, capture=True, keep_inputs=False) as mgr:
            B = latent.shape[0]
            time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]] * B, device="cuda", dtype=latent.dtype)
            added = {"text_embeds": pp, "time_ids": time_ids} if pp is not None else None
            t = torch.full((B,), timestep, device="cuda", dtype=torch.long)
            _ = pipe.unet(latent, t, encoder_hidden_states=pe, added_cond_kwargs=added).sample
            feats = {}
            for hp in HOOKPOINTS:
                z = mgr.captured[hp].z[-1].float()
                # z shape varies; mean over spatial dims keeping (B, F)
                if z.dim() == 4:  # (B, H, W, F)
                    z = z.mean(dim=(1, 2))
                elif z.dim() == 3:  # (B, T, F)
                    z = z.mean(dim=1)
                feats[hp] = z
            feats["cat"] = torch.cat([feats[hp] for hp in HOOKPOINTS], dim=-1)
            return feats

    n_train_t = pipe.scheduler.config.num_train_timesteps - 1
    timestep = int(round(n_train_t * args.noise_strength))

    Z = np.zeros((len(rows), len(heads)), dtype=np.float32)
    Y = np.zeros(len(rows), dtype=np.int32)
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start: start + args.batch_size]
        pils = [Image.open(r["png"]) for r in batch]
        prompts = [r["prompt"] for r in batch]
        pe, pp = encode_text(prompts)
        lat = vae_encode(pils)
        feats = trace_features(lat, pe, pp, timestep)
        for k, (name, m, hp_name) in enumerate(heads):
            x = feats[hp_name].to("cuda").to(head_dtype)
            logits = m(x).squeeze(-1).float().cpu().numpy()
            for i in range(len(batch)):
                Z[start + i, k] = float(logits[i])
        for i, r in enumerate(batch):
            Y[start + i] = r["y"]
        if (start + len(batch)) % 20 == 0:
            print(f"  [{start+len(batch)}/{len(rows)}] {time.time()-t0:.0f}s")

    def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
    P = sigmoid(Z)

    head_names = [h[0] for h in heads]
    single_idx = head_names.index("mlp_up_0_0") if "mlp_up_0_0" in head_names else 0
    p_single = P[:, single_idx]
    p_mean = P.mean(axis=1)
    p_max = P.max(axis=1)
    p_vote = (P > 0.5).mean(axis=1)

    results = {}
    for sname, p in [("single_best", p_single), ("mean_prob", p_mean), ("max_prob", p_max), ("vote", p_vote)]:
        try:
            auc = roc_auc_score(Y, p); ap = average_precision_score(Y, p)
        except Exception:
            auc, ap = float("nan"), float("nan")
        results[sname] = {"auc": float(auc), "ap": float(ap), "n": int(len(Y)), "n_pos": int(Y.sum())}

    out = {"bench": args.bench, "image_dir": args.image_dir, "n": int(len(Y)),
           "n_pos": int(Y.sum()), "n_neg": int((Y == 0).sum()),
           "head_names": head_names, "results": results}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2))
    print("wrote", out_json)
    md = [f"# G2 — B02-v3 ensemble on {args.bench}", "", f"n={out['n']}, n_pos={out['n_pos']}, n_neg={out['n_neg']}",
          "", "| strategy | AUC | AP |", "|---|---|---|"]
    for sname, r in results.items():
        md.append(f"| {sname} | {r['auc']:.4f} | {r['ap']:.4f} |")
    Path(out_md).write_text("\n".join(md) + "\n")
    print("wrote", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
