"""Find SAEUron features that activate differentially on nudity vs benign.

Loads SAEUron SAE (bcywinski/SAeUron_coco), encodes a small batch of
NSFW vs benign image activations from SD v1.4 at the upstream
hookpoint `unet.up_blocks.1.attentions.1`, computes per-feature
Fisher ratio, and reports the top-20 candidates.

Usage:
    python scripts/saeuron_feature_contrast.py \
        --nsfw-dir <path> --benign-dir <path> --n-each 30
"""
from pathlib import Path
import argparse, json, sys, time
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-each", type=int, default=30)
    ap.add_argument("--out-path", default="outputs/saeuron_feature_contrast.json")
    args = ap.parse_args()

    sys.path.insert(0, "/workspace/datasets/SAeUron")
    print("loading SD v1.4 + SAEUron pipeline")
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # load SAEUron SAE — use the published hookpoint
    print("loading SAEUron SAE checkpoint")
    from SAE.sae import Sae
    sae = Sae.load_from_hub(
        "bcywinski/SAeUron_coco",
        hookpoint="unet.up_blocks.1.attentions.1",
        device="cuda",
    ).eval()
    print(f"  Loaded SAE; n_features = {sae.encoder.weight.shape[0]}")

    # render nudity + benign and capture activations at up.1.1 hookpoint
    from dsi.data.i2p import i2p_nsfw_subset
    from dsi.data.coco import load_coco_captions

    nsfw_prompts = i2p_nsfw_subset(limit=args.n_each)
    benign_prompts = load_coco_captions(limit=args.n_each)

    # capture activations
    captured_nsfw, captured_benign = [], []
    
    def make_hook(target_list):
        def hook(module, args, kwargs, output):
            inp = args[0] if args else kwargs.get("hidden_states")
            if inp is None: return None
            # SAEUron expects (batch, sample_size, emb_size)
            if inp.dim() == 4:
                # (B, C, H, W) -> (B, H*W, C)
                B, C, H, W = inp.shape
                inp_flat = inp.permute(0, 2, 3, 1).reshape(B, H*W, C)
            else:
                inp_flat = inp
            with torch.no_grad():
                eo = sae.encode(inp_flat)
            # eo is EncoderOutput; access .top_acts or similar
            if hasattr(eo, "top_acts"):
                # SAEUron returns sparse (top_acts, top_indices, ...) — convert to dense via index_add
                top_acts = eo.top_acts  # (B, S, k)
                top_idx = eo.top_indices  # (B, S, k)
                B, S, k = top_acts.shape
                F = sae.encoder.weight.shape[0]
                z_dense = torch.zeros(B, S, F, device=top_acts.device)
                z_dense.scatter_add_(2, top_idx, top_acts)
                z_per_sample = z_dense.mean(dim=1)  # (B, F)
            else:
                z = eo
                z_per_sample = z.float().mean(dim=tuple(range(1, z.ndim - 1)))
            target_list.append(z_per_sample.cpu())
            return None
        return hook

    target_module = pipe.unet.up_blocks[1].attentions[1]

    print("rendering NSFW prompts with hook")
    handle = target_module.register_forward_hook(make_hook(captured_nsfw), with_kwargs=True)
    for i, p in enumerate(nsfw_prompts):
        gen = torch.Generator(device="cuda").manual_seed(420000 + i)
        _ = pipe(prompt=p.text, num_inference_steps=20, generator=gen, height=512, width=512).images[0]
        if (i + 1) % 5 == 0:
            print(f"  nsfw [{i+1}/{len(nsfw_prompts)}]")
    handle.remove()

    print("rendering benign prompts with hook")
    handle = target_module.register_forward_hook(make_hook(captured_benign), with_kwargs=True)
    for i, p in enumerate(benign_prompts):
        gen = torch.Generator(device="cuda").manual_seed(430000 + i)
        _ = pipe(prompt=p.text, num_inference_steps=20, generator=gen, height=512, width=512).images[0]
        if (i + 1) % 5 == 0:
            print(f"  benign [{i+1}/{len(benign_prompts)}]")
    handle.remove()

    import numpy as np
    nsfw_z = torch.cat(captured_nsfw, dim=0).numpy()  # (N*T, F)
    benign_z = torch.cat(captured_benign, dim=0).numpy()
    print(f"  nsfw_z shape: {nsfw_z.shape}, benign_z shape: {benign_z.shape}")

    mu_n = nsfw_z.mean(axis=0)
    mu_b = benign_z.mean(axis=0)
    var_n = nsfw_z.var(axis=0) + 1e-6
    var_b = benign_z.var(axis=0) + 1e-6
    fisher = (mu_n - mu_b) ** 2 / (var_n + var_b)
    top20 = np.argsort(-fisher)[:20]
    
    print(f"  top-20 features by Fisher ratio (nsfw vs benign):")
    for i, idx in enumerate(top20):
        print(f"    rank {i+1}: feature {int(idx)}, Fisher={fisher[idx]:.4f}, mu_nsfw={mu_n[idx]:.3f}, mu_benign={mu_b[idx]:.3f}")
    
    out = {
        "top20_features": [int(x) for x in top20],
        "top20_fishers": [float(fisher[x]) for x in top20],
        "top20_mu_nsfw": [float(mu_n[x]) for x in top20],
        "top20_mu_benign": [float(mu_b[x]) for x in top20],
        "n_nsfw": int(args.n_each),
        "n_benign": int(args.n_each),
    }
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(json.dumps(out, indent=2))
    print(f"  wrote {args.out_path}")


if __name__ == "__main__":
    main()
