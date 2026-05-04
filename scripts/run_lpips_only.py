#!/usr/bin/env python
"""C-10 lightweight: LPIPS-vgg on aligned (pre, post) image pairs.

Skip DreamSim (it hung previously). Just compute LPIPS distance per
(pre, post) pair and report mean ± std.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--pre-dir", required=True)
ap.add_argument("--post-dir", required=True)
ap.add_argument("--out-name", default="lpips.json")
ap.add_argument("--max-pairs", type=int, default=200)
ap.add_argument("--device", default="cpu")
args = ap.parse_args()

import torch
from PIL import Image
import numpy as np
import lpips
from torchvision import transforms

pre = sorted(Path(args.pre_dir).glob("*.png"))
post = sorted(Path(args.post_dir).glob("*.png"))
common = sorted(set(p.name for p in pre) & set(p.name for p in post))[: args.max_pairs]
print(f"  {len(common)} paired files")

tx = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

device = args.device
loss = lpips.LPIPS(net="vgg").to(device).eval()
dists = []
t0 = time.time()
with torch.no_grad():
    for i, n in enumerate(common):
        a = tx(Image.open(Path(args.pre_dir) / n).convert("RGB")).unsqueeze(0).to(device)
        b = tx(Image.open(Path(args.post_dir) / n).convert("RGB")).unsqueeze(0).to(device)
        d = loss(a, b).item()
        dists.append(d)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(common)}] {time.time()-t0:.1f}s mean={np.mean(dists):.4f}", flush=True)

mean = float(np.mean(dists)); std = float(np.std(dists))
out = {"pre_dir": args.pre_dir, "post_dir": args.post_dir, "n": len(dists),
       "lpips_mean": mean, "lpips_std": std, "lpips_per_pair": dists}
out_path = Path(args.post_dir).parent / args.out_name
out_path.write_text(json.dumps(out, indent=2))
print(f"LPIPS-vgg = {mean:.4f} ± {std:.4f} (n={len(dists)})")
print(f"wrote {out_path}")
