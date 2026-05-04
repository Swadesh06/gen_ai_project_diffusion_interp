#!/usr/bin/env python
"""Fast LPIPS-vgg on (pre, post) image pairs on GPU.

GPU LPIPS is ~50x faster than CPU. With small VRAM (~1 GB) it fits
alongside running attacks.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--pre-dir", required=True)
ap.add_argument("--post-dir", required=True)
ap.add_argument("--out-name", default="lpips_gpu.json")
ap.add_argument("--max-pairs", type=int, default=200)
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch", type=int, default=4)
args = ap.parse_args()

import torch
from PIL import Image
import numpy as np
import lpips
from torchvision import transforms

pre = sorted(Path(args.pre_dir).glob("*.png"))
post = sorted(Path(args.post_dir).glob("*.png"))
common = sorted(set(p.name for p in pre) & set(p.name for p in post))[: args.max_pairs]
print(f"  {len(common)} paired files, batch={args.batch}")

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
    for i in range(0, len(common), args.batch):
        names = common[i : i + args.batch]
        a_batch, b_batch = [], []
        for n in names:
            a_batch.append(tx(Image.open(Path(args.pre_dir) / n).convert("RGB")))
            b_batch.append(tx(Image.open(Path(args.post_dir) / n).convert("RGB")))
        a = torch.stack(a_batch).to(device)
        b = torch.stack(b_batch).to(device)
        d = loss(a, b).flatten().cpu().tolist()
        dists.extend(d)
        if (i // args.batch) % 5 == 0:
            print(f"  [{len(dists)}/{len(common)}] {time.time()-t0:.1f}s mean={np.mean(dists):.4f}", flush=True)

mean = float(np.mean(dists)); std = float(np.std(dists))
out = {"pre_dir": args.pre_dir, "post_dir": args.post_dir, "n": len(dists),
       "lpips_mean": mean, "lpips_std": std, "lpips_per_pair": dists}
out_path = Path(args.post_dir).parent / args.out_name
out_path.write_text(json.dumps(out, indent=2))
print(f"LPIPS-vgg (GPU) = {mean:.4f} ± {std:.4f} (n={len(dists)})")
print(f"wrote {out_path}")
