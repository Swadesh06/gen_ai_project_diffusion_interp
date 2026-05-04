#!/usr/bin/env python
"""Compute FID + CLIP-score on a directory of generated images.

FID: clean-fid library, against COCO val 2017 reference statistics.
CLIP-score: ViT-L/14 cosine similarity between image and its prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", required=True, help="dir with PNG generations")
    ap.add_argument("--prompts-source", choices=["i2p_nsfw", "coco", "auto"], default="auto",
                    help="how to recover the per-image prompt (auto = read .meta.json or .attack.json)")
    ap.add_argument("--ref-dir", default="/workspace/datasets/coco/val2017",
                    help="FID reference dir (default COCO val 2017)")
    ap.add_argument("--out-name", default="fid_clipscore.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-batch", type=int, default=16)
    ap.add_argument("--max-images", type=int, default=2000)
    args = ap.parse_args()

    import torch
    from PIL import Image

    gen_dir = Path(args.gen_dir)
    pngs = sorted(gen_dir.glob("*.png"))[: args.max_images]
    print(f"  {len(pngs)} images in {gen_dir}")

    # === FID via clean-fid ===
    print("computing FID via clean-fid (vs COCO val)")
    from cleanfid import fid

    ref = Path(args.ref_dir)
    if not ref.exists():
        print(f"WARN: ref dir {ref} missing; skipping FID")
        fid_score = None
    else:
        try:
            fid_score = float(fid.compute_fid(str(gen_dir), str(ref), mode="clean", num_workers=2))
        except Exception as e:
            print(f"FID error: {e}")
            fid_score = None
    print(f"  FID = {fid_score}")

    # === CLIP-score (per-image, then mean) ===
    print("computing CLIP-score (open_clip ViT-L/14)")
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=args.device)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()

    # Recover per-image prompt from sidecars
    prompts = []
    for p in pngs:
        meta = p.with_suffix(p.suffix + ".meta.json")
        atk = (gen_dir.parent / f"{p.stem}.attack.json")
        prompt = ""
        if meta.exists():
            try:
                prompt = json.loads(meta.read_text()).get("prompt", "")
            except Exception:
                pass
        if not prompt and atk.exists():
            try:
                prompt = json.loads(atk.read_text()).get("prompt", "")
            except Exception:
                pass
        prompts.append(prompt or "")

    sims = []
    from torch.nn.functional import normalize
    with torch.no_grad():
        for i in range(0, len(pngs), args.clip_batch):
            batch_imgs = [preprocess(Image.open(p).convert("RGB")) for p in pngs[i:i+args.clip_batch]]
            batch_prompts = prompts[i:i+args.clip_batch]
            xs = torch.stack(batch_imgs).to(args.device)
            ts = tokenizer(batch_prompts).to(args.device)
            ie = normalize(model.encode_image(xs), dim=-1)
            te = normalize(model.encode_text(ts), dim=-1)
            sims.extend((ie * te).sum(dim=-1).cpu().tolist())
    clip_mean = sum(sims) / max(1, len(sims))
    clip_std = (sum((s - clip_mean) ** 2 for s in sims) / max(1, len(sims))) ** 0.5
    print(f"  CLIP-score: {clip_mean:.4f} ± {clip_std:.4f} (n={len(sims)})")

    out = {
        "gen_dir": str(gen_dir), "ref_dir": str(ref), "n": len(pngs),
        "fid": fid_score, "clip_score_mean": clip_mean, "clip_score_std": clip_std,
        "clip_scores": sims,
    }
    out_path = gen_dir.parent / args.out_name
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
