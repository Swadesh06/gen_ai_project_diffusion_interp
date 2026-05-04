#!/usr/bin/env python
"""Quick FID-only compute (no CLIP) — much faster."""
import argparse, json, sys
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--gen-dir", required=True)
ap.add_argument("--out-name", default="fid.json")
ap.add_argument("--ref-dir", default="/workspace/datasets/coco/val2017")
args = ap.parse_args()
from cleanfid import fid
score = float(fid.compute_fid(args.gen_dir, args.ref_dir, mode="clean", num_workers=4))
print(f"FID = {score}")
out = {"gen_dir": args.gen_dir, "fid": score}
Path(args.gen_dir).parent.joinpath(args.out_name).write_text(json.dumps(out, indent=2))
print(f"wrote {args.out_name}")
