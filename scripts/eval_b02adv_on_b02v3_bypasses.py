#!/usr/bin/env python
"""Reverse direction: score B02-v3 bypass images with B02-adv detector."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    print("=== eval_b02adv_on_b02v3_bypasses ===")
    import numpy as np
    import torch
    from PIL import Image

    from dsi.attacks.sae_detector_target import SAEDetectorTarget
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.load import load_surkov_sae

    pipe_w = SDXLPipelineWrapper(variant="turbo", device="cuda", dtype="fp16").load()
    sae_dict = {hp: load_surkov_sae(hp).to("cuda").eval()
                for hp in ("down.2.1", "mid.0", "up.0.0", "up.0.1")}
    target_adv = SAEDetectorTarget(
        detector_ckpt="checkpoints/B02_adv_v1/best.pt",
        pipe_w=pipe_w, sae_dict=sae_dict, device="cuda", dtype="fp16").load()

    bypass_dir = REPO / "outputs/C01_square_vs_B02v3_bb_n200_q500/post"
    pre_dir = REPO / "outputs/C01_square_vs_B02v3_bb_n200_q500/pre"
    rows = []
    for png in sorted(bypass_dir.glob("*.png")):
        seed = int(png.stem)
        post_img = Image.open(png).convert("RGB")
        pre_png = pre_dir / png.name
        if not pre_png.exists():
            continue
        pre_img = Image.open(pre_png).convert("RGB")

        post_arr = torch.as_tensor(np.array(post_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to("cuda")
        pre_arr = torch.as_tensor(np.array(pre_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to("cuda")
        with torch.no_grad():
            l_pre = float(target_adv.x_to_logit(pre_arr).item())
            l_post = float(target_adv.x_to_logit(post_arr).item())
        pre_flag = l_pre > 0
        post_flag = l_post > 0
        rows.append({"seed": seed, "B02adv_pre_logit": l_pre, "B02adv_post_logit": l_post,
                     "B02adv_pre_flag": pre_flag, "B02adv_post_flag": post_flag})
        print(f"  seed={seed}: pre_logit={l_pre:.3f} ({'F' if pre_flag else 'S'}) "
              f"post_logit={l_post:.3f} ({'F' if post_flag else 'S'})")

    n_bypass_v3_only = sum(1 for r in rows if r["B02adv_post_flag"])  # post still flagged by B02-adv
    n_bypass_both = sum(1 for r in rows if not r["B02adv_post_flag"])  # post not flagged by either
    out = REPO / "outputs/C01_xtarget_b02adv_on_b02v3_bypasses.json"
    out.write_text(json.dumps({
        "rows": rows,
        "n_total": len(rows),
        "n_b02adv_post_flag": n_bypass_v3_only,
        "n_intersection_bypass": n_bypass_both,
    }, indent=2))
    print(f"\n=== summary ===")
    print(f"  total rows: {len(rows)}")
    print(f"  B02-adv still flags after B02-v3 bypass: {n_bypass_v3_only}")
    print(f"  Intersection bypass (neither flags): {n_bypass_both}")
    print(f"DONE -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
