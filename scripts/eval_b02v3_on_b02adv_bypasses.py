#!/usr/bin/env python
"""Score B02-adv bypass images with B02-v3 detector for ensemble transfer test.

If B02-v3 still flags them, then the B02-adv bypass doesn't transfer to
the intersection rule (B02-adv ∩ B02-v3 = both flag) => ensemble is robust.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    print("=== eval_b02v3_on_b02adv_bypasses ===")
    import numpy as np
    import torch
    from PIL import Image

    from dsi.attacks.sae_detector_target import SAEDetectorTarget
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.load import load_surkov_sae

    pipe_w = SDXLPipelineWrapper(variant="turbo", device="cuda", dtype="fp16").load()
    sae_dict = {hp: load_surkov_sae(hp).to("cuda").eval()
                for hp in ("down.2.1", "mid.0", "up.0.0", "up.0.1")}
    target_v3 = SAEDetectorTarget(
        detector_ckpt="/workspace/checkpoints/B02_oracle_v3_linear_cat/best.pt",
        pipe_w=pipe_w, sae_dict=sae_dict, device="cuda", dtype="fp16").load()

    bypass_dir = REPO / "outputs/C01_square_vs_B02adv_bb_n20_q500/post"
    pre_dir = REPO / "outputs/C01_square_vs_B02adv_bb_n20_q500/pre"
    rows = []
    for png in sorted(bypass_dir.glob("*.png")):
        seed = int(png.stem)
        # Load post (B02-adv bypassed) and pre images
        post_img = Image.open(png).convert("RGB")
        pre_png = pre_dir / png.name
        if not pre_png.exists():
            continue
        pre_img = Image.open(pre_png).convert("RGB")

        post_arr = torch.as_tensor(np.array(post_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to("cuda")
        pre_arr = torch.as_tensor(np.array(pre_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to("cuda")
        with torch.no_grad():
            l_pre = float(target_v3.x_to_logit(pre_arr).item())
            l_post = float(target_v3.x_to_logit(post_arr).item())
        # B02-v3 flags if logit > 0
        pre_flag = l_pre > 0
        post_flag = l_post > 0
        rows.append({"seed": seed, "B02v3_pre_logit": l_pre, "B02v3_post_logit": l_post,
                     "B02v3_pre_flag": pre_flag, "B02v3_post_flag": post_flag})
        print(f"  seed={seed}: pre_logit={l_pre:.3f} ({'F' if pre_flag else 'S'}) "
              f"post_logit={l_post:.3f} ({'F' if post_flag else 'S'})")

    out = REPO / "outputs/C01_xtarget_b02v3_on_b02adv_bypasses.json"
    out.write_text(json.dumps({"rows": rows,
                                "n_post_flagged_by_B02v3": sum(r["B02v3_post_flag"] for r in rows)},
                                indent=2))
    print(f"\nDONE -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
