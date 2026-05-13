#!/usr/bin/env python
"""Build outputs/figures/F1_sae_attribution.pdf — SAE attribution figure for Gate 1 cell 1.8.

For each attack space (A01 pixel, A02 latent, A03 embedding), pick the first
bypass case where SAE activations are recorded. Plot a 5-row × 3-column grid:
  rows: A01 / A02 / A03
  cols: post (PIL), top-10 feature bar at down.2.1, top-10 bar at up.0.1
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def collect_topk(sae_path: Path, k: int = 10) -> dict:
    import torch
    d = torch.load(sae_path, weights_only=False, map_location="cpu")
    out = {}
    for hp, t in d.items():
        # t shape: (H, W, F). Aggregate spatial → take max per feature, then top-k.
        agg = t.amax(dim=(0, 1))  # (F,)
        topv, topi = torch.topk(agg, k=k)
        out[hp] = {
            "feat_idx": topi.tolist(),
            "feat_val": topv.tolist(),
            "total_features": int(t.shape[-1]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/F1_sae_attribution.pdf")
    ap.add_argument("--exps", nargs="+", default=[
        "A01_pixel_eps4_n200", "A02_latent_eps0.1_n200",
    ])
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for exp in args.exps:
        edir = Path("outputs") / exp
        if not edir.exists():
            print(f"skip missing {exp}"); continue
        # Pick first bypass case (look for any post image where attack.json indicates bypass)
        found_idx = None
        for atk in sorted((edir).glob("*.attack.json")):
            try:
                a = json.loads(atk.read_text())
                if a.get("bypass"):
                    found_idx = int(atk.stem.split(".")[0])
                    break
            except Exception:
                continue
        if found_idx is None:
            print(f"no bypass in {exp}"); continue
        pre_p = edir / "pre" / f"{found_idx:08d}.png"
        post_p = edir / "post" / f"{found_idx:08d}.png"
        sae_p = edir / "sae" / f"{found_idx:08d}.sae.pt"
        if not (pre_p.exists() and post_p.exists() and sae_p.exists()):
            # Try to find any case where all three exist
            for atk in sorted((edir).glob("*.attack.json")):
                try:
                    a = json.loads(atk.read_text())
                    if not a.get("bypass"):
                        continue
                    i2 = int(atk.stem.split(".")[0])
                except Exception:
                    continue
                p2 = edir / "pre" / f"{i2:08d}.png"
                q2 = edir / "post" / f"{i2:08d}.png"
                s2 = edir / "sae" / f"{i2:08d}.sae.pt"
                if p2.exists() and q2.exists() and s2.exists():
                    found_idx = i2
                    pre_p, post_p, sae_p = p2, q2, s2
                    break
        if not (pre_p.exists() and post_p.exists() and sae_p.exists()):
            print(f"  no fully-paired bypass case in {exp}")
            continue
        rows.append({
            "exp": exp,
            "idx": found_idx,
            "pre_path": pre_p,
            "post_path": post_p,
            "sae_path": sae_p,
        })

    print(f"building figure with {len(rows)} rows")
    n_rows = max(1, len(rows))
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4.2 * n_rows), squeeze=False)

    space_pretty = {"A01_pixel_eps4_n200": "A01 pixel-PGD",
                    "A02_latent_eps0.1_n200": "A02 latent-PGD",
                    "A03_emb_eps0.5_n200": "A03 emb-PGD"}

    for r, row in enumerate(rows):
        pre_img = Image.open(row["pre_path"]).convert("RGB").resize((256, 256))
        post_img = Image.open(row["post_path"]).convert("RGB").resize((256, 256))
        topk = collect_topk(row["sae_path"], k=10)

        ax_pre = axes[r, 0]; ax_pre.imshow(pre_img); ax_pre.set_xticks([]); ax_pre.set_yticks([])
        ax_pre.set_title(f"{space_pretty.get(row['exp'], row['exp'])}\npre", fontsize=11)

        ax_post = axes[r, 1]; ax_post.imshow(post_img); ax_post.set_xticks([]); ax_post.set_yticks([])
        ax_post.set_title("post (bypass)", fontsize=11)

        for c, hp in enumerate(["down.2.1", "mid.0", "up.0.0"], start=2):
            ax = axes[r, c]
            d = topk[hp]
            ax.barh(range(10), d["feat_val"][::-1], color="#4477aa")
            ax.set_yticks(range(10)); ax.set_yticklabels([str(i) for i in d["feat_idx"][::-1]], fontsize=8)
            ax.set_title(f"{hp} top-10\n(idx of {d['total_features']})", fontsize=10)
            ax.set_xlabel("max activation", fontsize=9)

    plt.suptitle("F1 — SAE feature attribution on PGD bypass cases (post-attack image; top-10 firing features per UNet hookpoint)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, format="pdf", dpi=110, bbox_inches="tight")
    print(f"wrote {out}")
    # Also write a manifest of what was plotted
    manifest = []
    for r, row in enumerate(rows):
        topk = collect_topk(row["sae_path"], k=10)
        manifest.append({
            "row": r, "exp": row["exp"], "case_idx": row["idx"],
            "pre_path": str(row["pre_path"]),
            "post_path": str(row["post_path"]),
            "topk_features": topk,
        })
    (out.parent / "F1_sae_attribution_manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
