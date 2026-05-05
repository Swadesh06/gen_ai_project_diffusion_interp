#!/usr/bin/env python
"""Build a figure.png for D06_joint_mask_udatk_n50_lam5.

Layout: 4x4 grid showing:
- 4 corrected cases (top-left 2x2)
- 4 uncorrected (still-flagged) cases (top-right 2x2)
- 4 new-FP cases (benign pre, flagged post; bottom-left 2x2)
- 4 unchanged-benign cases (bottom-right 2x2)

Each cell shows pre + post side-by-side.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    print("=== build_figure_d6_uda ===")
    from PIL import Image, ImageDraw, ImageFont

    out_dir = REPO / "outputs/D06_joint_mask_udatk_n50_lam5"
    summary = json.loads((out_dir / "summary.json").read_text())
    rows = summary["rows"]

    # Categorize
    corrected = [r for r in rows if r["corrected"]]
    new_fps = [r for r in rows if r["new_fp"]]
    still_flagged = [r for r in rows if r["flag_pre"] and r["flag_post"] and not r["corrected"]]
    benign = [r for r in rows if not r["flag_pre"] and not r["flag_post"]]

    # Take first 4 of each
    corrected = corrected[:4]
    new_fps = new_fps[:4]
    still_flagged = still_flagged[:4]
    benign = benign[:4]

    print(f"  corrected={len(corrected)}, new_fps={len(new_fps)}, "
          f"still_flagged={len(still_flagged)}, benign={len(benign)}")

    # Build a 4x4 grid where each cell is pre|post (2 images per cell)
    cell_w, cell_h = 256, 128
    grid_w, grid_h = 4 * cell_w, 4 * cell_h
    fig = Image.new("RGB", (grid_w, grid_h + 60), (255, 255, 255))
    draw = ImageDraw.Draw(fig)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def cell_at(img_pre_path, img_post_path, x, y, label):
        if not Path(img_pre_path).exists() or not Path(img_post_path).exists():
            return
        pre = Image.open(img_pre_path).convert("RGB").resize((cell_w // 2, cell_h))
        post = Image.open(img_post_path).convert("RGB").resize((cell_w // 2, cell_h))
        fig.paste(pre, (x, y))
        fig.paste(post, (x + cell_w // 2, y))
        draw.text((x + 5, y + 5), label, fill=(255, 255, 0), font=font)

    categories = [
        ("corrected", corrected, 0, 0, "CORR"),
        ("new_fp", new_fps, 2 * cell_w, 0, "NEW-FP"),
        ("still_flagged", still_flagged, 0, 2 * cell_h, "STILL-F"),
        ("benign", benign, 2 * cell_w, 2 * cell_h, "BENIGN"),
    ]

    for cat_name, items, x_off, y_off, label in categories:
        for i, item in enumerate(items):
            row, col = i // 2, i % 2
            x = x_off + col * cell_w
            y = y_off + row * cell_h
            idx = item["i"]
            pre_path = out_dir / "pre" / f"{idx:04d}.png"
            post_path = out_dir / "post" / f"{idx:04d}.png"
            cell_label = f"{label}#{idx}"
            cell_at(pre_path, post_path, x, y, cell_label)

    # Title bar
    title = (f"D-6 joint-mask 41-feature UDA-nudity intervention | "
             f"corrected={summary['n_corrected']}/{summary['n_pre_flagged']} "
             f"new-FP={summary['n_new_fp']} | "
             f"each cell: pre|post")
    draw.text((10, grid_h + 10), title, fill=(0, 0, 0), font=font)

    fig_path = out_dir / "figure.png"
    fig.save(fig_path, dpi=(150, 150))
    print(f"  saved {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
