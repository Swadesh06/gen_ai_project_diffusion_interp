"""Score PixArt-Sigma 1024-MS i2p_nsfw outputs with safety_checker.

The d09_flux_smoke pipeline collects activations but doesn't run
safety_checker. Add it post-hoc.
"""
from pathlib import Path
import json, time
import torch
from PIL import Image

src_dir = Path("outputs/D09_pixart_i2p_n100/images")
files = sorted(src_dir.glob("i2p_nsfw_*.png"))
print(f"  {len(files)} i2p_nsfw images")

from dsi.detectors.baselines.safety_checker import SafetyCheckerWrapper
sc = SafetyCheckerWrapper(device="cuda").load()

t0 = time.time()
n_flag = 0
for i, f in enumerate(files):
    img = Image.open(f).convert("RGB")
    flagged = bool(sc.score([img])[0]["flagged"])
    n_flag += int(flagged)
    with open(f.with_suffix(".png.safety.json"), "w") as fh:
        json.dump({"flagged": flagged, "file": f.name}, fh)
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(files)}] {time.time()-t0:.1f}s flagged={n_flag}", flush=True)

print(f"  TOTAL: {n_flag}/{len(files)} = {n_flag/len(files):.3f}", flush=True)
out = src_dir.parent / "safety_summary.json"
out.write_text(json.dumps({"n": len(files), "flagged": n_flag, "rate": n_flag/len(files)}, indent=2))
print(f"  wrote {out}")
