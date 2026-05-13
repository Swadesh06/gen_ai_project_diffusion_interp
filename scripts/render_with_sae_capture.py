#!/usr/bin/env python
"""Render a prompt list through SDXL Turbo with Surkov SAE hook capture.

Saves per-sample images to <out>/renders/<idx>_<seed>.png and SAE activations
to <out>/sae/<idx>_<seed>.sae.pt (one dict-of-tensors per hookpoint).

Used by Gate 2 cell 2.12 (UDA/MMA AUC at full scale) and Gate 4 (intervention
on SDXL Base 4-step) to populate the matched-condition prompt list.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True,
                    help="path to JSON list of {prompt, seed, idx?, ...} dicts OR a benchmark name "
                         "(uda_nudity|uda_violence|mma|i2p_nsfw)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="turbo", choices=["turbo", "base"])
    ap.add_argument("--n-steps", type=int, default=4, help="diffusion steps for base; 1 for turbo")
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=42000000)
    args = ap.parse_args()

    out_dir = Path(args.out); (out_dir / "renders").mkdir(parents=True, exist_ok=True)
    (out_dir / "sae").mkdir(parents=True, exist_ok=True)

    print(f"=== render+SAE → {out_dir} ===")
    # Load prompts
    if args.prompts in ("uda_nudity", "uda_violence", "mma", "i2p_nsfw"):
        prompts = _load_benchmark(args.prompts, args.limit)
    else:
        rows = json.loads(Path(args.prompts).read_text())
        if isinstance(rows, dict) and "prompts" in rows:
            rows = rows["prompts"]
        prompts = []
        for i, r in enumerate(rows[: args.limit] if args.limit else rows):
            if isinstance(r, str):
                prompts.append({"idx": i, "prompt": r, "seed": args.seed_offset + i})
            else:
                prompts.append({
                    "idx": int(r.get("idx", i)),
                    "prompt": r["prompt"],
                    "seed": int(r.get("seed", args.seed_offset + i)),
                })
    print(f"  {len(prompts)} prompts")

    print("loading SDXL pipeline...")
    import torch
    from dsi.models.sdxl_pipeline import SDXLPipelineWrapper
    from dsi.sae.load import load_surkov_sae
    from dsi.sae.hooks import SurkovHookManager

    pipe_w = SDXLPipelineWrapper(variant=args.variant, device="cuda", dtype="fp16").load()
    print("loading 4 SAEs...")
    saes = {hp: load_surkov_sae(hp).to("cuda").eval() for hp in ("down.2.1", "mid.0", "up.0.0", "up.0.1")}

    completed = 0; t0 = time.time()
    for row in prompts:
        idx = row["idx"]; prompt = row["prompt"]; seed = int(row["seed"])
        png = out_dir / "renders" / f"{idx:08d}_{seed:08d}.png"
        sae_pt = out_dir / "sae" / f"{idx:08d}_{seed:08d}.sae.pt"
        if png.exists() and sae_pt.exists():
            completed += 1
            continue
        try:
            with SurkovHookManager(pipe_w.pipe.unet, saes) as mgr:
                g = torch.Generator(device="cuda").manual_seed(seed)
                with torch.no_grad():
                    img = pipe_w.pipe(
                        prompt=prompt,
                        num_inference_steps=args.n_steps,
                        guidance_scale=args.guidance,
                        generator=g,
                        height=512,
                        width=512,
                    ).images[0]
                img.resize((512, 512)).save(png)
                # Aggregate SAE activations: per hookpoint, take latents from step 0 reshape to (H, W, F)
                captures = {hp: mgr.captured.get(hp) for hp in saes}
                feats_out = {}
                for hp, cap in captures.items():
                    if cap is None or not cap.latents:
                        continue
                    # Each entry is per-step latents (B, H*W, F) — take last step, B=0
                    lat = cap.latents[-1]  # likely (B, T, F)
                    if lat.dim() == 3:
                        b, t, f = lat.shape
                        s = int(t ** 0.5)
                        if s * s == t:
                            lat = lat[0].view(s, s, f)
                        else:
                            lat = lat[0]
                    feats_out[hp] = lat.detach().cpu().to(torch.float32)
                torch.save(feats_out, sae_pt)
                completed += 1
                if completed % 10 == 0:
                    print(f"  [{completed}/{len(prompts)}] {time.time() - t0:.0f}s")
        except Exception as e:
            print(f"  err idx={idx} seed={seed}: {e}")

    # Save manifest
    manifest = {"out_dir": str(out_dir), "n_completed": completed, "n_total": len(prompts), "prompts": prompts}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done: {completed}/{len(prompts)} in {time.time() - t0:.0f}s")
    return 0


def _load_benchmark(name: str, limit=None) -> list[dict]:
    if name == "uda_nudity":
        # Already-rendered prompts in outputs/udatk_nudity_n143/renders/*.png.meta.json
        meta_dir = Path("outputs/udatk_nudity_n143/renders")
        rows = []
        for mp in sorted(meta_dir.glob("*.png.meta.json")):
            m = json.loads(mp.read_text())
            try:
                idx = int(mp.stem.split("_")[0])
            except Exception:
                idx = len(rows)
            rows.append({"idx": idx, "prompt": m["prompt"], "seed": int(m.get("seed", 42000000 + idx))})
        return rows[:limit] if limit else rows
    if name == "uda_violence":
        meta_dir = Path("outputs/udatk_violence_n200/renders")
        rows = []
        for mp in sorted(meta_dir.glob("*.png.meta.json")):
            m = json.loads(mp.read_text())
            try:
                idx = int(mp.stem.split("_")[0])
            except Exception:
                idx = len(rows)
            rows.append({"idx": idx, "prompt": m["prompt"], "seed": int(m.get("seed", 42000000 + idx))})
        return rows[:limit] if limit else rows
    if name == "mma":
        # MMA-Diffusion adversarial prompts: look in outputs/mma_diffusion_adv_n50 or similar
        rows = []
        ds = Path("/workspace/datasets/MMA-Diffusion-NSFW-adv-prompts-benchmark/MMA-Diffusion-NSFW-adv-prompts-benchmark.csv")
        if not ds.exists():
            ds = Path("outputs/mma_diffusion_adv_n50/manifest.json")
        if ds.exists() and ds.suffix == ".csv":
            import csv
            with ds.open() as f:
                r = csv.DictReader(f)
                for i, row in enumerate(r):
                    p = row.get("adv_prompt") or row.get("prompt") or row.get("text") or ""
                    if not p: continue
                    rows.append({"idx": i, "prompt": p, "seed": 42000000 + i})
        return rows[:limit] if limit else rows
    if name == "i2p_nsfw":
        from dsi.data.i2p import i2p_nsfw_subset
        sub = i2p_nsfw_subset(limit=None)
        rows = [{"idx": i, "prompt": p.text, "seed": 42000000 + i} for i, p in enumerate(sub)]
        return rows[:limit] if limit else rows
    raise ValueError(name)


if __name__ == "__main__":
    raise SystemExit(main())
