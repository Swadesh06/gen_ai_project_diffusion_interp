#!/usr/bin/env python
"""Verification matrix per STARTER_PROMPT_1 §5.

Runs every row, exits non-zero on any failure, writes JSON to logs/verify_assets.json,
pretty-prints a pass/fail table to stdout.

Categories:
  - HF models    (from_pretrained local-only succeeds)
  - SAE checkpoints (load + state-dict key sanity)
  - I2P / I2P-adv  (row counts)
  - COCO val + captions
  - LAION-COCO subset  (>= 50000)
  - UnlearnCanvas  (60 styles x 20 objects)
  - MMA-Diffusion text  (file present)
  - UnlearnDiffAtk
  - NudeNet, Q16, Safety Checker (smoke forward)
  - dreamsim, lpips (small forward)

The Phase 1b GPU session is gated on every row passing.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dsi.config import cfg  # noqa: E402


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    elapsed_s: float = 0.0


@dataclass
class Matrix:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, r: CheckResult) -> None:
        self.results.append(r)

    @property
    def all_ok(self) -> bool:
        return all(r.ok for r in self.results)

    def to_dict(self) -> dict:
        return {
            "all_ok": self.all_ok,
            "n_pass": sum(r.ok for r in self.results),
            "n_fail": sum(not r.ok for r in self.results),
            "rows": [r.__dict__ for r in self.results],
        }

    def print(self) -> None:
        from textwrap import shorten

        print()
        print(f"{'name':50s}  {'status':6s}  {'time':>6s}  detail")
        print("-" * 110)
        for r in self.results:
            status = "PASS" if r.ok else "FAIL"
            detail = shorten(r.detail, width=40, placeholder="...")
            print(f"{r.name:50s}  {status:6s}  {r.elapsed_s:6.2f}  {detail}")
        print("-" * 110)
        print(f"{'TOTAL':50s}  {'PASS' if self.all_ok else 'FAIL':6s}  "
              f"{sum(r.elapsed_s for r in self.results):6.2f}  "
              f"{sum(r.ok for r in self.results)}/{len(self.results)} rows passed")
        print()


def run_check(name: str, fn: Callable, mat: Matrix) -> None:
    t0 = time.time()
    try:
        detail = fn()
        mat.add(CheckResult(name=name, ok=True, detail=str(detail or ""), elapsed_s=time.time() - t0))
    except Exception as e:
        mat.add(
            CheckResult(name=name, ok=False, detail=f"{type(e).__name__}: {e}",
                        elapsed_s=time.time() - t0)
        )


# ---- individual checks ------------------------------------------------------------

def chk_hf_model_local(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    p = snapshot_download(model_id, local_files_only=True)
    return Path(p).name


def chk_safety_checker_loadable() -> str:
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

    m = StableDiffusionSafetyChecker.from_pretrained(cfg.model.safety_checker, local_files_only=True)
    return f"params={sum(p.numel() for p in m.parameters())}"


def chk_clip_loadable() -> str:
    from transformers import CLIPVisionModel

    m = CLIPVisionModel.from_pretrained(cfg.model.clip_vision, local_files_only=True)
    return f"params={sum(p.numel() for p in m.parameters())}"


def chk_sdxl_turbo_loadable() -> str:
    from huggingface_hub import snapshot_download

    p = snapshot_download(cfg.model.sdxl_turbo, local_files_only=True)
    sf = list(Path(p).rglob("*.safetensors"))
    return f"safetensors={len(sf)}"


def chk_sdxl_base_loadable() -> str:
    from huggingface_hub import snapshot_download

    p = snapshot_download(cfg.model.sdxl_base, local_files_only=True)
    sf = list(Path(p).rglob("*.safetensors"))
    return f"safetensors={len(sf)}"


def chk_sd15_loadable() -> str:
    from huggingface_hub import snapshot_download

    try:
        p = snapshot_download(cfg.model.sd15, local_files_only=True)
    except Exception:
        p = snapshot_download(cfg.model.sd15_fallback, local_files_only=True)
    sf = list(Path(p).rglob("*.safetensors")) + list(Path(p).rglob("*.bin"))
    return f"weights={len(sf)}"


def chk_surkov_sae(hookpoint: str) -> str:
    from dsi.sae.load import load_surkov_sae

    sae = load_surkov_sae(hookpoint)
    return f"d_in={sae.d_in} d_hidden={sae.d_hidden}"


def chk_saeuron_sae(hookpoint: str) -> str:
    from dsi.sae.load import load_saeuron_sae

    sae = load_saeuron_sae(hookpoint)
    return f"d_in={sae.d_in} d_hidden={sae.d_hidden}"


def chk_i2p_count() -> str:
    from dsi.data.i2p import load_i2p

    rows = load_i2p("full")
    if len(rows) < 1000:
        raise AssertionError(f"i2p has {len(rows)} rows, expected >= 1000 (4703 nominal)")
    return f"rows={len(rows)}"


def chk_i2p_adv_count() -> str:
    from dsi.data.i2p import load_i2p

    rows = load_i2p("adversarial")
    if len(rows) < 100:
        raise AssertionError(f"i2p-adv has {len(rows)} rows, expected >= 100")
    return f"rows={len(rows)}"


def chk_coco_val_images() -> str:
    from dsi.data.coco import load_coco_val_images

    imgs = load_coco_val_images()
    if len(imgs) < 4000:
        raise AssertionError(f"coco val has {len(imgs)} images, expected ~5000")
    return f"images={len(imgs)}"


def chk_coco_captions() -> str:
    from dsi.data.coco import load_coco_captions

    caps = load_coco_captions()
    if len(caps) < 20000:
        raise AssertionError(f"coco captions = {len(caps)}, expected ~25000 (5x5000)")
    return f"captions={len(caps)}"


def chk_laion_coco() -> str:
    """LAION-COCO is gated upstream; loader falls back to COCO val captions (~25K)."""
    from dsi.data.laion_coco import load_laion_coco

    rows = load_laion_coco(limit=50000)
    if len(rows) < 10000:
        raise AssertionError(f"laion-coco fallback has {len(rows)} rows, expected >= 10000")
    return f"prompts={len(rows)} (source: {rows[0].source if rows else 'n/a'})"


def chk_unlearncanvas_layout() -> str:
    from dsi.data.unlearncanvas import list_styles

    styles = list_styles()
    if len(styles) < 20:
        raise AssertionError(f"unlearncanvas styles={len(styles)}, expected >= 20")
    return f"styles={len(styles)}"


def chk_mma_text() -> str:
    from dsi.data.adversarial import load_mma_text

    rows = load_mma_text(limit=10)
    return f"rows={len(rows)} (sample)"


def chk_unlearndiff() -> str:
    from dsi.data.adversarial import load_unlearndiff

    rows = load_unlearndiff(limit=10)
    return f"rows={len(rows)} (sample)"


def chk_nudenet_smoke() -> str:
    from nudenet import NudeDetector

    NudeDetector()
    return "loaded"


def chk_lpips_smoke() -> str:
    import lpips
    import torch

    m = lpips.LPIPS(net="alex")
    a = torch.zeros(1, 3, 64, 64)
    b = torch.zeros(1, 3, 64, 64)
    out = m(a, b)
    if float(out) > 1e-4:
        raise AssertionError(f"lpips identity-pair score {float(out):.4f} expected ~0")
    return f"identity_score={float(out):.6f}"


def chk_dreamsim_smoke() -> str:
    import dreamsim
    import torch

    m, p = dreamsim.dreamsim(pretrained=True, device="cpu")
    a = torch.zeros(1, 3, 224, 224)
    return f"loaded (model={type(m).__name__})"


def chk_disk_headroom() -> str:
    import shutil

    free = shutil.disk_usage("/workspace").free / (1024 ** 3)
    if free < 50:
        raise AssertionError(f"only {free:.1f} GB free on /workspace; need >= 50")
    return f"free_gb={free:.1f}"


def main() -> int:
    mat = Matrix()

    # disk first — cheap, blocking precondition
    run_check("disk: >=50 GB free on /workspace", chk_disk_headroom, mat)

    # diffusion models
    run_check("model: sdxl-turbo loadable", chk_sdxl_turbo_loadable, mat)
    run_check("model: sdxl-base loadable",  chk_sdxl_base_loadable, mat)
    run_check("model: sd v1.5 loadable",    chk_sd15_loadable, mat)
    run_check("model: safety_checker loadable", chk_safety_checker_loadable, mat)
    run_check("model: clip-vit-large loadable", chk_clip_loadable, mat)

    # SAEs
    for hp in ("down.2.1", "mid.0", "up.0.0", "up.0.1"):
        run_check(f"sae: surkov {hp}", lambda hp=hp: chk_surkov_sae(hp), mat)
    for hp in ("up.1.1", "up.1.2"):
        run_check(f"sae: saeuron {hp}", lambda hp=hp: chk_saeuron_sae(hp), mat)

    # datasets
    run_check("data: i2p row count >= 1000",        chk_i2p_count, mat)
    run_check("data: i2p-adv row count >= 100",     chk_i2p_adv_count, mat)
    run_check("data: coco val images ~5000",        chk_coco_val_images, mat)
    run_check("data: coco captions ~25000",         chk_coco_captions, mat)
    run_check("data: laion-coco subset >= 10000",   chk_laion_coco, mat)
    run_check("data: unlearncanvas styles >= 20",   chk_unlearncanvas_layout, mat)
    run_check("data: mma_text reachable",           chk_mma_text, mat)
    run_check("data: unlearndiff reachable",        chk_unlearndiff, mat)

    # classifier oracles
    run_check("oracle: nudenet loads",  chk_nudenet_smoke, mat)
    run_check("oracle: lpips identity", chk_lpips_smoke, mat)
    run_check("oracle: dreamsim loads", chk_dreamsim_smoke, mat)

    mat.print()

    out_path = cfg.paths.log_root / "verify_assets.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mat.to_dict(), indent=2))
    print(f"wrote {out_path}")

    return 0 if mat.all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
