# reports/INDEX.md

Chronological log of every experiment. Each entry: `<exp_id> | date | best metric | status (keep/discard/crash) | one-line summary`.

| exp_id | date | metric | status | summary |
|---|---|---|---|---|
| phase1a_bootstrap | 2026-05-04 | 24/24 verify_assets rows green | keep | CPU scaffold; 66/66 tests pass; SAE checkpoints (Surkov 4 hookpoints, SAeUron 2 hookpoints) on disk; SDXL Turbo + SDXL Base + SD v1.5 + CompVis safety checker + CLIP ViT-L/14 + NudeNet + Q16 + LPIPS + DreamSim downloaded; SAEmnesia queued for reproduce-from-scratch (no public release as of 2026-05-04); UnlearnCanvas (81 GB, 153 parquet shards, 38+ styles) + I2P (4703) / I2P-adv (1104) / COCO val (5000 images / 25014 captions) / MMA-Diffusion text / UnlearnDiffAtk / Ring-A-Bell / Q16 repos cloned; LAION-COCO gated upstream → loader falls back to COCO captions (25014 prompts); env packed to /workspace/env-archives/dsi.tar.gz (672 MB, CPU torch wheel — GPU pod must reinstall + repack); GitHub repo not yet created — see PLAN.md "Open ops items for Phase 1b first hour". |
