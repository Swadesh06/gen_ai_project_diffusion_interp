# STARTER_PROMPT_1.md — Phase 1a (CPU-only) session start

You are starting the DiffSafeSAE research project. **This is a fresh start** — no code, no reports, no PLAN. The instance you're on right now is **CPU-only**. The GPU instance comes next, with a separate starter prompt. Your job in this session is to do everything that does not need a GPU: read all the docs, set up the conda environment, download all datasets and model checkpoints, scaffold the package, write the data loaders, the attack and intervention code skeletons, and the WandB / config / logging plumbing. By the end of this session, the next session (with GPU) should be able to start with `git pull` and immediately begin smoke-testing on hardware.

You operate by the rules in `gen_ai_project_diffusion_interp/CLAUDE.md`. Read it, then follow it.

---

## Step 1 — read the docs, in order, before doing anything

1. `gen_ai_project_diffusion_interp/project_brief.md` — one-page orientation. Read first to anchor the thesis.
2. `gen_ai_project_diffusion_interp/CLAUDE.md` — full read. The operating manual. **The parallelization section will not bind in this CPU-only session, but read it; the next session needs you to have planned co-scheduling already.**
3. `gen_ai_project_diffusion_interp/task_descriptions/task_description_v1.md` — full read. The four-contribution research spec, six work items, evaluation grids, pass criteria, dependency graph.
4. `gen_ai_project_diffusion_interp/task_descriptions/task_description_v1_appendix.md` — **binding ICLR-rigor extension**. Threat model, theoretical motivation, critical baselines v1 missed, three-tier generalization, statistical significance protocol, compute transparency, ten Phase-C experiment specs, reviewer-question checklist.
5. `gen_ai_project_diffusion_interp/Swadesh_Swain_22116091.pdf` — the original course-project proposal. Skim — you do not have to memorize it; the brief and the v1 spec already encode what survives.
6. `/workspace/conda_setup.md` — pack/unpack workflow for this pod. **Obey it.**
7. The Surkov et al. paper (arXiv:2410.22366) and the `surkovv/sdxl-unbox` README — required for SAE hook plumbing in step 4.
8. The SAeUron paper (arXiv:2501.18052) and `cywinski/SAeUron` README — required for the diffusion-native baseline reproduction.
9. The DSG paper (arXiv:2504.08192) and Arad et al. (arXiv:2505.20063) — required for Contribution 4 method.
10. The IGD paper (arXiv:2508.03006) — required for Contribution 2 baseline framing.

---

## Step 2 — make a plan, write it to PLAN.md

Create `gen_ai_project_diffusion_interp/PLAN.md`. Include:

- A "Phase 1a — CPU-only session start" section listing what you'll do this session (everything in steps 3 → 9 below).
- A "Phase 1b — first GPU session plan" section: which item in `task_description_v1.md` you'll run first, which next, what can be co-scheduled. **Write this now**, while you have time to think — the GPU session will run on it.
- A "Phase C — ideas" section, populated initially from `CLAUDE.md` §9 ("Phase 2 — your own ideas"). Add any of your own ideas as you read the literature.

Update `PLAN.md` at every meaningful step in this session.

---

## Step 3 — environment setup (CPU pod)

Set up the `dsi` conda env:

```bash
conda create -n dsi python=3.11 -y
conda activate dsi
```

Install dependencies. Pin versions where compatibility matters (Blackwell sm_120 needs torch 2.4+; check the appropriate CUDA wheel index for the GPU pod, but install the CPU torch wheel here so package resolution works — re-install GPU torch in the next session). Required packages (a minimum starting set; add as needed):

- `torch>=2.4` (CPU wheel for now), `torchvision`, `torchaudio`
- `diffusers>=0.27`, `transformers>=4.40`, `accelerate>=0.27`, `peft>=0.10`
- `safetensors`, `huggingface_hub`, `datasets`
- `einops`, `sae_lens` (or the EleutherAI SAE library Surkov et al. used — check their README)
- `nudenet`, `clean-fid`, `pytorch-fid`, `lpips`, `open_clip_torch`
- `wandb`, `python-dotenv`, `pyyaml`, `tqdm`, `pandas`, `numpy`, `scipy`, `scikit-learn`
- `matplotlib`, `seaborn`, `Pillow`
- `captum` (for attribution)
- `pytest` (test scaffolding)

After installing, **pack the env**: `bash /workspace/scripts/pack_env.sh dsi`. This is non-negotiable; the local env dies with the pod otherwise.

Set up `.env` at the repo root with `HF_TOKEN`, `HUGGINGFACE_TOKEN`, `WANDB_API_KEY`. Verify with `huggingface-cli whoami` and a `wandb login --relogin` smoke test (from `WANDB_API_KEY`).

---

## Step 4 — scaffold the repo

Create the directory tree exactly as specified in `CLAUDE.md` §15. Add stub `__init__.py` files in every package directory. Add empty placeholder modules for everything in the tree (every `.py` listed) — they should at least import cleanly. Wire `dsi.pth` per CLAUDE.md §3 so `import dsi` works from any cwd.

Create:
- `pyproject.toml` (basic; package name `dsi`, version `0.0.1`)
- `requirements.txt` (the dependency list you used)
- `README.md` (one paragraph stub; you'll fill it out properly at end of Phase 1)
- `.gitignore` (covers `.env`, `*.pt`, `*.safetensors`, `checkpoints/`, `logs/`, `outputs/` except `outputs/figures/`, `__pycache__/`, `*.egg-info`, `.pytest_cache`, `.ipynb_checkpoints`, etc.)
- `dsi/config.py` — load `.env`, expose paths (`DATA_ROOT`, `CHECKPOINT_ROOT`, `CACHE_ROOT`, `OUTPUT_ROOT`), expose hyperparameter defaults
- `scripts/bootstrap.sh` — downloads all datasets and checkpoints (Step 5 below)
- `reports/INDEX.md` (empty header table)

Initialize the git repo. Make the initial commit: `chore: phase 1 session start, scaffold repo + read v1 spec`. Set the remote to the SSH URL on the `Swadesh06` GitHub account (create the repo on github.com first if needed; name it `gen_ai_project_diffusion_interp`). Push.

---

## Step 5 — datasets and checkpoints, download in parallel

**Disk budget**: plan for ~150 GB on the persistent volume — ~50 GB models, ~60 GB datasets (COCO + UnlearnCanvas dominate), ~50 GB headroom for cached SAE activations and generations. Verify free space before starting: `df -h /workspace`.

**Cache locations**: HF models go to `$HF_HOME` (`/workspace/.cache/huggingface`). Datasets go to `/workspace/datasets/<name>/`. SAE checkpoints go to `/workspace/checkpoints/<name>/`. Set these in `dsi/config.py`.

Run the downloads in tmux sessions in parallel. CPU pod is fine for all of these; bandwidth is the bottleneck. Suggested sessions:

**Datasets**:
- `prep-i2p` — `huggingface-cli download AIML-TUDA/i2p --repo-type dataset` and `AIML-TUDA/i2p-adversarial-split`
- `prep-coco` — COCO 2017 val (5K images) for FID, COCO 2017 captions for prompts. Train images optional (only if you want a larger benign reference; 5K val is enough for FID per the standard protocol).
- `prep-laion` — LAION-COCO subset (≤ 50K prompts; captions only, no images needed)
- `prep-unlearncanvas` — `huggingface-cli download OPTML-Group/UnlearnCanvas` (dataset) **and** the released fine-tuned SD v1.5 checkpoints (`OPTML-Group/UnlearnCanvas` exposes both; check the model cards)
- `prep-mma` — `git clone git@github.com:cure-lab/MMA-Diffusion.git`. Text-modality adversarial prompts are immediate; image-modality benchmark requires upstream access request via `YijunYang280/MMA_Diffusion_adv_images_benchmark` on HF — **file the request now**, proceed without it
- `prep-unlearndiff` — `git clone git@github.com:OPTML-Group/Diffusion-MU-Attack.git`; pull their crafted prompt sets
- `prep-ringabell` — `git clone https://github.com/chiayi-hsu/Ring-A-Bell.git`; their adversarial concept-embedding prompts are needed for the I2P-adversarial split's embedding-attack subset

**Diffusion models**:
- `prep-models-sdxl` — `huggingface-cli download stabilityai/sdxl-turbo` and `stabilityai/stable-diffusion-xl-base-1.0`
- `prep-models-sd15` — `huggingface-cli download runwayml/stable-diffusion-v1-5`. **If 404 / gated** (the canonical mirror was restricted in late 2024), fall back to `benjamin-paine/stable-diffusion-v1-5` or `Lykon/dreamshaper-7` (note license terms differ; document which mirror you used in `reports/INDEX.md`).
- `prep-models-sd3` — `stabilityai/stable-diffusion-3-medium-diffusers` (Phase C-5 cross-model transfer; gated, requires accepting their license — log in via `huggingface-cli login` first). **Skip if Phase C-5 is descoped.**

**SAEs**:
- `prep-saes-surkov` — `git clone git@github.com:surkovv/sdxl-unbox.git`; fetch all four hookpoint checkpoints from their HF Space `surokpro2/Unboxing_SDXL_with_SAEs` (download via `huggingface-cli download surokpro2/Unboxing_SDXL_with_SAEs`)
- `prep-saes-saeuron` — `git clone git@github.com:cywinski/SAeUron.git`; for each released hookpoint, run their `python scripts/load_from_hub.py --name bcywinski/SAeUron --hookpoint <hookpoint> --save_path /workspace/checkpoints/saeuron/`
- `prep-saes-saemnesia` — check the SAEmnesia repo (most recent published, look it up at clone time — likely `OPTML-Group/SAEmnesia` or similar) for their released SAE checkpoints; if not released, mark this baseline as "reproduce-from-scratch" in `PLAN.md` and queue the SAE training as a Phase-1 prerequisite

**Classifiers and encoders**:
- `prep-models-clip` — `huggingface-cli download openai/clip-vit-large-patch14` (the safety checker's vision encoder)
- `prep-models-safety` — `huggingface-cli download CompVis/stable-diffusion-safety-checker`
- `prep-models-nudenet` — `pip install nudenet`, then trigger weights download with `from nudenet import NudeDetector; NudeDetector()`
- `prep-models-q16` — Schramowski et al. release; their repo's README. If the upstream link is dead, `ml-research/Q16` on GitHub usually mirrors.
- `prep-models-dreamsim` — for Phase C-10 generation-quality preservation: `pip install dreamsim` and trigger weight download (~500 MB)
- `prep-models-lpips` — for Phase C-10: `pip install lpips`, trigger weight download

### Verification matrix — must pass before declaring Step 5 done

Write `scripts/verify_assets.py` that runs the matrix below and exits non-zero on any failure. The agent runs it at the end of Step 5 and again at the start of every GPU session.

| Asset | Verification |
|---|---|
| Each HF model | `from_pretrained(path, local_files_only=True)` succeeds |
| Each SAE checkpoint | torch.load + state_dict key count matches upstream README |
| I2P / I2P-adv | row count: 4703 / ~600 |
| COCO val | 5000 images present; 5 captions per image |
| LAION-COCO subset | row count ≥ 50000 |
| UnlearnCanvas | 60 styles × 20 objects directories present |
| MMA-Diffusion text | adversarial prompt count matches paper Table 1 |
| UnlearnDiffAtk | crafted-prompt sets for nudity, violence, style each load |
| NudeNet | inference on a known clean image returns "safe" |
| Q16 | inference on a known clean image returns the expected label |
| Safety Checker | inference on a known unsafe demo image returns flagged |
| dreamsim, lpips | small forward pass on identical pair returns ~0 |

`scripts/verify_assets.py` writes a JSON report to `logs/verify_assets.json` and pretty-prints a pass/fail table. **The Phase 1b GPU session does not start until this matrix is fully green.**

While downloads run, do Step 6 in parallel.

---

## Step 6 — write the package skeleton (CPU-runnable parts)

Implement the modules that do not need a GPU. The goal: by end of session, `pytest` runs cleanly on a small CPU-only test suite covering the data loaders, attribution math, and config plumbing.

Concrete deliverables:

- **`dsi/data/`** (all loaders): I2P, COCO, LAION-COCO, UnlearnCanvas, adversarial sets. Each loader returns a typed iterable of `Prompt` / `Image` / `Pair` records. Add small fixtures and unit tests under `tests/`.
- **`dsi/config.py`**: full config object with `from_yaml(path)` and defaults. Wire `.env` loading.
- **`dsi/attacks/common.py`**: ASR computation, attack-runner skeleton (the runner accepts a callable `attack_step(image, target) -> perturbed_image` and abstracts over the three spaces). The actual `pixel.py` / `latent.py` / `embedding.py` PGD inner loops will fail without a GPU — write the structure and put a clear TODO at the GPU-call site.
- **`dsi/sae/hooks.py`**: SAE hook plumbing as a context manager, **purely structural** — registering hooks does not need a GPU, only the forward pass does. Read the Surkov repo's `SDLens/hooked_sd_pipeline.py` carefully and mirror their interface.
- **`dsi/sae/attribution.py`**: per-feature attribution math. This is NumPy / pure-PyTorch CPU-runnable. Implement the input-score $S_\text{in}$, the activation-delta computation, the cross-attack feature overlap metric. Unit-test with synthetic activations.
- **`dsi/interventions/stage1_fisher.py`**: Fisher-ratio scoring. CPU-runnable on cached activations. Unit-test.
- **`dsi/interventions/patches.py`**: mean / zero / resample patching primitives. CPU-runnable. Unit-test.
- **`dsi/eval/asr.py`**, **`dsi/eval/fid.py`** (wrap clean-fid), **`dsi/eval/clip_score.py`** (wrap open_clip; the encoder needs GPU but the API doesn't): write the wrappers; the GPU calls fail cleanly until next session.
- **`dsi/detectors/train.py`**: training loop skeleton — DataLoader plumbing, optimizer setup, checkpointing per CLAUDE.md §7 (every-N-steps, latest-4 retention, RNG state save/restore), WandB init per §6. Forward pass and loss step are GPU calls; everything around them is CPU-runnable.
- **`scripts/bootstrap.sh`**: a single shell script that does Step 5's downloads idempotently. The next session calls this on first launch.
- **WandB plumbing**: a `dsi/util/wandb.py` helper that wraps init with the standard tags / config dict / dir.
- **`tests/` directory**: pytest suite for everything CPU-runnable. Aim for at least 30 tests.

Commit progressively as you go: `scaffold(<scope>): <one-liner>` per commit.

---

## Step 7 — pre-write the experiment scripts

Create the script files in `scripts/` for items 1, 2, 3, 4, 5 (one per attack space, one per detector regime, one per intervention variant in the evaluation grid). Each is a CLI entrypoint with the standard `argparse` plus a `dry_run=True` mode that loads the config and prints what would happen. The actual GPU runs are gated.

This shifts the work of "what experiment to run next" out of the GPU session and into now, where you have time to think. The GPU session is then mechanical: pick a script, dry-run it, real-run it, monitor.

---

## Step 8 — pre-write the evaluation grid runner

`scripts/eval_full_grid.py` should take a `--row` arg matching one of the 12 rows in the §3 / §3.4 evaluation grid, run that row's full evaluation (all metrics, all benchmarks), and write to `reports/grid_row_<name>.md`. Wire it now; the GPU session populates the rows.

---

## Step 9 — final CPU-session deliverables checklist

Before declaring this session done:

- [ ] `PLAN.md` exists and is up to date with Phase 1b GPU schedule.
- [ ] Conda env `dsi` exists, packed to `/workspace/env-archives/dsi.tar.gz`.
- [ ] All datasets listed in Step 5 are downloaded; access requested for MMA-Diffusion image set.
- [ ] All model checkpoints in Step 5 are downloaded.
- [ ] All SAE checkpoints (Surkov, SAeUron, SAEmnesia) downloaded — or SAEmnesia queued for reproduce-from-scratch in `PLAN.md` if checkpoints are not released.
- [ ] **`scripts/verify_assets.py` runs green end-to-end** — every row of the verification matrix passes; `logs/verify_assets.json` written.
- [ ] Disk free space verified: ≥ 50 GB headroom remaining for cached activations and generations.
- [ ] Repo scaffolded per `CLAUDE.md` §15, every directory exists with at least an `__init__.py`.
- [ ] `import dsi` works from any cwd; `import dsi.attacks.pixel`, `import dsi.detectors.train`, etc. all import without error.
- [ ] CPU-runnable code is implemented and passes `pytest tests/` cleanly (≥ 30 tests passing).
- [ ] Experiment scripts in `scripts/` exist with `--dry-run` modes that pass.
- [ ] `scripts/bootstrap.sh` is idempotent — running it again should re-verify, not re-download.
- [ ] `.env` is set up with `HF_TOKEN` / `WANDB_API_KEY`; `huggingface-cli whoami` works; `wandb` logs in.
- [ ] Git repo initialized, remote on `git@github.com:Swadesh06/gen_ai_project_diffusion_interp.git` (or whatever the user named it), all work pushed to `origin/main`.
- [ ] `reports/INDEX.md` exists.
- [ ] `paper/main.tex` skeleton exists with section headers (introduction, related work, threat model, method, experiments, ablations, conclusion).
- [ ] Final commit: `chore: phase 1a complete, ready for gpu session`.

---

## Step 10 — when you're done

Stop. The next session will start with the GPU starter prompt and `git pull`, and pick up from here. Do not start GPU experiments yourself — there is no GPU on this pod.

You do not pause to ask permission for anything in steps 1–9. Read the docs, plan, scaffold, push, repeat. The human may be asleep.