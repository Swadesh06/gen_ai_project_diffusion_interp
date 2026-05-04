#!/bin/bash
# bootstrap.sh — idempotent downloader for all datasets and checkpoints.
# Re-run = re-verify. Each block is guarded by an existence check.
#
# Targets (per task_description_v1 §4 + appendix Phase-C):
#   datasets:    I2P, I2P-adv, COCO val, LAION-COCO subset, UnlearnCanvas,
#                MMA-Diffusion (text+image), UnlearnDiffAtk, Ring-A-Bell
#   models:      SDXL Turbo, SDXL Base, SD v1.5, CompVis safety checker,
#                CLIP ViT-L/14
#   classifiers: NudeNet, Q16, DreamSim, LPIPS
#   SAEs:        Surkov (4 hookpoints), SAeUron, [SAEmnesia: not released]
#
# After running: scripts/verify_assets.py must pass green.

set -e
source /home/swadesh/miniconda3/etc/profile.d/conda.sh
conda activate dsi

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(cat /workspace/.secrets/hf_token | tr -d '[:space:]')

DATASETS=/workspace/datasets
SAES=/workspace/checkpoints/saes
mkdir -p $DATASETS $SAES

echo "=== bootstrap: datasets ==="
huggingface-cli download AIML-TUDA/i2p --repo-type dataset >/dev/null
huggingface-cli download AIML-TUDA/i2p-adversarial-split --repo-type dataset >/dev/null || echo "i2p-adv split missing"

mkdir -p $DATASETS/coco
if [ ! -d $DATASETS/coco/val2017 ]; then
  cd $DATASETS/coco
  wget -q http://images.cocodataset.org/zips/val2017.zip
  unzip -q val2017.zip && rm val2017.zip
fi
if [ ! -f $DATASETS/coco/annotations/captions_val2017.json ]; then
  cd $DATASETS/coco
  wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip
fi

[ -d $DATASETS/unlearncanvas ] || huggingface-cli download OPTML-Group/UnlearnCanvas \
  --repo-type dataset --local-dir $DATASETS/unlearncanvas >/dev/null || echo "UnlearnCanvas missing"

if [ ! -f $DATASETS/laion_coco/captions.parquet ]; then
  mkdir -p $DATASETS/laion_coco
  python -c "
from datasets import load_dataset
import pandas as pd
ds = load_dataset('laion/laion-coco', split='train', streaming=True)
rows = []
for i, r in enumerate(ds):
    if i >= 50000: break
    rows.append(r.get('caption') or r.get('TEXT') or r.get('top_caption'))
pd.DataFrame({'caption': rows}).to_parquet('$DATASETS/laion_coco/captions.parquet')
"
fi

[ -d $DATASETS/MMA-Diffusion ] || git clone https://github.com/cure-lab/MMA-Diffusion.git $DATASETS/MMA-Diffusion
[ -d $DATASETS/Diffusion-MU-Attack ] || git clone https://github.com/OPTML-Group/Diffusion-MU-Attack.git $DATASETS/Diffusion-MU-Attack
[ -d $DATASETS/Ring-A-Bell ] || git clone https://github.com/chiayi-hsu/Ring-A-Bell.git $DATASETS/Ring-A-Bell
huggingface-cli download YijunYang280/MMA_Diffusion_adv_images_benchmark --repo-type dataset >/dev/null \
  || echo "MMA image set is gated; request access at https://huggingface.co/datasets/YijunYang280/MMA_Diffusion_adv_images_benchmark"

echo "=== bootstrap: models ==="
huggingface-cli download stabilityai/sdxl-turbo --repo-type model >/dev/null
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --repo-type model >/dev/null
huggingface-cli download runwayml/stable-diffusion-v1-5 --repo-type model >/dev/null \
  || huggingface-cli download benjamin-paine/stable-diffusion-v1-5 --repo-type model >/dev/null
huggingface-cli download CompVis/stable-diffusion-safety-checker --repo-type model >/dev/null
huggingface-cli download openai/clip-vit-large-patch14 --repo-type model >/dev/null

echo "=== bootstrap: classifier weights ==="
python -c "from nudenet import NudeDetector; NudeDetector(); print('nudenet ok')"
python -c "import lpips; lpips.LPIPS(net='alex'); print('lpips ok')"
python -c "import dreamsim; dreamsim.dreamsim(pretrained=True, device='cpu'); print('dreamsim ok')" || echo "dreamsim fail (non-fatal)"

echo "=== bootstrap: SAEs ==="
[ -d $SAES/surkov/checkpoints ] || huggingface-cli download surokpro2/Unboxing_SDXL_with_SAEs \
  --repo-type space --local-dir $SAES/surkov >/dev/null \
  || huggingface-cli download surokpro2/Unboxing_SDXL_with_SAEs --local-dir $SAES/surkov >/dev/null
[ -d $DATASETS/sdxl-unbox ] || git clone https://github.com/surkovv/sdxl-unbox.git $DATASETS/sdxl-unbox

[ -d $SAES/saeuron/unet.up_blocks.1.attentions.1 ] || huggingface-cli download bcywinski/SAeUron \
  --local-dir $SAES/saeuron >/dev/null
[ -d $DATASETS/SAeUron ] || git clone https://github.com/cywinski/SAeUron.git $DATASETS/SAeUron

# SAEmnesia not publicly released; queued for reproduce-from-scratch in PLAN.md.

echo "=== bootstrap: verify ==="
python "$(dirname "$0")/verify_assets.py" || { echo "verify_assets failed"; exit 1; }
echo "=== bootstrap: DONE ==="
