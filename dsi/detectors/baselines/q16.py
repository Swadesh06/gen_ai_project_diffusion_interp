"""Q16 wrapper (Schramowski et al., FAccT 2022). Independent ground-truth NSFW classifier.

Q16 = CLIP image embedding (ViT-L/14 by default; ViT-B/16 reproduces paper) projected
against a 2-class soft prompt learned via prompt-tuning on the SMID inappropriate-content
dataset. Prediction = argmax of cosine similarity with the two class prototypes.

The soft prompts ship as a (2, 768) float16 numpy array under
`<datasets>/Q16/data/ViT-L-14/prompts.p` (768-d for ViT-L/14).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from dsi.config import cfg


def _default_prompts_path(variant: str = "ViT-L-14") -> Path:
    return cfg.paths.data_root / "Q16" / "data" / variant / "prompts.p"


@dataclass
class Q16Wrapper:
    device: str = "cpu"
    variant: str = "ViT-L-14"  # "ViT-L-14" | "ViT-B-16" | "ViT-B-32"
    weights_path: str = ""

    def load(self):
        import open_clip
        import torch

        wp = Path(self.weights_path) if self.weights_path else _default_prompts_path(self.variant)
        if not wp.exists():
            raise FileNotFoundError(f"Q16 prompts file not found: {wp}. Clone https://github.com/ml-research/Q16 to {cfg.paths.data_root}/Q16.")
        with wp.open("rb") as f:
            soft_prompts = pickle.load(f)
        self.prompts = torch.as_tensor(soft_prompts, device=self.device, dtype=torch.float32)
        clip_model = {"ViT-L-14": "ViT-L-14", "ViT-B-16": "ViT-B-16", "ViT-B-32": "ViT-B-32"}[self.variant]
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained="openai", device=self.device,
        )
        self.model.eval()
        return self

    def score_image(self, pil_image) -> dict:
        import torch
        from torch.nn.functional import normalize

        with torch.no_grad():
            x = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            ie = normalize(self.model.encode_image(x).float(), dim=-1)
            te = normalize(self.prompts, dim=-1)
            sims = (ie @ te.T).squeeze(0)
            probs = sims.softmax(dim=-1)
            cls = int(probs.argmax().item())
        return {"flagged": cls == 1, "score": float(probs[1].item()), "class_index": cls}
