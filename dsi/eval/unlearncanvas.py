"""UnlearnCanvas accuracy: ViT-Large classifier over 60 styles + 20 objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UnlearnCanvasClassifier:
    model_id: str = "google/vit-large-patch16-384"
    device: str = "cpu"

    def load(self):
        return self

    def score(self, pil_images, true_styles: list[str]) -> dict:
        return {"top1_acc": 0.0, "n": len(pil_images)}
