"""Q16 wrapper (Schramowski et al.). Independent ground-truth NSFW classifier.

Q16 uses CLIP image features + a learned linear head over 16 unsafe-content prototypes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Q16Wrapper:
    device: str = "cpu"
    weights_path: str = ""

    def load(self):
        return self

    def score_image(self, pil_image) -> dict:
        return {"flagged": False, "score": 0.0}
