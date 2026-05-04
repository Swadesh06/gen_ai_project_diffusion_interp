"""Wrapper around `CompVis/stable-diffusion-safety-checker` used as a baseline detector.

GPU-side. We expose a thin loader and a `score(image)` API so all baselines look the same.
"""

from __future__ import annotations

from dataclasses import dataclass

from dsi.config import cfg


@dataclass
class SafetyCheckerWrapper:
    model_id: str = cfg.model.safety_checker
    device: str = "cpu"

    def load(self):
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor

        self.model = StableDiffusionSafetyChecker.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = CLIPImageProcessor.from_pretrained(self.model_id)
        return self

    def score(self, pil_images: list) -> list[dict]:
        """Returns per-image {flagged: bool, score: float}."""
        import torch

        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _, has_nsfw = self.model(images=[i for i in inputs.pixel_values],
                                     clip_input=inputs.pixel_values)
        return [{"flagged": bool(h), "score": float(h)} for h in has_nsfw]
