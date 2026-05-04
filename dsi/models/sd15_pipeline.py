"""SD v1.5 pipeline wrapper. Used for SAeUron baseline reproduction (their SAEs target SD v1.5)."""

from __future__ import annotations

from dataclasses import dataclass

from dsi.config import cfg


@dataclass
class SD15PipelineWrapper:
    device: str = "cpu"
    dtype: str = "fp16"
    fallback_id: str = ""

    @property
    def model_id(self) -> str:
        return cfg.model.sd15

    def load(self):
        import torch
        from diffusers import StableDiffusionPipeline

        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.float16 if self.dtype == "fp16" else torch.float32,
                safety_checker=None,
            ).to(self.device)
        except Exception:
            fb = self.fallback_id or cfg.model.sd15_fallback
            self.pipe = StableDiffusionPipeline.from_pretrained(
                fb, torch_dtype=torch.float16 if self.dtype == "fp16" else torch.float32,
                safety_checker=None,
            ).to(self.device)
        return self

    def generate(self, prompts: list[str], *, num_inference_steps: int = 25, guidance_scale: float = 7.5,
                 seed: int = 0, height: int = 512, width: int = 512):
        import torch

        gen = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(prompt=prompts, num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale, generator=gen, height=height, width=width)
        return out.images

    @property
    def unet(self):
        return self.pipe.unet
