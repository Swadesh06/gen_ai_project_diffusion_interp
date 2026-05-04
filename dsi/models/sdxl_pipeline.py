"""SDXL Turbo / SDXL Base pipeline wrapper exposing UNet hookpoints.

The SDXL UNet's block path layout (as exposed by `diffusers`):
  - `pipe.unet.down_blocks[2].attentions[1]`  -> "down.2.1"  (Surkov: composition, early)
  - `pipe.unet.mid_block.attentions[0]`       -> "mid.0"
  - `pipe.unet.up_blocks[0].attentions[0]`    -> "up.0.0"
  - `pipe.unet.up_blocks[0].attentions[1]`    -> "up.0.1"   (Surkov: local detail / colour)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dsi.config import cfg

SDXLVariant = Literal["turbo", "base"]


@dataclass
class SDXLPipelineWrapper:
    variant: SDXLVariant = "turbo"
    device: str = "cpu"
    dtype: str = "fp16"

    @property
    def model_id(self) -> str:
        return cfg.model.sdxl_turbo if self.variant == "turbo" else cfg.model.sdxl_base

    def load(self):
        import torch
        from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline

        if self.variant == "turbo":
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id, torch_dtype=torch.float16 if self.dtype == "fp16" else torch.float32,
                variant="fp16" if self.dtype == "fp16" else None,
            ).to(self.device)
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.float16 if self.dtype == "fp16" else torch.float32,
                variant="fp16" if self.dtype == "fp16" else None,
            ).to(self.device)
        if self.device == "cuda":
            self.pipe.set_progress_bar_config(disable=True)
        return self

    def generate(self, prompts: list[str], *, num_inference_steps: int = 1, guidance_scale: float = 0.0,
                 seed: int = 0, height: int = 512, width: int = 512):
        import torch

        gen = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt=prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
            height=height, width=width,
        )
        return out.images

    @property
    def unet(self):
        return self.pipe.unet

    @property
    def vae(self):
        return self.pipe.vae
