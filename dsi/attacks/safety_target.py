"""Wraps `CompVis/stable-diffusion-safety-checker` as differentiable target heads
for pixel / latent / embedding PGD attacks.

The CompVis safety checker is `StableDiffusionSafetyChecker`, a CLIP image encoder
(`openai/clip-vit-large-patch14`) followed by two prototypes (concept_embeds,
special_care_embeds) and a cosine-similarity test. We expose three callable
targets:

  - `pixel_to_logits(images_in_01)`: image (B, 3, H, W) in [0, 1] → 2-vector logit
  - `embedding_to_logits(emb_768)`: CLIP image embedding (B, 768) → 2-vector logit
  - `vae_latent_to_logits(z)`: SDXL VAE latent (B, 4, 64, 64) → 2-vector logit
    (decodes through VAE, then runs `pixel_to_logits` on the result)

The 2-vector logit is `[unsafe_logit, safe_logit]`. PGD maximises `safe_logit` to
make the safety checker stop flagging the image (untargeted bypass).
"""

from __future__ import annotations

from dataclasses import dataclass

from dsi.config import cfg

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class SafetyTarget:
    device: str = "cuda"
    dtype: str = "fp16"

    def load(self, vae=None):
        import torch

        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

        td = torch.float16 if self.dtype == "fp16" else torch.float32
        self.model = (
            StableDiffusionSafetyChecker.from_pretrained(cfg.model.safety_checker, torch_dtype=td)
            .to(self.device)
            .eval()
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.vision = self.model.vision_model
        self.visual_projection = self.model.visual_projection
        self.concept_embeds = self.model.concept_embeds
        self.special_care_embeds = self.model.special_care_embeds
        self.concept_embeds_weights = self.model.concept_embeds_weights
        self.special_care_embeds_weights = self.model.special_care_embeds_weights

        # CLIP normalisation (224x224, mean+std)
        self.mean = torch.tensor(CLIP_MEAN, device=self.device, dtype=td).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=self.device, dtype=td).view(1, 3, 1, 1)
        self.vae = vae
        return self

    @staticmethod
    def _interp_to_224(x):
        import torch.nn.functional as F

        return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

    def _preprocess_pixel(self, images_in_01):
        td = self.mean.dtype
        x = images_in_01.to(self.device).to(td)
        x = self._interp_to_224(x)
        x = (x - self.mean) / self.std
        return x

    def pixel_to_embedding(self, images_in_01):
        """(B,3,H,W) in [0,1] → 768-d normalized projected embedding (gradient flows)."""
        import torch.nn.functional as F

        x = self._preprocess_pixel(images_in_01)
        out = self.vision(pixel_values=x)
        pooled = out[1]
        emb = self.visual_projection(pooled)
        return F.normalize(emb, dim=-1)

    def embedding_to_concept_logits(self, emb_normalised):
        """(B, 768) normalised embedding → (B, 2): [max special_care, max concept].

        Higher = more unsafe.
        """
        import torch
        import torch.nn.functional as F

        special_norm = F.normalize(self.special_care_embeds, dim=-1)
        concept_norm = F.normalize(self.concept_embeds, dim=-1)
        sc_sim = emb_normalised @ special_norm.T
        cc_sim = emb_normalised @ concept_norm.T
        sc_score = sc_sim - self.special_care_embeds_weights[None, :]
        cc_score = cc_sim - self.concept_embeds_weights[None, :]
        unsafe_logit = torch.maximum(sc_score.max(dim=-1).values, cc_score.max(dim=-1).values)
        safe_logit = -unsafe_logit
        return torch.stack([unsafe_logit, safe_logit], dim=-1)

    def pixel_to_logits(self, images_in_01):
        emb = self.pixel_to_embedding(images_in_01)
        return self.embedding_to_concept_logits(emb)

    def embedding_to_logits(self, emb_unnormalised):
        """(B, 768) unnormalised → 2-vector logit."""
        import torch.nn.functional as F

        return self.embedding_to_concept_logits(F.normalize(emb_unnormalised, dim=-1))

    def vae_latent_to_pixel(self, z):
        """SDXL VAE decode: (B, 4, h, w) → (B, 3, H, W) in [0, 1].

        The VAE is fp32 (the SDXL fp16 VAE overflows). We cast `z` accordingly
        and cast the decoded image back to the safety-target dtype before clamping.
        """
        if self.vae is None:
            raise ValueError("SafetyTarget needs vae=... in load() to use VAE-latent attacks")
        scale = float(getattr(self.vae.config, "scaling_factor", 0.13025))
        z_vae = z.to(next(self.vae.parameters()).dtype)
        img = self.vae.decode(z_vae / scale).sample
        return ((img.clamp(-1, 1) + 1.0) / 2.0).to(self.mean.dtype)

    def vae_latent_to_logits(self, z):
        return self.pixel_to_logits(self.vae_latent_to_pixel(z))

    def flagged(self, images_in_01) -> list[bool]:
        """Boolean per-image: True iff the safety checker would flag (NSFW)."""
        import torch

        with torch.no_grad():
            logits = self.pixel_to_logits(images_in_01)
        return (logits[:, 0] > 0).cpu().tolist()
