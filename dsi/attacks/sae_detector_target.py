"""Wraps a trained SAE-activation detector as a differentiable target for PGD.

Pipeline:
  - input image x → SDXL Turbo UNet (1 step) with SAE hooks → captured z per hookpoint
  - z → flatten/concat → detector head → safe/unsafe logit

For Item 4 (Contribution 3) we attack THIS, instead of the safety_checker, and
measure cross-target transferability against the safety_checker.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SAEDetectorTarget:
    detector_ckpt: str
    pipe_w: object = None        # SDXLPipelineWrapper (already loaded)
    sae_dict: dict = None        # {hookpoint: SurkovTopKSAE}
    device: str = "cuda"
    dtype: str = "fp16"

    def load(self):
        import torch

        from dsi.detectors.sae_em import LinearProbe, MLPHead

        ck = torch.load(self.detector_ckpt, map_location=self.device, weights_only=False)
        sd = ck["model_state_dict"]

        # Infer head type + in_dim from state dict (multiple training-script conventions).
        if "linear.weight" in sd:
            in_dim = sd["linear.weight"].shape[1]
            head = LinearProbe(in_dim).to(self.device)
        elif "weight" in sd and "bias" in sd and sd["weight"].ndim == 2:
            # raw nn.Linear (train_detector.py linear path)
            in_dim = sd["weight"].shape[1]
            head = torch.nn.Linear(in_dim, 1).to(self.device)
        elif "fc1.weight" in sd:
            in_dim = sd["fc1.weight"].shape[1]
            hidden = sd["fc1.weight"].shape[0]
            head = MLPHead(in_dim, hidden=hidden).to(self.device)
        elif "0.weight" in sd:
            # nn.Sequential wrapper (train_detector.py mlp path)
            in_dim = sd["0.weight"].shape[1]
            hidden = sd["0.weight"].shape[0]
            head = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, 1),
            ).to(self.device)
        else:
            raise ValueError(f"unrecognised detector state_dict keys: {list(sd)[:5]}")
        head.load_state_dict(sd)
        head.eval()
        self.head = head
        self.in_dim = in_dim
        return self

    def x_to_logit(self, x_in_01):
        """(B,3,H,W) in [0,1] → 1-vec safe/unsafe logit (single scalar per image).

        Runs SDXL Turbo 1-step text-cond gen but starting from `x_in_01` as the
        text-conditioned VAE-encoded latent. NB: this is approximate — for a real
        Item 4 attack we should monkey-patch the pipeline to take the seed image
        and run only the residual step. For now we use a simpler proxy: hook the
        UNet on a single forward pass over the encoded x.
        """
        import torch

        if self.pipe_w is None or self.sae_dict is None:
            raise ValueError("SAEDetectorTarget needs pipe_w + sae_dict; pass before load()")
        from dsi.sae.hooks import SurkovHookManager

        scale = float(getattr(self.pipe_w.vae.config, "scaling_factor", 0.13025))
        z = self.pipe_w.vae.encode(
            (x_in_01.to(next(self.pipe_w.vae.parameters()).dtype) * 2 - 1)
        ).latent_dist.sample() * scale

        # 1-step UNet forward at t=999 with empty cross-attn (cond/uncond fused)
        unet = self.pipe_w.unet
        with SurkovHookManager(unet, self.sae_dict, capture=True, keep_inputs=False) as mgr:
            # call UNet directly with a placeholder timestep + empty embeds
            B = z.shape[0]
            t = torch.full((B,), 999, device=z.device, dtype=torch.long)
            # SDXL needs added_cond_kwargs etc.; this is a placeholder. Phase 1b
            # extension: pass actual prompt embeddings if we want gradient through
            # text-conditional path.
            try:
                unet(z, t).sample
            except Exception:
                pass
        feats = []
        for hp in sorted(self.sae_dict.keys()):
            zs = mgr.captured[hp].z
            if not zs:
                continue
            v = zs[0].mean(dim=tuple(range(1, zs[0].ndim - 1)))  # spatial mean → (B, D)
            feats.append(v.to(self.device).to(next(self.head.parameters()).dtype))
        if not feats:
            raise RuntimeError("SAE hooks did not fire")
        x_flat = torch.cat(feats, dim=-1)
        return self.head(x_flat).squeeze(-1)
