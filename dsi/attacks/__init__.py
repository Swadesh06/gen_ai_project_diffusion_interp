"""Three attack spaces against post-hoc safety classifiers.

pixel:     PGD on 3x512x512 image input
latent:    PGD on 4x64x64 SDXL VAE latent (decode-through-VAE)
embedding: PGD on the 768-d CLIP image embedding (attack ceiling)
"""

from dsi.attacks.common import (
    AttackBatchResult, AttackResult,
    asr_from_verdicts, run_attack,
)
from dsi.attacks.embedding import EmbeddingPGDConfig, pgd_attack_embedding
from dsi.attacks.latent import LatentPGDConfig, pgd_attack_latent
from dsi.attacks.pixel import PixelPGDConfig, pgd_attack_pixel, perturbation_norm

__all__ = [
    "AttackBatchResult", "AttackResult", "asr_from_verdicts", "run_attack",
    "PixelPGDConfig", "pgd_attack_pixel", "perturbation_norm",
    "LatentPGDConfig", "pgd_attack_latent",
    "EmbeddingPGDConfig", "pgd_attack_embedding",
]
