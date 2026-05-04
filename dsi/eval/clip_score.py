"""CLIP-score wrapper. Uses open_clip + cosine similarity between text and image embeddings.

GPU-side; the API is callable on CPU as a fallback (slow).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CLIPScorer:
    model_id: str = "ViT-L-14"
    pretrained: str = "openai"
    device: str = "cpu"

    def load(self):
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_id, pretrained=self.pretrained, device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_id)
        self.model.eval()
        return self

    def score(self, images: list, texts: list[str]) -> list[float]:
        import torch
        from torch.nn.functional import normalize

        with torch.no_grad():
            xs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            ts = self.tokenizer(texts).to(self.device)
            ie = normalize(self.model.encode_image(xs), dim=-1)
            te = normalize(self.model.encode_text(ts), dim=-1)
            sims = (ie * te).sum(dim=-1).cpu().tolist()
        return sims
