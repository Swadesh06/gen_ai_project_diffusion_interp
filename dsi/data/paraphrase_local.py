"""Local Llama 3.1 70B int8 paraphrase (Path B) for the counterfactual Strategy 3.

v2 §3 Item 1c-0 Strategy 3 Path B. Runs alongside Gemini Path A; the local model
handles whatever Gemini refuses (Llama has no usage policy enforcement at the
weights level). 70B int8 ~ 40 GB on the new pod's 96 GB GPU.

Public API:
    from dsi.data.paraphrase_local import LlamaParaphraser
    p = LlamaParaphraser(model_id='meta-llama/Meta-Llama-3.1-70B-Instruct', load_in_8bit=True)
    rows = p.paraphrase_anchors(anchors, cell='i2p_unsafe', n_per_anchor=1)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Literal, Optional

CellKind = Literal["i2p_safe", "i2p_unsafe", "coco_safe", "coco_unsafe"]


def _build_prompt_messages(anchor: str, cell: CellKind, n: int) -> list[dict]:
    style = "I2P-style fantasy / illustration / digital painting prompt" if "i2p" in cell else "MS-COCO-style natural-language caption"
    safety = "with explicit unsafe content (nudity / violence / gore / hate)" if cell.endswith("unsafe") else "with strictly safe content (no nudity, no violence, no gore, no hate)"
    sys = "You are a paraphrasing assistant. Output only the requested rewrites with no commentary, numbering, or quotation."
    user = (
        f"Rewrite the following anchor as a single {style} {safety}. "
        f"Match the anchor's topic but vary phrasing. Output exactly {n} version(s), one per line.\n\n"
        f"Anchor: {anchor}\n"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


class LlamaParaphraser:
    """Lazy-loaded Llama paraphraser. Loading is deferred to `load()` to keep import cheap."""

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        load_in_8bit: bool = True,
        device_map: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        torch_dtype: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.torch_dtype = torch_dtype
        self._tok = None
        self._mdl = None

    def load(self) -> "LlamaParaphraser":
        if self._tok is not None and self._mdl is not None:
            return self
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        kwargs: dict = {"device_map": self.device_map}
        if self.load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16 if self.torch_dtype == "auto" else self.torch_dtype
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        if self._tok.pad_token_id is None:
            self._tok.pad_token_id = self._tok.eos_token_id
        self._mdl = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._mdl.eval()
        return self

    def _generate(self, messages: list[dict]) -> str:
        import torch

        assert self._tok is not None and self._mdl is not None
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(self._mdl.device)
        with torch.no_grad():
            out = self._mdl.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tok.pad_token_id,
            )
        decoded = self._tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded.strip()

    def paraphrase_anchors(
        self,
        anchors: Iterable[str],
        cell: CellKind,
        *,
        n_per_anchor: int = 1,
        log_path: Optional[Path] = None,
    ) -> list[dict]:
        if self._mdl is None:
            self.load()
        rows: list[dict] = []
        for i, a in enumerate(anchors):
            messages = _build_prompt_messages(a, cell, n_per_anchor)
            try:
                txt = self._generate(messages)
                outs = self._parse_lines(txt, n_per_anchor)
                err = None
            except Exception as e:
                txt = ""
                outs = []
                err = f"{type(e).__name__}: {e}"
            rows.append({
                "anchor_id": i,
                "anchor": a,
                "cell": cell,
                "paraphrases": outs,
                "model_used": self.model_id,
                "refused": len(outs) == 0,
                "error": err,
            })
            if log_path is not None:
                with log_path.open("a") as f:
                    f.write(json.dumps(rows[-1]) + "\n")
        return rows

    @staticmethod
    def _parse_lines(text: str, n: int) -> list[str]:
        lines = [re.sub(r"^[\-\*\d\.\)]+\s*", "", l).strip() for l in text.splitlines()]
        out = [l for l in lines if l and not l.lower().startswith(("here ", "sure", "certainly", "i'm", "i can"))]
        return out[:n]
