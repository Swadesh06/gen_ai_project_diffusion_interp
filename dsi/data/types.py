"""Typed records consumed by all data loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class Prompt:
    text: str
    source: str
    label: Literal["nsfw", "benign", "unknown"] = "unknown"
    category: str = ""
    seed: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class ImageRecord:
    path: str
    label: Literal["nsfw", "benign", "unknown"] = "unknown"
    caption: Optional[str] = None
    source: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class Pair:
    prompt: Prompt
    image: Optional[ImageRecord] = None
