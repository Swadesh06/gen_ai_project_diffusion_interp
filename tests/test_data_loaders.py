"""Data-loader smoke. We only check the API and label heuristics; actual data may not be downloaded yet."""

from __future__ import annotations

from dsi.data import ImageRecord, Pair, Prompt
from dsi.data.adversarial import load_adversarial, load_mma_text, load_ringabell, load_unlearndiff
from dsi.data.coco import load_coco_captions, load_coco_val_images
from dsi.data.i2p import load_i2p, _category_label
from dsi.data.laion_coco import load_laion_coco
from dsi.data.unlearncanvas import list_styles, load_images


def test_types():
    p = Prompt(text="a", source="t", label="nsfw")
    assert p.label == "nsfw" and p.text == "a"
    img = ImageRecord(path="/x.png")
    pair = Pair(prompt=p, image=img)
    assert pair.prompt is p


def test_i2p_category_label():
    assert _category_label("nudity") == "nsfw"
    assert _category_label("violence,hate") == "nsfw"
    assert _category_label("photography") == "benign"
    assert _category_label("") == "benign"


def test_loaders_no_crash_when_no_data():
    """All loaders return empty list (not crash) when assets are missing."""
    assert isinstance(load_coco_captions(limit=2), list)
    assert isinstance(load_coco_val_images(limit=2), list)
    assert isinstance(load_laion_coco(limit=2), list)
    assert isinstance(list_styles(), list)
    assert isinstance(load_images(style="van_gogh"), list)
    assert isinstance(load_mma_text(limit=2), list)
    assert isinstance(load_unlearndiff(limit=2), list)
    assert isinstance(load_ringabell(limit=2), list)


def test_adversarial_dispatch():
    for src in ("mma_text", "mma_image", "unlearndiff", "ringabell"):
        out = load_adversarial(src, limit=1)
        assert isinstance(out, list)


def test_adversarial_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        load_adversarial("nonexistent", limit=1)  # type: ignore


def test_i2p_loads_or_empty():
    out = load_i2p("full", limit=1)
    assert isinstance(out, list)
