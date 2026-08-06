from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from openmed.core.labels import DATE, ID_NUM, ID_SUBTYPE_MRN, PERSON
from openmed.training.synthetic import BurnedInGenerator, generate_burned_in_example


def test_rendered_case_has_aligned_canonical_spans_and_controlled_boxes():
    positions = ((20, 20), (40, 120), (60, 220))
    example = BurnedInGenerator(seed=1790).generate(
        "en",
        positions=positions,
        font_sizes=(24, 22, 20),
        font_names=("DejaVuSans.ttf",) * 3,
        background="solid",
    )

    assert example.image.mode == "L"
    assert example.image.size == (640, 320)
    assert example.metadata["synthetic"] is True
    assert example.metadata["contains_real_phi"] is False
    assert example.metadata["augmentation_only"] is True
    assert example.metadata["synthetic_source"] == "burned_in"
    assert example.image.info["synthetic"] is True
    assert tuple(span.label for span in example.gold_spans) == (
        PERSON,
        DATE,
        ID_NUM,
    )
    assert len(example.gold_boxes) == len(example.gold_spans) == 3

    for position, span, box in zip(
        positions, example.gold_spans, example.gold_boxes, strict=True
    ):
        assert example.text[span.start : span.end] == span.text
        assert (box.start, box.end, box.label, box.text) == (
            span.start,
            span.end,
            span.label,
            span.text,
        )
        assert box.bbox[:2] == position
        assert box.canonical_label == span.label
        assert span.metadata["synthetic"] is True
        assert box.metadata["synthetic"] is True

    mrn_span = example.gold_spans[-1]
    assert mrn_span.text.startswith("MRN-")
    assert mrn_span.metadata["id_subtype"] == ID_SUBTYPE_MRN
    assert example.gold_boxes[-1].metadata["source"] == "clinical_ids"

    item = example.to_training_item()
    assert item["is_synthetic"] is True
    assert item["synthetic_source"] == "burned_in"
    assert item["labels"] == [span.to_dict() for span in example.gold_spans]
    assert item["pixel_boxes"] == [box.to_dict() for box in example.gold_boxes]


def test_gold_boxes_bound_every_drawn_text_pixel():
    example = BurnedInGenerator(seed=41).generate(
        positions=((16, 16), (16, 112), (16, 208)),
        font_sizes=(26, 26, 26),
        font_names=("DejaVuSansMono.ttf",) * 3,
        background="solid",
    )
    background = Image.new(
        "L", example.image.size, color=example.metadata["background_value"]
    )
    changed = ImageChops.difference(example.image, background)

    assert changed.getbbox() is not None
    for box in example.gold_boxes:
        x0, y0, x1, y1 = box.bbox
        assert 0 <= x0 < x1 <= example.image.width
        assert 0 <= y0 < y1 <= example.image.height
        assert changed.crop(box.bbox).getbbox() == (0, 0, x1 - x0, y1 - y0)

    for y in range(example.image.height):
        for x in range(example.image.width):
            if changed.getpixel((x, y)):
                assert any(
                    x0 <= x < x1 and y0 <= y < y1
                    for x0, y0, x1, y1 in (
                        annotation.bbox for annotation in example.gold_boxes
                    )
                )


def test_generation_is_deterministic_per_seed_and_varies_across_seeds():
    first = generate_burned_in_example(seed=101)
    second = generate_burned_in_example(seed=101)
    different = generate_burned_in_example(seed=102)

    assert first.text == second.text
    assert first.gold_spans == second.gold_spans
    assert first.gold_boxes == second.gold_boxes
    assert first.metadata == second.metadata
    assert first.image.tobytes() == second.image.tobytes()
    assert (
        first.text,
        first.gold_boxes,
        first.image.tobytes(),
    ) != (
        different.text,
        different.gold_boxes,
        different.image.tobytes(),
    )


def test_importing_core_and_burned_in_module_does_not_import_pillow():
    code = (
        "import sys\n"
        "import openmed\n"
        "import openmed.training.synthetic.burned_in\n"
        "print(any(name == 'PIL' or name.startswith('PIL.') "
        "for name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_renderer_names_multimodal_extra_when_pillow_is_missing(monkeypatch):
    module = importlib.import_module("openmed.training.synthetic.burned_in")
    real_import_module = module.importlib.import_module

    def import_without_pillow(name: str):
        if name == "PIL" or name.startswith("PIL."):
            raise ModuleNotFoundError(name)
        return real_import_module(name)

    monkeypatch.setattr(module.importlib, "import_module", import_without_pillow)

    with pytest.raises(ImportError, match=r"openmed\[multimodal\]"):
        BurnedInGenerator(seed=7).generate()


def test_multimodal_extra_owns_pillow_dependency():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
        import tomli as tomllib

    repository_root = Path(__file__).parents[3]
    with (repository_root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    core_dependencies = project["dependencies"]
    multimodal_dependencies = project["optional-dependencies"]["multimodal"]
    assert not any(
        dependency.lower().startswith("pillow") for dependency in core_dependencies
    )
    assert any(
        dependency.lower().startswith("pillow")
        for dependency in multimodal_dependencies
    )
