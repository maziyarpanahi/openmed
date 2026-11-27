"""Generate synthetic label-reading and counting photos; never use patient data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

FIXTURE_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (320, 192)
BOX_COORDINATE_SYSTEM = "half-open pixels [x0, y0, x1, y1] in the RGB raster"
SCHEMA_VERSION = "openmed.photo_qa_fixtures.v1"

_LABEL_SCALE = 4
_FONT = ImageFont.load_default_imagefont()
_COLOR_RGB = {
    "blue": (55, 105, 180),
    "green": (60, 145, 95),
    "red": (190, 70, 65),
}
_PLURALS = {"circle": "circles", "square": "squares", "triangle": "triangles"}
_SHAPES = tuple(_PLURALS)


@dataclass(frozen=True)
class LabelCase:
    """One fictitious label-reading question and its printed answer."""

    case_id: str
    text: str
    question: str

    @property
    def answer(self) -> str:
        return self.text


@dataclass(frozen=True)
class CountCase:
    """One synthetic shape-counting question."""

    case_id: str
    shape: str
    color: str
    count: int

    @property
    def question(self) -> str:
        return f"How many {self.color} {_PLURALS[self.shape]} are visible?"

    @property
    def answer(self) -> str:
        return str(self.count)


Case = LabelCase | CountCase

LABEL_CASES = (
    LabelCase("label_lot_2718", "LOT 2718", "What code is printed on the label?"),
    LabelCase("label_lot_4409", "LOT 4409", "Read the printed code."),
    LabelCase("label_kit_7b2", "KIT 7B2", "What label text is visible?"),
    LabelCase("label_dose_05", "DOSE 05", "What dose text is printed?"),
    LabelCase("label_dose_12", "DOSE 12", "Read the dose shown on the label."),
    LabelCase("label_ref_83c", "REF 83C", "What reference code is visible?"),
    LabelCase("label_set_44", "SET 44", "What set code is printed?"),
    LabelCase("label_unit_b12", "UNIT B12", "What unit code is visible?"),
    LabelCase("label_code_qx9", "CODE QX9", "Read the code on this label."),
    LabelCase("label_lot_a17f", "LOT A17F", "What lot text is printed?"),
)

COUNT_CASES = (
    CountCase("count_red_circles_1", "circle", "red", 1),
    CountCase("count_red_circles_2", "circle", "red", 2),
    CountCase("count_red_circles_3", "circle", "red", 3),
    CountCase("count_red_circles_4", "circle", "red", 4),
    CountCase("count_blue_squares_1", "square", "blue", 1),
    CountCase("count_blue_squares_2", "square", "blue", 2),
    CountCase("count_blue_squares_3", "square", "blue", 3),
    CountCase("count_green_triangles_2", "triangle", "green", 2),
    CountCase("count_green_triangles_3", "triangle", "green", 3),
    CountCase("count_red_squares_2", "square", "red", 2),
)

CASES: tuple[Case, ...] = LABEL_CASES + COUNT_CASES
_CASES_BY_ID = {case.case_id: case for case in CASES}

# Four columns by three rows. Targets use a fixed subset of slots and the
# remaining slots receive contrasting distractors.
_CENTERS = tuple(
    (48 + 72 * column, 50 + 50 * row) for row in range(3) for column in range(4)
)
_TARGET_SLOTS = (1, 6, 8, 3)
_DISTRACTOR_SLOTS = tuple(
    index for index in range(len(_CENTERS)) if index not in _TARGET_SLOTS
)


def _digest(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _shape_bounds(
    center: tuple[int, int], *, radius: int = 20
) -> tuple[int, int, int, int]:
    x, y = center
    return (x - radius, y - radius, x + radius + 1, y + radius + 1)


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    shape: str,
    color: str,
) -> None:
    x, y = center
    radius = 20
    fill = _COLOR_RGB[color]
    outline = tuple(max(0, channel - 70) for channel in fill)
    draw.ellipse(
        (x - radius + 3, y - radius + 5, x + radius + 3, y + radius + 5),
        fill=(208, 212, 214),
    )
    bounds = _shape_bounds(center, radius=radius)
    if shape == "circle":
        draw.ellipse(bounds, fill=fill, outline=outline, width=2)
    elif shape == "square":
        draw.rounded_rectangle(bounds, radius=4, fill=fill, outline=outline, width=2)
    else:
        draw.polygon(
            ((x, y - radius), (x - radius, y + radius), (x + radius, y + radius)),
            fill=fill,
            outline=outline,
        )
    draw.ellipse((x - 11, y - 11, x - 6, y - 6), fill=(235, 238, 238))


def _render_label(case: LabelCase) -> tuple[Image.Image, tuple[int, int, int, int]]:
    base = Image.new(
        "RGB",
        (IMAGE_SIZE[0] // _LABEL_SCALE, IMAGE_SIZE[1] // _LABEL_SCALE),
        (231, 237, 241),
    )
    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle(
        (4, 4, 75, 43), radius=3, fill=(252, 252, 249), outline=(177, 185, 192)
    )
    text_origin = (8, 15)
    text_box = draw.textbbox(text_origin, case.text, font=_FONT)
    draw.text(text_origin, case.text, font=_FONT, fill=(22, 27, 31))
    for index, height in enumerate((7, 11, 8, 12, 6, 10, 8)):
        x = 12 + index * 8
        draw.line((x, 36, x, 36 - height), fill=(70, 76, 82))
        draw.line((x + 2, 36, x + 2, 36 - height), fill=(70, 76, 82))
    image = base.resize(IMAGE_SIZE, Image.Resampling.NEAREST)
    box = tuple(value * _LABEL_SCALE for value in text_box)
    return image, box


def _render_count(case: CountCase) -> tuple[Image.Image, tuple[int, int, int, int]]:
    image = Image.new("RGB", IMAGE_SIZE, (229, 235, 239))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (10, 10, 309, 181),
        radius=12,
        fill=(245, 246, 242),
        outline=(202, 208, 212),
        width=2,
    )
    target_slots = _TARGET_SLOTS[: case.count]
    for slot in target_slots:
        _draw_shape(draw, _CENTERS[slot], case.shape, case.color)

    distractors = [
        (shape, color)
        for shape in _SHAPES
        for color in _COLOR_RGB
        if (shape, color) != (case.shape, case.color)
    ]
    for index, slot in enumerate(_DISTRACTOR_SLOTS[:6]):
        shape, color = distractors[index % len(distractors)]
        _draw_shape(draw, _CENTERS[slot], shape, color)

    bounds = [_shape_bounds(_CENTERS[slot]) for slot in target_slots]
    box = (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )
    return image, box


def render_case(case: Case) -> tuple[Image.Image, tuple[int, int, int, int], str]:
    """Render one deterministic RGB image, answer box and fixture kind."""
    if isinstance(case, LabelCase):
        image, box = _render_label(case)
        return image, box, "label_reading"
    image, box = _render_count(case)
    return image, box, "counting"


def build_fixtures(
    output_dir: str | Path = FIXTURE_DIR,
) -> dict[str, Any]:
    """Write all PNGs and the truth manifest, returning the manifest."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    items = []
    for case in CASES:
        image, box, kind = render_case(case)
        filename = f"{case.case_id}.png"
        image.save(root / filename, format="PNG", optimize=False)
        items.append(
            {
                "id": case.case_id,
                "kind": kind,
                "image": filename,
                "question": case.question,
                "answer": case.answer,
                "box": list(box),
                "sha256": _digest(image),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Synthetic photo-style label-reading and counting fixtures. "
            "Every label text and code is invented."
        ),
        "box_coordinate_system": BOX_COORDINATE_SYSTEM,
        "items": items,
    }
    (root / "truth.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def case_by_id(case_id: str) -> Case:
    """Return a case by its stable manifest id."""
    return _CASES_BY_ID[case_id]


if __name__ == "__main__":
    build_fixtures()
