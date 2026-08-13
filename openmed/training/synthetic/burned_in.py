"""Synthetic burned-in PHI images with aligned text and pixel gold labels.

Pillow is deliberately imported only when a case is rendered. Importing this
module, :mod:`openmed.training.synthetic`, or the core :mod:`openmed` package
therefore remains safe for installations that do not include the
``multimodal`` extra.
"""

from __future__ import annotations

import hashlib
import importlib
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Mapping, Sequence

from faker import Faker

from openmed.core import labels as L
from openmed.core.anonymizer.providers import clinical_ids

from .locale_phi import LocalePhiGenerator, SyntheticPhiSpan

if TYPE_CHECKING:
    from PIL.Image import Image as PillowImage
else:
    PillowImage = Any

BURNED_IN_LABELS: Final[tuple[str, ...]] = (L.PERSON, L.DATE, L.ID_NUM)
DEFAULT_CANVAS_SIZE: Final[tuple[int, int]] = (640, 320)
DEFAULT_FONT_SIZES: Final[tuple[int, ...]] = (18, 22, 26)
DEFAULT_FONT_NAMES: Final[tuple[str, ...]] = (
    "DejaVuSans.ttf",
    "DejaVuSansMono.ttf",
    "DejaVuSerif.ttf",
)
BACKGROUND_MODES: Final[tuple[str, ...]] = ("solid", "gradient", "phantom")

_FIELD_BY_LABEL: Final[Mapping[str, str]] = {
    L.PERSON: "person",
    L.DATE: "date",
    L.ID_NUM: "medical_record_number",
}
_MULTIMODAL_INSTALL_HINT: Final = 'Install with: pip install "openmed[multimodal]".'


@dataclass(frozen=True)
class BurnedInTextBox:
    """Pixel-space gold box for one rendered canonical PHI span.

    ``bbox`` uses Pillow's half-open ``(x0, y0, x1, y1)`` convention. The
    character offsets address :attr:`BurnedInExample.text` and are identical
    to those on the corresponding :class:`SyntheticPhiSpan`.
    """

    bbox: tuple[int, int, int, int]
    start: int
    end: int
    label: str
    text: str
    font_name: str
    font_size: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def canonical_label(self) -> str:
        """Return the canonical OpenMed label for the rendered text."""

        return self.label

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible annotation record."""

        return {
            "bbox": list(self.bbox),
            "end": self.end,
            "font_name": self.font_name,
            "font_size": self.font_size,
            "label": self.label,
            "metadata": dict(self.metadata),
            "start": self.start,
            "text": self.text,
        }


@dataclass(frozen=True)
class BurnedInExample:
    """One synthetic image and its aligned pixel/text annotations."""

    image: PillowImage
    text: str
    gold_boxes: tuple[BurnedInTextBox, ...]
    gold_spans: tuple[SyntheticPhiSpan, ...]
    language: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def boxes(self) -> tuple[BurnedInTextBox, ...]:
        """Return the gold boxes using the concise training-data alias."""

        return self.gold_boxes

    def to_training_item(self) -> dict[str, Any]:
        """Return a training-ready record while keeping the image in memory."""

        return {
            "image": self.image,
            "is_synthetic": True,
            "labels": [span.to_dict() for span in self.gold_spans],
            "language": self.language,
            "metadata": dict(self.metadata),
            "pixel_boxes": [box.to_dict() for box in self.gold_boxes],
            "synthetic_source": "burned_in",
            "text": self.text,
        }


class BurnedInGenerator:
    """Render deterministic names, dates, and MRNs into synthetic image pixels.

    Args:
        seed: Seed controlling PHI values, background, font, size, and layout.
        canvas_size: Output image size as ``(width, height)`` pixels.
        font_sizes: Positive font-size pool used for generated layouts.
        font_names: TrueType font-name pool used for generated layouts. The
            default DejaVu choices fall back to Pillow's bundled default font
            when unavailable on the current platform.
        margin: Minimum edge margin for automatically positioned text.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
        font_sizes: Sequence[int] = DEFAULT_FONT_SIZES,
        font_names: Sequence[str] = DEFAULT_FONT_NAMES,
        margin: int = 16,
    ) -> None:
        width, height = _validate_canvas_size(canvas_size)
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if width <= margin * 2 or height <= margin * 2:
            raise ValueError("margin leaves no drawable canvas area")

        self.seed = seed
        self.canvas_size = (width, height)
        self.font_sizes = _validate_font_sizes(font_sizes)
        self.font_names = _validate_font_names(font_names)
        self.margin = margin
        self._rng = random.Random(seed)

    def generate(
        self,
        language: str = "en",
        *,
        positions: Sequence[tuple[int, int]] | None = None,
        font_sizes: Sequence[int] | None = None,
        font_names: Sequence[str] | None = None,
        background: str | None = None,
    ) -> BurnedInExample:
        """Generate one synthetic burned-in PHI image.

        ``positions`` controls the top-left pixel of each gold box in
        ``(PERSON, DATE, ID_NUM)`` order. Per-item ``font_sizes`` and
        ``font_names`` use that same order. Omitting these controls selects
        deterministic variations from the configured pools.

        Args:
            language: Locale-PHI language used for the name and date strings.
            positions: Optional three exact ``(x, y)`` gold-box origins.
            font_sizes: Optional three exact positive font sizes.
            font_names: Optional three exact TrueType font names or paths.
            background: Optional one of :data:`BACKGROUND_MODES`.

        Returns:
            A synthetic Pillow image with aligned canonical spans and boxes.

        Raises:
            ImportError: If Pillow from the ``multimodal`` extra is missing.
            ValueError: If a rendering control is invalid or text cannot fit.
        """

        image_mod, draw_mod, font_mod = _import_pillow()
        controlled_positions = _validate_positions(positions)
        controlled_sizes = (
            _validate_exact_controls(font_sizes, "font_sizes", _validate_font_size)
            if font_sizes is not None
            else None
        )
        controlled_fonts = (
            _validate_exact_controls(font_names, "font_names", _validate_font_name)
            if font_names is not None
            else None
        )
        if background is not None and background not in BACKGROUND_MODES:
            raise ValueError(
                f"unsupported background {background!r}; "
                f"supported={list(BACKGROUND_MODES)!r}"
            )

        case_seed = self._case_seed(language)
        rng = random.Random(case_seed)
        values = self._synthetic_values(language, case_seed)
        text, spans = _canonical_text(values)
        width, height = self.canvas_size
        background_mode = background or rng.choice(BACKGROUND_MODES)
        image, background_metadata = _render_background(
            image_mod,
            draw_mod,
            rng,
            size=self.canvas_size,
            mode=background_mode,
        )

        line_height = (height - (2 * self.margin)) // len(values)
        if line_height <= 0:
            raise ValueError("canvas is too short for the burned-in PHI labels")

        boxes: list[BurnedInTextBox] = []
        for index, (value, label, value_metadata) in enumerate(values):
            requested_size = (
                controlled_sizes[index]
                if controlled_sizes is not None
                else rng.choice(self.font_sizes)
            )
            requested_font = (
                controlled_fonts[index]
                if controlled_fonts is not None
                else rng.choice(self.font_names)
            )

            if controlled_positions is None:
                lane_top = self.margin + (line_height * index)
                max_width = width - (2 * self.margin)
                max_height = line_height
            else:
                lane_top = controlled_positions[index][1]
                max_width = width - controlled_positions[index][0]
                max_height = height - lane_top

            font, actual_font_name, actual_size, measured_box = _fit_font(
                image_mod,
                draw_mod,
                font_mod,
                text=value,
                font_name=requested_font,
                font_size=requested_size,
                max_width=max_width,
                max_height=max_height,
                allow_resize=controlled_sizes is None,
            )
            text_width = measured_box[2] - measured_box[0]
            text_height = measured_box[3] - measured_box[1]

            if controlled_positions is None:
                max_x = width - self.margin - text_width
                max_y = lane_top + line_height - text_height
                if max_x < self.margin or max_y < lane_top:
                    raise ValueError("rendered PHI text does not fit the canvas")
                target_x = rng.randint(self.margin, max_x)
                target_y = rng.randint(lane_top, max_y)
            else:
                target_x, target_y = controlled_positions[index]

            mask = image_mod.new("L", self.canvas_size, color=0)
            mask_draw = draw_mod.Draw(mask)
            origin = (
                target_x - measured_box[0],
                target_y - measured_box[1],
            )
            mask_draw.text(origin, value, fill=255, font=font)
            raster_box = mask.getbbox()
            if raster_box is None:
                raise ValueError("rendered PHI text produced an empty pixel mask")
            if raster_box[2] > width or raster_box[3] > height:
                raise ValueError("rendered PHI text exceeds the canvas bounds")

            image.paste(255, (0, 0, width, height), mask)
            span = spans[index]
            box_metadata = {
                "field": _FIELD_BY_LABEL[label],
                "synthetic": True,
                **dict(value_metadata),
            }
            boxes.append(
                BurnedInTextBox(
                    bbox=tuple(int(coordinate) for coordinate in raster_box),
                    start=span.start,
                    end=span.end,
                    label=label,
                    text=value,
                    font_name=actual_font_name,
                    font_size=actual_size,
                    metadata=box_metadata,
                )
            )

        metadata = {
            "augmentation_only": True,
            "background": background_mode,
            **background_metadata,
            "canvas_size": self.canvas_size,
            "case_seed": case_seed,
            "contains_real_phi": False,
            "language": language,
            "seed": self.seed,
            "synthetic": True,
            "synthetic_source": "burned_in",
        }
        image.info.update(
            {
                "contains_real_phi": False,
                "synthetic": True,
                "synthetic_source": "burned_in",
            }
        )
        return BurnedInExample(
            image=image,
            text=text,
            gold_boxes=tuple(boxes),
            gold_spans=spans,
            language=language,
            metadata=metadata,
        )

    def _case_seed(self, language: str) -> int:
        if self.seed is None:
            return self._rng.getrandbits(64)
        material = f"{self.seed}|{language}|burned_in".encode()
        digest = hashlib.blake2b(material, digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=False)

    @staticmethod
    def _synthetic_values(
        language: str, case_seed: int
    ) -> tuple[tuple[str, str, Mapping[str, Any]], ...]:
        locale_example = LocalePhiGenerator(seed=case_seed).generate(language)
        name = next(
            span for span in locale_example.gold_spans if span.label == L.PERSON
        )
        date = next(span for span in locale_example.gold_spans if span.label == L.DATE)

        faker = Faker("en_US")
        faker.seed_instance(case_seed)
        faker.add_provider(clinical_ids.MedicalRecordNumberProvider)
        mrn = faker.medical_record_number()

        return (
            (name.text, L.PERSON, {"source": "locale_phi"}),
            (date.text, L.DATE, {"source": "locale_phi"}),
            (
                mrn,
                L.ID_NUM,
                {
                    "generator": "clinical_ids.MedicalRecordNumberProvider",
                    "id_subtype": L.ID_SUBTYPE_MRN,
                    "source": "clinical_ids",
                },
            ),
        )


def generate_burned_in_example(
    *,
    seed: int | None = None,
    language: str = "en",
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
) -> BurnedInExample:
    """Generate one synthetic burned-in PHI example with default variation."""

    return BurnedInGenerator(seed=seed, canvas_size=canvas_size).generate(language)


def _canonical_text(
    values: Sequence[tuple[str, str, Mapping[str, Any]]],
) -> tuple[str, tuple[SyntheticPhiSpan, ...]]:
    chunks: list[str] = []
    spans: list[SyntheticPhiSpan] = []
    cursor = 0
    for index, (value, label, metadata) in enumerate(values):
        if index:
            chunks.append("\n")
            cursor += 1
        start = cursor
        chunks.append(value)
        cursor += len(value)
        spans.append(
            SyntheticPhiSpan(
                start=start,
                end=cursor,
                label=label,
                text=value,
                metadata={"synthetic": True, **dict(metadata)},
            )
        )
    return "".join(chunks), tuple(spans)


def _render_background(
    image_mod: Any,
    draw_mod: Any,
    rng: random.Random,
    *,
    size: tuple[int, int],
    mode: str,
) -> tuple[Any, Mapping[str, Any]]:
    width, height = size
    if mode == "solid":
        value = rng.choice((0, 12, 24, 36, 48))
        return image_mod.new("L", size, color=value), {"background_value": value}

    image = image_mod.new("L", size, color=0)
    draw = draw_mod.Draw(image)
    if mode == "gradient":
        start = rng.randint(0, 20)
        end = rng.randint(48, 96)
        vertical = bool(rng.getrandbits(1))
        steps = height if vertical else width
        for offset in range(steps):
            ratio = offset / max(steps - 1, 1)
            value = round(start + ((end - start) * ratio))
            if vertical:
                draw.line((0, offset, width, offset), fill=value)
            else:
                draw.line((offset, 0, offset, height), fill=value)
        return image, {
            "background_end": end,
            "background_start": start,
            "background_vertical": vertical,
        }

    base = rng.randint(0, 18)
    image.paste(base, (0, 0, width, height))
    ellipse_count = rng.randint(2, 4)
    for _ in range(ellipse_count):
        x0 = rng.randint(0, max(width // 2, 1))
        y0 = rng.randint(0, max(height // 2, 1))
        x1 = rng.randint(max(x0 + 1, width // 2), width)
        y1 = rng.randint(max(y0 + 1, height // 2), height)
        draw.ellipse((x0, y0, x1, y1), fill=rng.randint(35, 115))
    return image, {
        "background_base": base,
        "background_ellipses": ellipse_count,
    }


def _fit_font(
    image_mod: Any,
    draw_mod: Any,
    font_mod: Any,
    *,
    text: str,
    font_name: str,
    font_size: int,
    max_width: int,
    max_height: int,
    allow_resize: bool,
) -> tuple[Any, str, int, tuple[int, int, int, int]]:
    if max_width <= 0 or max_height <= 0:
        raise ValueError("text position leaves no drawable canvas area")

    size = font_size
    while size >= 8:
        font, actual_font_name = _load_font(font_mod, font_name, size)
        measure_draw = draw_mod.Draw(image_mod.new("L", (1, 1), color=0))
        typographic_box = measure_draw.textbbox((0, 0), text, font=font)
        typographic_width = int(typographic_box[2] - typographic_box[0])
        typographic_height = int(typographic_box[3] - typographic_box[1])
        padding = max(size, 8)
        probe = image_mod.new(
            "L",
            (typographic_width + (2 * padding), typographic_height + (2 * padding)),
            color=0,
        )
        probe_origin = (
            padding - typographic_box[0],
            padding - typographic_box[1],
        )
        draw_mod.Draw(probe).text(probe_origin, text, fill=255, font=font)
        raster_box = probe.getbbox()
        if raster_box is None:
            raise ValueError("rendered PHI text produced an empty pixel mask")
        measured = (
            int(raster_box[0] - probe_origin[0]),
            int(raster_box[1] - probe_origin[1]),
            int(raster_box[2] - probe_origin[0]),
            int(raster_box[3] - probe_origin[1]),
        )
        width = int(measured[2] - measured[0])
        height = int(measured[3] - measured[1])
        if width <= max_width and height <= max_height:
            return font, actual_font_name, size, tuple(map(int, measured))
        if not allow_resize:
            break
        size -= 1
    raise ValueError(
        f"rendered PHI text does not fit within {max_width}x{max_height} pixels"
    )


def _load_font(font_mod: Any, font_name: str, font_size: int) -> tuple[Any, str]:
    candidates = (font_name,) + tuple(
        candidate for candidate in DEFAULT_FONT_NAMES if candidate != font_name
    )
    for candidate in candidates:
        try:
            return font_mod.truetype(candidate, font_size), candidate
        except OSError:
            if font_name not in DEFAULT_FONT_NAMES:
                raise ValueError(f"unable to load font {font_name!r}") from None

    try:
        return font_mod.load_default(size=font_size), "PillowDefault"
    except TypeError:  # Pillow 10.0 does not accept the size keyword.
        return font_mod.load_default(), "PillowDefault"


def _import_pillow() -> tuple[Any, Any, Any]:
    try:
        return (
            importlib.import_module("PIL.Image"),
            importlib.import_module("PIL.ImageDraw"),
            importlib.import_module("PIL.ImageFont"),
        )
    except ImportError as exc:  # pragma: no cover - exercised via import blocking.
        from openmed.multimodal.exceptions import MissingDependencyError

        raise MissingDependencyError(
            dependency="Pillow", instruction=_MULTIMODAL_INSTALL_HINT
        ) from exc


def _validate_canvas_size(size: tuple[int, int]) -> tuple[int, int]:
    if len(size) != 2:
        raise ValueError("canvas_size must contain exactly width and height")
    width, height = size
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("canvas width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("canvas height must be a positive integer")
    return width, height


def _validate_font_size(size: Any) -> int:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("font sizes must be positive integers")
    return size


def _validate_font_sizes(sizes: Sequence[int]) -> tuple[int, ...]:
    if not sizes:
        raise ValueError("font_sizes must not be empty")
    return tuple(_validate_font_size(size) for size in sizes)


def _validate_font_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("font names must be non-empty strings")
    return name


def _validate_font_names(names: Sequence[str]) -> tuple[str, ...]:
    if not names:
        raise ValueError("font_names must not be empty")
    return tuple(_validate_font_name(name) for name in names)


def _validate_exact_controls(
    values: Sequence[Any], name: str, validator: Any
) -> tuple[Any, ...]:
    if len(values) != len(BURNED_IN_LABELS):
        raise ValueError(f"{name} must contain exactly {len(BURNED_IN_LABELS)} items")
    return tuple(validator(value) for value in values)


def _validate_positions(
    positions: Sequence[tuple[int, int]] | None,
) -> tuple[tuple[int, int], ...] | None:
    if positions is None:
        return None
    if len(positions) != len(BURNED_IN_LABELS):
        raise ValueError(
            f"positions must contain exactly {len(BURNED_IN_LABELS)} items"
        )

    validated: list[tuple[int, int]] = []
    for position in positions:
        if len(position) != 2:
            raise ValueError("each position must contain exactly x and y")
        x, y = position
        if (
            isinstance(x, bool)
            or not isinstance(x, int)
            or isinstance(y, bool)
            or not isinstance(y, int)
            or x < 0
            or y < 0
        ):
            raise ValueError("positions must contain non-negative integer pixels")
        validated.append((x, y))
    return tuple(validated)


__all__ = [
    "BACKGROUND_MODES",
    "BURNED_IN_LABELS",
    "DEFAULT_CANVAS_SIZE",
    "DEFAULT_FONT_NAMES",
    "DEFAULT_FONT_SIZES",
    "BurnedInExample",
    "BurnedInGenerator",
    "BurnedInTextBox",
    "generate_burned_in_example",
]
