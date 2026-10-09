"""Fixed synthetic OpenPHI-Vision layouts; no scoring or training export.

Only the checked-in catalog is accepted. Rendering reuses burned-in font fitting
and raster measurement with a bundled, hash-pinned OFL font and BASIC layout,
so neither host fonts nor optional text-shaping engines select fixture pixels.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from importlib.resources import files
from types import SimpleNamespace
from typing import Any, Iterator

CATALOG_VERSION = "openphi-vision-v1"
_CATALOG_SHA256 = "c8834836fddc11a9d0c298be4a83bcdb95bacc1a9c1600e33c04e7049756056d"


@dataclass(frozen=True)
class VisionFixtureBox:
    """Half-open pixel box and character span for a synthetic identifier.

    Text is accessible for local evaluation but excluded from representations.
    Offsets address the enclosing page's newline-joined text.
    """

    bbox: tuple[int, int, int, int]
    start: int
    end: int
    label: str
    critical: bool
    region: str
    text: str = field(repr=False)


@dataclass(frozen=True)
class VisionFixturePage:
    """Immutable synthetic page with verified PGM bytes and gold annotations."""

    page_id: str
    size: tuple[int, int]
    sha256: str
    gold_boxes: tuple[VisionFixtureBox, ...]
    text: str = field(repr=False)
    pgm_bytes: bytes = field(repr=False)

    def to_image(self) -> Any:
        """Return a detached Pillow image for an offline vision/OCR provider.

        Returns:
            An independent grayscale image; mutations do not alter the fixture.
        """
        from openmed.training.synthetic.burned_in import _import_pillow

        image_mod, _, _ = _import_pillow()
        with image_mod.open(io.BytesIO(self.pgm_bytes)) as image:
            return image.copy()


@dataclass(frozen=True)
class VisionFixtureCase:
    """Versioned fixed layout and expected zero-based packet page groups.

    Packet groups encode synthetic patient membership, not a split algorithm.
    All nested representations omit source values and pixel payloads.
    """

    case_id: str
    track: str
    version: str
    sha256: str
    pages: tuple[VisionFixturePage, ...]
    expected_packet_splits: tuple[tuple[int, ...], ...]


def load_openphi_vision_fixtures() -> Iterator[VisionFixtureCase]:
    """Yield all fixed cases in catalog order, verifying every pinned digest.

    The catalog is synthetic-only and performs no I/O outside bundled resources.
    It takes no seed, layout, font, transport, or export controls. Gold text is
    available in memory but must not be copied into diagnostics or evidence.

    Yields:
        Immutable cases with gold boxes, critical flags and packet membership.

    Raises:
        ImportError: If the optional Pillow dependency is unavailable.
        ValueError: If bundled declarations, font, boxes or pixels have drifted.
    """
    root = files("openmed.eval").joinpath("fixtures", "openphi_vision_v1")
    catalog_bytes = root.joinpath("catalog.json").read_bytes()
    if _digest(catalog_bytes) != _CATALOG_SHA256:
        raise ValueError("vision fixture catalog digest mismatch")
    catalog = json.loads(catalog_bytes)
    font_bytes = root.joinpath("IBMPlexMono-Regular.ttf").read_bytes()
    if _digest(font_bytes) != catalog["font_sha256"]:
        raise ValueError("vision fixture font digest mismatch")
    for declaration in catalog["cases"]:
        yield _load_case(declaration, font_bytes)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _case_digest(declaration: dict[str, Any]) -> str:
    # Bind gold text, boxes, critical flags and splits as well as page pixels.
    material = {key: value for key, value in declaration.items() if key != "sha256"}
    return _digest(
        json.dumps(
            material, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode("ascii")
    )


def _render_page(
    declaration: dict[str, Any], font_bytes: bytes
) -> tuple[bytes, tuple[VisionFixtureBox, ...], str]:
    from openmed.training.synthetic.burned_in import _fit_font, _import_pillow

    image_mod, draw_mod, font_mod = _import_pillow()
    # Force the same bundled face and shaping engine on every platform. The
    # existing helper never reaches its fallback because this face is present.
    fixed_font_mod = SimpleNamespace(
        truetype=lambda _name, size: font_mod.truetype(
            io.BytesIO(font_bytes), size, layout_engine=font_mod.Layout.BASIC
        )
    )
    width, height = declaration["size"]
    image = image_mod.new("L", (width, height), color=0)
    boxes = []
    chunks = []
    cursor = 0
    for item in declaration["items"]:
        value = item["text"]
        x, y = item["origin"]
        font, _, _, measured = _fit_font(
            image_mod,
            draw_mod,
            fixed_font_mod,
            text=value,
            font_name="fixture-face",
            font_size=18,
            max_width=width - x,
            max_height=height - y,
            allow_resize=False,
        )
        mask = image_mod.new("L", (width, height), color=0)
        draw_mod.Draw(mask).text(
            (x - measured[0], y - measured[1]), value, fill=255, font=font
        )
        bbox = mask.getbbox()
        if bbox != tuple(item["bbox"]):
            raise ValueError("vision fixture gold box mismatch")
        image.paste(255, (0, 0, width, height), mask)
        if chunks:
            cursor += 1
        boxes.append(
            VisionFixtureBox(
                bbox=tuple(item["bbox"]),
                start=cursor,
                end=cursor + len(value),
                label=item["label"],
                critical=item["critical"],
                region=item["region"],
                text=value,
            )
        )
        chunks.append(value)
        cursor += len(value)
    # PGM has a literal header and raw pixels: no timestamps, encoders, EXIF,
    # compression-library versions or private paths enter the digest.
    pgm = f"P5\n{width} {height}\n255\n".encode("ascii") + image.tobytes()
    return pgm, tuple(boxes), "\n".join(chunks)


def _load_case(declaration: dict[str, Any], font_bytes: bytes) -> VisionFixtureCase:
    if _case_digest(declaration) != declaration["sha256"]:
        raise ValueError("vision fixture case digest mismatch")
    pages = []
    for page in declaration["pages"]:
        try:
            payload, boxes, text = _render_page(page, font_bytes)
        except (OSError, ValueError):
            raise ValueError("vision fixture rendering mismatch") from None
        if _digest(payload) != page["sha256"]:
            raise ValueError("vision fixture page digest mismatch")
        pages.append(
            VisionFixturePage(
                page_id=page["page_id"],
                size=tuple(page["size"]),
                sha256=page["sha256"],
                gold_boxes=boxes,
                text=text,
                pgm_bytes=payload,
            )
        )
    return VisionFixtureCase(
        case_id=declaration["case_id"],
        track=declaration["track"],
        version=CATALOG_VERSION,
        sha256=declaration["sha256"],
        pages=tuple(pages),
        expected_packet_splits=tuple(
            tuple(group) for group in declaration["expected_packet_splits"]
        ),
    )


__all__ = [
    "CATALOG_VERSION",
    "VisionFixtureBox",
    "VisionFixturePage",
    "VisionFixtureCase",
    "load_openphi_vision_fixtures",
]
