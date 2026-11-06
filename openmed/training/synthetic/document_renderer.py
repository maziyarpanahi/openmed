"""Actual synthetic page rendering, source-linked layout truth and scan corruption.

Only internally generated note grammars enter the renderer. Returned images and
text are explicit synthetic test data; ordinary receipts contain no text values.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, TypeVar

from openmed.core.labels import CANONICAL_LABELS, PERSON
from openmed.multimodal.evidence.fused_graph import (
    DerivationEdge,
    EdgeKind,
    EvidenceNode,
    FusedEvidenceGraph,
    NodeKind,
    PageGeometry,
)

from .burned_in import (
    COMPLEX_LAYOUT_SCRIPTS,
    PERMISSIVE_FONT_LICENSES,
    PILLOW_DEFAULT_FONT_LICENSE,
    BurnedInFont,
    BurnedInGenerator,
    _check_glyph_coverage,
    _layout_engine_name,
    _load_face,
    _span_scripts,
    font_supports_complex_layout,
)
from .note_grammar import (
    GRAMMARS,
    render_synthetic_note,
    synthetic_note_facts,
    validate_synthetic_note,
)

SCHEMA_VERSION = "openmed.synthetic.document-renderer.v1"
__all__ = [
    "DocumentLayout",
    "DocumentRenderError",
    "DocumentRenderLimits",
    "DocumentRenderSpec",
    "GeneratedSpan",
    "HandwritingPlaceholder",
    "RenderedDocument",
    "RenderedPage",
    "RenderedWord",
    "RenderStage",
    "ScanArtifacts",
    "WordRole",
    "WordVisibility",
    "render_synthetic_document",
    "reproduce_synthetic_document",
]
_OPERATIONS = ("skew", "blur", "scan", "occlude", "jpeg", "encode")
T = TypeVar("T")
Box = tuple[int, int, int, int]


class DocumentRenderError(ValueError):
    """Fixed-message rejection with no source, path or backend exception context."""


class DocumentLayout(str, Enum):
    """Executed visual page templates, not labels on the same raster."""

    FORM = "form"
    LETTER = "letter"
    LABEL = "label"
    FAX = "fax"


class WordRole(str, Enum):
    """Generated text purpose, distinct from a handwriting placeholder region."""

    BODY = "body"
    STAMP = "stamp"
    PHI_OVERLAY = "phi_overlay"
    FOOTER = "footer"


class WordVisibility(str, Enum):
    """Geometric glyph-support visibility, not OCR readability after compression."""

    VISIBLE = "visible"
    PARTIAL = "partial"
    OCCLUDED = "occluded"
    OUTSIDE = "outside"


def _require(condition: bool) -> None:
    if not condition:
        raise DocumentRenderError("synthetic_document_render_rejected")


def _safe(call: Callable[[], T]) -> T:
    failed = False
    result = None
    try:
        result = call()
    except Exception:
        failed = True
    if failed:
        raise DocumentRenderError("synthetic_document_render_rejected")
    return result  # type: ignore[return-value]


def _integer(value, low, high):
    _require(type(value) is int and low <= value <= high)


def _finite(value, low, high):
    _require(
        type(value) in (int, float) and math.isfinite(value) and low <= value <= high
    )


def _hex(value, size=64):
    _require(
        type(value) is str
        and len(value) == size
        and all(c in "0123456789abcdef" for c in value)
    )


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _bytes_digest(value):
    _require(type(value) is bytes)
    return hashlib.sha256(value).hexdigest()


def _box(value, size):
    _require(type(value) is tuple and len(value) == 4)
    for number in value:
        _integer(number, 0, max(size))
    x0, y0, x1, y1 = value
    _require(0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1])


@dataclass(frozen=True, slots=True, kw_only=True)
class ScanArtifacts:
    """Actual seeded corruption operations, applied in their recorded order.

    Args:
        skew_degrees: Horizontal shear angle, around the page's vertical midpoint.
        blur_radius: Pillow Gaussian blur radius in pixels.
        scan_strength: Number of seeded speckle batches and scanner dropout lines.
        occlusion: Optional normalized half-open white rectangle on each page.
        jpeg_quality: Optional actual JPEG encode/decode quality before final PNG.
    """

    skew_degrees: float = 0.0
    blur_radius: float = 0.0
    scan_strength: int = 0
    occlusion: tuple[float, float, float, float] | None = None
    jpeg_quality: int | None = None

    def __post_init__(self) -> None:
        _finite(self.skew_degrees, -8, 8)
        _finite(self.blur_radius, 0, 3)
        _integer(self.scan_strength, 0, 32)
        if self.occlusion is not None:
            _require(type(self.occlusion) is tuple and len(self.occlusion) == 4)
            for value in self.occlusion:
                _finite(value, 0, 1)
            x0, y0, x1, y1 = self.occlusion
            _require(x0 < x1 and y0 < y1)
        if self.jpeg_quality is not None:
            _integer(self.jpeg_quality, 5, 100)


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentRenderSpec:
    """Closed synthetic generation and layout choices; no real text input.

    Args:
        seed: Seed for generated facts, spacing, placeholder strokes and scan noise.
        layout: Actual form, letter, label or fax visual layout.
        grammar: Existing synthetic clinical-note grammar name.
        page_size: Optional rendered (width, height); template defaults otherwise.
        font_size: Body font size; footer uses a bounded smaller face.
        stamp: Draw an actual synthetic-copy stamp, with word truth.
        handwriting_placeholder: Draw explicitly untranscribed synthetic strokes.
        phi_overlay: Repeat a generated identifier in the footer with PHI truth.
        artifacts: Executed post-render transformations and corruption.
    """

    seed: int
    layout: DocumentLayout = DocumentLayout.FORM
    grammar: str = "encounter"
    page_size: tuple[int, int] | None = None
    font_size: int = 18
    stamp: bool = True
    handwriting_placeholder: bool = True
    phi_overlay: bool = True
    artifacts: ScanArtifacts = ScanArtifacts()

    def __post_init__(self) -> None:
        _integer(self.seed, 0, 2**63 - 1)
        _require(type(self.layout) is DocumentLayout)
        _require(type(self.grammar) is str and self.grammar in GRAMMARS)
        _integer(self.font_size, 12, 28)
        for value in (self.stamp, self.handwriting_placeholder, self.phi_overlay):
            _require(type(value) is bool)
        _require(type(self.artifacts) is ScanArtifacts)
        self.artifacts.__post_init__()
        if self.page_size is not None:
            _require(type(self.page_size) is tuple and len(self.page_size) == 2)
            _integer(self.page_size[0], 320, 1600)
            _integer(self.page_size[1], 240, 2000)


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentRenderLimits:
    """Cooperative allocation/work ceilings, not native font/codec process isolation.

    Args:
        max_pages: Refuse the entire document if pagination exceeds this count.
        max_words: Maximum words, including stamps and repeated identifiers.
        max_page_pixels: Maximum area of each raster.
        max_mask_work_pixels: Budget for per-word affine/blur/support processing.
        max_png_bytes: Maximum bytes for each PNG result or mask tile.
    """

    max_pages: int = 8
    max_words: int = 384
    max_page_pixels: int = 2_000_000
    max_mask_work_pixels: int = 256_000_000
    max_png_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        _integer(self.max_pages, 1, 16)
        _integer(self.max_words, 1, 480)
        _integer(self.max_page_pixels, 1, 4_000_000)
        _integer(self.max_mask_work_pixels, 1, 512_000_000)
        _integer(self.max_png_bytes, 1, 32 * 1024 * 1024)


@dataclass(frozen=True, slots=True)
class GeneratedSpan:
    """Exact generated-text fact span; the value itself remains in protected text."""

    start: int
    end: int
    label: str
    fact_ref: str
    is_phi: bool

    def __post_init__(self) -> None:
        _integer(self.start, 0, 100_000)
        _integer(self.end, self.start + 1, 100_000)
        _require(type(self.label) is str and self.label in CANONICAL_LABELS)
        _require(
            type(self.fact_ref) is str
            and re.fullmatch(r"fact_[0-9a-f]{24}", self.fact_ref) is not None
        )
        _require(type(self.is_phi) is bool)


@dataclass(frozen=True, slots=True)
class RenderStage:
    """Canonical executed-operation record with actual output and predecessor hashes."""

    operation: str
    parameters_digest: str
    output_digest: str
    previous_digest: str

    def __post_init__(self) -> None:
        _require(type(self.operation) is str and self.operation in _OPERATIONS)
        for value in (self.parameters_digest, self.output_digest, self.previous_digest):
            _hex(value)

    @property
    def digest(self) -> str:
        """Bind one actual transformation to its parameters and previous stage."""
        self.__post_init__()
        return _hash([SCHEMA_VERSION, asdict(self)])


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderedWord:
    """Clean glyph truth and post-transform geometric support for one actual word."""

    word_id: str
    page: int
    reading_order: int
    span: tuple[int, int]
    role: WordRole
    labels: tuple[str, ...]
    fact_refs: tuple[str, ...]
    is_phi: bool
    clean_box: Box
    transformed_polygon: tuple[tuple[float, float], ...]
    support_box: Box | None
    visible_box: Box | None
    visible_fraction: float
    visibility: WordVisibility
    clipped: bool
    clean_mask_png: bytes = field(repr=False)
    visible_mask_png: bytes = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize coordinates and hashes, excluding both glyph-mask payloads."""
        _safe(lambda: _validate_word(self))
        return {
            name: getattr(self, name)
            for name in (
                "word_id",
                "page",
                "reading_order",
                "span",
                "role",
                "labels",
                "fact_refs",
                "is_phi",
                "clean_box",
                "transformed_polygon",
                "support_box",
                "visible_box",
                "visible_fraction",
                "visibility",
                "clipped",
            )
        } | {
            "clean_mask_digest": _bytes_digest(self.clean_mask_png),
            "visible_mask_digest": _bytes_digest(self.visible_mask_png),
        }


@dataclass(frozen=True, slots=True)
class HandwritingPlaceholder:
    """Explicitly untranscribed synthetic strokes, never claimed handwriting OCR gold."""

    clean_box: Box
    transformed_polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """Protected clean/final PNGs with word truth, artifact chain and evidence graphs."""

    index: int
    size: tuple[int, int]
    words: tuple[RenderedWord, ...]
    handwriting: HandwritingPlaceholder | None
    stages: tuple[RenderStage, ...]
    clean_graph: FusedEvidenceGraph
    graph: FusedEvidenceGraph
    clean_png: bytes = field(repr=False)
    png: bytes = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return only generated-source hashes, coordinates and controlled metadata."""
        _safe(lambda: _validate_page(self))
        return {
            "index": self.index,
            "size": self.size,
            "clean_image_digest": _bytes_digest(self.clean_png),
            "image_digest": _bytes_digest(self.png),
            "words": [word.to_dict() for word in self.words],
            "handwriting_placeholder": asdict(self.handwriting)
            if self.handwriting
            else None,
            "stages": [asdict(stage) for stage in self.stages],
            "clean_graph": self.clean_graph.to_dict(),
            "graph": self.graph.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RenderedDocument:
    """Explicit synthetic test artifact; ordinary reports omit all generated values."""

    spec: DocumentRenderSpec
    font_digest: str
    font_license: str
    runtime_digest: str
    spans: tuple[GeneratedSpan, ...]
    pages: tuple[RenderedPage, ...]
    text: str = field(repr=False)

    def to_json(self) -> str:
        """Serialize value-free layout truth and receipts; never text, pixels or paths."""
        _safe(lambda: _validate(self))
        return _safe(
            lambda: _json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "synthetic": True,
                    "spec": asdict(self.spec),
                    "font_digest": self.font_digest,
                    "font_license": self.font_license,
                    "runtime_digest": self.runtime_digest,
                    "text_digest": _bytes_digest(self.text.encode()),
                    "spans": [asdict(span) for span in self.spans],
                    "pages": [page.to_dict() for page in self.pages],
                    "clinical_validation": False,
                }
            )
        )

    @property
    def digest(self) -> str:
        """Bind generated content, executed rendering and all layout/visibility truth."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()


class _Output(BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, value):
        _require(self.tell() + len(value) <= self.limit)
        return super().write(value)


def _png(image, limits):
    with _Output(limits.max_png_bytes) as stream:
        image.save(stream, format="PNG", optimize=False)
        return stream.getvalue()


def _pixels(image):
    digest = hashlib.sha256(_json([image.mode, image.size]).encode())
    digest.update(b"\x00")
    digest.update(image.tobytes())
    return digest.hexdigest()


def _font(requested, spec):
    from PIL import ImageFont

    helper = BurnedInGenerator(seed=spec.seed)
    if requested is None:
        default = ImageFont.load_default(size=spec.font_size)
        data = default.font_bytes
        _require(type(data) is bytes and len(data) <= 16 * 1024 * 1024)
        resolved = helper._describe_font(
            data,
            index=0,
            display_name="PillowDefault",
            license_id=PILLOW_DEFAULT_FONT_LICENSE,
            fallback=False,
        )
    else:
        _require(type(requested) is BurnedInFont)
        requested.__post_init__()
        path = Path(requested.path)
        _require(path.is_file() and path.stat().st_size <= 16 * 1024 * 1024)
        with path.open("rb") as stream:
            data = stream.read(16 * 1024 * 1024 + 1)
        _require(0 < len(data) <= 16 * 1024 * 1024)
        resolved = helper._describe_font(
            data,
            index=requested.index,
            display_name="LocalFont",
            license_id=requested.license,
            fallback=False,
        )
    return (
        resolved,
        _load_face(ImageFont, resolved, spec.font_size),
        _load_face(ImageFont, resolved, max(12, spec.font_size - 4)),
    )


def _runtime(font):
    import faker
    import PIL
    from PIL import ImageFont, features

    return _hash(
        [
            PIL.__version__,
            faker.VERSION,
            sys.version_info[:2],
            zlib.ZLIB_RUNTIME_VERSION,
            features.version_codec("jpg"),
            features.version_module("freetype2"),
            features.version_feature("raqm"),
            _layout_engine_name(ImageFont, font),
        ]
    )


def _glyph(value, font):
    from PIL import Image, ImageDraw

    left, top, right, bottom = font.getbbox(value)
    _require(0 < right - left <= 4096 and 0 < bottom - top <= 256)
    with Image.new("L", (right - left + 4, bottom - top + 4)) as mask:
        ImageDraw.Draw(mask).text((2 - left, 2 - top), value, font=font, fill=255)
        ink = mask.getbbox()
        _require(ink is not None)
        relative = (
            left + ink[0] - 2,
            top + ink[1] - 2,
            left + ink[2] - 2,
            top + ink[3] - 2,
        )
        return mask.crop(ink), relative, math.ceil(font.getlength(value))


def _forward(box, shear, height):
    x0, y0, x1, y1 = box
    return tuple(
        (float(x + shear * (y - height / 2)), float(y))
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    )


def _warp(image, shear, fill):
    from PIL import Image

    if not shear:
        return image.copy()
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, -shear, shear * image.height / 2, 0, 1, 0),
        resample=Image.Resampling.BILINEAR,
        fillcolor=fill,
    )


def _scan_plan(spec, index, size):
    rng = random.Random(_hash([spec.seed, index, "scan"]))
    width, height = size
    strength = spec.artifacts.scan_strength
    speckles = tuple(
        (rng.randrange(width), rng.randrange(height), rng.randrange(100, 245))
        for _ in range(strength * 32)
    )
    dropouts = tuple(
        (0, y, width, min(height, y + 1 + strength // 16))
        for y in (rng.randrange(height) for _ in range(strength))
    )
    return speckles, dropouts


def _occlusion(spec, size):
    if spec.artifacts.occlusion is None:
        return None
    x0, y0, x1, y1 = spec.artifacts.occlusion
    return (
        math.floor(x0 * size[0]),
        math.floor(y0 * size[1]),
        math.ceil(x1 * size[0]),
        math.ceil(y1 * size[1]),
    )


def _erase(image, rectangles, value):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for x0, y0, x1, y1 in rectangles:
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=value)


def _graph(words, image_digest, index, size, clean):
    geometry = PageGeometry(page=index, width=size[0], height=size[1])
    page_id = _hash([image_digest, index, "page"])[:32]
    nodes = [
        EvidenceNode(
            node_id=page_id,
            kind=NodeKind.PAGE,
            source_digest=image_digest,
            geometry=geometry,
        )
    ]
    edges = []
    for word in words:
        box = word.clean_box if clean else word.visible_box
        if box is None:
            continue
        pixel_id = _hash([image_digest, word.word_id, "pixels"])[:32]
        text_id = _hash([image_digest, word.word_id, "gold_ocr"])[:32]
        nodes.extend(
            (
                EvidenceNode(
                    node_id=pixel_id,
                    kind=NodeKind.PIXELS,
                    source_digest=image_digest,
                    geometry=geometry,
                    region=box,
                ),
                EvidenceNode(
                    node_id=text_id,
                    kind=NodeKind.OCR,
                    source_digest=image_digest,
                    geometry=geometry,
                    region=box,
                    span=word.span,
                ),
            )
        )
        edges.extend(
            (
                DerivationEdge(source=page_id, target=pixel_id, kind=EdgeKind.LOCATES),
                DerivationEdge(source=pixel_id, target=text_id, kind=EdgeKind.OCR),
            )
        )
    return FusedEvidenceGraph(nodes=tuple(nodes), edges=tuple(edges))


def _render(spec, limits, requested_font):
    from PIL import Image, ImageDraw, ImageFilter

    _require(type(spec) is DocumentRenderSpec and type(limits) is DocumentRenderLimits)
    spec.__post_init__()
    limits.__post_init__()
    size = spec.page_size or (
        (512, 320) if spec.layout is DocumentLayout.LABEL else (768, 1024)
    )
    width, height = size
    _require(width * height <= limits.max_page_pixels)
    resolved, font, footer_font = _font(requested_font, spec)
    runtime = _runtime(font)
    owned_facts = {key for _, keys in GRAMMARS[spec.grammar] for key in keys}
    note = render_synthetic_note(
        spec.grammar,
        tuple(
            fact
            for fact in synthetic_note_facts(seed=spec.seed)
            if fact.key in owned_facts
        ),
    )
    _require(not validate_synthetic_note(note))
    prefix = f"SYNTHETIC CLINICAL {spec.layout.value.upper()}\n"
    text = prefix + note.text
    spans = [
        GeneratedSpan(
            span.start + len(prefix),
            span.end + len(prefix),
            span.label,
            span.metadata["fact_id"],
            span.metadata["role"] == "identifier",
        )
        for span in note.gold_spans
    ]
    identifier = next(fact for fact in note.facts if fact.key == "identifier")
    all_text = text + "SYNTHETIC COPY ID: PAGE 0123456789/" + identifier.value
    _check_glyph_coverage(resolved, all_text, label=PERSON)
    if any(
        script in COMPLEX_LAYOUT_SCRIPTS or script == "Unknown"
        for script in _span_scripts(all_text)
    ):
        _require(font_supports_complex_layout(font))
    rng = random.Random(spec.seed)
    margin = 24 + 2 * rng.randrange(3)
    line_height = sum(font.getmetrics()) + 4
    footer_count = 1 + int(spec.stamp) + int(spec.phi_overlay)
    footer_height = max(106, 20 + footer_count * (sum(footer_font.getmetrics()) + 7))
    bottom = height - footer_height
    columns = 2 if spec.layout is DocumentLayout.FORM else 1
    column_width = (width - (columns + 1) * margin) // columns
    _require(bottom - margin >= 2 * line_height)
    canvases, records, handwriting = [], [], []
    masks = []
    page_index, column, y = -1, 0, margin

    def new_page():
        nonlocal page_index, column, y
        _require(len(canvases) < limits.max_pages)
        image = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(image)
        if spec.layout is DocumentLayout.FORM:
            draw.rectangle(
                (margin - 8, margin - 8, width - margin + 8, bottom),
                outline=(130, 145, 155),
                width=2,
            )
            draw.line(
                (width // 2, margin - 8, width // 2, bottom), fill=(180, 190, 195)
            )
        elif spec.layout is DocumentLayout.LETTER:
            draw.rectangle((margin, 10, width - margin, 15), fill=(32, 71, 93))
        elif spec.layout is DocumentLayout.LABEL:
            draw.rounded_rectangle(
                (4, 4, width - 5, height - 5), radius=12, outline=(20, 20, 20), width=3
            )
        else:
            draw.rectangle(
                (margin - 8, margin - 10, width - margin + 8, margin + line_height),
                fill=(225, 225, 225),
            )
            for x in range(margin, width - margin, 12):
                draw.line(
                    (x, bottom - 2, min(x + 6, width - margin), bottom - 2),
                    fill=(60, 60, 60),
                )
        canvases.append(image)
        records.append([])
        handwriting.append(None)
        page_index, column, y = len(canvases) - 1, 0, margin

    def place(value, start, end, origin, face, role):
        _require(len(masks) < limits.max_words)
        mask, relative, advance = _glyph(value, face)
        x, baseline = origin
        box = tuple(
            int(coordinate + offset)
            for coordinate, offset in zip(
                relative, (x, baseline, x, baseline), strict=True
            )
        )
        try:
            _box(box, size)
            color = (150, 35, 45) if role is WordRole.STAMP else (25, 34, 43)
            canvases[page_index].paste(color, box, mask)
            spans_here = [
                span for span in spans if span.start < end and start < span.end
            ]
            record = {
                "word_id": _hash([spec.seed, start, end, page_index, role.value])[:32],
                "page": page_index,
                "reading_order": len(masks),
                "span": (start, end),
                "role": role,
                "labels": tuple(sorted({span.label for span in spans_here})),
                "fact_refs": tuple(sorted({span.fact_ref for span in spans_here})),
                "is_phi": any(span.is_phi for span in spans_here),
                "clean_box": box,
                "clean_mask_png": _png(mask, limits),
            }
            records[page_index].append(record)
            masks.append((record, mask.copy()))
        finally:
            mask.close()
        return advance

    image = support = None
    try:
        new_page()
        cursor = 0
        for line in text.splitlines(keepends=True):
            x = margin + column * (column_width + margin)
            for match in re.finditer(r"\S+", line):
                value = match[0]
                bounds = font.getbbox(value)
                _require(bounds[2] - min(0, bounds[0]) <= column_width)
                if (
                    x + bounds[2]
                    > margin + column * (column_width + margin) + column_width
                ):
                    y += line_height
                    x = margin + column * (column_width + margin)
                if y + line_height > bottom:
                    column += 1
                    y = margin
                    if column >= columns:
                        new_page()
                    x = margin + column * (column_width + margin)
                advance = place(
                    value,
                    cursor + match.start(),
                    cursor + match.end(),
                    (x, y),
                    font,
                    WordRole.BODY,
                )
                x += advance + math.ceil(font.getlength(" "))
            if spec.layout is DocumentLayout.FORM:
                ImageDraw.Draw(canvases[page_index]).line(
                    (
                        margin + column * (column_width + margin),
                        y + line_height - 2,
                        margin + column * (column_width + margin) + column_width,
                        y + line_height - 2,
                    ),
                    fill=(225, 230, 234),
                )
            y += line_height
            cursor += len(line)
        body_count = len(masks)
        _require(body_count == len(re.findall(r"\S+", text)))

        for page_index, canvas in enumerate(canvases):
            draw = ImageDraw.Draw(canvas)
            footer_lines = []
            if spec.stamp:
                footer_lines.append(("SYNTHETIC COPY", WordRole.STAMP))
            if spec.phi_overlay:
                footer_lines.append(("ID: " + identifier.value, WordRole.PHI_OVERLAY))
            footer_lines.append(
                (f"PAGE {page_index + 1}/{len(canvases)}", WordRole.FOOTER)
            )
            y = bottom + 10
            for line, role in footer_lines:
                start = len(text) + 1
                text += "\n" + line
                if role is WordRole.PHI_OVERLAY:
                    spans.append(
                        GeneratedSpan(
                            start + 4,
                            start + len(line),
                            identifier.label,
                            identifier.fact_id,
                            True,
                        )
                    )
                x = margin
                for match in re.finditer(r"\S+", line):
                    advance = place(
                        match[0],
                        start + match.start(),
                        start + match.end(),
                        (x, y),
                        footer_font,
                        role,
                    )
                    x += advance + math.ceil(footer_font.getlength(" "))
                _require(x <= width - margin)
                if role is WordRole.STAMP:
                    draw.rectangle(
                        (
                            margin - 4,
                            y - 1,
                            x + 3,
                            y + sum(footer_font.getmetrics()) + 2,
                        ),
                        outline=(150, 35, 45),
                        width=1,
                    )
                y += sum(footer_font.getmetrics()) + 7
            if spec.handwriting_placeholder:
                box = (width - margin - 100, height - 42, width - margin, height - 15)
                # A placeholder must not overwrite a word while leaving its
                # clean glyph mask falsely labelled as unobscured gold.
                for record in records[page_index]:
                    x0, y0, x1, y1 = record["clean_box"]
                    _require(
                        x1 <= box[0] or box[2] <= x0 or y1 <= box[1] or box[3] <= y0
                    )
                points = [
                    (box[0] + offset, rng.randrange(box[1] + 2, box[3] - 2))
                    for offset in range(0, 100, 5)
                ]
                draw.line(points, fill=(60, 75, 110), width=2)
                handwriting[page_index] = box

        # Read each page's body in column order, then its footer, before the next
        # page. Text offsets remain independent of that visual reading order.
        ordinal = 0
        for page_records in records:
            for record in page_records:
                record["reading_order"] = ordinal
                ordinal += 1
        _require(width * height * len(masks) * 3 <= limits.max_mask_work_pixels)
        shear = math.tan(math.radians(spec.artifacts.skew_degrees))
        occlusion = _occlusion(spec, size)
        pages = []
        for page_index, clean in enumerate(canvases):
            clean_png = _png(clean, limits)
            stages = []
            image = clean.copy()

            def stage(operation, parameters, output_digest=None):
                stages.append(
                    RenderStage(
                        operation,
                        _hash(parameters),
                        output_digest or _pixels(image),
                        stages[-1].digest if stages else _bytes_digest(clean_png),
                    )
                )

            warped = _warp(image, shear, (255, 255, 255))
            image.close()
            image = warped
            stage(
                "skew",
                [spec.artifacts.skew_degrees, shear, size, "bilinear_fixed_canvas"],
            )
            if spec.artifacts.blur_radius:
                blurred = image.filter(
                    ImageFilter.GaussianBlur(spec.artifacts.blur_radius)
                )
                image.close()
                image = blurred
            stage("blur", [spec.artifacts.blur_radius])
            speckles, dropouts = _scan_plan(spec, page_index, size)
            draw = ImageDraw.Draw(image)
            for x, y, tone in speckles:
                draw.point((x, y), fill=(tone, tone, tone))
            _erase(image, dropouts, (255, 255, 255))
            stage(
                "scan",
                [
                    spec.seed,
                    page_index,
                    spec.artifacts.scan_strength,
                    speckles,
                    dropouts,
                ],
            )
            _erase(image, (occlusion,) if occlusion else (), (255, 255, 255))
            stage("occlude", occlusion)
            jpeg_digest = None
            if spec.artifacts.jpeg_quality is not None:
                with _Output(limits.max_png_bytes) as stream:
                    image.save(
                        stream,
                        format="JPEG",
                        quality=spec.artifacts.jpeg_quality,
                        subsampling=0,
                        optimize=False,
                        progressive=False,
                    )
                    compressed = stream.getvalue()
                jpeg_digest = _bytes_digest(compressed)
                with Image.open(BytesIO(compressed)) as decoded:
                    roundtrip = decoded.convert("RGB")
                image.close()
                image = roundtrip
            stage("jpeg", [spec.artifacts.jpeg_quality, jpeg_digest, "encode_decode"])
            png = _png(image, limits)
            image.close()
            stage("encode", ["PNG", "RGB", "fresh_no_metadata"], _bytes_digest(png))
            words = []
            for record, tile in masks:
                if record["page"] != page_index:
                    continue
                with Image.new("L", size) as source_mask:
                    source_mask.paste(tile, record["clean_box"])
                    support = _warp(source_mask, shear, 0)
                if spec.artifacts.blur_radius:
                    blurred = support.filter(
                        ImageFilter.GaussianBlur(spec.artifacts.blur_radius)
                    )
                    support.close()
                    support = blurred
                support_box = support.getbbox()
                mass = sum(
                    value * count for value, count in enumerate(support.histogram())
                )
                _erase(support, dropouts + ((occlusion,) if occlusion else ()), 0)
                visible_box = support.getbbox()
                remaining = sum(
                    value * count for value, count in enumerate(support.histogram())
                )
                if visible_box is not None:
                    with support.crop(visible_box) as visible:
                        visible_png = _png(visible, limits)
                else:
                    visible_png = b""
                support.close()
                polygon = _forward(record["clean_box"], shear, height)
                clipped = any(
                    not (0 <= x <= width and 0 <= y <= height) for x, y in polygon
                )
                visibility = (
                    WordVisibility.OUTSIDE
                    if not mass
                    else WordVisibility.OCCLUDED
                    if not remaining
                    else WordVisibility.PARTIAL
                    if remaining < mass or clipped
                    else WordVisibility.VISIBLE
                )
                words.append(
                    RenderedWord(
                        **record,
                        transformed_polygon=polygon,
                        support_box=support_box,
                        visible_box=visible_box,
                        visible_fraction=remaining / mass if mass else 0.0,
                        visibility=visibility,
                        clipped=clipped,
                        visible_mask_png=visible_png,
                    )
                )
            placeholder = handwriting[page_index]
            pages.append(
                RenderedPage(
                    page_index,
                    size,
                    tuple(words),
                    HandwritingPlaceholder(
                        placeholder, _forward(placeholder, shear, height)
                    )
                    if placeholder
                    else None,
                    tuple(stages),
                    _graph(words, _bytes_digest(clean_png), page_index, size, True),
                    _graph(words, _bytes_digest(png), page_index, size, False),
                    clean_png,
                    png,
                )
            )
        document = RenderedDocument(
            spec,
            resolved.sha256,
            resolved.license,
            runtime,
            tuple(spans),
            tuple(pages),
            text,
        )
        _validate(document)
        return document
    finally:
        if image is not None:
            image.close()
        if support is not None:
            support.close()
        for canvas in canvases:
            canvas.close()
        for _, mask in masks:
            mask.close()


def _polygon(value):
    _require(type(value) is tuple and len(value) == 4)
    for point in value:
        _require(type(point) is tuple and len(point) == 2)
        for coordinate in point:
            _finite(coordinate, -10_000, 10_000)


def _image_payload(payload, size, mode, *, tight=False):
    from PIL import Image

    _require(type(payload) is bytes and 0 < len(payload) <= 32 * 1024 * 1024)
    with Image.open(BytesIO(payload)) as image:
        _require(image.format == "PNG" and image.size == size and image.mode == mode)
        _require(not image.info)
        image.load()
        if tight:
            _require(image.getbbox() == (0, 0, *size))


def _validate_word(word, size=(1600, 2000)):
    _require(type(word) is RenderedWord)
    _hex(word.word_id, 32)
    _integer(word.page, 0, 15)
    _integer(word.reading_order, 0, 479)
    _require(type(word.span) is tuple and len(word.span) == 2)
    _integer(word.span[0], 0, 99_999)
    _integer(word.span[1], word.span[0] + 1, 100_000)
    _require(type(word.role) is WordRole and type(word.visibility) is WordVisibility)
    _box(word.clean_box, size)
    _polygon(word.transformed_polygon)
    for box in (word.support_box, word.visible_box):
        if box is not None:
            _box(box, size)
    if word.visible_box is not None:
        _require(word.support_box is not None)
        sx0, sy0, sx1, sy1 = word.support_box
        vx0, vy0, vx1, vy1 = word.visible_box
        _require(sx0 <= vx0 < vx1 <= sx1 and sy0 <= vy0 < vy1 <= sy1)
        _image_payload(word.visible_mask_png, (vx1 - vx0, vy1 - vy0), "L", tight=True)
    else:
        _require(word.visible_mask_png == b"")
    x0, y0, x1, y1 = word.clean_box
    _image_payload(word.clean_mask_png, (x1 - x0, y1 - y0), "L", tight=True)
    _finite(word.visible_fraction, 0, 1)
    _require(type(word.is_phi) is bool and type(word.clipped) is bool)
    _require(
        type(word.labels) is tuple
        and all(
            type(label) is str and label in CANONICAL_LABELS for label in word.labels
        )
    )
    _require(word.labels == tuple(sorted(set(word.labels))))
    _require(
        type(word.fact_refs) is tuple
        and all(
            type(ref) is str and re.fullmatch(r"fact_[0-9a-f]{24}", ref)
            for ref in word.fact_refs
        )
    )
    _require(word.fact_refs == tuple(sorted(set(word.fact_refs))))
    if word.support_box is None:
        _require(
            word.visible_box is None
            and word.visible_fraction == 0
            and word.visibility is WordVisibility.OUTSIDE
        )
    elif word.visible_box is None:
        _require(
            word.visible_fraction == 0 and word.visibility is WordVisibility.OCCLUDED
        )
    else:
        _require(word.visible_fraction > 0)
        expected = (
            WordVisibility.PARTIAL
            if word.visible_fraction < 1 or word.clipped
            else WordVisibility.VISIBLE
        )
        _require(word.visibility is expected)


def _validate_page(page):
    _require(type(page) is RenderedPage)
    _integer(page.index, 0, 15)
    _require(type(page.words) is tuple and 0 < len(page.words) <= 480)
    _require(type(page.size) is tuple and len(page.size) == 2)
    _integer(page.size[0], 320, 1600)
    _integer(page.size[1], 240, 2000)
    for payload in (page.clean_png, page.png):
        _image_payload(payload, page.size, "RGB")
    if page.handwriting is not None:
        _require(type(page.handwriting) is HandwritingPlaceholder)
        _box(page.handwriting.clean_box, page.size)
        _polygon(page.handwriting.transformed_polygon)
    _require(type(page.stages) is tuple and len(page.stages) == len(_OPERATIONS))
    previous = _bytes_digest(page.clean_png)
    for operation, stage in zip(_OPERATIONS, page.stages, strict=True):
        _require(
            type(stage) is RenderStage
            and stage.operation == operation
            and stage.previous_digest == previous
        )
        previous = stage.digest
    _require(page.stages[-1].output_digest == _bytes_digest(page.png))
    for word in page.words:
        _validate_word(word, page.size)
        _require(word.page == page.index)
    for graph, payload, clean in (
        (page.clean_graph, page.clean_png, True),
        (page.graph, page.png, False),
    ):
        _require(type(graph) is FusedEvidenceGraph)
        graph.__post_init__()
        _require(
            graph
            == _graph(page.words, _bytes_digest(payload), page.index, page.size, clean)
        )


def _validate(document):
    _require(type(document) is RenderedDocument)
    _require(type(document.spec) is DocumentRenderSpec)
    document.spec.__post_init__()
    _require(type(document.text) is str and len(document.text) <= 100_000)
    _hex(document.font_digest)
    _hex(document.runtime_digest)
    _require(
        type(document.font_license) is str
        and document.font_license in PERMISSIVE_FONT_LICENSES
    )
    _require(type(document.pages) is tuple and 0 < len(document.pages) <= 16)
    _require(type(document.spans) is tuple and len(document.spans) <= 1024)
    for span in document.spans:
        _require(type(span) is GeneratedSpan)
        span.__post_init__()
        _require(span.end <= len(document.text))
    orders, offsets = [], []
    for index, page in enumerate(document.pages):
        _validate_page(page)
        _require(page.index == index)
        size = document.spec.page_size or (
            (512, 320) if document.spec.layout is DocumentLayout.LABEL else (768, 1024)
        )
        _require(page.size == size)
        shear = math.tan(math.radians(document.spec.artifacts.skew_degrees))
        if page.handwriting:
            _require(
                page.handwriting.transformed_polygon
                == _forward(page.handwriting.clean_box, shear, page.size[1])
            )
        for word in page.words:
            start, end = word.span
            _require(end <= len(document.text))
            _require(re.fullmatch(r"\S+", document.text[start:end]) is not None)
            overlaps = [
                span for span in document.spans if span.start < end and start < span.end
            ]
            _require(word.labels == tuple(sorted({span.label for span in overlaps})))
            _require(
                word.fact_refs == tuple(sorted({span.fact_ref for span in overlaps}))
            )
            _require(word.is_phi == any(span.is_phi for span in overlaps))
            polygon = _forward(word.clean_box, shear, page.size[1])
            _require(word.transformed_polygon == polygon)
            _require(
                word.clipped
                == any(
                    not (0 <= x <= size[0] and 0 <= y <= size[1]) for x, y in polygon
                )
            )
            orders.append(word.reading_order)
            offsets.append(word.span)
    _require(orders == list(range(len(orders))))
    _require(
        sorted(offsets)
        == [
            (match.start(), match.end()) for match in re.finditer(r"\S+", document.text)
        ]
    )


def render_synthetic_document(
    *,
    spec: DocumentRenderSpec,
    limits: DocumentRenderLimits = DocumentRenderLimits(),
    font: BurnedInFont | None = None,
) -> RenderedDocument:
    """Generate clinical content, draw actual pages and retain transformed gold truth.

    Args:
        spec: Explicit seed, note grammar, visual template and actual scan artifacts.
        limits: Allocation/work ceilings; overflow refuses the whole document.
        font: Optional caller-supplied local permissively licensed font. None uses
            Pillow's explicit embedded face, never platform font fallback.

    Returns:
        Explicit synthetic text and protected PNGs, with value-free receipts and
        clean/final geometry. Handwriting placeholders are untranscribed strokes.
        Visibility records describe glyph geometry, not OCR accuracy or clinical
        validity. No files are written and no network requests are made.
    """
    return _safe(lambda: _render(spec, limits, font))


def reproduce_synthetic_document(
    document: RenderedDocument,
    *,
    limits: DocumentRenderLimits = DocumentRenderLimits(),
    font: BurnedInFont | None = None,
) -> RenderedDocument:
    """Actually regenerate pixels and truth, rejecting stale or edited receipts.

    Args:
        document: Prior synthetic result; receipts are not signatures.
        limits: Current work ceilings; a previous result cannot override them.
        font: Same local font asset, if the original render used a custom font.
    """

    def reproduce():
        _validate(document)
        actual = _render(document.spec, limits, font)
        _require(actual.to_json() == document.to_json())
        return actual

    return _safe(reproduce)
