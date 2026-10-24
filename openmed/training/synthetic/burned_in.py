"""Synthetic burned-in PHI images with aligned text and pixel gold labels.

Pillow is deliberately imported only when a case is rendered. Importing this
module, :mod:`openmed.training.synthetic`, or the core :mod:`openmed` package
therefore remains safe for installations that do not include the
``multimodal`` extra.

Every rendered span is checked against the character map of the exact font
bytes used to draw it, and spans in scripts that need complex text layout
(joining, reordering, conjuncts, or mark positioning) also require a
shaping-capable layout engine. Failures raise :class:`BurnedInFontError`
subclasses whose messages carry only reason codes, script names, canonical
labels, counts, and font digests. A missing font is never replaced unless the
caller opts in with ``font_fallback=True``, and every example records the
SHA-256 digest and declared license of each font it used.
"""

from __future__ import annotations

import bisect
import hashlib
import importlib
import io
import os
import random
import struct
import unicodedata
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Container,
    Final,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)

from faker import Faker

from openmed.core import labels as L
from openmed.core.anonymizer.providers import clinical_ids

from .locale_phi import LocalePhiGenerator, SyntheticPhiSpan
from .spdx_identifier import PERMISSIVE_SPDX_IDENTIFIERS

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

#: Declared license of the default DejaVu font names (Bitstream Vera terms;
#: DejaVu changes are public domain). It labels the name, not a verified file.
DEFAULT_FONT_LICENSE: Final = "Bitstream-Vera"
#: Name and declared license of Pillow's embedded Aileron Regular font, the
#: last resort when ``font_fallback=True`` and no default font is installed.
PILLOW_DEFAULT_FONT_NAME: Final = "PillowDefault"
PILLOW_DEFAULT_FONT_LICENSE: Final = "CC0-1.0"
#: SPDX "no assertion" value recorded for plain font names without a license.
UNDECLARED_FONT_LICENSE: Final = "NOASSERTION"
#: Licenses accepted for :class:`BurnedInFont`. Rendering text with a font
#: does not redistribute it, so the permissive font licenses are allowed in
#: addition to the committed permissive SPDX allowlist.
PERMISSIVE_FONT_LICENSES: Final[frozenset[str]] = PERMISSIVE_SPDX_IDENTIFIERS | {
    "Bitstream-Vera",
    "CC0-1.0",
    "OFL-1.1",
    "OFL-1.1-RFN",
    "OFL-1.1-no-RFN",
}

_COMMON_SCRIPT: Final = "Common"
_INHERITED_SCRIPT: Final = "Inherited"
_UNKNOWN_SCRIPT: Final = "Unknown"

# Block-level approximation of the Unicode Script property for the scripts
# the locale PHI generator and common clinical documents use. Sorted, with no
# overlaps. Letters and marks outside these ranges resolve to ``Unknown``,
# which is treated as needing complex layout so the check fails closed.
_SCRIPT_RANGES: Final[tuple[tuple[int, int, str], ...]] = (
    (0x0041, 0x005A, "Latin"),
    (0x0061, 0x007A, "Latin"),
    (0x00AA, 0x00AA, "Latin"),
    (0x00BA, 0x00BA, "Latin"),
    (0x00C0, 0x00D6, "Latin"),
    (0x00D8, 0x00F6, "Latin"),
    (0x00F8, 0x02AF, "Latin"),
    (0x0300, 0x036F, _INHERITED_SCRIPT),
    (0x0370, 0x03FF, "Greek"),
    (0x0400, 0x052F, "Cyrillic"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0750, 0x077F, "Arabic"),
    (0x0780, 0x07BF, "Thaana"),
    (0x07C0, 0x07FF, "Nko"),
    (0x0860, 0x086F, "Syriac"),
    (0x0870, 0x08FF, "Arabic"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"),
    (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Hangul"),
    (0x1200, 0x139F, "Ethiopic"),
    (0x1780, 0x17FF, "Khmer"),
    (0x1800, 0x18AF, "Mongolian"),
    (0x19E0, 0x19FF, "Khmer"),
    (0x1AB0, 0x1AFF, _INHERITED_SCRIPT),
    (0x1C80, 0x1C8F, "Cyrillic"),
    (0x1C90, 0x1CBF, "Georgian"),
    (0x1D00, 0x1D7F, "Latin"),
    (0x1DC0, 0x1DFF, _INHERITED_SCRIPT),
    (0x1E00, 0x1EFF, "Latin"),
    (0x1F00, 0x1FFF, "Greek"),
    (0x200C, 0x200D, _INHERITED_SCRIPT),
    (0x20D0, 0x20FF, _INHERITED_SCRIPT),
    (0x2C60, 0x2C7F, "Latin"),
    (0x2D00, 0x2D2F, "Georgian"),
    (0x2D80, 0x2DDF, "Ethiopic"),
    (0x2DE0, 0x2DFF, "Cyrillic"),
    (0x2E80, 0x2FDF, "Han"),
    (0x3005, 0x3005, "Han"),
    (0x3007, 0x3007, "Han"),
    (0x3021, 0x3029, "Han"),
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x3130, 0x318F, "Hangul"),
    (0x31F0, 0x31FF, "Katakana"),
    (0x3400, 0x4DBF, "Han"),
    (0x4E00, 0x9FFF, "Han"),
    (0xA640, 0xA69F, "Cyrillic"),
    (0xA720, 0xA7FF, "Latin"),
    (0xA8E0, 0xA8FF, "Devanagari"),
    (0xA960, 0xA97F, "Hangul"),
    (0xA9E0, 0xA9FF, "Myanmar"),
    (0xAA60, 0xAA7F, "Myanmar"),
    (0xAB00, 0xAB2F, "Ethiopic"),
    (0xAB30, 0xAB6F, "Latin"),
    (0xAC00, 0xD7FF, "Hangul"),
    (0xF900, 0xFAFF, "Han"),
    (0xFB00, 0xFB06, "Latin"),
    (0xFB1D, 0xFB4F, "Hebrew"),
    (0xFB50, 0xFDFF, "Arabic"),
    (0xFE00, 0xFE0F, _INHERITED_SCRIPT),
    (0xFE20, 0xFE2F, _INHERITED_SCRIPT),
    (0xFE70, 0xFEFC, "Arabic"),
    (0xFF21, 0xFF3A, "Latin"),
    (0xFF41, 0xFF5A, "Latin"),
    (0xFF66, 0xFF9F, "Katakana"),
    (0xFFA0, 0xFFDC, "Hangul"),
    (0x20000, 0x323AF, "Han"),
)
_SCRIPT_RANGE_STARTS: Final[tuple[int, ...]] = tuple(
    start for start, _, _ in _SCRIPT_RANGES
)

#: Script names accepted as ``script_fonts`` keys.
FONT_SCRIPTS: Final[tuple[str, ...]] = tuple(
    sorted({script for _, _, script in _SCRIPT_RANGES} - {_INHERITED_SCRIPT})
)
#: Scripts whose legible rendering needs shaping, bidirectional reordering,
#: or mark positioning. Spans in these scripts (and in unrecognised scripts)
#: fail closed unless the font's layout engine shapes text.
COMPLEX_LAYOUT_SCRIPTS: Final[frozenset[str]] = frozenset(
    {
        "Arabic",
        "Bengali",
        "Devanagari",
        "Ethiopic",
        "Gujarati",
        "Gurmukhi",
        "Hebrew",
        "Kannada",
        "Khmer",
        "Lao",
        "Malayalam",
        "Mongolian",
        "Myanmar",
        "Nko",
        "Oriya",
        "Sinhala",
        "Syriac",
        "Tamil",
        "Telugu",
        "Thaana",
        "Thai",
        "Tibetan",
    }
)

# Default-ignorable code points are formatting controls that shaping engines
# consume without a glyph, so fonts need not map them.
_DEFAULT_IGNORABLE_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)

_SFNT_VERSIONS: Final[frozenset[bytes]] = frozenset(
    {b"\x00\x01\x00\x00", b"OTTO", b"true"}
)
_UCS4_CMAP_ENCODINGS: Final[frozenset[tuple[int, int]]] = frozenset({(3, 10), (0, 4)})
_PROBE_FONT_SIZE: Final = 12

_FIELD_BY_LABEL: Final[Mapping[str, str]] = {
    L.PERSON: "person",
    L.DATE: "date",
    L.ID_NUM: "medical_record_number",
}
_MULTIMODAL_INSTALL_HINT: Final = 'Install with: pip install "openmed[multimodal]".'

CharacterMapProvider = Callable[[bytes, int], Container[int]]
"""Return the code points a font face (file bytes, face index) maps to glyphs."""

LayoutCapabilityProvider = Callable[[Any], bool]
"""Return whether a loaded Pillow font face shapes complex-script text."""


class BurnedInFontError(ValueError):
    """Value-free failure to render burned-in PHI with a verifiable font.

    Messages and attributes carry only a stable reason code, script names,
    canonical labels, counts, and font digests or opaque font references;
    never span text, code points, or file-system paths.

    Attributes:
        reason: Stable machine-readable failure code.
        script: Script that triggered the failure, when applicable.
        label: Canonical label of the affected span, when applicable.
        font_sha256: SHA-256 digest of the font file bytes, when known.
        font_ref: Opaque reference to a font that could not be read.
        missing_count: Number of code points without a glyph in the font.
        layout_engine: Layout engine of the loaded font face.
    """

    default_reason: ClassVar[str] = "font_error"

    def __init__(
        self,
        reason: str | None = None,
        *,
        script: str | None = None,
        label: str | None = None,
        font_sha256: str | None = None,
        font_ref: str | None = None,
        missing_count: int | None = None,
        layout_engine: str | None = None,
    ) -> None:
        self.reason = reason or self.default_reason
        self.script = script
        self.label = label
        self.font_sha256 = font_sha256
        self.font_ref = font_ref
        self.missing_count = missing_count
        self.layout_engine = layout_engine
        details = ", ".join(
            f"{name}={value}"
            for name, value in (
                ("script", script),
                ("label", label),
                ("font_sha256", font_sha256),
                ("font_ref", font_ref),
                ("missing_count", missing_count),
                ("layout_engine", layout_engine),
            )
            if value is not None
        )
        super().__init__(f"{self.reason}: {details}" if details else self.reason)


class BurnedInFontUnavailableError(BurnedInFontError):
    """A requested font could not be loaded and no fallback was requested."""

    default_reason: ClassVar[str] = "font_unavailable"


class BurnedInGlyphCoverageError(BurnedInFontError):
    """The selected font has no glyph for some code points of a span."""

    default_reason: ClassVar[str] = "glyph_coverage_missing"


class BurnedInShapingUnavailableError(BurnedInFontError):
    """A complex-script span cannot be shaped by the font's layout engine."""

    default_reason: ClassVar[str] = "complex_layout_unavailable"


@dataclass(frozen=True)
class BurnedInFont:
    """Caller-supplied font file with a declared permissive license.

    Fonts are never bundled with OpenMed. Supply a TrueType, OpenType, or
    collection file you are licensed to use; examples record only its file
    name, SHA-256 digest, and declared license, never the full path.

    Args:
        path: Font file path, or a file name Pillow resolves from the
            platform font directories.
        license: SPDX identifier of the font license. It must match one of
            :data:`PERMISSIVE_FONT_LICENSES` (case-insensitive) and is stored
            in its canonical spelling.
        index: Face index inside a font collection such as ``.ttc``.

    Raises:
        BurnedInFontError: If the license is not an allowed permissive font
            license (reason ``font_license_not_permissive``).
        ValueError: If the path or face index is invalid.
    """

    path: str | os.PathLike[str]
    license: str
    index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, os.PathLike)):
            raise ValueError("font path must be a string or path-like object")
        raw_path = os.fspath(self.path)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("font path must be a non-empty string path")
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("font index must be a non-negative integer")
        canonical = _canonical_font_license(self.license)
        if canonical is None:
            raise BurnedInFontError("font_license_not_permissive")
        object.__setattr__(self, "license", canonical)

    @property
    def file_name(self) -> str:
        """Return the font file name recorded in example metadata."""

        return os.path.basename(os.fspath(self.path))


class FontCharacterMap:
    """Code points one font face maps to a real (non-``.notdef``) glyph.

    Instances are immutable, support ``in`` and ``len``, and store coverage as
    merged inclusive ranges so large CJK fonts stay compact.

    Args:
        ranges: Inclusive ``(first, last)`` code-point ranges in any order.
    """

    __slots__ = ("_ends", "_size", "_starts")

    def __init__(self, ranges: Iterable[tuple[int, int]] = ()) -> None:
        merged: list[list[int]] = []
        for start, end in sorted(ranges):
            if start > end:
                continue
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self._starts = tuple(start for start, _ in merged)
        self._ends = tuple(end for _, end in merged)
        self._size = sum(end - start + 1 for start, end in merged)

    def __contains__(self, code_point: object) -> bool:
        if isinstance(code_point, bool) or not isinstance(code_point, int):
            return False
        index = bisect.bisect_right(self._starts, code_point) - 1
        return index >= 0 and code_point <= self._ends[index]

    def __len__(self) -> int:
        return self._size


def read_font_character_map(data: bytes, index: int = 0) -> FontCharacterMap:
    """Read the Unicode character map FreeType selects for one font face.

    This is the default character-map provider. It mirrors FreeType's
    charmap choice (the last UCS-4 Unicode subtable, otherwise the last
    Unicode subtable). Code points mapped to glyph 0 or to glyph ids beyond
    ``maxp.numGlyphs`` are not covered. A selected subtable in a format this
    reader does not decode yields an empty map, so coverage checks fail closed
    instead of guessing.

    Args:
        data: Complete TrueType, OpenType, or font-collection file bytes.
        index: Face index inside a collection; must be 0 for single fonts.

    Returns:
        The covered code points.

    Raises:
        BurnedInFontError: With reason ``font_unparseable`` when the bytes are
            not a readable sfnt font face.
    """

    try:
        return _read_character_map(bytes(data), index)
    except (_MalformedFontError, struct.error):
        raise BurnedInFontError("font_unparseable") from None


def font_supports_complex_layout(font: Any) -> bool:
    """Return whether a loaded Pillow font lays text out with Raqm shaping.

    This is the default layout-capability provider. Pillow shapes text,
    reorders bidirectional runs, and positions marks only through its Raqm
    layout engine (HarfBuzz and FriBiDi); the basic engine draws one glyph per
    code point, and bitmap fonts never shape.

    Args:
        font: A font object returned by :mod:`PIL.ImageFont`.

    Returns:
        ``True`` only when the font uses the Raqm layout engine.
    """

    try:
        image_font = importlib.import_module("PIL.ImageFont")
    except ImportError:
        return False
    raqm = getattr(getattr(image_font, "Layout", None), "RAQM", None)
    return raqm is not None and getattr(font, "layout_engine", None) == raqm


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


@dataclass(frozen=True)
class _ResolvedFont:
    display_name: str
    sha256: str
    license: str
    fallback: bool
    data: bytes
    index: int
    coverage: Container[int]
    pillow_default: bool = False

    def provenance(self) -> dict[str, Any]:
        return {
            "font_fallback": self.fallback,
            "font_license": self.license,
            "font_name": self.display_name,
            "font_sha256": self.sha256,
        }


class BurnedInGenerator:
    """Render deterministic names, dates, and MRNs into synthetic image pixels.

    Each span is drawn only when the selected font maps every code point to a
    glyph and, for complex scripts, when the font's layout engine shapes text.
    Font choice per span, in order: an exact per-call ``font_names`` control,
    then the first ``script_fonts`` entry matching a script in the span, then
    the configured ``font_names`` pool.

    Args:
        seed: Seed controlling PHI values, background, font, size, and layout.
        canvas_size: Output image size as ``(width, height)`` pixels.
        font_sizes: Positive font-size pool used for generated layouts.
        font_names: Font pool used for generated layouts: TrueType names or
            paths, or :class:`BurnedInFont` entries with a declared license.
        margin: Minimum edge margin for automatically positioned text.
        script_fonts: Optional mapping from a script in :data:`FONT_SCRIPTS`
            to a caller-supplied :class:`BurnedInFont` used for spans in that
            script. Fonts are never bundled.
        font_fallback: Opt in to replacing a missing default DejaVu font with
            another default font or Pillow's embedded font. The replacement is
            still coverage-checked and is recorded in metadata. Without it, a
            missing font raises :class:`BurnedInFontUnavailableError`.
        character_map_provider: Optional override returning the code points a
            font face covers, given its file bytes and face index. Defaults to
            :func:`read_font_character_map`.
        layout_capability_provider: Optional override returning whether a
            loaded font face shapes complex scripts. Defaults to
            :func:`font_supports_complex_layout`.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
        font_sizes: Sequence[int] = DEFAULT_FONT_SIZES,
        font_names: Sequence[str | BurnedInFont] = DEFAULT_FONT_NAMES,
        margin: int = 16,
        script_fonts: Mapping[str, BurnedInFont] | None = None,
        font_fallback: bool = False,
        character_map_provider: CharacterMapProvider | None = None,
        layout_capability_provider: LayoutCapabilityProvider | None = None,
    ) -> None:
        width, height = _validate_canvas_size(canvas_size)
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if width <= margin * 2 or height <= margin * 2:
            raise ValueError("margin leaves no drawable canvas area")
        if not isinstance(font_fallback, bool):
            raise ValueError("font_fallback must be a boolean")
        for provider in (character_map_provider, layout_capability_provider):
            if provider is not None and not callable(provider):
                raise TypeError("font capability providers must be callable")

        self.seed = seed
        self.canvas_size = (width, height)
        self.font_sizes = _validate_font_sizes(font_sizes)
        self.font_names = _validate_font_names(font_names)
        self.margin = margin
        self.script_fonts = _validate_script_fonts(script_fonts)
        self.font_fallback = font_fallback
        self._character_map_provider: CharacterMapProvider = (
            character_map_provider or read_font_character_map
        )
        self._layout_capability_provider: LayoutCapabilityProvider = (
            layout_capability_provider or font_supports_complex_layout
        )
        self._resolved_fonts: dict[str | BurnedInFont, _ResolvedFont] = {}
        self._rng = random.Random(seed)

    def generate(
        self,
        language: str = "en",
        *,
        positions: Sequence[tuple[int, int]] | None = None,
        font_sizes: Sequence[int] | None = None,
        font_names: Sequence[str | BurnedInFont] | None = None,
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
            font_names: Optional three exact fonts (names, paths, or
                :class:`BurnedInFont` entries). They take precedence over
                ``script_fonts``.
            background: Optional one of :data:`BACKGROUND_MODES`.

        Returns:
            A synthetic Pillow image with aligned canonical spans and boxes.
            Each box's metadata records the font digest, declared license,
            fallback use, layout engine, and scripts of its span.

        Raises:
            ImportError: If Pillow from the ``multimodal`` extra is missing.
            BurnedInFontUnavailableError: If a font cannot be loaded and no
                fallback was requested.
            BurnedInGlyphCoverageError: If the selected font lacks a glyph
                for a code point of a span.
            BurnedInShapingUnavailableError: If a complex-script span would be
                drawn without a shaping layout engine.
            BurnedInFontError: If a font file's character map is unreadable.
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
        fonts_used: dict[tuple[str, str, bool], dict[str, Any]] = {}
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
            scripts = _span_scripts(value)
            if controlled_fonts is None:
                requested_font = self._script_font(scripts) or requested_font
            resolved = self._resolve_font(font_mod, requested_font, label=label)
            _check_glyph_coverage(resolved, value, label=label)
            layout_engine = self._check_layout(
                font_mod, resolved, requested_size, scripts, label=label
            )

            if controlled_positions is None:
                lane_top = self.margin + (line_height * index)
                max_width = width - (2 * self.margin)
                max_height = line_height
            else:
                lane_top = controlled_positions[index][1]
                max_width = width - controlled_positions[index][0]
                max_height = height - lane_top

            font, actual_size, measured_box = _fit_font(
                image_mod,
                draw_mod,
                load_font=lambda size, resolved=resolved: _load_face(
                    font_mod, resolved, size
                ),
                text=value,
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
            fonts_used.setdefault(
                (resolved.sha256, resolved.display_name, resolved.fallback),
                resolved.provenance(),
            )
            box_metadata = {
                "field": _FIELD_BY_LABEL[label],
                "font_fallback": resolved.fallback,
                "font_license": resolved.license,
                "font_sha256": resolved.sha256,
                "layout_engine": layout_engine,
                "scripts": list(scripts),
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
                    font_name=resolved.display_name,
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
            "font_fallback_allowed": self.font_fallback,
            "font_fallback_used": any(box.metadata["font_fallback"] for box in boxes),
            "fonts": list(fonts_used.values()),
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

    def _script_font(self, scripts: Sequence[str]) -> BurnedInFont | None:
        for script in scripts:
            font = self.script_fonts.get(script)
            if font is not None:
                return font
        return None

    def _resolve_font(
        self, font_mod: Any, requested: str | BurnedInFont, *, label: str
    ) -> _ResolvedFont:
        cached = self._resolved_fonts.get(requested)
        if cached is not None:
            return cached
        resolved = self._read_font(font_mod, requested, fallback=False)
        reason = "font_unavailable"
        if resolved is None and self.font_fallback and requested in DEFAULT_FONT_NAMES:
            resolved = self._fallback_font(font_mod, requested)
            reason = "fallback_font_unavailable"
        if resolved is None:
            raise BurnedInFontUnavailableError(
                reason, label=label, font_ref=_font_reference(requested)
            )
        self._resolved_fonts[requested] = resolved
        return resolved

    def _read_font(
        self, font_mod: Any, requested: str | BurnedInFont, *, fallback: bool
    ) -> _ResolvedFont | None:
        if isinstance(requested, BurnedInFont):
            name = os.fspath(requested.path)
            face_index = requested.index
            license_id = requested.license
        else:
            name = requested
            face_index = 0
            license_id = (
                DEFAULT_FONT_LICENSE
                if requested in DEFAULT_FONT_NAMES
                else UNDECLARED_FONT_LICENSE
            )
        try:
            # Pillow resolves bare names from the platform font directories;
            # the resolved file's bytes are what gets digested and rendered.
            probe = font_mod.truetype(name, _PROBE_FONT_SIZE, index=face_index)
            data = _font_file_bytes(probe)
        except OSError:
            return None
        if data is None:
            return None
        return self._describe_font(
            data,
            index=face_index,
            display_name=os.path.basename(name),
            license_id=license_id,
            fallback=fallback,
        )

    def _fallback_font(self, font_mod: Any, requested: str) -> _ResolvedFont | None:
        for candidate in DEFAULT_FONT_NAMES:
            if candidate == requested:
                continue
            resolved = self._read_font(font_mod, candidate, fallback=True)
            if resolved is not None:
                return resolved
        try:
            default_font = font_mod.load_default(size=_PROBE_FONT_SIZE)
        except TypeError:  # Pillow 10.0 does not accept the size keyword.
            default_font = font_mod.load_default()
        data = getattr(default_font, "font_bytes", None)
        if not isinstance(data, bytes):
            # A bitmap default font has no character map to verify coverage.
            return None
        return self._describe_font(
            data,
            index=0,
            display_name=PILLOW_DEFAULT_FONT_NAME,
            license_id=PILLOW_DEFAULT_FONT_LICENSE,
            fallback=True,
            pillow_default=True,
        )

    def _describe_font(
        self,
        data: bytes,
        *,
        index: int,
        display_name: str,
        license_id: str,
        fallback: bool,
        pillow_default: bool = False,
    ) -> _ResolvedFont:
        sha256 = hashlib.sha256(data).hexdigest()
        try:
            coverage = self._character_map_provider(data, index)
        except BurnedInFontError as exc:
            raise BurnedInFontError(exc.reason, font_sha256=sha256) from None
        return _ResolvedFont(
            display_name=display_name,
            sha256=sha256,
            license=license_id,
            fallback=fallback,
            data=data,
            index=index,
            coverage=coverage,
            pillow_default=pillow_default,
        )

    def _check_layout(
        self,
        font_mod: Any,
        resolved: _ResolvedFont,
        size: int,
        scripts: Sequence[str],
        *,
        label: str,
    ) -> str:
        face = _load_face(font_mod, resolved, size)
        engine = _layout_engine_name(font_mod, face)
        complex_scripts = [
            script
            for script in scripts
            if script in COMPLEX_LAYOUT_SCRIPTS or script == _UNKNOWN_SCRIPT
        ]
        if complex_scripts and not self._layout_capability_provider(face):
            raise BurnedInShapingUnavailableError(
                script=complex_scripts[0],
                label=label,
                font_sha256=resolved.sha256,
                layout_engine=engine,
            )
        return engine

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
    script_fonts: Mapping[str, BurnedInFont] | None = None,
    font_fallback: bool = False,
) -> BurnedInExample:
    """Generate one synthetic burned-in PHI example with default variation.

    Args:
        seed: Seed controlling PHI values, background, font, size, and layout.
        language: Locale-PHI language used for the name and date strings.
        canvas_size: Output image size as ``(width, height)`` pixels.
        script_fonts: Optional caller-supplied fonts keyed by script.
        font_fallback: Opt in to replacing missing default fonts.

    Returns:
        A synthetic Pillow image with aligned canonical spans and boxes.
    """

    return BurnedInGenerator(
        seed=seed,
        canvas_size=canvas_size,
        script_fonts=script_fonts,
        font_fallback=font_fallback,
    ).generate(language)


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
    *,
    load_font: Callable[[int], Any],
    text: str,
    font_size: int,
    max_width: int,
    max_height: int,
    allow_resize: bool,
) -> tuple[Any, int, tuple[int, int, int, int]]:
    if max_width <= 0 or max_height <= 0:
        raise ValueError("text position leaves no drawable canvas area")

    size = font_size
    while size >= 8:
        font = load_font(size)
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
            return font, size, tuple(map(int, measured))
        if not allow_resize:
            break
        size -= 1
    raise ValueError(
        f"rendered PHI text does not fit within {max_width}x{max_height} pixels"
    )


def _load_face(font_mod: Any, resolved: _ResolvedFont, size: int) -> Any:
    if resolved.pillow_default:
        # Only resolved when Pillow exposes the embedded font's bytes, which
        # implies a Pillow release whose ``load_default`` accepts ``size``.
        return font_mod.load_default(size=size)
    # Render from the digested bytes so the recorded digest is the drawn font.
    return font_mod.truetype(io.BytesIO(resolved.data), size, index=resolved.index)


def _font_file_bytes(font: Any) -> bytes | None:
    data = getattr(font, "font_bytes", None)
    if isinstance(data, bytes):
        return data
    path = getattr(font, "path", None)
    if isinstance(path, (str, bytes, os.PathLike)):
        with open(path, "rb") as stream:
            return stream.read()
    return None


def _layout_engine_name(font_mod: Any, font: Any) -> str:
    layout = getattr(font_mod, "Layout", None)
    engine = getattr(font, "layout_engine", None)
    if layout is not None and engine == getattr(layout, "RAQM", object()):
        return "raqm"
    if layout is not None and engine == getattr(layout, "BASIC", object()):
        return "basic"
    return "bitmap"


def _font_reference(requested: str | BurnedInFont) -> str:
    if isinstance(requested, str) and requested in DEFAULT_FONT_NAMES:
        return requested
    raw = (
        os.fspath(requested.path) if isinstance(requested, BurnedInFont) else requested
    )
    digest = hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()
    return f"name-sha256:{digest[:16]}"


def _canonical_font_license(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    folded = value.strip().casefold()
    for allowed in sorted(PERMISSIVE_FONT_LICENSES):
        if allowed.casefold() == folded:
            return allowed
    return None


def _script_of(code_point: int) -> str:
    index = bisect.bisect_right(_SCRIPT_RANGE_STARTS, code_point) - 1
    if index >= 0:
        _, end, script = _SCRIPT_RANGES[index]
        if code_point <= end:
            return script
    if unicodedata.category(chr(code_point))[0] in "LM":
        return _UNKNOWN_SCRIPT
    return _COMMON_SCRIPT


def _span_scripts(text: str) -> tuple[str, ...]:
    """Return the specific scripts in ``text`` in first-appearance order."""

    scripts: list[str] = []
    for char in text:
        script = _script_of(ord(char))
        if script in (_COMMON_SCRIPT, _INHERITED_SCRIPT) or script in scripts:
            continue
        scripts.append(script)
    return tuple(scripts)


def _needs_glyph(code_point: int) -> bool:
    if unicodedata.category(chr(code_point)) in ("Cc", "Cs"):
        return False
    return not any(
        start <= code_point <= end for start, end in _DEFAULT_IGNORABLE_RANGES
    )


def _check_glyph_coverage(resolved: _ResolvedFont, text: str, *, label: str) -> None:
    missing_scripts: list[str] = []
    missing_count = 0
    for char in text:
        code_point = ord(char)
        if not _needs_glyph(code_point) or code_point in resolved.coverage:
            continue
        missing_count += 1
        script = _script_of(code_point)
        if script not in missing_scripts:
            missing_scripts.append(script)
    if not missing_count:
        return
    specific = [
        script
        for script in missing_scripts
        if script not in (_COMMON_SCRIPT, _INHERITED_SCRIPT)
    ]
    raise BurnedInGlyphCoverageError(
        script=(specific or missing_scripts)[0],
        label=label,
        font_sha256=resolved.sha256,
        missing_count=missing_count,
    )


class _MalformedFontError(Exception):
    """Internal marker for font bytes the sfnt reader cannot decode."""


def _read_character_map(data: bytes, index: int) -> FontCharacterMap:
    offset = 0
    if data[:4] == b"ttcf":
        (face_count,) = struct.unpack_from(">I", data, 8)
        if not 0 <= index < face_count:
            raise _MalformedFontError
        (offset,) = struct.unpack_from(">I", data, 12 + (4 * index))
    elif index != 0:
        raise _MalformedFontError
    if data[offset : offset + 4] not in _SFNT_VERSIONS:
        raise _MalformedFontError

    (table_count,) = struct.unpack_from(">H", data, offset + 4)
    tables: dict[bytes, int] = {}
    for record in range(table_count):
        position = offset + 12 + (16 * record)
        tag = data[position : position + 4]
        (table_offset,) = struct.unpack_from(">I", data, position + 8)
        tables.setdefault(tag, table_offset)
    cmap_offset = tables.get(b"cmap")
    if cmap_offset is None:
        raise _MalformedFontError
    glyph_count = 0x10000
    if b"maxp" in tables:
        (glyph_count,) = struct.unpack_from(">H", data, tables[b"maxp"] + 4)

    (subtable_count,) = struct.unpack_from(">H", data, cmap_offset + 2)
    unicode_subtables: list[tuple[tuple[int, int], int]] = []
    for record in range(subtable_count):
        platform, encoding, subtable_offset = struct.unpack_from(
            ">HHI", data, cmap_offset + 4 + (8 * record)
        )
        # FreeType exposes these as Unicode charmaps; (0, 5) holds variation
        # sequences only.
        if (platform == 0 and encoding != 5) or (platform == 3 and encoding in (1, 10)):
            unicode_subtables.append(
                ((platform, encoding), cmap_offset + subtable_offset)
            )
    ucs4 = [
        position
        for encoding, position in unicode_subtables
        if encoding in _UCS4_CMAP_ENCODINGS
    ]
    if ucs4:
        selected = ucs4[-1]
    elif unicode_subtables:
        selected = unicode_subtables[-1][1]
    else:
        return FontCharacterMap()
    return FontCharacterMap(_subtable_ranges(data, selected, glyph_count))


def _subtable_ranges(
    data: bytes, offset: int, glyph_count: int
) -> Iterator[tuple[int, int]]:
    (subtable_format,) = struct.unpack_from(">H", data, offset)
    if subtable_format == 4:
        return _format4_ranges(data, offset, glyph_count)
    if subtable_format == 12:
        return _format12_ranges(data, offset, glyph_count)
    if subtable_format == 6:
        return _format6_ranges(data, offset, glyph_count)
    if subtable_format == 0:
        return _format0_ranges(data, offset, glyph_count)
    # Other formats are not decoded; an empty map fails coverage closed.
    return iter(())


def _format0_ranges(
    data: bytes, offset: int, glyph_count: int
) -> Iterator[tuple[int, int]]:
    glyphs = data[offset + 6 : offset + 6 + 256]
    if len(glyphs) != 256:
        raise _MalformedFontError
    for code_point, glyph in enumerate(glyphs):
        if 0 < glyph < glyph_count:
            yield code_point, code_point


def _format4_ranges(
    data: bytes, offset: int, glyph_count: int
) -> Iterator[tuple[int, int]]:
    (segment_bytes,) = struct.unpack_from(">H", data, offset + 6)
    end_codes = offset + 14
    start_codes = end_codes + segment_bytes + 2
    deltas = start_codes + segment_bytes
    range_offsets = deltas + segment_bytes
    for segment in range(segment_bytes // 2):
        (end,) = struct.unpack_from(">H", data, end_codes + (2 * segment))
        (start,) = struct.unpack_from(">H", data, start_codes + (2 * segment))
        (delta,) = struct.unpack_from(">H", data, deltas + (2 * segment))
        range_offset_position = range_offsets + (2 * segment)
        (range_offset,) = struct.unpack_from(">H", data, range_offset_position)
        for code_point in range(start, min(end, 0xFFFE) + 1):
            if range_offset == 0:
                glyph = (code_point + delta) & 0xFFFF
            else:
                address = range_offset_position + range_offset
                address += 2 * (code_point - start)
                if address + 2 > len(data):
                    continue
                (glyph,) = struct.unpack_from(">H", data, address)
                if glyph:
                    glyph = (glyph + delta) & 0xFFFF
            if 0 < glyph < glyph_count:
                yield code_point, code_point


def _format6_ranges(
    data: bytes, offset: int, glyph_count: int
) -> Iterator[tuple[int, int]]:
    first_code, entry_count = struct.unpack_from(">HH", data, offset + 6)
    for entry in range(entry_count):
        (glyph,) = struct.unpack_from(">H", data, offset + 10 + (2 * entry))
        if 0 < glyph < glyph_count:
            yield first_code + entry, first_code + entry


def _format12_ranges(
    data: bytes, offset: int, glyph_count: int
) -> Iterator[tuple[int, int]]:
    (group_count,) = struct.unpack_from(">I", data, offset + 12)
    for group in range(group_count):
        start, end, first_glyph = struct.unpack_from(
            ">III", data, offset + 16 + (12 * group)
        )
        end = min(end, 0x10FFFF)
        if first_glyph == 0:
            start += 1
            first_glyph = 1
        if start > end or first_glyph >= glyph_count:
            continue
        last = min(end, start + (glyph_count - 1 - first_glyph))
        yield start, last


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


def _validate_font_name(name: Any) -> str | BurnedInFont:
    if isinstance(name, BurnedInFont):
        return name
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "font names must be non-empty strings or BurnedInFont instances"
        )
    return name


def _validate_font_names(
    names: Sequence[str | BurnedInFont],
) -> tuple[str | BurnedInFont, ...]:
    if not names:
        raise ValueError("font_names must not be empty")
    return tuple(_validate_font_name(name) for name in names)


def _validate_script_fonts(
    script_fonts: Mapping[str, BurnedInFont] | None,
) -> dict[str, BurnedInFont]:
    if script_fonts is None:
        return {}
    validated: dict[str, BurnedInFont] = {}
    for script, font in script_fonts.items():
        if script not in FONT_SCRIPTS:
            raise ValueError(
                f"unsupported font script {script!r}; supported={list(FONT_SCRIPTS)!r}"
            )
        if not isinstance(font, BurnedInFont):
            raise ValueError("script_fonts values must be BurnedInFont instances")
        validated[script] = font
    return validated


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
    "COMPLEX_LAYOUT_SCRIPTS",
    "DEFAULT_CANVAS_SIZE",
    "DEFAULT_FONT_LICENSE",
    "DEFAULT_FONT_NAMES",
    "DEFAULT_FONT_SIZES",
    "FONT_SCRIPTS",
    "PERMISSIVE_FONT_LICENSES",
    "PILLOW_DEFAULT_FONT_LICENSE",
    "PILLOW_DEFAULT_FONT_NAME",
    "UNDECLARED_FONT_LICENSE",
    "BurnedInExample",
    "BurnedInFont",
    "BurnedInFontError",
    "BurnedInFontUnavailableError",
    "BurnedInGenerator",
    "BurnedInGlyphCoverageError",
    "BurnedInShapingUnavailableError",
    "BurnedInTextBox",
    "CharacterMapProvider",
    "FontCharacterMap",
    "LayoutCapabilityProvider",
    "font_supports_complex_layout",
    "generate_burned_in_example",
    "read_font_character_map",
]
