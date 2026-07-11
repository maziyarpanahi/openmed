"""RTL-aware rendering of de-identified output with bidi-safe masks.

When de-identifying right-to-left text (Arabic, Persian, Hebrew), the mask
tokens and Latin surrogates injected in place of detected PII sit inside a
right-to-left run. Under the Unicode Bidirectional Algorithm (UAX #9) an LTR
token such as ``[NAME]`` embedded in an RTL paragraph can reorder visually
against the surrounding text, producing garbled or misleading redacted output.

This module wraps each injected replacement in Unicode isolate marks
(:data:`FSI` ... :data:`PDI`) so the replacement renders in its own bidi
context and cannot reorder the surrounding RTL text. The transformation is a
no-op for pure left-to-right documents: their output is returned byte-for-byte
unchanged so existing de-identification behaviour is unaffected.

The helpers here are stdlib-only, local-first, and never inspect or store
plaintext identifiers beyond the replacement tokens they are asked to wrap.
Script detection is delegated to :mod:`openmed.core.script_detect`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Union

from .script_detect import detect_script, segment_by_script

# --- Unicode bidi control characters (UAX #9) --------------------------------

FSI = "\u2068"  # First Strong Isolate
LRI = "\u2066"  # Left-to-Right Isolate
RLI = "\u2067"  # Right-to-Left Isolate
PDI = "\u2069"  # Pop Directional Isolate

# Legacy embedding/override controls plus explicit directional marks. These are
# stripped by :func:`strip_bidi_controls` so plain-text consumers recover clean
# output regardless of which control scheme produced it.
_LRE = "\u202a"  # Left-to-Right Embedding
_RLE = "\u202b"  # Right-to-Left Embedding
_PDF = "\u202c"  # Pop Directional Formatting
_LRO = "\u202d"  # Left-to-Right Override
_RLO = "\u202e"  # Right-to-Left Override
_LRM = "\u200e"  # Left-to-Right Mark
_RLM = "\u200f"  # Right-to-Left Mark
_ALM = "\u061c"  # Arabic Letter Mark

#: All bidi control code points recognised by :func:`strip_bidi_controls`.
BIDI_CONTROL_CHARS = frozenset(
    {
        FSI,
        LRI,
        RLI,
        PDI,
        _LRE,
        _RLE,
        _PDF,
        _LRO,
        _RLO,
        _LRM,
        _RLM,
        _ALM,
    }
)

#: Unicode scripts that lay out right-to-left. Kept local to this module so it
#: stays aligned with the RTL languages OpenMed supports (Arabic covers ar/fa,
#: Hebrew covers he).
RTL_SCRIPTS = frozenset({"Arabic", "Hebrew"})

_LTR = "ltr"
_RTL = "rtl"
_AUTO = "auto"


def _normalize_direction(direction: str) -> str:
    """Validate and normalize a rendering direction."""

    if not isinstance(direction, str):
        raise TypeError("direction must be a string")
    normalized = direction.lower()
    if normalized not in {_AUTO, _RTL, _LTR}:
        raise ValueError("direction must be one of 'auto', 'rtl', or 'ltr'")
    return normalized


# --- Base direction ----------------------------------------------------------


def base_direction(text: str) -> str:
    """Return the document base direction implied by ``text``.

    The direction is derived from the dominant Unicode script: a document whose
    dominant script is right-to-left (Arabic or Hebrew) reports ``"rtl"``;
    everything else, including script-free text, reports ``"ltr"``.

    Args:
        text: The text to inspect.

    Returns:
        Either ``"rtl"`` or ``"ltr"``.
    """

    return _RTL if detect_script(text) in RTL_SCRIPTS else _LTR


def contains_rtl(text: str) -> bool:
    """Return whether ``text`` contains any right-to-left script run."""

    return any(
        script in RTL_SCRIPTS for _start, _end, script in segment_by_script(text)
    )


# --- Result container --------------------------------------------------------


@dataclass(frozen=True)
class RenderedRedaction:
    """Result of rendering de-identified text with bidi-safe masks.

    Attributes:
        text: The rendered output. Identical to the input for LTR documents;
            for RTL/mixed documents each replacement span is wrapped in
            ``FSI ... PDI`` isolates.
        base_direction: ``"rtl"`` or ``"ltr"`` for the document as a whole, so a
            downstream renderer can set ``dir=rtl`` on its container.
        isolated: Whether any isolates were inserted (always ``False`` for the
            LTR no-op path).
        source_offsets: For each wrapped span, the ``(start, end)`` offsets in
            the *input* ``text``, in ascending order. Empty when nothing was
            wrapped.
    """

    text: str
    base_direction: str
    isolated: bool = False
    source_offsets: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.text

    @property
    def is_rtl(self) -> bool:
        """Whether the document base direction is right-to-left."""

        return self.base_direction == _RTL


# --- Span normalisation ------------------------------------------------------

SpanLike = Union[Sequence[int], Mapping[str, Any], Any]


def _span_offsets(span: SpanLike, text_len: int) -> tuple[int, int]:
    """Extract ``(start, end)`` offsets from a span-like value.

    Accepts ``(start, end)`` / ``(start, end, ...)`` sequences, mappings with
    ``start``/``end`` keys, and objects exposing ``start``/``end`` attributes.
    Every offset must index the already-redacted ``text`` passed to
    :func:`render_redacted`. In particular, ``PIIEntity`` offsets index the
    original input and must not be passed through without remapping them to the
    rendered replacement positions.
    """

    if isinstance(span, Mapping):
        start = span.get("start")
        end = span.get("end")
    elif isinstance(span, Sequence) and not isinstance(span, (str, bytes)):
        if len(span) < 2:
            raise ValueError("span sequence must provide start and end offsets")
        start, end = span[0], span[1]
    else:
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)

    if start is None or end is None:
        raise ValueError("span must provide integer start and end offsets")

    start = int(start)
    end = int(end)
    if start < 0 or end < start:
        raise ValueError("span offsets must be non-negative with end >= start")
    if end > text_len:
        raise ValueError("span offsets must fall within text")
    return start, end


def _normalize_spans(spans: Sequence[SpanLike], text_len: int) -> list[tuple[int, int]]:
    """Return non-overlapping ``(start, end)`` spans sorted by position."""

    offsets = [_span_offsets(span, text_len) for span in spans]
    offsets.sort(key=lambda pair: (pair[0], pair[1]))

    ordered: list[tuple[int, int]] = []
    previous_end = -1
    for start, end in offsets:
        if start == end:
            # Zero-width spans carry no replacement text to isolate. Ignore
            # them before overlap checks, including when they sit inside a
            # non-empty replacement span.
            continue
        if start < previous_end:
            raise ValueError("spans must not overlap")
        ordered.append((start, end))
        previous_end = end
    return ordered


# --- Public rendering API ----------------------------------------------------


def render_redacted(
    text: str,
    spans: Sequence[SpanLike] | None = None,
    direction: str = _AUTO,
) -> RenderedRedaction:
    """Render de-identified ``text`` with bidi-safe masks.

    For a right-to-left or mixed-script document, each replacement span (the
    mask token or surrogate injected in place of PII) is wrapped in
    ``FSI ... PDI`` isolates so it renders in its own bidi context and cannot
    visually reorder the surrounding text. A pure left-to-right document is
    returned byte-for-byte unchanged, with no isolate characters added.

    Args:
        text: The de-identified output text (masks/surrogates already
            substituted). This is *not* the original document.
        spans: The spans covering each injected replacement, expressed as
            offsets into ``text``. Each item may be an ``(start, end)`` tuple,
            a mapping with ``start``/``end`` keys, or an object with
            ``start``/``end`` attributes. These must be replacement offsets in
            the already-redacted ``text``; offsets from the original document
            (including ``DeidentificationResult.pii_entities`` offsets) must be
            remapped first. Overlapping spans are rejected; zero-width spans
            are ignored. When omitted or empty, no spans are wrapped but the
            base direction is still reported.
        direction: ``"auto"`` (default) detects the base direction from the
            dominant script. Pass ``"rtl"`` or ``"ltr"`` to force it.

    Returns:
        A :class:`RenderedRedaction` with the rendered text, the reported base
        direction, and the source offsets of every wrapped span.

    Raises:
        TypeError: If ``text`` or ``direction`` is not a string.
        ValueError: If ``direction`` is invalid or spans overlap or fall
            outside ``text``.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    direction = _normalize_direction(direction)

    if direction == _AUTO:
        resolved = _RTL if contains_rtl(text) else _LTR
        reported = base_direction(text)
    else:
        resolved = direction
        reported = direction

    span_offsets = _normalize_spans(list(spans or ()), len(text))

    # Pure-LTR path (or an RTL document with nothing to wrap): return the input
    # unchanged so existing output stays byte-for-byte identical.
    if resolved == _LTR or not span_offsets:
        return RenderedRedaction(
            text=text,
            base_direction=reported,
            isolated=False,
            source_offsets=(),
        )

    pieces: list[str] = []
    cursor = 0
    for start, end in span_offsets:
        pieces.append(text[cursor:start])
        pieces.append(FSI)
        pieces.append(text[start:end])
        pieces.append(PDI)
        cursor = end
    pieces.append(text[cursor:])

    return RenderedRedaction(
        text="".join(pieces),
        base_direction=reported,
        isolated=True,
        source_offsets=tuple(span_offsets),
    )


def wrap_mask(mask: str, direction: str = _AUTO) -> str:
    """Wrap a single mask/surrogate token in ``FSI ... PDI`` isolates.

    A convenience for callers that build redacted output token by token rather
    than by span offsets. For ``direction="ltr"`` the mask is returned
    unchanged.

    Args:
        mask: The replacement token, for example ``"[NAME]"``.
        direction: ``"auto"`` always isolates (the mask is assumed to be placed
            in a bidi-sensitive context); ``"rtl"`` isolates; ``"ltr"`` is a
            no-op.

    Returns:
        The isolated mask, or the mask unchanged for the LTR no-op path.

    Raises:
        TypeError: If ``mask`` or ``direction`` is not a string.
        ValueError: If ``direction`` is invalid.
    """

    if not isinstance(mask, str):
        raise TypeError("mask must be a string")
    direction = _normalize_direction(direction)
    if direction == _LTR:
        return mask
    return f"{FSI}{mask}{PDI}"


def strip_bidi_controls(text: str) -> str:
    """Remove bidi isolate/embedding/override controls and directional marks.

    This is the inverse of the isolation applied by :func:`render_redacted`: it
    recovers the plain replacement text for consumers that cannot interpret
    bidi controls (logs, exact-match comparisons, downstream tokenizers).

    Args:
        text: Text that may contain bidi control characters.

    Returns:
        ``text`` with every character in :data:`BIDI_CONTROL_CHARS` removed.
        Text with no such controls is returned unchanged.

    Raises:
        TypeError: If ``text`` is not a string.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not any(char in BIDI_CONTROL_CHARS for char in text):
        return text
    return "".join(char for char in text if char not in BIDI_CONTROL_CHARS)


__all__ = [
    "BIDI_CONTROL_CHARS",
    "FSI",
    "LRI",
    "PDI",
    "RLI",
    "RTL_SCRIPTS",
    "RenderedRedaction",
    "base_direction",
    "contains_rtl",
    "render_redacted",
    "strip_bidi_controls",
    "wrap_mask",
]
