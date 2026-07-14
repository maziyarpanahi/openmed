"""Full-width/half-width normalization with offset preservation (roadmap v1.9).

Chinese documents freely mix full-width Latin letters, digits, and punctuation
(``ＡＢＣ１２３，：``) with half-width ASCII, so a phone number or ID written in
full-width digits never matches the half-width-oriented PHI regexes or
tokenizers. Blunt NFKC normalization fixes matching but can change string length
and merge characters, silently shifting every downstream PHI offset.

``normalize_width`` maps full-width variants to half-width while leaving genuine
full-width CJK content untouched, and returns an explicit per-character offset
alignment so a span detected on the normalized text maps back to the exact
original code points. The mapping is 1:1 in the common case; the rare
compatibility expansions (``㎏`` -> ``kg``) are handled by recording, for every
normalized character, the original code point it came from.

Two conventions are supported: the recommended CJK convention (Latin, digits,
and symbols to half-width; everything else, including Han, left as-is) and strict
per-character NFKC.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

#: Recommended CJK convention: full-width ASCII/space to half-width, Han as-is.
CJK_CONVENTION = "cjk"
#: Strict per-character NFKC normalization.
STRICT_NFKC = "nfkc"
WIDTH_CONVENTIONS = (CJK_CONVENTION, STRICT_NFKC)

# Full-width ASCII variants U+FF01-FF5E map to U+0021-007E by this offset.
_FULLWIDTH_START = 0xFF01
_FULLWIDTH_END = 0xFF5E
_WIDTH_OFFSET = 0xFEE0
#: Ideographic space, normalized to a configurable target.
IDEOGRAPHIC_SPACE = "　"


@dataclass(frozen=True)
class WidthNormalization:
    """Width-normalized text with an offset alignment to the original.

    ``char_origins[i]`` is the ``(start, end)`` code-point span in ``original``
    that normalized character ``i`` was produced from.
    """

    text: str
    original: str
    char_origins: tuple[tuple[int, int], ...]

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a ``[start, end)`` span on the normalized text to the original.

        The returned span is expressed in original code-point offsets and
        covers exactly the source characters the normalized span came from.
        """

        if not (0 <= start <= end <= len(self.text)):
            raise ValueError("span must satisfy 0 <= start <= end <= len(text)")
        if start == end:
            if start < len(self.char_origins):
                anchor = self.char_origins[start][0]
            elif self.char_origins:
                anchor = self.char_origins[-1][1]
            else:
                anchor = 0
            return anchor, anchor
        return self.char_origins[start][0], self.char_origins[end - 1][1]


def _convert_char(ch: str, convention: str, space_target: str) -> str:
    if convention == CJK_CONVENTION:
        code_point = ord(ch)
        if _FULLWIDTH_START <= code_point <= _FULLWIDTH_END:
            return chr(code_point - _WIDTH_OFFSET)
        if ch == IDEOGRAPHIC_SPACE:
            return space_target
        return ch
    if convention == STRICT_NFKC:
        return unicodedata.normalize("NFKC", ch)
    raise ValueError(f"unknown width convention: {convention!r}")


def normalize_width(
    text: str,
    *,
    convention: str = CJK_CONVENTION,
    space_target: str = " ",
) -> WidthNormalization:
    """Normalize full-width characters to half-width, preserving offsets.

    ``convention`` selects the recommended :data:`CJK_CONVENTION` (default) or
    strict :data:`STRICT_NFKC`. ``space_target`` is the replacement for the
    ideographic space under the CJK convention. Returns a
    :class:`WidthNormalization` whose ``char_origins`` map every normalized
    character back to its source code points.
    """

    if convention not in WIDTH_CONVENTIONS:
        raise ValueError(f"unknown width convention: {convention!r}")

    chars: list[str] = []
    origins: list[tuple[int, int]] = []
    for index, char in enumerate(text):
        for normalized_char in _convert_char(char, convention, space_target):
            chars.append(normalized_char)
            origins.append((index, index + 1))

    return WidthNormalization(
        text="".join(chars),
        original=text,
        char_origins=tuple(origins),
    )


def detect_width_normalized(
    text: str,
    matcher: Callable[[str], Iterable[Sequence[object]]],
    *,
    convention: str = CJK_CONVENTION,
    space_target: str = " ",
) -> list[tuple[object, ...]]:
    """Run ``matcher`` on width-normalized ``text`` and map spans back.

    This is the early width pre-pass: half-width-oriented PHI patterns match
    full-width phone/ID inputs once the text is normalized. ``matcher`` takes
    the normalized text and returns an iterable of items whose first two
    elements are the ``(start, end)`` span on the normalized text; any trailing
    elements (label, score, ...) are preserved. Each returned tuple is
    ``(original_start, original_end, *trailing)`` in original code-point offsets.
    """

    normalization = normalize_width(
        text, convention=convention, space_target=space_target
    )
    results: list[tuple[object, ...]] = []
    for item in matcher(normalization.text):
        start = int(item[0])  # type: ignore[call-overload]
        end = int(item[1])  # type: ignore[call-overload]
        original_start, original_end = normalization.to_original_span(start, end)
        results.append((original_start, original_end, *tuple(item[2:])))
    return results


__all__ = [
    "CJK_CONVENTION",
    "STRICT_NFKC",
    "WIDTH_CONVENTIONS",
    "IDEOGRAPHIC_SPACE",
    "WidthNormalization",
    "normalize_width",
    "detect_width_normalized",
]
