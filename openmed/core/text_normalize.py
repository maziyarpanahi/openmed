"""Small, opt-in text normalizers for deterministic PII detection.

Detection patterns in the language packs intentionally remain ASCII-oriented
by default.  Callers that want native digit support can normalize their input
with :func:`normalize_for_detection`, run the existing patterns on the
returned text, and project each match back through its offset map.

The map is expressed in Python code-point offsets, matching the offsets used
by the rest of the core package.  NFKC can expand one source character into
several output characters (for example, ``ﬁ`` becomes ``fi``), so each output
character keeps both its source index and its source span.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

_DIGIT_TRANSLATION = str.maketrans(
    {
        **{
            chr(base + digit): str(digit)
            for base in (
                0x0660,  # Arabic-Indic
                0x06F0,  # Eastern Arabic-Indic (Persian)
                0x0966,  # Devanagari
                0x09E6,  # Bengali
                0x0E50,  # Thai
                0xFF10,  # Fullwidth
                0x0C66,  # Telugu
            )
            for digit in range(10)
        }
    }
)


@dataclass(frozen=True)
class OffsetMap(Sequence[int]):
    """Map normalized code points back to source code-point offsets.

    ``offset_map[index]`` is the source index that produced the normalized
    code point at ``index``.  Expanded or composed output can cover multiple
    source code points; :meth:`to_original_span` uses the retained source
    spans to project a complete normalized span without losing those offsets.

    The class deliberately behaves like a read-only sequence of integers so a
    caller can use the simple index-map form while still getting an exact span
    helper for length-changing normalization.
    """

    indices: tuple[int, ...]
    source_spans: tuple[tuple[int, int], ...]
    original_length: int

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.source_spans):
            raise ValueError("indices and source_spans must have equal lengths")
        if self.original_length < 0:
            raise ValueError("original_length must be non-negative")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        return self.indices[index]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OffsetMap):
            return (
                self.indices == other.indices
                and self.source_spans == other.source_spans
                and self.original_length == other.original_length
            )
        if isinstance(other, Sequence):
            return self.indices == tuple(other)
        return NotImplemented

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        """Translate a normalized half-open span to source offsets.

        Args:
            start: Inclusive normalized code-point offset.
            end: Exclusive normalized code-point offset.

        Returns:
            The smallest source half-open span covering the normalized span.

        Raises:
            ValueError: If the span is outside the normalized text or reversed.
        """

        if not (0 <= start <= end <= len(self)):
            raise ValueError("span must satisfy 0 <= start <= end <= normalized length")
        if start == end:
            if start < len(self.source_spans):
                anchor = self.source_spans[start][0]
            elif self.source_spans:
                anchor = self.source_spans[-1][1]
            else:
                anchor = self.original_length
            return anchor, anchor

        spans = self.source_spans[start:end]
        return min(span[0] for span in spans), max(span[1] for span in spans)

    # These aliases make the map convenient for callers that use the naming
    # conventions of the other offset-aware normalization helpers.
    map_span = to_original_span
    normalized_span_to_original_offsets = to_original_span


def _require_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text


def normalize_digits(text: str) -> str:
    """Fold supported native decimal digits to ASCII.

    Arabic-Indic, Persian/Eastern Arabic-Indic, Devanagari, Telugu, Bengali,
    Thai, and fullwidth decimal digits are mapped one code point at a time.
    Letters, punctuation, whitespace, and digits from other scripts are left
    unchanged.  The operation is length-preserving.

    Args:
        text: Text that may contain supported native decimal digits.

    Returns:
        ``text`` with the supported digits rendered as ``0`` through ``9``.
    """

    return _require_text(text).translate(_DIGIT_TRANSLATION)


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Apply a Unicode normalization form to ``text``.

    ``NFKC`` is the default because it folds compatibility forms such as
    fullwidth Latin characters and ligatures.  This function does not fold
    native decimal digits that are not affected by the selected Unicode form;
    use :func:`normalize_digits` or :func:`normalize_for_detection` for that
    opt-in detection behavior.
    """

    return unicodedata.normalize(form, _require_text(text))


def _normalization_map(
    text: str,
    normalized: str,
    per_character_text: str,
    per_character_spans: Sequence[tuple[int, int]],
) -> OffsetMap:
    """Align global normalization output with independently normalized input.

    Per-code-point normalization is exact for the common compatibility and
    width folds.  Canonical composition can also combine adjacent source code
    points, so the uncommon case is aligned with a bounded fallback.  Changed
    regions conservatively cover every source code point that contributed to
    the region; this is the only safe mapping when normalization reorders or
    composes characters.
    """

    if per_character_text == normalized:
        spans = tuple(per_character_spans)
        return OffsetMap(
            indices=tuple(start for start, _end in spans),
            source_spans=spans,
            original_length=len(text),
        )

    matcher = SequenceMatcher(
        a=per_character_text,
        b=normalized,
        autojunk=False,
    )
    output_spans: list[tuple[int, int]] = []
    previous_source_end = 0
    previous_output_end = 0

    for old_start, new_start, match_size in matcher.get_matching_blocks():
        old_end = old_start + match_size
        new_end = new_start + match_size
        if previous_output_end < new_start:
            changed_spans = per_character_spans[previous_source_end:old_start]
            if changed_spans:
                source_span = (
                    min(span[0] for span in changed_spans),
                    max(span[1] for span in changed_spans),
                )
            elif per_character_spans:
                anchor = (
                    per_character_spans[old_start][0]
                    if old_start < len(per_character_spans)
                    else per_character_spans[-1][1]
                )
                source_span = (anchor, anchor)
            else:
                source_span = (0, 0)
            output_spans.extend([source_span] * (new_start - previous_output_end))

        if old_start < old_end:
            output_spans.extend(per_character_spans[old_start:old_end])

        previous_source_end = old_end
        previous_output_end = new_end

    if len(output_spans) != len(normalized):
        raise RuntimeError("failed to build Unicode normalization offset map")

    return OffsetMap(
        indices=tuple(start for start, _end in output_spans),
        source_spans=tuple(output_spans),
        original_length=len(text),
    )


def normalize_unicode_with_offsets(
    text: str,
    form: str = "NFKC",
) -> tuple[str, OffsetMap]:
    """Normalize Unicode and retain a source map for every output code point.

    The returned map uses source character offsets, not UTF-8 byte offsets.
    For a one-to-many expansion, every output index points to the source
    character that expanded.  For a many-to-one composition, the map's
    :meth:`OffsetMap.to_original_span` method covers all source characters.
    """

    text = _require_text(text)
    normalized = normalize_unicode(text, form=form)

    per_character_parts: list[str] = []
    per_character_spans: list[tuple[int, int]] = []
    for index, character in enumerate(text):
        normalized_character = unicodedata.normalize(form, character)
        per_character_parts.append(normalized_character)
        per_character_spans.extend((index, index + 1) for _ in normalized_character)

    return normalized, _normalization_map(
        text,
        normalized,
        "".join(per_character_parts),
        per_character_spans,
    )


def normalize_for_detection(
    text: str,
    *,
    form: str = "NFKC",
) -> tuple[str, OffsetMap]:
    """Return normalized detection text and a map back to the original text.

    This is the opt-in boundary for ASCII-oriented language-pack patterns:
    callers can pass the returned text to an existing detector and use
    ``offset_map.to_original_span(start, end)`` for every match.  Digit folding
    is length-preserving, so the Unicode map remains valid after that final
    stage.
    """

    normalized, offset_map = normalize_unicode_with_offsets(text, form=form)
    return normalize_digits(normalized), offset_map


__all__ = [
    "OffsetMap",
    "normalize_digits",
    "normalize_for_detection",
    "normalize_unicode",
    "normalize_unicode_with_offsets",
]
