"""Chinese text normalization with offset preservation (roadmap v2.0).

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

Two width conventions are supported: the recommended CJK convention (Latin,
digits, and symbols to half-width; everything else, including Han, left as-is)
and strict per-character NFKC.

The module also wraps the optional Apache-2.0 OpenCC package for Simplified,
Traditional, Taiwan, and Hong Kong conversions. OpenCC phrase conversions are
not always one code point in to one code point out, so :func:`convert_script`
aligns independently reproducible conversion units. Context-dependent phrase
rewrites map conservatively to the complete source phrase. Downstream PHI
detectors can therefore run on a canonical Chinese script and safely project
detected spans onto the original document before redaction.
"""

from __future__ import annotations

import unicodedata
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

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
# OpenCC's standard dictionary phrases are short. Limiting independent chunk
# probes keeps alignment linear in document length; an unresolved longer rule
# falls back to one conservative source span rather than guessing offsets.
_MAX_ALIGNMENT_SOURCE_CHARS = 32


class OpenCCConfig(str, Enum):
    """Supported OpenCC conversion configurations."""

    S2T = "s2t"
    T2S = "t2s"
    S2TW = "s2tw"
    TW2S = "tw2s"
    S2HK = "s2hk"
    HK2S = "hk2s"
    T2TW = "t2tw"
    T2HK = "t2hk"
    S2TWP = "s2twp"
    TW2SP = "tw2sp"


class ChineseTargetScript(str, Enum):
    """Canonical Chinese script used by variant normalization."""

    SIMPLIFIED = "simplified"
    TRADITIONAL = "traditional"


class OpenCCUnavailableWarning(RuntimeWarning):
    """Warning emitted when optional OpenCC conversion is unavailable."""


_OPENCC_NOTICE_EMITTED = False


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


@dataclass(frozen=True)
class ScriptConversion:
    """OpenCC-converted text with code-point alignment to the original.

    ``char_origins[i]`` is the ``(start, end)`` source span that produced
    converted character ``i``. Unchanged anchors map one-to-one. Characters in
    a context-dependent phrase replacement share the complete source-phrase
    span so partial detections cannot point at adjacent text.
    """

    text: str
    original: str
    char_origins: tuple[tuple[int, int], ...]
    config: OpenCCConfig
    opencc_available: bool = True

    @property
    def changed(self) -> bool:
        """Return whether OpenCC changed the input text."""

        return self.text != self.original

    @property
    def alignment(self) -> tuple[tuple[int, int], ...]:
        """Alias for the converted-character to source-span alignment map."""

        return self.char_origins

    @property
    def offset_map(self) -> tuple[tuple[int, int], ...]:
        """Alias using the public issue terminology for the alignment map."""

        return self.char_origins

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a ``[start, end)`` converted-text span to source offsets."""

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
        spans = self.char_origins[start:end]
        return min(span[0] for span in spans), max(span[1] for span in spans)

    def __iter__(self) -> Iterator[object]:
        """Allow ``converted_text, alignment = convert_script(...)``."""

        yield self.text
        yield self.char_origins


def _coerce_opencc_config(config: OpenCCConfig | str) -> OpenCCConfig:
    if isinstance(config, OpenCCConfig):
        return config
    value = str(config).lower()
    if value.endswith(".json"):
        value = value[:-5]
    try:
        return OpenCCConfig(value)
    except ValueError as exc:
        supported = ", ".join(item.value for item in OpenCCConfig)
        raise ValueError(
            f"unknown OpenCC config {config!r}; supported configs: {supported}"
        ) from exc


def _coerce_target_script(
    target: ChineseTargetScript | str,
) -> ChineseTargetScript:
    if isinstance(target, ChineseTargetScript):
        return target
    try:
        return ChineseTargetScript(str(target).lower())
    except ValueError as exc:
        raise ValueError(
            f"target script must be 'simplified' or 'traditional', got {target!r}"
        ) from exc


@lru_cache(maxsize=len(OpenCCConfig))
def _opencc_converter(config: OpenCCConfig) -> Any:
    import opencc

    return opencc.OpenCC(f"{config.value}.json")


def _notice_opencc_unavailable() -> None:
    global _OPENCC_NOTICE_EMITTED
    if _OPENCC_NOTICE_EMITTED:
        return
    warnings.warn(
        "OpenCC is not installed; Chinese script conversion was skipped. "
        "Install it with `pip install 'openmed[zh]'`.",
        OpenCCUnavailableWarning,
        stacklevel=3,
    )
    _OPENCC_NOTICE_EMITTED = True


def _identity_char_origins(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((index, index + 1) for index in range(len(text)))


def _align_replacement(
    source: str,
    converted: str,
    source_start: int,
    source_end: int,
    converted_start: int,
    converted_end: int,
    convert_chunk: Callable[[str], str],
) -> list[tuple[int, int]]:
    source_chunk = source[source_start:source_end]
    converted_chunk = converted[converted_start:converted_end]
    converted_length = len(converted_chunk)
    source_length = source_end - source_start
    if converted_length <= 0:
        return []
    if source_length <= 0:
        return [(source_start, source_start)] * converted_length

    if source_length == converted_length:
        character_converted = [convert_chunk(character) for character in source_chunk]
        if (
            all(len(value) == 1 for value in character_converted)
            and "".join(character_converted) == converted_chunk
        ):
            return [
                (source_start + index, source_start + index + 1)
                for index in range(source_length)
            ]

    # A phrase-level rewrite can reorder, expand, or contract source characters.
    # Without token provenance from OpenCC, assigning proportional sub-spans can
    # point a detected identifier at adjacent text. Map every output character
    # to the complete rewritten source chunk instead so redaction fails closed.
    return [(source_start, source_end)] * converted_length


def _align_converted_text(
    source: str,
    converted: str,
    converter: Any,
) -> tuple[tuple[int, int], ...]:
    """Align independent conversion units and fail closed for phrase rewrites."""

    if not converted:
        return ()
    if not source:
        return tuple((0, 0) for _ in converted)

    conversion_cache: dict[str, str] = {}

    def convert_chunk(chunk: str) -> str:
        if chunk not in conversion_cache:
            conversion_cache[chunk] = str(converter.convert(chunk))
        return conversion_cache[chunk]

    origins: list[tuple[int, int]] = []
    source_cursor = 0
    converted_cursor = 0
    while source_cursor < len(source) and converted_cursor < len(converted):
        probe_limit = min(
            len(source),
            source_cursor + _MAX_ALIGNMENT_SOURCE_CHARS,
        )
        matched = False
        for source_end in range(source_cursor + 1, probe_limit + 1):
            chunk_converted = convert_chunk(source[source_cursor:source_end])
            if not chunk_converted or not converted.startswith(
                chunk_converted,
                converted_cursor,
            ):
                continue
            converted_end = converted_cursor + len(chunk_converted)
            origins.extend(
                _align_replacement(
                    source,
                    converted,
                    source_cursor,
                    source_end,
                    converted_cursor,
                    converted_end,
                    convert_chunk,
                )
            )
            source_cursor = source_end
            converted_cursor = converted_end
            matched = True
            break

        if matched:
            continue

        origins.extend(
            _align_replacement(
                source,
                converted,
                source_cursor,
                len(source),
                converted_cursor,
                len(converted),
                convert_chunk,
            )
        )
        source_cursor = len(source)
        converted_cursor = len(converted)

    if converted_cursor < len(converted):
        origins.extend(
            [(len(source), len(source))] * (len(converted) - converted_cursor)
        )

    if len(origins) != len(converted):
        raise RuntimeError("OpenCC alignment did not cover converted text")
    return tuple(origins)


def convert_script(
    text: str,
    config: OpenCCConfig | str,
) -> ScriptConversion:
    """Convert Chinese script with OpenCC and preserve source offsets.

    Args:
        text: Source text in Simplified, Traditional, or mixed Chinese.
        config: One of the standard OpenCC configurations represented by
            :class:`OpenCCConfig` (a matching string is also accepted).

    Returns:
        Converted text plus a per-character source alignment. If OpenCC is not
        installed, returns the input with identity alignment and emits one
        process-wide optional-dependency warning.
    """

    resolved_config = _coerce_opencc_config(config)
    try:
        converter = _opencc_converter(resolved_config)
    except (ImportError, ModuleNotFoundError):
        _notice_opencc_unavailable()
        return ScriptConversion(
            text=text,
            original=text,
            char_origins=_identity_char_origins(text),
            config=resolved_config,
            opencc_available=False,
        )

    converted = str(converter.convert(text))
    return ScriptConversion(
        text=converted,
        original=text,
        char_origins=_align_converted_text(text, converted, converter),
        config=resolved_config,
    )


def normalize_chinese_variants(
    text: str,
    target: ChineseTargetScript | str,
) -> ScriptConversion:
    """Canonicalize mixed Chinese variants to ``target`` script."""

    resolved_target = _coerce_target_script(target)
    config = (
        OpenCCConfig.T2S
        if resolved_target is ChineseTargetScript.SIMPLIFIED
        else OpenCCConfig.S2T
    )
    return convert_script(text, config)


def detect_variant_normalized(
    text: str,
    matcher: Callable[[str], Iterable[Sequence[object]]],
    *,
    target: ChineseTargetScript | str,
) -> list[tuple[object, ...]]:
    """Run ``matcher`` on canonical Chinese text and restore source spans."""

    conversion = normalize_chinese_variants(text, target)
    results: list[tuple[object, ...]] = []
    for item in matcher(conversion.text):
        start = int(item[0])  # type: ignore[call-overload]
        end = int(item[1])  # type: ignore[call-overload]
        original_start, original_end = conversion.to_original_span(start, end)
        results.append((original_start, original_end, *tuple(item[2:])))
    return results


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
    "ChineseTargetScript",
    "OpenCCConfig",
    "OpenCCUnavailableWarning",
    "ScriptConversion",
    "WidthNormalization",
    "convert_script",
    "detect_variant_normalized",
    "normalize_width",
    "normalize_chinese_variants",
    "detect_width_normalized",
]
