"""Conservative Indic morphology helpers with source-offset preservation.

The rules in this module intentionally cover only frequent case markers and
postpositions whose surface form can be removed at an existing grapheme
boundary.  They are not a general-purpose stemmer.  A rule is applied only
when both a confidence threshold and a caller-supplied stem allow-list agree,
which keeps the default behavior precision-first for PHI name boundaries.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Final

SUPPORTED_INDIC_MORPHOLOGY_LANGUAGES: Final[tuple[str, ...]] = (
    "hi",
    "mr",
    "ta",
    "te",
    "ml",
    "kn",
)


@dataclass(frozen=True)
class _SuffixRule:
    surface: str
    separated: str
    separated_char_origins: tuple[tuple[int, int], ...]


def _suffix_rule(surface: str) -> _SuffixRule:
    return _SuffixRule(
        surface=surface,
        separated=surface,
        separated_char_origins=tuple(
            (index, index + 1) for index in range(len(surface))
        ),
    )


def _tamil_glide_rule(surface: str, separated: str) -> _SuffixRule:
    """Reverse an initial Tamil ``வ`` glide and retain its source alignment."""
    return _SuffixRule(
        surface=surface,
        separated=separated,
        separated_char_origins=(
            (0, 2),
            *((index + 1, index + 2) for index in range(1, len(separated))),
        ),
    )


# Longest forms precede their suffixes so matching is deterministic.  These
# tables are deliberately small: expanding them requires precision fixtures.
_SUFFIX_RULES: Final[Mapping[str, tuple[_SuffixRule, ...]]] = MappingProxyType(
    {
        "hi": tuple(
            _suffix_rule(surface)
            for surface in ("में", "पर", "को", "से", "ने", "का", "की", "के")
        ),
        "mr": (
            *(
                _suffix_rule(surface)
                for surface in (
                    "पासून",
                    "मध्ये",
                    "बरोबर",
                    "साठी",
                    "कडे",
                    "वरून",
                    "ला",
                    "ने",
                    "चा",
                    "ची",
                    "चे",
                    "वर",
                )
            ),
        ),
        "ta": (
            _tamil_glide_rule("விடமிருந்து", "இடமிருந்து"),
            _tamil_glide_rule("விடம்", "இடம்"),
            _tamil_glide_rule("வுடன்", "உடன்"),
            _tamil_glide_rule("வுக்கு", "உக்கு"),
            _tamil_glide_rule("வால்", "ஆல்"),
            _tamil_glide_rule("வில்", "இல்"),
            _tamil_glide_rule("வின்", "இன்"),
            _tamil_glide_rule("வை", "ஐ"),
        ),
        "te": tuple(
            _suffix_rule(surface)
            for surface in ("నుంచి", "యొక్క", "దగ్గర", "తో", "లో", "కు", "కి", "ని", "ను")
        ),
        "ml": tuple(
            _suffix_rule(surface)
            for surface in (
                "യിൽനിന്ന്",
                "യോടൊപ്പം",
                "യുടെ",
                "യിൽ",
                "യോട്",
                "യ്ക്ക്",
                "യെ",
                "ൽ",
            )
        ),
        "kn": tuple(
            _suffix_rule(surface) for surface in ("ಜೊತೆಗೆ", "ದಿಂದ", "ದಲ್ಲಿ", "ಕ್ಕೆ", "ನ್ನು", "ಗೆ")
        ),
    }
)
INDIC_SUFFIX_TABLES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        language: tuple(rule.surface for rule in rules)
        for language, rules in _SUFFIX_RULES.items()
    }
)

_LANGUAGE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "hi": "hi",
        "hin": "hi",
        "hindi": "hi",
        "mr": "mr",
        "mar": "mr",
        "marathi": "mr",
        "ta": "ta",
        "tam": "ta",
        "tamil": "ta",
        "te": "te",
        "tel": "te",
        "telugu": "te",
        "ml": "ml",
        "mal": "ml",
        "malayalam": "ml",
        "kn": "kn",
        "kan": "kn",
        "kannada": "kn",
    }
)

_VIRAMAS: Final[frozenset[str]] = frozenset(
    {
        "\u094d",  # Devanagari sign virama
        "\u0bcd",  # Tamil sign virama
        "\u0c4d",  # Telugu sign virama
        "\u0ccd",  # Kannada sign virama
        "\u0d4d",  # Malayalam sign virama
    }
)
_JOINERS: Final[frozenset[str]] = frozenset({"\u200c", "\u200d"})


@dataclass(frozen=True)
class MorphologyOffsetMap:
    """Map each output code point to its source code-point span.

    ``char_origins[i]`` is the ``[start, end)`` source span that produced
    output character ``text[i]``.  Inserted sandhi separators use a zero-width
    source span at the split boundary.
    """

    text: str
    source: str
    char_origins: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.char_origins) != len(self.text):
            raise ValueError("char_origins must contain one entry per output character")
        for start, end in self.char_origins:
            if not (0 <= start <= end <= len(self.source)):
                raise ValueError("char origin falls outside the source text")

    def to_source_span(self, start: int, end: int) -> tuple[int, int]:
        """Map an output ``[start, end)`` span back to source offsets."""
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
class StemResult:
    """Result of conservative suffix stripping for one token."""

    token: str
    stem: str
    language: str
    confidence: float
    stem_span: tuple[int, int]
    stripped_suffix_spans: tuple[tuple[int, int], ...]
    offset_map: MorphologyOffsetMap
    rule: str | None = None
    applied: bool = False

    @property
    def stripped_suffixes(self) -> tuple[str, ...]:
        """Return source suffix surfaces selected by the rule."""
        return tuple(self.token[start:end] for start, end in self.stripped_suffix_spans)


@dataclass(frozen=True)
class SandhiSplitResult:
    """Offset-preserving split of a fused or already-separated suffix."""

    token: str
    text: str
    language: str
    confidence: float
    parts: tuple[str, ...]
    part_spans: tuple[tuple[int, int], ...]
    offset_map: MorphologyOffsetMap
    rule: str | None = None
    applied: bool = False
    already_split: bool = False


def grapheme_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return tailored extended-grapheme spans for the supported Indic scripts.

    The implementation keeps combining marks, virama-linked consonants, and
    optional joiners in one cluster.  This covers the boundary safety needed by
    the Devanagari and Dravidian rules without adding a runtime dependency.

    Args:
        text: Source Unicode text.

    Returns:
        Ordered ``[start, end)`` code-point spans for each grapheme.
    """
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        start = index
        index += 1
        links_next_letter = False
        while index < len(text):
            char = text[index]
            category = unicodedata.category(char)
            if category in {"Mc", "Me", "Mn"}:
                links_next_letter = links_next_letter or char in _VIRAMAS
                index += 1
                continue
            if char in _JOINERS:
                index += 1
                continue
            if links_next_letter and category.startswith("L"):
                links_next_letter = False
                index += 1
                continue
            break
        spans.append((start, index))
    return tuple(spans)


def grapheme_boundaries(text: str) -> tuple[int, ...]:
    """Return all safe code-point boundaries for ``text``."""
    spans = grapheme_spans(text)
    return (0, *(end for _, end in spans))


def stem_token(
    token: str,
    language: str,
    *,
    confidence: float,
    allowed_stems: Iterable[str],
    minimum_confidence: float = 0.9,
    minimum_stem_graphemes: int = 2,
) -> StemResult:
    """Strip one allow-listed Indic inflectional suffix conservatively.

    Both the confidence and exact stem allow-list gates are mandatory.  The
    returned stem is always an exact prefix of ``token`` and its offset map
    therefore reconstructs directly from the source.

    Args:
        token: One potentially inflected name surface.
        language: Supported ISO language code or language name.
        confidence: Upstream entity confidence in the inclusive range 0..1.
        allowed_stems: Caller-approved name stems.  An empty collection is a
            fail-closed no-op.
        minimum_confidence: Minimum upstream confidence required to strip.
        minimum_stem_graphemes: Minimum number of graphemes retained.

    Returns:
        A :class:`StemResult` with explicit source offsets.
    """
    _validate_token(token)
    normalized_language = _normalize_language(language)
    _validate_thresholds(confidence, minimum_confidence, minimum_stem_graphemes)
    unchanged = _unchanged_stem(token, normalized_language, confidence)
    if not token or any(char.isspace() for char in token):
        return unchanged
    match = _joined_suffix_match(
        token,
        normalized_language,
        confidence=confidence,
        minimum_confidence=minimum_confidence,
        allowed_stems=allowed_stems,
        minimum_stem_graphemes=minimum_stem_graphemes,
    )
    if match is None:
        return unchanged

    stem, rule, boundary = match
    origins = tuple((index, index + 1) for index in range(boundary))
    return StemResult(
        token=token,
        stem=stem,
        language=normalized_language,
        confidence=confidence,
        stem_span=(0, boundary),
        stripped_suffix_spans=((boundary, len(token)),),
        offset_map=MorphologyOffsetMap(
            text=stem,
            source=token,
            char_origins=origins,
        ),
        rule=rule.surface,
        applied=True,
    )


def split_sandhi(
    token: str,
    language: str,
    *,
    confidence: float,
    allowed_stems: Iterable[str],
    minimum_confidence: float = 0.9,
    minimum_stem_graphemes: int = 2,
) -> SandhiSplitResult:
    """Split a frequent fused case marker while preserving source offsets.

    A fused input is rendered with one inserted ASCII space.  Re-applying the
    function to that rendered text detects the existing split and leaves it
    byte-for-byte unchanged.

    Args:
        token: Fused or already-separated surface.
        language: Supported ISO language code or language name.
        confidence: Upstream entity confidence in the inclusive range 0..1.
        allowed_stems: Caller-approved name stems.
        minimum_confidence: Minimum confidence required to split.
        minimum_stem_graphemes: Minimum number of graphemes in the stem.

    Returns:
        A :class:`SandhiSplitResult` with part spans and an output offset map.
    """
    _validate_token(token)
    normalized_language = _normalize_language(language)
    _validate_thresholds(confidence, minimum_confidence, minimum_stem_graphemes)
    allowed = _normalized_allowlist(allowed_stems)

    separated = _already_split_match(
        token,
        normalized_language,
        confidence=confidence,
        minimum_confidence=minimum_confidence,
        allowed_stems=allowed,
        minimum_stem_graphemes=minimum_stem_graphemes,
    )
    if separated is not None:
        stem, rule, stem_end, suffix_start = separated
        return SandhiSplitResult(
            token=token,
            text=token,
            language=normalized_language,
            confidence=confidence,
            parts=(stem, rule.separated),
            part_spans=((0, stem_end), (suffix_start, len(token))),
            offset_map=_identity_offset_map(token),
            rule=rule.surface,
            already_split=True,
        )

    joined = _joined_suffix_match(
        token,
        normalized_language,
        confidence=confidence,
        minimum_confidence=minimum_confidence,
        allowed_stems=allowed,
        minimum_stem_graphemes=minimum_stem_graphemes,
    )
    if joined is None:
        return SandhiSplitResult(
            token=token,
            text=token,
            language=normalized_language,
            confidence=confidence,
            parts=(token,),
            part_spans=((0, len(token)),),
            offset_map=_identity_offset_map(token),
        )

    stem, rule, boundary = joined
    rendered = f"{stem} {rule.separated}"
    origins = (
        *((index, index + 1) for index in range(boundary)),
        (boundary, boundary),
        *(
            (boundary + source_start, boundary + source_end)
            for source_start, source_end in rule.separated_char_origins
        ),
    )
    return SandhiSplitResult(
        token=token,
        text=rendered,
        language=normalized_language,
        confidence=confidence,
        parts=(stem, rule.separated),
        part_spans=((0, boundary), (boundary, len(token))),
        offset_map=MorphologyOffsetMap(
            text=rendered,
            source=token,
            char_origins=tuple(origins),
        ),
        rule=rule.surface,
        applied=True,
    )


def _normalize_language(language: str) -> str:
    if not isinstance(language, str):
        raise TypeError("language must be text")
    normalized = _LANGUAGE_ALIASES.get(language.strip().casefold())
    if normalized is None:
        supported = ", ".join(SUPPORTED_INDIC_MORPHOLOGY_LANGUAGES)
        raise ValueError(
            f"unsupported morphology language; expected one of: {supported}"
        )
    return normalized


def _validate_thresholds(
    confidence: float,
    minimum_confidence: float,
    minimum_stem_graphemes: int,
) -> None:
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, Real)
        or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError("confidence must be between 0 and 1")
    if (
        isinstance(minimum_confidence, bool)
        or not isinstance(minimum_confidence, Real)
        or not 0.0 <= minimum_confidence <= 1.0
    ):
        raise ValueError("minimum_confidence must be between 0 and 1")
    if (
        isinstance(minimum_stem_graphemes, bool)
        or not isinstance(minimum_stem_graphemes, int)
        or minimum_stem_graphemes < 1
    ):
        raise ValueError("minimum_stem_graphemes must be at least 1")


def _normalized_allowlist(allowed_stems: Iterable[str]) -> frozenset[str]:
    if isinstance(allowed_stems, str):
        raise TypeError("allowed_stems must be an iterable of strings")
    normalized: set[str] = set()
    for stem in allowed_stems:
        if not isinstance(stem, str):
            raise TypeError("allowed_stems must contain only strings")
        if stem:
            normalized.add(unicodedata.normalize("NFC", stem).casefold())
    return frozenset(normalized)


def _validate_token(token: str) -> None:
    if not isinstance(token, str):
        raise TypeError("token must be text")


def _allowed(stem: str, allowed_stems: frozenset[str]) -> bool:
    return unicodedata.normalize("NFC", stem).casefold() in allowed_stems


def _joined_suffix_match(
    token: str,
    language: str,
    *,
    confidence: float,
    minimum_confidence: float,
    allowed_stems: Iterable[str],
    minimum_stem_graphemes: int,
) -> tuple[str, _SuffixRule, int] | None:
    if confidence < minimum_confidence:
        return None
    allowed = (
        allowed_stems
        if isinstance(allowed_stems, frozenset)
        else _normalized_allowlist(allowed_stems)
    )
    if not allowed:
        return None
    boundaries = frozenset(grapheme_boundaries(token))
    for rule in _SUFFIX_RULES[language]:
        if not token.endswith(rule.surface) or len(token) <= len(rule.surface):
            continue
        boundary = len(token) - len(rule.surface)
        stem = token[:boundary]
        if boundary not in boundaries:
            continue
        if len(grapheme_spans(stem)) < minimum_stem_graphemes:
            continue
        if _allowed(stem, allowed):
            return stem, rule, boundary
    return None


def _already_split_match(
    token: str,
    language: str,
    *,
    confidence: float,
    minimum_confidence: float,
    allowed_stems: frozenset[str],
    minimum_stem_graphemes: int,
) -> tuple[str, _SuffixRule, int, int] | None:
    if confidence < minimum_confidence or not allowed_stems:
        return None
    for rule in _SUFFIX_RULES[language]:
        if not token.endswith(rule.separated):
            continue
        suffix_start = len(token) - len(rule.separated)
        stem_end = suffix_start
        while stem_end > 0 and token[stem_end - 1].isspace():
            stem_end -= 1
        if stem_end == suffix_start:
            continue
        stem = token[:stem_end]
        if len(grapheme_spans(stem)) >= minimum_stem_graphemes and _allowed(
            stem, allowed_stems
        ):
            return stem, rule, stem_end, suffix_start
    return None


def _identity_offset_map(text: str) -> MorphologyOffsetMap:
    return MorphologyOffsetMap(
        text=text,
        source=text,
        char_origins=tuple((index, index + 1) for index in range(len(text))),
    )


def _unchanged_stem(token: str, language: str, confidence: float) -> StemResult:
    return StemResult(
        token=token,
        stem=token,
        language=language,
        confidence=confidence,
        stem_span=(0, len(token)),
        stripped_suffix_spans=(),
        offset_map=_identity_offset_map(token),
    )


__all__ = [
    "INDIC_SUFFIX_TABLES",
    "SUPPORTED_INDIC_MORPHOLOGY_LANGUAGES",
    "MorphologyOffsetMap",
    "SandhiSplitResult",
    "StemResult",
    "grapheme_boundaries",
    "grapheme_spans",
    "split_sandhi",
    "stem_token",
]
