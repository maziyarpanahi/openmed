"""Unicode script detection helpers for mixed-script PII routing.

The helpers in this module are intentionally lightweight and stdlib-only. They
use explicit Unicode block ranges plus :mod:`unicodedata` character categories
to identify dominant scripts and preserve exact offsets while segmenting text
into script-oriented runs.

The curated confusable mappings are derived from Unicode UTS #39
``confusables.txt`` version 17.0.0. Unicode data files are distributed under
the Unicode License v3 (SPDX: ``Unicode-3.0``). The full data file is not
embedded: only mappings needed for the supported Latin/Cyrillic/Greek/CJK PII
evasion defense are retained.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass

from .language_pack_catalog import SCRIPT_LANGUAGE_HINTS

UNKNOWN_SCRIPT = "Unknown"

CJK_SCRIPTS = frozenset({"Han", "Hiragana/Katakana", "Hangul"})
INDIC_SCRIPTS = frozenset(
    {
        "Devanagari",
        "Bengali",
        "Gurmukhi",
        "Gujarati",
        "Odia",
        "Tamil",
        "Telugu",
        "Kannada",
        "Malayalam",
    }
)
CONFUSABLE_DATA_VERSION = "17.0.0"
CONFUSABLE_DATA_URL = "https://www.unicode.org/Public/17.0.0/security/confusables.txt"
CONFUSABLE_DATA_LICENSE = "Unicode-3.0"

SUPPORTED_SCRIPTS = (
    "Latin",
    "Arabic",
    "Ethiopic",
    "Han",
    "Hiragana/Katakana",
    "Hangul",
    "Cyrillic",
    "Devanagari",
    "Bengali",
    "Gurmukhi",
    "Gujarati",
    "Odia",
    "Tamil",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Greek",
    "Hebrew",
    "Thai",
)

ZERO_WIDTH_CHARS = frozenset(
    {
        "\u200b",  # zero width space
        "\u200c",  # zero width non-joiner
        "\u200d",  # zero width joiner
        "\u2060",  # word joiner
        "\ufeff",  # zero width no-break space
    }
)

_CONFUSABLE_FOLD: dict[str, str] = {
    "\u0391": "A",
    "\u0392": "B",
    "\u0395": "E",
    "\u0397": "H",
    "\u0399": "I",
    "\u039a": "K",
    "\u039c": "M",
    "\u039d": "N",
    "\u039f": "O",
    "\u03a1": "P",
    "\u03a4": "T",
    "\u03a7": "X",
    "\u03b1": "a",
    "\u03b5": "e",
    "\u03b7": "n",
    "\u03b9": "i",
    "\u03ba": "k",
    "\u03bc": "u",
    "\u03bf": "o",
    "\u03c1": "p",
    "\u03c4": "t",
    "\u03c5": "u",
    "\u03c7": "x",
    "\u0410": "A",
    "\u0412": "B",
    "\u0415": "E",
    "\u041a": "K",
    "\u041c": "M",
    "\u041d": "H",
    "\u041e": "O",
    "\u0420": "P",
    "\u0421": "C",
    "\u0422": "T",
    "\u0425": "X",
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0445": "x",
    "\u0456": "i",
    "\u3007": "O",
}


@dataclass(frozen=True)
class MixedScriptSpan:
    """An identifier-like source span containing more than one script."""

    start: int
    end: int
    scripts: tuple[str, ...]
    confusable_count: int = 0
    invisible_count: int = 0

    def to_metadata(self) -> dict[str, object]:
        """Return a raw-text-free representation suitable for audit metadata."""
        return {
            "confusable_count": self.confusable_count,
            "end": self.end,
            "invisible_count": self.invisible_count,
            "scripts": list(self.scripts),
            "start": self.start,
        }


@dataclass(frozen=True)
class DetectionNormalization:
    """Offset-preserving Unicode normalization for PII detection."""

    text: str
    original_length: int
    offset_starts: tuple[int, ...]
    offset_ends: tuple[int, ...]
    removed_zero_width: int = 0
    stripped_combining_marks: int = 0
    folded_confusables: int = 0
    folded_native_digits: int = 0
    indic_changes: int = 0
    indic_scripts: tuple[str, ...] = ()
    scripts: tuple[str, ...] = ()
    mixed_script: bool = False

    @property
    def changed(self) -> bool:
        """Return whether the normalized text differs structurally."""
        return (
            self.removed_zero_width > 0
            or self.stripped_combining_marks > 0
            or self.folded_confusables > 0
            or self.folded_native_digits > 0
            or self.indic_changes > 0
        )

    def remap_span(self, start: int, end: int) -> tuple[int, int]:
        """Map normalized-text offsets back to original-text offsets."""
        safe_start = max(0, min(int(start), len(self.text)))
        safe_end = max(safe_start, min(int(end), len(self.text)))
        if not self.offset_starts:
            return 0, 0
        if safe_start >= len(self.offset_starts):
            original_start = self.original_length
        else:
            original_start = self.offset_starts[safe_start]
        if safe_end <= 0:
            original_end = original_start
        elif safe_end - 1 >= len(self.offset_ends):
            original_end = self.original_length
        else:
            original_end = self.offset_ends[safe_end - 1]
        return original_start, max(original_start, original_end)

    def to_metadata(self) -> dict[str, object]:
        """Return PHI-free normalization metadata."""
        return {
            "changed": self.changed,
            "folded_confusables": self.folded_confusables,
            "folded_native_digits": self.folded_native_digits,
            "indic_changes": self.indic_changes,
            "indic_scripts": list(self.indic_scripts),
            "mixed_script": self.mixed_script,
            "removed_zero_width": self.removed_zero_width,
            "scripts": list(self.scripts),
            "stripped_combining_marks": self.stripped_combining_marks,
        }


_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    (
        "Latin",
        (
            (0x0041, 0x005A),
            (0x0061, 0x007A),
            (0x00C0, 0x00FF),
            (0x0100, 0x017F),
            (0x0180, 0x024F),
            (0x1E00, 0x1EFF),
            (0x2C60, 0x2C7F),
            (0xA720, 0xA7FF),
            (0xAB30, 0xAB6F),
            (0xFF21, 0xFF3A),
            (0xFF41, 0xFF5A),
        ),
    ),
    (
        "Arabic",
        (
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ),
    ),
    (
        "Ethiopic",
        (
            (0x1200, 0x137F),
            (0x1380, 0x139F),
            (0x2D80, 0x2DDF),
            (0xAB00, 0xAB2F),
            (0x1E7E0, 0x1E7FF),
        ),
    ),
    (
        "Han",
        (
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F),
            (0x2B820, 0x2CEAF),
            (0x2CEB0, 0x2EBEF),
            (0x30000, 0x3134F),
            (0x31350, 0x323AF),
        ),
    ),
    (
        "Hiragana/Katakana",
        (
            (0x3040, 0x309F),
            (0x30A0, 0x30FF),
            (0x31F0, 0x31FF),
            (0x1B000, 0x1B16F),
            (0xFF65, 0xFF9F),
        ),
    ),
    (
        "Hangul",
        (
            (0x1100, 0x11FF),
            (0x3130, 0x318F),
            (0xA960, 0xA97F),
            (0xAC00, 0xD7AF),
            (0xD7B0, 0xD7FF),
        ),
    ),
    (
        "Cyrillic",
        (
            (0x0400, 0x04FF),
            (0x0500, 0x052F),
            (0x1C80, 0x1C8F),
            (0x2DE0, 0x2DFF),
            (0xA640, 0xA69F),
        ),
    ),
    (
        "Devanagari",
        (
            (0x0900, 0x097F),
            (0xA8E0, 0xA8FF),
            (0x11B00, 0x11B5F),
        ),
    ),
    ("Bengali", ((0x0980, 0x09FF),)),
    ("Gurmukhi", ((0x0A00, 0x0A7F),)),
    ("Gujarati", ((0x0A80, 0x0AFF),)),
    ("Odia", ((0x0B00, 0x0B7F),)),
    ("Tamil", ((0x0B80, 0x0BFF),)),
    ("Telugu", ((0x0C00, 0x0C7F),)),
    ("Kannada", ((0x0C80, 0x0CFF),)),
    ("Malayalam", ((0x0D00, 0x0D7F),)),
    (
        "Greek",
        (
            (0x0370, 0x03FF),
            (0x1F00, 0x1FFF),
        ),
    ),
    (
        "Hebrew",
        (
            (0x0590, 0x05FF),
            (0xFB1D, 0xFB4F),
        ),
    ),
    ("Thai", ((0x0E00, 0x0E7F),)),
)


def detect_script(text: str) -> str:
    """Return the dominant Unicode script in ``text``.

    Neutral characters such as whitespace, punctuation, symbols, and digits do
    not affect the decision. If no supported script-bearing code point is
    present, ``"Unknown"`` is returned.
    """

    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}

    for index, char in enumerate(text):
        script = _script_for_char(char)
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
        first_seen.setdefault(script, index)

    if not counts:
        return UNKNOWN_SCRIPT

    return max(counts, key=lambda script: (counts[script], -first_seen[script]))


def is_han_dominant(text: str) -> bool:
    """Return whether Han is the dominant supported script in ``text``."""

    return detect_script(text) == "Han"


def segment_by_script(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield contiguous ``(start, end, script)`` runs covering ``text``.

    Neutral characters are assigned to the surrounding run: leading neutral
    characters attach to the first detected script, and later neutral characters
    attach to the preceding script. This keeps offsets exact while avoiding
    stand-alone whitespace or punctuation runs.
    """

    if not text:
        return

    run_start = 0
    current_script: str | None = None

    for index, char in enumerate(text):
        script = _script_for_char(char)
        if script is None:
            continue
        if current_script is None:
            current_script = script
            continue
        if script == current_script:
            continue

        yield run_start, index, current_script
        run_start = index
        current_script = script

    if current_script is None:
        yield 0, len(text), UNKNOWN_SCRIPT
        return

    yield run_start, len(text), current_script


def candidate_languages_for_script(script: str) -> tuple[str, ...]:
    """Return candidate language codes for a detected script."""

    return SCRIPT_LANGUAGE_HINTS.get(script, SCRIPT_LANGUAGE_HINTS[UNKNOWN_SCRIPT])


def confusable_skeleton(text: str) -> str:
    """Return a curated UTS #39-style skeleton for PII matching.

    The mapper is deliberately narrower than general-purpose Unicode
    normalization: it folds the supported cross-script lookalikes and ASCII
    full-width forms, and removes the invisible controls used by the evasion
    generator. It does not case-fold or strip diacritics.
    """

    output: list[str] = []
    for char in text:
        if char in ZERO_WIDTH_CHARS:
            continue
        output.append(_fold_confusable_char(char))
    return "".join(output)


def mixed_script_spans(text: str) -> tuple[MixedScriptSpan, ...]:
    """Return identifier-like spans that mix Unicode scripts.

    Script changes separated by whitespace or ordinary punctuation are normal
    multilingual prose and are not flagged. Invisible controls stay attached
    to the surrounding token so an inserted zero-width character cannot split
    an otherwise suspicious identifier.
    """

    findings: list[MixedScriptSpan] = []
    token_start: int | None = None
    for index in range(len(text) + 1):
        char = text[index] if index < len(text) else ""
        if char and _is_identifier_char(char):
            if token_start is None:
                token_start = index
            continue
        if token_start is None:
            continue

        token = text[token_start:index]
        scripts = tuple(sorted(_script_counts(token)))
        if len(scripts) > 1:
            findings.append(
                MixedScriptSpan(
                    start=token_start,
                    end=index,
                    scripts=scripts,
                    confusable_count=sum(
                        _fold_confusable_char(item) != item for item in token
                    ),
                    invisible_count=sum(item in ZERO_WIDTH_CHARS for item in token),
                )
            )
        token_start = None
    return tuple(findings)


def detect_mixed_script(text: str) -> bool:
    """Return whether an identifier-like span mixes Unicode scripts."""

    return bool(mixed_script_spans(text))


def normalize_for_pii_detection(
    text: str,
    *,
    width_convention: str = "cjk",
) -> DetectionNormalization:
    """Fold adversarial Unicode artifacts while preserving offset remapping.

    Indic script runs first receive script-specific NFC canonicalization. The
    defense then strips zero-width controls and standalone non-Indic combining
    marks, while retaining Ethiopic marks attached to a preceding Ethiopic
    grapheme. It folds common Latin-lookalike Greek/Cyrillic/full-width
    characters and Indic decimal digits, and records a script-consistency
    summary without storing source text. ``width_convention`` selects the
    CJK-safe width fold or strict per-character NFKC normalization.
    """

    # Local imports keep the lightweight script helpers from importing the
    # broader processing package during module initialization.
    from ..processing.text import INDIC_SCRIPTS, IndicNormalizer, fold_indic_digits
    from ..processing.zh_normalize import normalize_width

    scripts = tuple(sorted(_script_counts(text)))
    mixed_script = detect_mixed_script(text)
    indic_normalizer = IndicNormalizer()
    routed_chars: list[str] = []
    routed_starts: list[int] = []
    routed_ends: list[int] = []
    indic_changes = 0
    indic_scripts: list[str] = []
    removed_zero_width = 0

    for run_start, run_end, script in segment_by_script(text):
        run = text[run_start:run_end]
        if script in INDIC_SCRIPTS:
            normalized = indic_normalizer.normalize_with_offsets(run, script=script)
            routed_chars.extend(normalized.text)
            routed_starts.extend(
                run_start + offset for offset in normalized.offset_starts
            )
            routed_ends.extend(run_start + offset for offset in normalized.offset_ends)
            indic_changes += normalized.changes
            removed_zero_width += normalized.removed_joiners
            if script not in indic_scripts:
                indic_scripts.append(script)
            continue

        routed_chars.extend(run)
        routed_starts.extend(range(run_start, run_end))
        routed_ends.extend(range(run_start + 1, run_end + 1))

    routed_text = "".join(routed_chars)
    width_normalization = normalize_width(
        routed_text,
        convention=width_convention,
    )
    digit_folding = fold_indic_digits(width_normalization.text)
    output: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    stripped_combining_marks = 0
    normalized_by_routed_source: list[list[str]] = [[] for _ in routed_text]
    for char, (routed_start, _routed_end) in zip(
        width_normalization.text,
        width_normalization.char_origins,
    ):
        normalized_by_routed_source[routed_start].append(char)
    changed_source_indices = {
        routed_starts[index]
        for index, (char, normalized_chars) in enumerate(
            zip(routed_text, normalized_by_routed_source)
        )
        if "".join(normalized_chars) != char
    }
    folded_native_digit_sources = {
        routed_starts[width_normalization.char_origins[index][0]]
        for index, (width_char, folded_char) in enumerate(
            zip(width_normalization.text, digit_folding.text)
        )
        if width_char != folded_char
    }

    for index, char in enumerate(digit_folding.text):
        routed_start, routed_end = width_normalization.char_origins[index]
        original_start = routed_starts[routed_start]
        original_end = routed_ends[routed_end - 1]
        if char in ZERO_WIDTH_CHARS:
            removed_zero_width += 1
            continue
        category = unicodedata.category(char)
        attached_ethiopic_mark = (
            category == "Mn"
            and _script_for_char(char) == "Ethiopic"
            and original_start > 0
            and _script_for_char(text[original_start - 1]) == "Ethiopic"
        )
        if (
            category == "Mn"
            and _script_for_char(char) not in INDIC_SCRIPTS
            and not attached_ethiopic_mark
        ):
            stripped_combining_marks += 1
            continue

        replacement = _fold_confusable_char(char)
        if replacement != char:
            changed_source_indices.add(original_start)
        for replacement_char in replacement:
            output.append(replacement_char)
            starts.append(original_start)
            ends.append(original_end)

    return DetectionNormalization(
        text="".join(output),
        original_length=len(text),
        offset_starts=tuple(starts),
        offset_ends=tuple(ends),
        removed_zero_width=removed_zero_width,
        stripped_combining_marks=stripped_combining_marks,
        folded_confusables=len(changed_source_indices),
        folded_native_digits=len(folded_native_digit_sources),
        indic_changes=indic_changes,
        indic_scripts=tuple(indic_scripts),
        scripts=scripts,
        mixed_script=mixed_script,
    )


def _script_for_char(char: str) -> str | None:
    codepoint = ord(char)
    if codepoint == 0x3007:
        return "Han"
    # Python 3.10's Unicode 13 database predates Ethiopic Extended-B. Route the
    # explicit Unicode block independently of ``unicodedata.category`` so the
    # same text is detected consistently across supported Python versions.
    if 0x1E7E0 <= codepoint <= 0x1E7FF:
        return "Ethiopic"

    category = unicodedata.category(char)
    if category[0] not in {"L", "M"}:
        return None

    for script, ranges in _SCRIPT_RANGES:
        if any(start <= codepoint <= end for start, end in ranges):
            return script
    return None


def _script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in text:
        script = _script_for_char(char)
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
    return counts


def _fold_confusable_char(char: str) -> str:
    folded = _CONFUSABLE_FOLD.get(char)
    if folded is not None:
        return folded

    codepoint = ord(char)
    if 0xFF01 <= codepoint <= 0xFF5E:
        return chr(codepoint - 0xFEE0)

    return char


def _is_identifier_char(char: str) -> bool:
    if char in ZERO_WIDTH_CHARS:
        return True
    return unicodedata.category(char)[0] in {"L", "M", "N"}


__all__ = [
    "CJK_SCRIPTS",
    "CONFUSABLE_DATA_LICENSE",
    "CONFUSABLE_DATA_URL",
    "CONFUSABLE_DATA_VERSION",
    "DetectionNormalization",
    "INDIC_SCRIPTS",
    "MixedScriptSpan",
    "SCRIPT_LANGUAGE_HINTS",
    "SUPPORTED_SCRIPTS",
    "UNKNOWN_SCRIPT",
    "ZERO_WIDTH_CHARS",
    "candidate_languages_for_script",
    "confusable_skeleton",
    "detect_mixed_script",
    "detect_script",
    "is_han_dominant",
    "mixed_script_spans",
    "normalize_for_pii_detection",
    "segment_by_script",
]
