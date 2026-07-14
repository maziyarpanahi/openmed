"""Unicode script detection helpers for mixed-script PII routing.

The helpers in this module are intentionally lightweight and stdlib-only. They
use explicit Unicode block ranges plus :mod:`unicodedata` character categories
to identify dominant scripts and preserve exact offsets while segmenting text
into script-oriented runs.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass

UNKNOWN_SCRIPT = "Unknown"

SUPPORTED_SCRIPTS = (
    "Latin",
    "Arabic",
    "Han",
    "Hiragana/Katakana",
    "Hangul",
    "Cyrillic",
    "Devanagari",
    "Telugu",
    "Greek",
    "Hebrew",
    "Thai",
)

SCRIPT_LANGUAGE_HINTS: dict[str, tuple[str, ...]] = {
    "Latin": ("en", "fr", "de", "it", "es", "nl", "pt", "tr"),
    "Arabic": ("ar",),
    "Han": ("ja",),
    "Hiragana/Katakana": ("ja",),
    "Hangul": ("ko",),
    "Cyrillic": ("en",),
    "Devanagari": ("hi",),
    "Telugu": ("te",),
    "Greek": ("en",),
    "Hebrew": ("en",),
    "Thai": ("en",),
    UNKNOWN_SCRIPT: ("en",),
}

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
    scripts: tuple[str, ...] = ()
    mixed_script: bool = False

    @property
    def changed(self) -> bool:
        """Return whether the normalized text differs structurally."""
        return (
            self.removed_zero_width > 0
            or self.stripped_combining_marks > 0
            or self.folded_confusables > 0
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
    ("Telugu", ((0x0C00, 0x0C7F),)),
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


def normalize_for_pii_detection(
    text: str,
    *,
    width_convention: str = "cjk",
) -> DetectionNormalization:
    """Fold adversarial Unicode artifacts while preserving offset remapping.

    The defense strips zero-width controls and standalone combining marks, folds
    common Latin-lookalike Greek/Cyrillic/full-width characters, and records a
    script-consistency summary without storing source text. ``width_convention``
    selects the CJK-safe width fold or strict per-character NFKC normalization.
    """

    # Keep the reusable width-normalization API in ``processing`` while
    # composing its explicit source map with this existing detection defense.
    # The local import avoids making the lightweight script helpers import the
    # broader processing package during module initialization.
    from ..processing.zh_normalize import normalize_width

    scripts = tuple(sorted(_script_counts(text)))
    width_normalization = normalize_width(text, convention=width_convention)
    output: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    removed_zero_width = 0
    stripped_combining_marks = 0
    normalized_by_source: list[list[str]] = [[] for _ in text]
    for char, (original_start, _original_end) in zip(
        width_normalization.text,
        width_normalization.char_origins,
    ):
        normalized_by_source[original_start].append(char)
    changed_source_indices = {
        index
        for index, (char, normalized_chars) in enumerate(
            zip(text, normalized_by_source)
        )
        if "".join(normalized_chars) != char
    }

    for index, char in enumerate(width_normalization.text):
        original_start, original_end = width_normalization.char_origins[index]
        if char in ZERO_WIDTH_CHARS:
            removed_zero_width += 1
            continue
        if unicodedata.category(char) == "Mn":
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
        scripts=scripts,
        mixed_script=len(scripts) > 1,
    )


def _script_for_char(char: str) -> str | None:
    category = unicodedata.category(char)
    if category[0] not in {"L", "M"}:
        return None

    codepoint = ord(char)
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


__all__ = [
    "DetectionNormalization",
    "SCRIPT_LANGUAGE_HINTS",
    "SUPPORTED_SCRIPTS",
    "UNKNOWN_SCRIPT",
    "ZERO_WIDTH_CHARS",
    "candidate_languages_for_script",
    "detect_script",
    "normalize_for_pii_detection",
    "segment_by_script",
]
