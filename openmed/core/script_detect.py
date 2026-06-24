"""Unicode script detection helpers for mixed-script PII routing.

The helpers in this module are intentionally lightweight and stdlib-only. They
use explicit Unicode block ranges plus :mod:`unicodedata` character categories
to identify dominant scripts and preserve exact offsets while segmenting text
into script-oriented runs.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator

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
    "Hangul": ("en",),
    "Cyrillic": ("en",),
    "Devanagari": ("hi",),
    "Telugu": ("te",),
    "Greek": ("en",),
    "Hebrew": ("en",),
    "Thai": ("en",),
    UNKNOWN_SCRIPT: ("en",),
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


def _script_for_char(char: str) -> str | None:
    category = unicodedata.category(char)
    if category[0] not in {"L", "M"}:
        return None

    codepoint = ord(char)
    for script, ranges in _SCRIPT_RANGES:
        if any(start <= codepoint <= end for start, end in ranges):
            return script
    return None


__all__ = [
    "SCRIPT_LANGUAGE_HINTS",
    "SUPPORTED_SCRIPTS",
    "UNKNOWN_SCRIPT",
    "candidate_languages_for_script",
    "detect_script",
    "segment_by_script",
]
