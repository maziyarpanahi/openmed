"""Synthetic script-correct name providers.

The generators use Unicode letter inventories rather than real-name lists, so
no source or demographic dictionary is bundled. Every output code point is
selected outside the source surface's code-point set. Han output also preserves
the number of Han characters in the detected name.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Final

from ...language_pack import LanguagePack, get_language_pack
from ..locales import generate_urdu_name


def _require_language_pack(code: str) -> LanguagePack:
    """Return a catalog-backed language pack or fail during registration."""

    language_pack = get_language_pack(code)
    if language_pack is None:  # pragma: no cover - catalog import is mandatory
        raise RuntimeError(f"language pack {code!r} is not registered")
    return language_pack


HAN_LANGUAGE_PACK: Final = _require_language_pack("zh")
DEVANAGARI_LANGUAGE_PACK: Final = _require_language_pack("hi")
TELUGU_LANGUAGE_PACK: Final = _require_language_pack("te")
URDU_LANGUAGE_PACK: Final = _require_language_pack("ur")


def _unicode_letters(start: int, end: int, name_prefix: str) -> tuple[str, ...]:
    return tuple(
        character
        for codepoint in range(start, end + 1)
        if (character := chr(codepoint)).isalpha()
        and unicodedata.category(character).startswith("L")
        and unicodedata.name(character, "").startswith(name_prefix)
    )


_HAN_LETTERS: Final[Sequence[str]] = tuple(
    chr(codepoint) for codepoint in range(0x4E00, 0x9FA6)
)
_HAN_RANGES: Final = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2EBEF),
    (0x30000, 0x323AF),
)
_DEVANAGARI_RANGES: Final = (
    (0x0900, 0x097F),
    (0xA8E0, 0xA8FF),
    (0x11B00, 0x11B5F),
)
_TELUGU_RANGES: Final = ((0x0C00, 0x0C7F),)
_DEVANAGARI_LETTERS: Final[Sequence[str]] = _unicode_letters(
    0x0904,
    0x0939,
    "DEVANAGARI LETTER",
)
_TELUGU_LETTERS: Final[Sequence[str]] = _unicode_letters(
    0x0C05,
    0x0C39,
    "TELUGU LETTER",
)


def _script_length(
    original: str,
    *,
    ranges: tuple[tuple[int, int], ...],
) -> int:
    return sum(
        any(start <= ord(character) <= end for start, end in ranges)
        for character in original
    )


def _draw_disjoint_name(
    faker,
    original: str,
    *,
    alphabet: Sequence[str],
    length: int,
) -> str:
    source_codepoints = set(original)
    available = tuple(
        character for character in alphabet if character not in source_codepoints
    )
    if not available:  # pragma: no cover - real names cannot exhaust a script block
        raise ValueError("source surface exhausts the synthetic script inventory")
    return "".join(faker.random.choice(available) for _ in range(max(1, length)))


def generate_han_name(faker, original: str, *, locale: str) -> str:
    """Return a synthetic Han name with the source's Han-character count."""

    return _draw_disjoint_name(
        faker,
        original,
        alphabet=_HAN_LETTERS,
        length=_script_length(original, ranges=_HAN_RANGES),
    )


def generate_devanagari_name(faker, original: str, *, locale: str) -> str:
    """Return a synthetic Devanagari name disjoint from the source surface."""

    return _draw_disjoint_name(
        faker,
        original,
        alphabet=_DEVANAGARI_LETTERS,
        length=_script_length(original, ranges=_DEVANAGARI_RANGES),
    )


def generate_telugu_name(faker, original: str, *, locale: str) -> str:
    """Return a synthetic Telugu name disjoint from the source surface."""

    return _draw_disjoint_name(
        faker,
        original,
        alphabet=_TELUGU_LETTERS,
        length=_script_length(original, ranges=_TELUGU_RANGES),
    )


SCRIPT_NAME_PACKS: Final = (
    (HAN_LANGUAGE_PACK, "Han", generate_han_name),
    (DEVANAGARI_LANGUAGE_PACK, "Devanagari", generate_devanagari_name),
    (TELUGU_LANGUAGE_PACK, "Telugu", generate_telugu_name),
    (URDU_LANGUAGE_PACK, "Arabic", generate_urdu_name),
)


__all__ = [
    "DEVANAGARI_LANGUAGE_PACK",
    "HAN_LANGUAGE_PACK",
    "SCRIPT_NAME_PACKS",
    "TELUGU_LANGUAGE_PACK",
    "URDU_LANGUAGE_PACK",
    "generate_devanagari_name",
    "generate_han_name",
    "generate_telugu_name",
    "generate_urdu_name",
]
