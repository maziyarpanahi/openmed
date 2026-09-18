"""Synthetic script-correct name providers.

The generators use Unicode letter inventories rather than real-name lists, so
no source or demographic dictionary is bundled. Every output code point is
selected outside the source surface's code-point set. Han output also preserves
the number of Han characters in the detected name. Gujarati uses Faker's
native ``gu_IN`` name provider and preserves the language-specific ``ભાઈ`` /
``બેન`` suffix when the source contains one.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Final

from ...language_pack import LanguagePack, get_language_pack


def _require_language_pack(code: str) -> LanguagePack:
    """Return a catalog-backed language pack or fail during registration."""

    language_pack = get_language_pack(code)
    if language_pack is None:  # pragma: no cover - catalog import is mandatory
        raise RuntimeError(f"language pack {code!r} is not registered")
    return language_pack


HAN_LANGUAGE_PACK: Final = _require_language_pack("zh")
DEVANAGARI_LANGUAGE_PACK: Final = _require_language_pack("hi")
TELUGU_LANGUAGE_PACK: Final = _require_language_pack("te")
GUJARATI_LANGUAGE_PACK: Final = _require_language_pack("gu")
KANNADA_LANGUAGE_PACK: Final = _require_language_pack("kn")


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
_GUJARATI_LETTERS: Final[Sequence[str]] = _unicode_letters(
    0x0A85,
    0x0AB9,
    "GUJARATI LETTER",
)
_GUJARATI_SUFFIXES: Final = ("ભાઈ", "બેન")
_GUJARATI_HONORIFICS: Final = ("શ્રીમતી", "શ્રી")
_KANNADA_INITIAL_LED_RE = re.compile(
    r"^(?P<prefix>(?:(?:[A-Za-z]+|[\u0C80-\u0CFF]+)\.[ \t]*){1,2})"
    r"(?P<given>.+)$"
)
_KANNADA_INITIAL_RE = re.compile(r"(?P<token>[A-Za-z]+|[\u0C80-\u0CFF]+)\.")
_KANNADA_SCRIPT_RE = re.compile(r"[\u0C80-\u0CFF]")


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


def generate_gujarati_name(faker, original: str, *, locale: str) -> str:
    """Return a native Gujarati surrogate while preserving a gender suffix.

    ``ભાઈ`` is the masculine and ``બેન`` the feminine fused suffix used in
    Gujarati records.  The suffix carries useful grammatical information but
    is not the person's identifying stem, so only that suffix is retained.
    The base name is drawn from Faker's installed ``gu_IN`` provider.
    """

    source = str(original).strip()
    honorific = ""
    stem = source
    for candidate in _GUJARATI_HONORIFICS:
        prefix = f"{candidate} "
        if source.startswith(prefix):
            honorific = prefix
            stem = source[len(prefix) :]
            break

    suffix = next(
        (candidate for candidate in _GUJARATI_SUFFIXES if stem.endswith(candidate)),
        "",
    )
    source_stem = stem[: -len(suffix)] if suffix else stem

    method = "name" if " " in source_stem else "first_name"
    if suffix:
        gender = "male" if suffix == "ભાઈ" else "female"
        method = f"{method}_{gender}"

    for _ in range(20):
        candidate = str(getattr(faker, method)())
        candidate = candidate.strip()
        for candidate_honorific in _GUJARATI_HONORIFICS:
            candidate_prefix = f"{candidate_honorific} "
            if candidate.startswith(candidate_prefix):
                candidate = candidate[len(candidate_prefix) :]
                break
        for candidate_suffix in _GUJARATI_SUFFIXES:
            if candidate.endswith(candidate_suffix):
                candidate = candidate[: -len(candidate_suffix)]
                break
        if candidate and candidate != source_stem:
            result = f"{honorific}{candidate}{suffix}"
            if result != source:
                return result

    fallback = _draw_disjoint_name(
        faker,
        source_stem,
        alphabet=_GUJARATI_LETTERS,
        length=2,
    )
    return f"{honorific}{fallback}{suffix}"


def _kannada_values(key: str) -> tuple[str, ...]:
    """Return the small synthetic Kannada provider vocabulary for ``key``."""

    from ...pii_i18n import LOCALE_FAKE_DATA

    return tuple(LOCALE_FAKE_DATA["kn_IN"][key])


def _draw_kannada_value(faker, key: str, original: str) -> str:
    values = _kannada_values(key)
    alternatives = tuple(value for value in values if value != original)
    return str(faker.random_element(alternatives or values))


def _generate_kannada_initials(faker, prefix: str) -> str:
    rendered: list[str] = []
    for match in _KANNADA_INITIAL_RE.finditer(prefix):
        token = match.group("token")
        key = "INITIAL_KANNADA" if _KANNADA_SCRIPT_RE.search(token) else "INITIAL_LATIN"
        rendered.append(_draw_kannada_value(faker, key, token) + ".")
        following = prefix[match.end() :]
        separator = re.match(r"[ \t]*", following)
        rendered.append(separator.group(0) if separator else "")
    return "".join(rendered)


def generate_kannada_name(faker, original: str, *, locale: str) -> str:
    """Return a Kannada surrogate preserving place/father initial structure.

    Kannada PERSON spans commonly contain one or two initials before the given
    name. The bundled synthetic provider keeps that count, dot punctuation,
    whitespace, and the script of each initial while drawing the given name
    from a small local vocabulary. The ``ಅವರು`` suffix is outside the span and
    therefore never reaches this generator.
    """

    source = original.strip()
    match = _KANNADA_INITIAL_LED_RE.fullmatch(source)
    if match is not None:
        given = match.group("given").strip()
        given_key = (
            "FIRST_NAME" if _KANNADA_SCRIPT_RE.search(given) else "FIRST_NAME_LATIN"
        )
        replacement = _draw_kannada_value(faker, given_key, given)
        return _generate_kannada_initials(faker, match.group("prefix")) + replacement

    key = "NAME" if " " in source else "FIRST_NAME"
    if not _KANNADA_SCRIPT_RE.search(source):
        key = "NAME_LATIN" if key == "NAME" else "FIRST_NAME_LATIN"
    return _draw_kannada_value(faker, key, source)


SCRIPT_NAME_PACKS: Final = (
    (HAN_LANGUAGE_PACK, "Han", generate_han_name),
    (DEVANAGARI_LANGUAGE_PACK, "Devanagari", generate_devanagari_name),
    (TELUGU_LANGUAGE_PACK, "Telugu", generate_telugu_name),
    (GUJARATI_LANGUAGE_PACK, "Gujarati", generate_gujarati_name),
    (KANNADA_LANGUAGE_PACK, "Kannada", generate_kannada_name),
)


__all__ = [
    "DEVANAGARI_LANGUAGE_PACK",
    "GUJARATI_LANGUAGE_PACK",
    "HAN_LANGUAGE_PACK",
    "KANNADA_LANGUAGE_PACK",
    "SCRIPT_NAME_PACKS",
    "TELUGU_LANGUAGE_PACK",
    "generate_devanagari_name",
    "generate_gujarati_name",
    "generate_han_name",
    "generate_kannada_name",
    "generate_telugu_name",
]
