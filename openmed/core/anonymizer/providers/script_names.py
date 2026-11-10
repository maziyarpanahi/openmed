"""Synthetic script-correct name providers.

The generators use Unicode letter inventories rather than real-name lists, so
no source or demographic dictionary is bundled. Every output code point is
selected outside the source surface's code-point set. Han output also preserves
the number of Han characters in the detected name. Gujarati uses Faker's
native ``gu_IN`` name provider and preserves the language-specific ``ભાઈ`` /
``બેન`` suffix when the source contains one.

Arabic-script names come from Faker's generic ``ar_AA`` person inventory,
because the default ``ar_EG`` locale ships no Arabic-script person data. A
regional override whose Faker locale does ship Arabic-script names, such as
``ar-DZ``, keeps its own regional names. Ethiopic names are recombined from the
synthetic curated values in ``LANGUAGE_FAKE_DATA["am"]``, because the ``en_KE``
backend behind ``am_ET`` ships no Ethiopic data. Neither path bundles a list of
real people, and no output name part reuses a token of the source surface.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Final

from ... import labels as L
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
PUNJABI_LANGUAGE_PACK: Final = _require_language_pack("pa")
TELUGU_LANGUAGE_PACK: Final = _require_language_pack("te")
GUJARATI_LANGUAGE_PACK: Final = _require_language_pack("gu")
KANNADA_LANGUAGE_PACK: Final = _require_language_pack("kn")
URDU_LANGUAGE_PACK: Final = _require_language_pack("ur")
ARABIC_LANGUAGE_PACK: Final = _require_language_pack("ar")
AMHARIC_LANGUAGE_PACK: Final = _require_language_pack("am")


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


_PUNJABI_MASCULINE_GIVEN_NAMES: Final = (
    "ਅਮਨਦੀਪ",
    "ਗੁਰਕੀਰਤ",
    "ਹਰਜੀਤ",
    "ਮਨਦੀਪ",
    "ਨਵਦੀਪ",
    "ਰਣਜੀਤ",
)
_PUNJABI_FEMININE_GIVEN_NAMES: Final = (
    "ਗੁਰਲੀਨ",
    "ਜਸਲੀਨ",
    "ਨਵਜੋਤ",
    "ਰਵਨੀਤ",
    "ਸਿਮਰਨ",
    "ਹਰਲੀਨ",
)
_PUNJABI_MASCULINE_SUFFIX: Final = "ਸਿੰਘ"
_PUNJABI_FEMININE_SUFFIX: Final = "ਕੌਰ"


def generate_gurmukhi_name(faker, original: str, *, locale: str) -> str:
    """Return a bundled Punjabi name with a gender-matched Sikh suffix.

    Full names ending in ``ਸਿੰਘ`` or ``ਕੌਰ`` retain that gender marker while
    replacing the given name. Name-part labels without a Sikh suffix receive a
    standalone Gurmukhi given name.
    """

    source = original.strip()
    source_tokens = source.split()
    suffix = (
        source_tokens[-1]
        if len(source_tokens) >= 2
        and source_tokens[-1] in {_PUNJABI_MASCULINE_SUFFIX, _PUNJABI_FEMININE_SUFFIX}
        else ""
    )
    if suffix == _PUNJABI_FEMININE_SUFFIX:
        pool = _PUNJABI_FEMININE_GIVEN_NAMES
    elif suffix == _PUNJABI_MASCULINE_SUFFIX:
        pool = _PUNJABI_MASCULINE_GIVEN_NAMES
    else:
        pool = (
            *_PUNJABI_MASCULINE_GIVEN_NAMES,
            *_PUNJABI_FEMININE_GIVEN_NAMES,
        )

    source_folded = source.casefold()
    candidates = tuple(name for name in pool if name.casefold() not in source_folded)
    given_name = str(faker.random.choice(candidates or pool))
    return f"{given_name} {suffix}" if suffix else given_name


_ARABIC_SCRIPT_RANGES: Final = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
"""Unicode blocks that hold Arabic-script letters."""

_ETHIOPIC_SCRIPT_RANGES: Final = (
    (0x1200, 0x137F),
    (0x1380, 0x139F),
    (0x2D80, 0x2DDF),
    (0xAB00, 0xAB2F),
)
"""Unicode blocks that hold Ethiopic syllables."""

_NATIVE_DRAW_ATTEMPTS: Final = 8


def _fold_surface(value: str) -> str:
    """Return ``value`` in NFKC form without combining marks.

    Arabic harakat and Ethiopic combining marks do not change which name or
    place a token denotes, so token comparisons ignore them.
    """

    normalized = unicodedata.normalize("NFKC", str(value))
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("M")
    )


def _surface_tokens(value: str) -> frozenset[str]:
    """Return the mark-folded word tokens of ``value``."""

    return frozenset(re.findall(r"\w+", _fold_surface(value)))


def _is_script_text(value: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    """Return whether ``value`` has letters and every letter is in ``ranges``."""

    letters = [character for character in str(value) if character.isalpha()]
    return bool(letters) and all(
        any(start <= ord(character) <= end for start, end in ranges)
        for character in letters
    )


def _draw_unshared_value(
    faker,
    values: Sequence[str],
    original: str,
    *,
    exclude: frozenset[str] = frozenset(),
) -> str:
    """Draw a value sharing no token with ``original`` or ``exclude``.

    If the curated inventory is exhausted, use synthetic letters in the same
    script rather than returning a source token.
    """

    forbidden = _surface_tokens(original) | exclude
    eligible = tuple(
        value for value in values if _surface_tokens(value).isdisjoint(forbidden)
    )
    if eligible:
        return str(faker.random.choice(eligible))
    if values and _is_script_text(values[0], _ARABIC_SCRIPT_RANGES):
        alphabet = _unicode_letters(0x0621, 0x064A, "ARABIC LETTER")
    else:
        alphabet = _unicode_letters(0x1200, 0x135A, "ETHIOPIC SYLLABLE")
    return _draw_disjoint_name(
        faker,
        " ".join((original, *sorted(forbidden))),
        alphabet=alphabet,
        length=4,
    )


@lru_cache(maxsize=1)
def _arabic_name_inventory() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return Faker's generic Arabic-script given and family names."""

    from faker.providers.person.ar_AA import Provider

    given_names = tuple(
        dict.fromkeys((*Provider.first_names_male, *Provider.first_names_female))
    )
    return given_names, tuple(Provider.last_names)


def _draw_arabic_name_part(
    faker,
    original: str,
    *,
    method: str,
    values: Sequence[str],
    exclude: frozenset[str] = frozenset(),
) -> str:
    """Prefer the locale's own Arabic-script names, else the ``ar_AA`` inventory."""

    forbidden = _surface_tokens(original) | exclude
    for _ in range(_NATIVE_DRAW_ATTEMPTS):
        candidate = str(getattr(faker, method)()).strip()
        if not _is_script_text(candidate, _ARABIC_SCRIPT_RANGES):
            # This Faker locale has no Arabic-script names; stop asking it.
            break
        if _surface_tokens(candidate).isdisjoint(forbidden):
            return candidate
    return _draw_unshared_value(faker, values, original, exclude=exclude)


def generate_arabic_first_name(faker, original: str, *, locale: str) -> str:
    """Return an Arabic-script given name sharing no token with ``original``."""

    del locale
    given_names, _family_names = _arabic_name_inventory()
    return _draw_arabic_name_part(
        faker,
        original,
        method="first_name",
        values=given_names,
    )


def generate_arabic_last_name(faker, original: str, *, locale: str) -> str:
    """Return an Arabic-script family name sharing no token with ``original``."""

    del locale
    _given_names, family_names = _arabic_name_inventory()
    return _draw_arabic_name_part(
        faker,
        original,
        method="last_name",
        values=family_names,
    )


def generate_arabic_person(faker, original: str, *, locale: str) -> str:
    """Return an Arabic-script given and family name distinct from ``original``.

    The locale's own Faker names are used when they are Arabic-script, which
    keeps regional overrides such as ``ar-DZ`` in-region. Otherwise the parts
    come from Faker's generic ``ar_AA`` inventory.
    """

    _given_names, family_names = _arabic_name_inventory()
    given = generate_arabic_first_name(faker, original, locale=locale)
    family = _draw_arabic_name_part(
        faker,
        original,
        method="last_name",
        values=family_names,
        exclude=_surface_tokens(given),
    )
    return f"{given} {family}"


def _amharic_values(key: str) -> tuple[str, ...]:
    """Return the synthetic curated Amharic values for ``key``."""

    from ...pii_i18n import LANGUAGE_FAKE_DATA

    return tuple(LANGUAGE_FAKE_DATA["am"][key])


def generate_ethiopic_first_name(faker, original: str, *, locale: str) -> str:
    """Return a curated Ethiopic given name sharing no token with ``original``."""

    del locale
    return _draw_unshared_value(faker, _amharic_values("FIRST_NAME"), original)


def generate_ethiopic_last_name(faker, original: str, *, locale: str) -> str:
    """Return a curated Ethiopic father's name sharing no source token."""

    del locale
    return _draw_unshared_value(faker, _amharic_values("LAST_NAME"), original)


def generate_ethiopic_person(faker, original: str, *, locale: str) -> str:
    """Return a curated Ethiopic given name plus father's name.

    Ethiopian names pair a given name with the father's given name. Both parts
    are drawn from the synthetic curated Amharic values, never from a list of
    real people, and neither part reuses a token of ``original``.
    """

    given = generate_ethiopic_first_name(faker, original, locale=locale)
    father = _draw_unshared_value(
        faker,
        _amharic_values("LAST_NAME"),
        original,
        exclude=_surface_tokens(given),
    )
    return f"{given} {father}"


SCRIPT_NAME_PACKS: Final = (
    (HAN_LANGUAGE_PACK, "Han", generate_han_name),
    (DEVANAGARI_LANGUAGE_PACK, "Devanagari", generate_devanagari_name),
    (PUNJABI_LANGUAGE_PACK, "Gurmukhi", generate_gurmukhi_name),
    (TELUGU_LANGUAGE_PACK, "Telugu", generate_telugu_name),
    (GUJARATI_LANGUAGE_PACK, "Gujarati", generate_gujarati_name),
    (KANNADA_LANGUAGE_PACK, "Kannada", generate_kannada_name),
    (URDU_LANGUAGE_PACK, "Arabic", generate_urdu_name),
)

SCRIPT_NAME_PART_PACKS: Final[
    tuple[tuple[LanguagePack, str, Mapping[str, Callable[..., str]]], ...]
] = (
    (
        ARABIC_LANGUAGE_PACK,
        "Arabic",
        {
            L.PERSON: generate_arabic_person,
            L.FIRST_NAME: generate_arabic_first_name,
            L.MIDDLE_NAME: generate_arabic_first_name,
            L.LAST_NAME: generate_arabic_last_name,
        },
    ),
    (
        AMHARIC_LANGUAGE_PACK,
        "Ethiopic",
        {
            L.PERSON: generate_ethiopic_person,
            L.FIRST_NAME: generate_ethiopic_first_name,
            L.MIDDLE_NAME: generate_ethiopic_first_name,
            L.LAST_NAME: generate_ethiopic_last_name,
        },
    ),
)
"""Per-label script name generators, keyed by language pack and script."""


__all__ = [
    "AMHARIC_LANGUAGE_PACK",
    "ARABIC_LANGUAGE_PACK",
    "DEVANAGARI_LANGUAGE_PACK",
    "GUJARATI_LANGUAGE_PACK",
    "HAN_LANGUAGE_PACK",
    "KANNADA_LANGUAGE_PACK",
    "PUNJABI_LANGUAGE_PACK",
    "SCRIPT_NAME_PACKS",
    "SCRIPT_NAME_PART_PACKS",
    "TELUGU_LANGUAGE_PACK",
    "URDU_LANGUAGE_PACK",
    "generate_arabic_first_name",
    "generate_arabic_last_name",
    "generate_arabic_person",
    "generate_devanagari_name",
    "generate_ethiopic_first_name",
    "generate_ethiopic_last_name",
    "generate_ethiopic_person",
    "generate_gujarati_name",
    "generate_han_name",
    "generate_kannada_name",
    "generate_gurmukhi_name",
    "generate_telugu_name",
    "generate_urdu_name",
]
