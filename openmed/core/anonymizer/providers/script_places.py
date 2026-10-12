"""Synthetic script-correct place providers for Arabic and Amharic.

Faker's default ``ar_EG`` locale ships no Arabic-script city or street data,
and the ``en_KE`` backend behind ``am_ET`` ships no Ethiopic data, so the
generic Faker path would insert Latin-script places into Arabic or Ethiopic
text. These providers keep ``LOCATION`` and ``STREET_ADDRESS`` surrogates in the
source script.

Arabic places use the locale's own Faker data when it is entirely
Arabic-script, which keeps regional overrides such as ``ar-DZ`` in-region.
Otherwise they use curated Egyptian templates. Ethiopic places recombine the
curated values in ``LANGUAGE_FAKE_DATA["am"]``.

Provenance: every curated value in this module is synthetic surrogate material
chosen by hand. The values are public city names and generic street nouns. None
comes from a patient record, and none is a person's name.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Final

from ... import labels as L
from ...language_pack import LanguagePack
from .script_names import (
    _ARABIC_SCRIPT_RANGES,
    _NATIVE_DRAW_ATTEMPTS,
    AMHARIC_LANGUAGE_PACK,
    ARABIC_LANGUAGE_PACK,
    _draw_unshared_value,
    _fold_surface,
    _is_script_text,
    _surface_tokens,
)

EGYPTIAN_CITIES: Final = (
    "القاهرة",
    "الإسكندرية",
    "الجيزة",
    "المنصورة",
    "طنطا",
    "أسيوط",
    "الأقصر",
    "أسوان",
    "بورسعيد",
    "الإسماعيلية",
    "الزقازيق",
    "دمياط",
)
"""Synthetic curated Egyptian city surrogates (public place names)."""

EGYPTIAN_STREET_NAMES: Final = (
    "النيل",
    "التحرير",
    "الجمهورية",
    "السلام",
    "الحرية",
    "النصر",
    "الجامعة",
    "المحطة",
    "الزهور",
    "البحر",
)
"""Synthetic curated street names built from generic nouns, not people."""

EGYPTIAN_STREET_TEMPLATES: Final = (
    "{building} شارع {street}",
    "شارع {street} {building}",
)
"""Egyptian street-level address shapes: building number and street name."""

_ARABIC_GENERIC_PLACE_WORDS: Final = frozenset(
    {"شارع", "طريق", "حي", "نهج", "زنقة", "دوار", "ميدان"}
)
_AMHARIC_KEBELE: Final = "ቀበሌ"
_ARABIC_BUILDING_RANGE: Final = (1, 199)
_AMHARIC_KEBELE_RANGE: Final = (1, 30)


def _source_numbers(original: str) -> frozenset[int]:
    """Return the numeric values written in ``original`` in any digit script."""

    return frozenset(int(run) for run in re.findall(r"\d+", original))


def _draw_number(faker, original: str, *, bounds: tuple[int, int]) -> int:
    """Draw a number in ``bounds`` that does not appear in ``original``."""

    excluded = _source_numbers(original)
    low, high = bounds
    candidates = [number for number in range(low, high + 1) if number not in excluded]
    return faker.random.choice(candidates)


def _distinctive_tokens(value: str) -> frozenset[str]:
    """Return place tokens other than numbers and generic street words."""

    return frozenset(
        token
        for token in _surface_tokens(value)
        if not token.isdigit() and token not in _ARABIC_GENERIC_PLACE_WORDS
    )


def _native_arabic_place(faker, original: str, *, method: str) -> str | None:
    """Return the locale's own Arabic-script place when it shares no source token."""

    source_tokens = _distinctive_tokens(original)
    source_numbers = _source_numbers(original)
    folded_original = _fold_surface(original)
    for _ in range(_NATIVE_DRAW_ATTEMPTS):
        candidate = " ".join(str(getattr(faker, method)()).split())
        if not _is_script_text(candidate, _ARABIC_SCRIPT_RANGES):
            # This Faker locale has no Arabic-script places; stop asking it.
            return None
        if (
            _fold_surface(candidate) != folded_original
            and _distinctive_tokens(candidate).isdisjoint(source_tokens)
            and _source_numbers(candidate).isdisjoint(source_numbers)
        ):
            return candidate
    return None


def generate_arabic_location(faker, original: str, *, locale: str) -> str:
    """Return an Arabic-script city sharing no token with ``original``."""

    del locale
    native = _native_arabic_place(faker, original, method="city")
    if native is not None:
        return native
    return _draw_unshared_value(faker, EGYPTIAN_CITIES, original)


def generate_arabic_street_address(faker, original: str, *, locale: str) -> str:
    """Return an Arabic-script street address distinct from ``original``.

    The curated Egyptian template never reuses the source street name or a
    number written in the source.
    """

    del locale
    native = _native_arabic_place(faker, original, method="street_address")
    if native is not None:
        return native
    street = _draw_unshared_value(faker, EGYPTIAN_STREET_NAMES, original)
    building = _draw_number(faker, original, bounds=_ARABIC_BUILDING_RANGE)
    template = faker.random.choice(EGYPTIAN_STREET_TEMPLATES)
    return template.format(building=building, street=street)


def _amharic_locations() -> tuple[str, ...]:
    """Return the synthetic curated Ethiopic place values."""

    from ...pii_i18n import LANGUAGE_FAKE_DATA

    return tuple(LANGUAGE_FAKE_DATA["am"]["LOCATION"])


def generate_ethiopic_location(faker, original: str, *, locale: str) -> str:
    """Return a curated Ethiopic city sharing no token with ``original``."""

    del locale
    return _draw_unshared_value(faker, _amharic_locations(), original)


def generate_ethiopic_street_address(faker, original: str, *, locale: str) -> str:
    """Return an Ethiopic city, kebele, and number distinct from ``original``.

    The shape follows the curated ``city kebele number`` address values. The
    city and number never repeat the source city or a number in the source.
    """

    city = generate_ethiopic_location(faker, original, locale=locale)
    kebele = _draw_number(faker, original, bounds=_AMHARIC_KEBELE_RANGE)
    return f"{city} {_AMHARIC_KEBELE} {kebele}"


SCRIPT_PLACE_PACKS: Final[
    tuple[tuple[LanguagePack, str, Mapping[str, Callable[..., str]]], ...]
] = (
    (
        ARABIC_LANGUAGE_PACK,
        "Arabic",
        {
            L.LOCATION: generate_arabic_location,
            L.STREET_ADDRESS: generate_arabic_street_address,
        },
    ),
    (
        AMHARIC_LANGUAGE_PACK,
        "Ethiopic",
        {
            L.LOCATION: generate_ethiopic_location,
            L.STREET_ADDRESS: generate_ethiopic_street_address,
        },
    ),
)
"""Per-label script place generators, keyed by language pack and script."""


__all__ = [
    "EGYPTIAN_CITIES",
    "EGYPTIAN_STREET_NAMES",
    "EGYPTIAN_STREET_TEMPLATES",
    "SCRIPT_PLACE_PACKS",
    "generate_arabic_location",
    "generate_arabic_street_address",
    "generate_ethiopic_location",
    "generate_ethiopic_street_address",
]
