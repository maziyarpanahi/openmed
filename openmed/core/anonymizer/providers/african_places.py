"""Local-convention address and town surrogates for African language packs.

Faker ships person names but no address provider for the ``sw``, ``ha_NG``,
``ig_NG`` and ``yo_NG`` locales, and OpenMed backs the conceptual ``rw_RW``
locale with ``en_US``. The generic ``STREET_ADDRESS`` and ``LOCATION`` path
therefore produced US templates such as a numbered "Squares" street with an
"Apt." unit, or an invented "-furt" town. The generators below follow how
addresses are written locally instead:

- Swahili: ``Mtaa wa`` and ``Barabara ya`` streets with ``Na.`` house numbers,
  ``Kitalu Na.`` plots, ``S.L.P.`` post-office boxes and landmark directions.
- Hausa: ``Titin`` and ``Layin`` streets with ``Lamba`` house numbers,
  ``Unguwar`` and ``Tudun`` areas, ``P.M.B.`` and ``Akwatin Gidan Waya`` boxes
  and landmark directions.
- Yoruba: ``Òpópónà`` and ``Ọ̀nà`` streets with ``Ilé`` house numbers,
  ``Òkè-`` and ``Ìsàlẹ̀-`` areas, ``P.M.B.`` boxes and landmark directions.
- Igbo: ``Okporo ụzọ`` streets with ``Nọmba`` house numbers, ``Ama-``,
  ``Umu-`` and market-day areas, ``P.M.B.`` boxes and landmark directions.
- Kinyarwanda: the ``Umudugudu`` / ``Akagari`` / ``Umurenge`` village, cell and
  sector hierarchy, ``Inzu No.`` house numbers, ``B.P.`` boxes and landmark
  directions.

A source span that is a post-office box keeps a box shape; any other span
receives a street, plot, area or landmark address. Every street, area and town
name in the surrogate shares no distinctive token with the source, and every
number differs from the numbers written in the source.

Provenance: every value in this module is repository-authored synthetic
surrogate material. Street and area names combine public generic vocabulary
(tree, plant, landform and settlement words) with common local place-name
patterns. They were not copied from a gazetteer or address register and are
not people's names; any match with a real locality is incidental. House, plot
and box numbers are drawn at random, so no street is paired with a real
number and no value is a real address.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from ... import labels as L


@dataclass(frozen=True)
class AddressConvention:
    """Synthetic address vocabulary and templates for one language pack.

    Attributes:
        language: OpenMed ISO 639-1 language code.
        street_names: Street-name roots used by ``{street}`` slots.
        areas: Area, ward or village names used by ``{area}``, ``{cell}`` and
            ``{sector}`` slots. Each template draws distinct values.
        towns: Town names used by ``{town}`` slots and ``LOCATION`` surrogates.
        street_templates: Street, plot, area and landmark address shapes.
        box_templates: Post-office box address shapes.
    """

    language: str
    street_names: tuple[str, ...]
    areas: tuple[str, ...]
    towns: tuple[str, ...]
    street_templates: tuple[str, ...]
    box_templates: tuple[str, ...]


SWAHILI_ADDRESSES: Final = AddressConvention(
    language="sw",
    street_names=(
        "Mvule",
        "Mkuyu",
        "Mpingo",
        "Mninga",
        "Mwembe",
        "Mbuyu",
        "Mkoko",
        "Mchikichi",
    ),
    areas=(
        "Bondeni",
        "Kisimani",
        "Mlimani",
        "Mtoni",
        "Mabatini",
        "Mnarani",
        "Majengo Mapya",
        "Kichangani",
    ),
    towns=(
        "Mperani",
        "Mchungwani",
        "Mnazini",
        "Mtendeni",
        "Mlimauni",
        "Mpapaini",
        "Mkoroshoni",
        "Mzambarauni",
    ),
    street_templates=(
        "Mtaa wa {street}, Nyumba Na. {number}, {area}",
        "Barabara ya {street} Na. {number}, {town}",
        "Kitalu Na. {number}, {area}, {town}",
        "Karibu na soko kuu, {area}, {town}",
        "Nyuma ya shule ya msingi {area}, {town}",
        "Jirani na kituo cha afya {area}, {town}",
    ),
    box_templates=(
        "S.L.P. {box}, {town}",
        "Sanduku la Posta {box}, {town}",
    ),
)
"""Swahili streets, plots, ``S.L.P.`` boxes and landmark directions."""

HAUSA_ADDRESSES: Final = AddressConvention(
    language="ha",
    street_names=(
        "Dorawa",
        "Tsamiya",
        "Kuka",
        "Rimi",
        "Marke",
        "Gamji",
        "Kadanya",
        "Zogale",
    ),
    areas=(
        "Unguwar Rafi",
        "Unguwar Fadama",
        "Tudun Yashi",
        "Tudun Dutse",
        "Kofar Fada",
        "Sabon Layi",
        "Unguwar Rijiya",
        "Bakin Kogi",
    ),
    towns=(
        "Garin Gona",
        "Garin Tafki",
        "Kauyen Dawa",
        "Garin Gero",
        "Sabon Fili",
        "Rafin Yamma",
        "Dutsen Gabas",
        "Kauyen Rogo",
    ),
    street_templates=(
        "Lamba {number}, Titin {street}, {area}",
        "Gida mai lamba {number}, Layin {street}, {town}",
        "{number} Titin {street}, {town}",
        "Kusa da kasuwa, {area}, {town}",
        "Bayan masallacin Juma'a, {area}, {town}",
        "Gaban asibiti, {area}, {town}",
    ),
    box_templates=(
        "P.M.B. {box}, {town}",
        "Akwatin Gidan Waya {box}, {town}",
    ),
)
"""Hausa streets, wards, Nigerian ``P.M.B.`` boxes and landmark directions."""

YORUBA_ADDRESSES: Final = AddressConvention(
    language="yo",
    street_names=(
        "Ìrókò",
        "Àràbà",
        "Ọ̀pẹ",
        "Àgbọn",
        "Obì",
        "Ọ̀gẹ̀dẹ̀",
        "Ìbẹ́pẹ",
        "Ọsàn",
    ),
    areas=(
        "Òkè-Òkúta",
        "Ìsàlẹ̀-Odò",
        "Ìdí-Ìgbó",
        "Òkè-Àfọ̀n",
        "Ìsàlẹ̀-Ọ̀gbà",
        "Agbo-Ilé Tuntun",
        "Ìdí-Òrombó",
        "Òkè-Ẹ̀lú",
    ),
    towns=(
        "Ìlú-Odòdó",
        "Òkè-Ìràwọ̀",
        "Ìgbó-Òṣùpá",
        "Ilé-Òjò",
        "Ìlú-Afẹ́fẹ́",
        "Òkè-Ìrì",
    ),
    street_templates=(
        "Ilé {number}, Òpópónà {street}, {area}",
        "{number} Ọ̀nà {street}, {town}",
        "Nítòsí ọjà, {area}, {town}",
        "Lẹ́yìn ilé-ìwé alákọ̀ọ́bẹ̀rẹ̀, {area}, {town}",
        "Ní iwájú ilé-ìwòsàn, {area}, {town}",
    ),
    box_templates=(
        "P.M.B. {box}, {town}",
        "Àpótí Ìfìwéránṣẹ́ {box}, {town}",
    ),
)
"""Yoruba streets, house numbers, Nigerian ``P.M.B.`` boxes and landmarks."""

IGBO_ADDRESSES: Final = AddressConvention(
    language="ig",
    street_names=(
        "Ọjị",
        "Ụkwa",
        "Akị",
        "Ube",
        "Ugba",
        "Nkwụ",
        "Ọka",
        "Oroma",
    ),
    areas=(
        "Amaọhịa",
        "Amaagụ",
        "Umuenyi",
        "Eke Mmiri",
        "Orie Ugbo",
        "Nkwo Ọhịa",
        "Afọ Ụzụ",
        "Umuezi",
    ),
    towns=(
        "Umuọnwa",
        "Amaosisi",
        "Amanzu",
        "Umuakpụ",
        "Amaokpa",
        "Umuegbe",
    ),
    street_templates=(
        "Nọmba {number}, Okporo ụzọ {street}, {area}",
        "{number} Okporo Ụzọ {street}, {town}",
        "N'akụkụ ahịa, {area}, {town}",
        "N'azụ ụlọ akwụkwọ, {area}, {town}",
        "N'ihu ụlọ ọgwụ, {area}, {town}",
    ),
    box_templates=(
        "P.M.B. {box}, {town}",
        "P.O. Box {box}, {town}",
    ),
)
"""Igbo streets, compound and market-day areas, Nigerian boxes and landmarks."""

KINYARWANDA_ADDRESSES: Final = AddressConvention(
    language="rw",
    street_names=(),
    areas=(
        "Nyamivumu",
        "Rwamabuye",
        "Nyakiyaga",
        "Rusozi",
        "Nyamashyamba",
        "Rwamazi",
        "Nyarubingo",
        "Gatare",
    ),
    towns=(
        "Rwinyange",
        "Nyakigera",
        "Gisovumu",
        "Rukiyaga",
        "Kanyabuye",
        "Nyamirambi",
    ),
    street_templates=(
        "Umudugudu wa {area}, Akagari ka {cell}, Umurenge wa {sector}",
        "Inzu No. {number}, Umudugudu wa {area}, {town}",
        "Hafi y'isoko, Umudugudu wa {area}, {town}",
        "Inyuma y'ishuri ribanza, Akagari ka {cell}, {town}",
        "Imbere y'ikigo nderabuzima, Umurenge wa {sector}, {town}",
    ),
    box_templates=(
        "B.P. {box} {town}",
        "Agasanduku k'iposita {box}, {town}",
    ),
)
"""Kinyarwanda village, cell and sector hierarchy, ``B.P.`` boxes and landmarks."""

ADDRESS_CONVENTIONS: Final[Mapping[str, AddressConvention]] = {
    convention.language: convention
    for convention in (
        SWAHILI_ADDRESSES,
        HAUSA_ADDRESSES,
        YORUBA_ADDRESSES,
        IGBO_ADDRESSES,
        KINYARWANDA_ADDRESSES,
    )
}
"""Curated address conventions keyed by OpenMed language code."""

AFRICAN_ADDRESS_LOCALES: Final[Mapping[str, str]] = {
    "sw": "sw",
    "sw_TZ": "sw",
    "ha_NG": "ha",
    "ig_NG": "ig",
    "yo_NG": "yo",
    "rw_RW": "rw",
}
"""Resolved anonymizer locale -> language whose address convention it uses.

Every other locale, including ``en_KE``, ``en_NG``, ``zu_ZA`` and ``xh_ZA``,
keeps the previously registered address and location generators.
"""

AFRICAN_ADDRESS_APPROXIMATIONS: Final[Mapping[str, str]] = {
    "sw": (
        "curated synthetic Swahili streets, plots, S.L.P. boxes, landmarks and "
        "towns; Faker sw ships no address data"
    ),
    "ha": (
        "curated synthetic Hausa streets, wards, P.M.B. boxes, landmarks and "
        "towns; Faker ha_NG ships no address data"
    ),
    "ig": (
        "curated synthetic Igbo streets, compound and market-day areas, P.M.B. "
        "boxes, landmarks and towns; Faker ig_NG ships no address data"
    ),
    "yo": (
        "curated synthetic Yoruba streets, areas, P.M.B. boxes, landmarks and "
        "towns; Faker yo_NG ships no address data"
    ),
    "rw": (
        "curated synthetic Kinyarwanda village, cell and sector names, B.P. "
        "boxes, landmarks and towns instead of the en_US backend"
    ),
    "xh": (
        "isiZulu street words and South African towns from the zu_ZA backend; "
        "no curated isiXhosa address provider"
    ),
}
"""Documented address-surrogate sources for African language packs.

Reported as ``address_approximation`` by
:func:`openmed.core.anonymizer.locales.locale_coherence_report`. Languages
absent from this mapping use their native Faker address data or are outside
the African pack set.
"""

_HOUSE_NUMBER_RANGE: Final = (1, 199)
_BOX_NUMBER_RANGE: Final = (1, 9999)
_MAX_ATTEMPTS: Final = 16
_AREA_SLOTS: Final = 3  # {area}, {cell} and {sector}

# Structural settlement words that recur across many curated names. They may
# repeat between source and surrogate; the name-specific tokens may not.
_GENERIC_PLACE_TOKENS: Final = frozenset(
    {
        # Hausa ward, hill, gate, riverbank, town and direction words.
        "unguwar",
        "tudun",
        "kofar",
        "sabon",
        "bakin",
        "garin",
        "kauyen",
        "gabas",
        "yamma",
        "arewa",
        "kudu",
        # Yoruba hill, lower, under, house, town and compound words.
        "oke",
        "isale",
        "idi",
        "ile",
        "ilu",
        "agbo",
        # Igbo market-day names.
        "eke",
        "nkwo",
        "orie",
        "afo",
    }
)

_POSTAL_BOX_RE: Final = re.compile(
    r"\bp\.?\s*o\.?\s*box\b"
    r"|\bbox\s*\d"
    r"|\bs\.\s*l\.\s*p\b"
    r"|\bslp\s*\d"
    r"|\bp\.\s*m\.\s*b\b"
    r"|\bpmb\s*\d"
    r"|\bb\.\s*p\.?\s*\d"
    r"|\bbp\s*\d"
    r"|\b(?:agasanduku|sanduku|akwatin|apoti)\b"
)

_FOLDED_ADDRESS_LOCALES: Final[Mapping[str, str]] = {
    locale.casefold(): language for locale, language in AFRICAN_ADDRESS_LOCALES.items()
}


def _fold(value: str) -> str:
    """Return ``value`` without combining marks, case-folded."""

    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).casefold()


def _compact(value: str) -> str:
    """Return the folded alphanumeric characters of ``value``."""

    return "".join(character for character in _fold(value) if character.isalnum())


def _distinctive_tokens(value: str) -> frozenset[str]:
    """Return name-specific letter tokens of a curated value."""

    return frozenset(
        token
        for token in re.findall(r"[^\W\d_]+", _fold(value))
        if len(token) >= 3 and token not in _GENERIC_PLACE_TOKENS
    )


def _source_numbers(original: str) -> frozenset[int]:
    """Return the numeric values written in ``original``."""

    return frozenset(int(run) for run in re.findall(r"\d+", original))


def _is_postal_box(original: str) -> bool:
    """Return whether ``original`` is written as a post-office box."""

    return _POSTAL_BOX_RE.search(_fold(original)) is not None


def _eligible(pool: Sequence[str], compact_source: str) -> tuple[str, ...]:
    """Return pool values sharing no distinctive token with the source."""

    return tuple(
        value
        for value in pool
        if not any(token in compact_source for token in _distinctive_tokens(value))
    )


def _draw_one(faker, pool: Sequence[str], compact_source: str) -> str:
    """Draw one pool value that shares no distinctive token with the source."""

    if not pool:
        return ""
    return faker.random.choice(_eligible(pool, compact_source) or tuple(pool))


def _draw_distinct(
    faker,
    pool: Sequence[str],
    compact_source: str,
    count: int,
) -> list[str]:
    """Draw ``count`` distinct pool values, preferring source-disjoint ones."""

    eligible = _eligible(pool, compact_source)
    return faker.random.sample(eligible if len(eligible) >= count else pool, count)


def _draw_number(faker, excluded: frozenset[int], bounds: tuple[int, int]) -> int:
    """Draw a number within ``bounds`` that is not written in the source."""

    low, high = bounds
    for _ in range(32):
        number = faker.random.randint(low, high)
        if number not in excluded:
            return number
    for number in range(low, high + 1):
        if number not in excluded:
            return number
    raise RuntimeError("no surrogate number is available outside the source")


def _retains_source(original: str, candidate: str) -> bool:
    """Return whether ``candidate`` reproduces the normalized source surface."""

    folded_source = " ".join(_fold(original).split())
    if not folded_source:
        return False
    if folded_source == " ".join(_fold(candidate).split()):
        return True
    compact_source = _compact(original)
    return len(compact_source) >= 4 and compact_source in _compact(candidate)


def african_address_convention(locale: str) -> AddressConvention | None:
    """Return the curated address convention for an anonymizer locale.

    Args:
        locale: Resolved anonymizer locale such as ``ha_NG`` or ``sw``.
            Hyphenated and differently cased spellings are accepted.

    Returns:
        The matching :class:`AddressConvention`, or ``None`` when ``locale`` is
        not an African address locale.
    """

    normalized = str(locale or "").replace("-", "_").casefold()
    language = _FOLDED_ADDRESS_LOCALES.get(normalized)
    return ADDRESS_CONVENTIONS.get(language) if language is not None else None


def is_african_address_locale(locale: str) -> bool:
    """Return whether ``locale`` uses a curated African address convention."""

    return african_address_convention(locale) is not None


def _require_convention(locale: str) -> AddressConvention:
    convention = african_address_convention(locale)
    if convention is None:
        raise ValueError(f"no African address convention for locale {locale!r}")
    return convention


def generate_african_street_address(faker, original: str, *, locale: str) -> str:
    """Return a local-convention street address distinct from ``original``.

    Args:
        faker: Seeded Faker instance; only ``faker.random`` is used.
        original: Source address span.
        locale: Resolved anonymizer locale listed in
            :data:`AFRICAN_ADDRESS_LOCALES`.

    Returns:
        A post-office box when ``original`` is written as one, otherwise a
        street, plot, area or landmark address.

    Raises:
        ValueError: If ``locale`` has no curated African address convention.
        RuntimeError: If no candidate avoids reproducing the source.
    """

    convention = _require_convention(locale)
    templates = (
        convention.box_templates
        if _is_postal_box(original)
        else convention.street_templates
    )
    compact_source = _compact(original)
    excluded_numbers = _source_numbers(original)
    for _ in range(_MAX_ATTEMPTS):
        template = faker.random.choice(templates)
        street = _draw_one(faker, convention.street_names, compact_source)
        area, cell, sector = _draw_distinct(
            faker,
            convention.areas,
            compact_source,
            _AREA_SLOTS,
        )
        town = _draw_one(faker, convention.towns, compact_source)
        number = _draw_number(faker, excluded_numbers, _HOUSE_NUMBER_RANGE)
        box = _draw_number(faker, excluded_numbers, _BOX_NUMBER_RANGE)
        candidate = template.format(
            street=street,
            area=area,
            cell=cell,
            sector=sector,
            town=town,
            number=number,
            box=box,
        )
        if not _retains_source(original, candidate):
            return candidate
    raise RuntimeError("could not generate an address surrogate distinct from source")


def generate_african_location(faker, original: str, *, locale: str) -> str:
    """Return a curated local-pattern town name distinct from ``original``.

    Args:
        faker: Seeded Faker instance; only ``faker.random`` is used.
        original: Source location span.
        locale: Resolved anonymizer locale listed in
            :data:`AFRICAN_ADDRESS_LOCALES`.

    Returns:
        A town name sharing no distinctive token with ``original``.

    Raises:
        ValueError: If ``locale`` has no curated African address convention.
        RuntimeError: If no candidate avoids reproducing the source.
    """

    convention = _require_convention(locale)
    compact_source = _compact(original)
    for _ in range(_MAX_ATTEMPTS):
        town = _draw_one(faker, convention.towns, compact_source)
        if not _retains_source(original, town):
            return town
    raise RuntimeError("could not generate a location surrogate distinct from source")


AFRICAN_PLACE_GENERATORS: Final[Mapping[str, Callable[..., str]]] = {
    L.STREET_ADDRESS: generate_african_street_address,
    L.LOCATION: generate_african_location,
}
"""Curated generators per canonical label for :data:`AFRICAN_ADDRESS_LOCALES`."""


@dataclass(frozen=True)
class AfricanPlaceDispatcher:
    """Route African address locales to curated generators.

    Every other locale reaches ``fallback`` unchanged, with the same Faker
    state, so non-African surrogates stay byte-for-byte identical.

    Attributes:
        generator: Curated generator for :data:`AFRICAN_ADDRESS_LOCALES`.
        fallback: Generator previously registered for the label.
    """

    generator: Callable[..., str]
    fallback: Callable[..., str]

    def __call__(self, faker, original: str, *, locale: str) -> str:
        """Return a surrogate from the curated or fallback generator."""

        if is_african_address_locale(locale):
            return self.generator(faker, original, locale=locale)
        return self.fallback(faker, original, locale=locale)


__all__ = [
    "ADDRESS_CONVENTIONS",
    "AFRICAN_ADDRESS_APPROXIMATIONS",
    "AFRICAN_ADDRESS_LOCALES",
    "AFRICAN_PLACE_GENERATORS",
    "AddressConvention",
    "AfricanPlaceDispatcher",
    "HAUSA_ADDRESSES",
    "IGBO_ADDRESSES",
    "KINYARWANDA_ADDRESSES",
    "SWAHILI_ADDRESSES",
    "YORUBA_ADDRESSES",
    "african_address_convention",
    "generate_african_location",
    "generate_african_street_address",
    "is_african_address_locale",
]
