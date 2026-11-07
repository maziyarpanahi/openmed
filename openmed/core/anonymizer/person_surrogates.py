"""Shape generated PERSON surrogates to the source span's conventions.

The label generators return a full name for a PERSON span, and Faker's name
formats randomly add titles to it (``Bi.``, ``Chị``, ``Univ.Prof.``,
``جناب آقای``, ``MBA.``). :func:`shape_person_surrogate` post-processes that
generated name for non-CJK spans so that:

- it never gains an honorific the source name did not carry,
- family-name-first languages (Vietnamese) keep family-first order, and
- when the source carried a gender cue (see
  :func:`openmed.core.person_honorifics.peel_person_honorifics`), the given
  name agrees with it.

Gendered given names come from the active Faker locale's own
``first_names_female``/``first_names_male`` inventories, Faker's generic
``ar_AA`` inventory for Arabic, and small synthetic Urdu and Amharic lists
below. Replacement names are never drawn from the source span's own tokens,
and nothing here logs or stores the source text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final, Optional

from ..name_order import CJK_LANGUAGES
from ..person_honorifics import (
    Gender,
    NameConvention,
    fold_name_token,
    name_convention_for,
)
from ..script_detect import detect_script
from .locales import URDU_FAMILY_NAMES

__all__ = [
    "AMHARIC_FEMININE_GIVEN_NAMES",
    "AMHARIC_MASCULINE_GIVEN_NAMES",
    "URDU_FEMININE_GIVEN_NAMES",
    "URDU_MASCULINE_GIVEN_NAMES",
    "shape_person_surrogate",
    "strip_inserted_honorifics",
]

_MAX_STRIPPED_TITLES: Final = 3

# Faker's generic ``prefixes`` lists for es_ES, fr_FR and pt_PT hold surname
# particles rather than titles; they are never stripped.
_NAME_PARTICLES: Final = frozenset(
    {"da", "das", "de", "de la", "del", "der", "di", "do", "dos", "du", "la", "le"}
)

URDU_FEMININE_GIVEN_NAMES: Final = ("آمنہ", "زہرہ", "فاطمہ", "مریم", "نادیہ")
"""Feminine entries of :data:`openmed.core.anonymizer.locales.URDU_GIVEN_NAMES`."""

URDU_MASCULINE_GIVEN_NAMES: Final = ("عارف", "حسن", "سلمان")
"""Masculine entries of :data:`openmed.core.anonymizer.locales.URDU_GIVEN_NAMES`."""

AMHARIC_FEMININE_GIVEN_NAMES: Final = (
    "ሜሮን",
    "ሔዋን",
    "ትዕግስት",
    "ማርታ",
    "አስቴር",
    "ብርቱካን",
)
"""Small synthetic-safe inventory of feminine Ethiopic given names."""

AMHARIC_MASCULINE_GIVEN_NAMES: Final = (
    "ዳዊት",
    "ሚካኤል",
    "ዮሐንስ",
    "ብርሃኑ",
    "አበበ",
    "ተስፋዬ",
)
"""Small synthetic-safe inventory of masculine Ethiopic given names."""

# Gendered tokens that generators may place outside the given-name slot, with
# the neutral alternatives used when they contradict the source's gender cue.
_SURROGATE_GENDER_TOKENS: Final[Mapping[str, Mapping[str, Gender]]] = {
    "ur": {"بیگم": "female"},
}
_NEUTRAL_FAMILY_NAMES: Final[Mapping[str, tuple[str, ...]]] = {
    "ur": tuple(name for name in URDU_FAMILY_NAMES if name != "بیگم"),
}

_PROVIDER_PREFIX_ATTRIBUTES: Final = (
    "prefixes_female",
    "prefixes_male",
    "prefixes_nonbinary",
)
_PROVIDER_SUFFIX_ATTRIBUTES: Final = (
    "suffixes",
    "suffixes_female",
    "suffixes_male",
    "suffixes_nonbinary",
)
_PROVIDER_FIRST_NAME_ATTRIBUTES: Final = (
    "first_names",
    "first_names_female",
    "first_names_male",
    "first_names_nonbinary",
    "first_names_unisex",
)
_PROVIDER_LAST_NAME_ATTRIBUTES: Final = (
    "last_names",
    "last_names_female",
    "last_names_male",
)


@dataclass(frozen=True)
class _TitleVocabulary:
    """Folded honorific and name vocabularies for one Faker person provider."""

    prefixes: tuple[tuple[str, ...], ...] = ()
    suffixes: tuple[tuple[str, ...], ...] = ()
    first_names: frozenset[str] = frozenset()
    last_names: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _GivenNameInventory:
    """Gender-disjoint names in one script, as ``(name, folded key)`` pairs."""

    script: str
    female: tuple[tuple[str, str], ...]
    male: tuple[tuple[str, str], ...]
    female_keys: frozenset[str]
    male_keys: frozenset[str]

    def names(self, gender: Gender) -> tuple[tuple[str, str], ...]:
        return self.female if gender == "female" else self.male


def _values(source: Any, attribute: str) -> tuple[str, ...]:
    """Return the string entries of a Faker list or weighted mapping."""

    value = getattr(source, attribute, None)
    if isinstance(value, Mapping):
        value = tuple(value)
    elif isinstance(value, (set, frozenset)):
        # Sets iterate in hash order; sort so seeded draws stay reproducible.
        value = tuple(sorted(str(item) for item in value))
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _faker_person_provider(faker: Any) -> Any:
    """Return the person provider behind ``faker``, or ``None``."""

    lookup = getattr(faker, "provider", None)
    if lookup is None:
        return None
    try:
        return lookup("faker.providers.person")
    except Exception:  # noqa: BLE001 - stubs and custom proxies lack providers
        return None


def _keys(values: Iterable[str], fold_marks: bool) -> frozenset[str]:
    return frozenset(fold_name_token(value, fold_marks=fold_marks) for value in values)


def _title_keys(
    values: Iterable[str],
    fold_marks: bool,
) -> frozenset[tuple[str, ...]]:
    return frozenset(
        tuple(fold_name_token(part, fold_marks=fold_marks) for part in value.split())
        for value in values
        if value.split()
    )


def _longest_first(
    entries: Iterable[tuple[str, ...]],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        sorted(set(entries), key=lambda key: (-len(key), -len("".join(key)), key))
    )


@lru_cache(maxsize=128)
def _provider_vocabulary(provider_cls: type, fold_marks: bool) -> _TitleVocabulary:
    prefixes = {
        value
        for attribute in _PROVIDER_PREFIX_ATTRIBUTES
        for value in _values(provider_cls, attribute)
    }
    prefixes.update(
        value
        for value in _values(provider_cls, "prefixes")
        if value.casefold() not in _NAME_PARTICLES
    )
    # Multi-word suffixes (Thai ``ณ <place>``) are surname particles, not
    # post-nominal titles, so only single-token suffixes are stripped.
    suffixes = {
        value
        for attribute in _PROVIDER_SUFFIX_ATTRIBUTES
        for value in _values(provider_cls, attribute)
        if len(value.split()) == 1
    }
    return _TitleVocabulary(
        prefixes=_longest_first(_title_keys(prefixes, fold_marks)),
        suffixes=_longest_first(_title_keys(suffixes, fold_marks)),
        first_names=_keys(
            (
                value
                for attribute in _PROVIDER_FIRST_NAME_ATTRIBUTES
                for value in _values(provider_cls, attribute)
            ),
            fold_marks,
        ),
        last_names=_keys(
            (
                value
                for attribute in _PROVIDER_LAST_NAME_ATTRIBUTES
                for value in _values(provider_cls, attribute)
            ),
            fold_marks,
        ),
    )


def _title_vocabulary(
    faker: Any,
    convention: Optional[NameConvention],
    fold_marks: bool,
) -> _TitleVocabulary:
    provider = _faker_person_provider(faker)
    base = (
        _provider_vocabulary(type(provider), fold_marks)
        if provider is not None
        else _TitleVocabulary()
    )
    if convention is None:
        return base
    registered = {
        position: _title_keys(
            (
                honorific.text
                for honorific in convention.honorifics
                if honorific.position == position
            ),
            fold_marks,
        )
        for position in ("prefix", "suffix")
    }
    return _TitleVocabulary(
        prefixes=_longest_first((*base.prefixes, *registered["prefix"])),
        suffixes=_longest_first((*base.suffixes, *registered["suffix"])),
        first_names=base.first_names,
        last_names=base.last_names,
    )


def _occurs_in(entry: tuple[str, ...], reference: Sequence[str]) -> bool:
    """Return whether the source already carries ``entry`` (spaced or glued)."""

    width = len(entry)
    if any(
        tuple(reference[start : start + width]) == entry
        for start in range(len(reference) - width + 1)
    ):
        return True
    return width == 1 and any(
        token.startswith(entry[0]) or token.endswith(entry[0]) for token in reference
    )


def strip_inserted_honorifics(
    surrogate: str,
    *,
    reference: str,
    lang: Optional[str],
    faker: Any = None,
) -> str:
    """Remove titles a generator added that ``reference`` does not carry.

    Leading titles and trailing post-nominals are matched as whole tokens
    against the Faker locale's prefix and suffix inventories plus the
    language's registered honorifics. A title glued to a known first name
    (Thai ``นายสมชาย``) or glued suffix on a known surname (Hindi
    ``शर्माजी``) is split off as well. At least one name token always remains,
    and any title that also occurs in ``reference`` is kept.

    Args:
        surrogate: The generated name.
        reference: The source name the surrogate replaces (honorifics already
            peeled when the language has a registered convention).
        lang: ISO 639-1 language code of the span.
        faker: The Faker instance that generated ``surrogate``. Without it only
            the registered honorifics are considered.

    Returns:
        ``surrogate`` without inserted titles; the input object unchanged when
        nothing was removed.
    """

    if not surrogate or not surrogate.strip():
        return surrogate
    convention = name_convention_for(lang)
    fold_marks = bool(convention and convention.fold_marks)
    vocabulary = _title_vocabulary(faker, convention, fold_marks)
    if not vocabulary.prefixes and not vocabulary.suffixes:
        return surrogate

    tokens = surrogate.split()
    keys = [fold_name_token(token, fold_marks=fold_marks) for token in tokens]
    reference_keys = [
        fold_name_token(token, fold_marks=fold_marks) for token in reference.split()
    ]
    changed = False

    for _ in range(_MAX_STRIPPED_TITLES):
        title = next(
            (
                entry
                for entry in vocabulary.prefixes
                if len(entry) < len(tokens)
                and tuple(keys[: len(entry)]) == entry
                and not _occurs_in(entry, reference_keys)
            ),
            None,
        )
        if title is not None:
            del tokens[: len(title)], keys[: len(title)]
            changed = True
            continue
        glued = _glued_title(
            keys[0],
            tokens[0],
            vocabulary.prefixes,
            vocabulary.first_names,
            reference_keys,
            leading=True,
        )
        if glued is None:
            break
        tokens[0], keys[0] = glued
        changed = True

    for _ in range(_MAX_STRIPPED_TITLES):
        title = next(
            (
                entry
                for entry in vocabulary.suffixes
                if len(entry) < len(tokens)
                and tuple(keys[-len(entry) :]) == entry
                and not _occurs_in(entry, reference_keys)
            ),
            None,
        )
        if title is not None:
            del tokens[-len(title) :], keys[-len(title) :]
            # Some locales write post-nominals after a comma ("Name, S.Kom").
            tokens[-1], keys[-1] = tokens[-1].rstrip(","), keys[-1].rstrip(",")
            changed = True
            if not tokens[-1] and len(tokens) > 1:
                del tokens[-1], keys[-1]
            continue
        glued = _glued_title(
            keys[-1],
            tokens[-1],
            vocabulary.suffixes,
            vocabulary.last_names,
            reference_keys,
            leading=False,
        )
        if glued is None:
            break
        tokens[-1], keys[-1] = glued
        changed = True

    return " ".join(tokens) if changed else surrogate


def _glued_title(
    key: str,
    token: str,
    titles: Sequence[tuple[str, ...]],
    known_names: frozenset[str],
    reference_keys: Sequence[str],
    *,
    leading: bool,
) -> Optional[tuple[str, str]]:
    """Split a title glued to a known name, returning ``(token, key)``."""

    if len(key) != len(token) or not known_names:
        return None
    for entry in titles:
        if len(entry) != 1 or len(entry[0]) >= len(key):
            continue
        title = entry[0]
        if leading and key.startswith(title):
            rest_key, rest = key[len(title) :], token[len(title) :]
        elif not leading and key.endswith(title):
            rest_key, rest = key[: -len(title)], token[: -len(title)]
        else:
            continue
        if rest_key in known_names and not _occurs_in(entry, reference_keys):
            return rest, rest_key
    return None


def _inventory(
    female: Iterable[str],
    male: Iterable[str],
    fold_marks: bool,
) -> Optional[_GivenNameInventory]:
    """Build a gender-disjoint, single-token name inventory."""

    def pairs(names: Iterable[str]) -> tuple[tuple[str, str], ...]:
        unique = dict.fromkeys(name for name in names if name and " " not in name)
        return tuple(
            (name, fold_name_token(name, fold_marks=fold_marks)) for name in unique
        )

    female_pairs, male_pairs = pairs(female), pairs(male)
    shared = {key for _, key in female_pairs} & {key for _, key in male_pairs}
    female_pairs = tuple(pair for pair in female_pairs if pair[1] not in shared)
    male_pairs = tuple(pair for pair in male_pairs if pair[1] not in shared)
    if not female_pairs or not male_pairs:
        return None
    sample = [name for name, _ in (*female_pairs[:16], *male_pairs[:16])]
    return _GivenNameInventory(
        script=detect_script(" ".join(sample)),
        female=female_pairs,
        male=male_pairs,
        female_keys=frozenset(key for _, key in female_pairs),
        male_keys=frozenset(key for _, key in male_pairs),
    )


@lru_cache(maxsize=128)
def _provider_inventory(
    provider_cls: type,
    fold_marks: bool,
) -> Optional[_GivenNameInventory]:
    return _inventory(
        _values(provider_cls, "first_names_female"),
        _values(provider_cls, "first_names_male"),
        fold_marks,
    )


@lru_cache(maxsize=128)
def _provider_family_inventory(
    provider_cls: type,
    fold_marks: bool,
) -> Optional[_GivenNameInventory]:
    """Return gendered family names (Indonesian Faker data has them)."""

    return _inventory(
        _values(provider_cls, "last_names_female"),
        _values(provider_cls, "last_names_male"),
        fold_marks,
    )


@lru_cache(maxsize=32)
def _language_inventory(
    lang: str,
    fold_marks: bool,
) -> Optional[_GivenNameInventory]:
    """Return the extra in-script inventory for ``lang``, if one exists."""

    if lang == "ar":
        try:
            from faker.providers.person.ar_AA import Provider
        except ImportError:  # pragma: no cover - ar_AA ships with Faker
            return None
        return _inventory(
            _values(Provider, "first_names_female"),
            _values(Provider, "first_names_male"),
            fold_marks,
        )
    if lang == "ur":
        return _inventory(
            URDU_FEMININE_GIVEN_NAMES, URDU_MASCULINE_GIVEN_NAMES, fold_marks
        )
    if lang == "am":
        return _inventory(
            AMHARIC_FEMININE_GIVEN_NAMES, AMHARIC_MASCULINE_GIVEN_NAMES, fold_marks
        )
    return None


def _inventories(
    faker: Any,
    convention: NameConvention,
) -> tuple[_GivenNameInventory, ...]:
    provider = _faker_person_provider(faker)
    candidates = (
        _provider_inventory(type(provider), convention.fold_marks)
        if provider is not None
        else None,
        _language_inventory(convention.lang, convention.fold_marks),
    )
    return tuple(inventory for inventory in candidates if inventory is not None)


def _known_gender(
    key: str,
    inventories: Sequence[_GivenNameInventory],
) -> Optional[Gender]:
    genders: set[Gender] = set()
    for inventory in inventories:
        if key in inventory.female_keys:
            genders.add("female")
        if key in inventory.male_keys:
            genders.add("male")
    return genders.pop() if len(genders) == 1 else None


def _draw(faker: Any, candidates: Sequence[str]) -> Optional[str]:
    if not candidates:
        return None
    return str(faker.random.choice(tuple(candidates)))


def _to_family_first(
    tokens: list[str],
    keys: list[str],
    vocabulary: _TitleVocabulary,
) -> None:
    """Rotate a given-first ``given [middle] family`` name to family-first.

    The name is rotated when it ends in a family name and either starts with
    a word that is not a family name or ends in a word that is only a family
    name. Names whose both ends are plausible family names are left alone.
    """

    if len(tokens) < 2 or keys[-1] not in vocabulary.last_names:
        return
    if keys[0] in vocabulary.last_names and keys[-1] in vocabulary.first_names:
        return
    tokens[0], tokens[-1] = tokens[-1], tokens[0]
    keys[0], keys[-1] = keys[-1], keys[0]


def shape_person_surrogate(
    surrogate: str,
    *,
    source_name: str,
    lang: Optional[str],
    gender: Optional[Gender],
    faker: Any,
    preserve_unknown_gender: bool = False,
) -> str:
    """Align a generated non-CJK PERSON surrogate with its source name.

    Steps, each a no-op when it does not apply:

    1. Strip titles the generator inserted (:func:`strip_inserted_honorifics`).
    2. For family-name-first languages, rotate a given-first Faker name so the
       family name leads.
    3. With a ``gender`` cue, replace a given name of the other (or of an
       unknown) gender with one of the cued gender drawn in the same script,
       replace family names the locale data marks as the other gender
       (Indonesian), and swap contradicting name-internal markers
       (Vietnamese ``Văn`` -> ``Thị``).

    Args:
        surrogate: The generated name for the bare source name.
        source_name: The bare source name (honorifics peeled). Its tokens are
            never reused as replacement names.
        lang: ISO 639-1 language code of the span.
        gender: Gender cue from the source span's own honorific or marker.
        faker: The seeded Faker instance used for generation; replacement
            draws use its RNG so deterministic mode stays deterministic.
        preserve_unknown_gender: Keep a given name whose gender is unknown.
            Script-specific providers set this so their synthetic names (which
            carry no gender) are not replaced.

    Returns:
        The shaped surrogate. Japanese, Korean and Chinese surrogates (including
        region-qualified codes such as ``zh-TW``) are returned unchanged; their
        honorifics are handled by :mod:`openmed.core.name_order`.
    """

    primary_lang = re.split(r"[-_]", str(lang or ""), maxsplit=1)[0].casefold()
    if primary_lang in CJK_LANGUAGES:
        return surrogate
    shaped = strip_inserted_honorifics(
        surrogate,
        reference=source_name,
        lang=lang,
        faker=faker,
    )
    convention = name_convention_for(lang)
    if convention is None or not shaped.split():
        return shaped
    if not convention.family_first and gender is None:
        return shaped

    fold_marks = convention.fold_marks
    tokens = shaped.split()
    keys = [fold_name_token(token, fold_marks=fold_marks) for token in tokens]
    changed = False

    if convention.family_first:
        before = list(tokens)
        _to_family_first(tokens, keys, _title_vocabulary(faker, None, fold_marks))
        changed = tokens != before

    if gender is not None:
        given_index = len(tokens) - 1 if convention.family_first else 0
        excluded = {
            fold_name_token(token, fold_marks=fold_marks)
            for token in source_name.split()
        }
        excluded.update(key for index, key in enumerate(keys) if index != given_index)
        changed |= _match_given_gender(
            tokens,
            keys,
            given_index=given_index,
            gender=gender,
            inventories=_inventories(faker, convention),
            excluded=excluded,
            faker=faker,
            preserve_unknown=preserve_unknown_gender,
            fold_marks=fold_marks,
        )
        changed |= _match_family_gender(
            tokens,
            keys,
            given_index=given_index,
            gender=gender,
            faker=faker,
            excluded=excluded,
            fold_marks=fold_marks,
        )
        changed |= _match_marker_gender(
            tokens,
            keys,
            given_index=given_index,
            gender=gender,
            convention=convention,
            excluded=excluded,
            faker=faker,
        )

    return " ".join(tokens) if changed else shaped


def _match_given_gender(
    tokens: list[str],
    keys: list[str],
    *,
    given_index: int,
    gender: Gender,
    inventories: Sequence[_GivenNameInventory],
    excluded: set[str],
    faker: Any,
    preserve_unknown: bool,
    fold_marks: bool,
) -> bool:
    known = _known_gender(keys[given_index], inventories)
    if known == gender or (known is None and preserve_unknown):
        return False
    script = detect_script(tokens[given_index])
    candidates = [
        name
        for inventory in inventories
        if inventory.script == script
        for name, key in inventory.names(gender)
        if key not in excluded
    ]
    replacement = _draw(faker, candidates)
    if replacement is None:
        return False
    tokens[given_index] = replacement
    keys[given_index] = fold_name_token(replacement, fold_marks=fold_marks)
    excluded.add(keys[given_index])
    return True


def _match_family_gender(
    tokens: list[str],
    keys: list[str],
    *,
    given_index: int,
    gender: Gender,
    faker: Any,
    excluded: set[str],
    fold_marks: bool,
) -> bool:
    """Replace family names that the locale data marks as the other gender."""

    provider = _faker_person_provider(faker)
    inventory = (
        _provider_family_inventory(type(provider), fold_marks)
        if provider is not None
        else None
    )
    if inventory is None:
        return False
    changed = False
    for index, key in enumerate(keys):
        if index == given_index:
            continue
        known = _known_gender(key, (inventory,))
        if known is None or known == gender:
            continue
        if detect_script(tokens[index]) != inventory.script:
            continue
        replacement = _draw(
            faker,
            [
                name
                for name, name_key in inventory.names(gender)
                if name_key not in excluded
            ],
        )
        if replacement is None:
            continue
        tokens[index] = replacement
        keys[index] = fold_name_token(replacement, fold_marks=fold_marks)
        excluded.add(keys[index])
        changed = True
    return changed


def _match_marker_gender(
    tokens: list[str],
    keys: list[str],
    *,
    given_index: int,
    gender: Gender,
    convention: NameConvention,
    excluded: set[str],
    faker: Any,
) -> bool:
    markers = {
        fold_name_token(marker, fold_marks=convention.fold_marks): (
            marker,
            marker_gender,
        )
        for marker, marker_gender in (
            *convention.gender_markers.items(),
            *_SURROGATE_GENDER_TOKENS.get(convention.lang, {}).items(),
        )
    }
    if not markers:
        return False
    same_gender = [marker for marker, value in markers.values() if value == gender]
    neutral = [
        name
        for name in _NEUTRAL_FAMILY_NAMES.get(convention.lang, ())
        if fold_name_token(name, fold_marks=convention.fold_marks) not in excluded
    ]
    changed = False
    for index, key in enumerate(keys):
        if index == given_index or key not in markers:
            continue
        if markers[key][1] == gender:
            continue
        replacement = same_gender[0] if same_gender else _draw(faker, neutral)
        if replacement is None:
            continue
        tokens[index] = replacement
        keys[index] = fold_name_token(replacement, fold_marks=convention.fold_marks)
        excluded.add(keys[index])
        changed = True
    return changed
