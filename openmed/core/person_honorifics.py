"""Honorific, gender-cue and name-order conventions for non-CJK PERSON spans.

:mod:`openmed.core.name_order` covers Japanese, Korean and Chinese. This module
is its counterpart for other languages whose PERSON spans routinely carry a
title that encodes grammatical gender: Thai ``นางสาว``, Amharic ``ወይዘሮ``,
Swahili ``Bi.``, Arabic ``السيدة``, Persian ``خانم``, Urdu ``محترمہ``, Hindi
``श्रीमती``, Vietnamese ``Chị``, Indonesian ``Ibu``, Yoruba ``Arábìnrin`` and
Hausa ``Hajiya``. Replacing such a span wholesale drops the title or swaps in a
name of the other gender, which then contradicts the surrounding pronouns and
grammar.

The registry is declarative: each :class:`NameConvention` lists the
language's honorific prefixes and suffixes with their grammatical gender,
whether names are written family-name-first (Vietnamese), and any grammatical
gender markers carried inside a name (Vietnamese ``Thị``/``Văn``).
:func:`peel_person_honorifics` splits a span into the verbatim honorific text
and the bare name so the surrogate step can replace only the name and
re-attach the title unchanged.

Gender cues come only from the span's own honorific or grammatical marker. The
module never infers gender from a given name, and it never adds a title the
span did not carry. Ambiguous words stay part of the name: a title is peeled
only at the start or end of the span, only as a whole token (Thai titles are
written attached, so Thai alone accepts an attached title followed by a valid
syllable start), and entries marked with ``min_name_tokens`` need that many
name tokens after them. Everything here is offline, dependency-free and
operates on the span text alone; it never logs or stores the span.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Dict, Final, Literal, Mapping, Optional, Tuple

__all__ = [
    "Gender",
    "Honorific",
    "NameConvention",
    "PERSON_NAME_CONVENTIONS",
    "PeeledPersonSpan",
    "fold_name_token",
    "is_family_first",
    "name_convention_for",
    "peel_person_honorifics",
    "register_person_honorific",
]

Gender = Literal["female", "male"]
"""Grammatical gender carried by an honorific or a name-internal marker."""

HonorificPosition = Literal["prefix", "suffix"]

_GENDERS: Final = frozenset({"female", "male"})
_POSITIONS: Final = frozenset({"prefix", "suffix"})
_MAX_PEELED_HONORIFICS: Final = 3
_TOKEN_RE: Final = re.compile(r"\S+")

# Thai titles are written attached to the given name (``นางสาวสมหญิง``). An
# attached title is only peeled when the remainder starts like a Thai syllable
# (a consonant or a leading vowel), so names such as ``คุณากร`` stay whole.
_THAI_SYLLABLE_START_RE: Final = re.compile(r"[ก-ฮเ-ไ]")

# Arabic-script spelling variants that do not change which word is written:
# Arabic/Persian yeh and kaf, heh/teh marbuta/heh goal, hamza-carrying alefs,
# and the tatweel elongation mark.
_ARABIC_FOLD: Final = str.maketrans(
    {
        "ي": "ی",  # ي -> ی
        "ى": "ی",  # ى -> ی
        "ك": "ک",  # ك -> ک
        "ة": "ه",  # ة -> ه
        "ہ": "ه",  # ہ -> ه
        "ە": "ه",  # ە -> ه
        "آ": "ا",  # آ -> ا
        "أ": "ا",  # أ -> ا
        "إ": "ا",  # إ -> ا
        "ـ": None,  # tatweel
    }
)


@dataclass(frozen=True)
class Honorific:
    """One honorific registered for a language.

    Attributes:
        text: The honorific as written, possibly several whitespace-separated
            words (``"Quý bà"``, ``"سرکار خانم"``).
        gender: Grammatical gender the honorific marks, or ``None`` for a
            gender-neutral title such as a doctor title.
        position: ``"prefix"`` for a title before the name, ``"suffix"`` for
            one after it (Urdu ``صاحب``, Hindi ``जी``).
        min_name_tokens: Minimum number of name tokens that must follow a
            prefix (or precede a suffix) before it is treated as a title. Use
            it for words that are also common name parts, such as Vietnamese
            ``Anh``.
    """

    text: str
    gender: Optional[Gender] = None
    position: HonorificPosition = "prefix"
    min_name_tokens: int = 1

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("honorific text must be non-empty")
        if self.gender is not None and self.gender not in _GENDERS:
            raise ValueError("gender must be 'female', 'male' or None")
        if self.position not in _POSITIONS:
            raise ValueError("position must be 'prefix' or 'suffix'")
        if self.min_name_tokens < 1:
            raise ValueError("min_name_tokens must be at least 1")


@dataclass(frozen=True)
class NameConvention:
    """Declarative PERSON-name conventions for one language.

    Attributes:
        lang: ISO 639-1 language code.
        honorifics: Registered honorific prefixes and suffixes.
        family_first: ``True`` when full names are written family name first
            (Vietnamese ``Nguyễn Thị Lan``).
        attached_prefixes: ``True`` when titles are written attached to the
            given name without a space (Thai).
        fold_marks: ``True`` to ignore combining marks when matching, so tone
            marks and harakat do not hide an honorific (Yoruba, Arabic script).
        gender_markers: Name-internal grammatical gender markers, such as the
            Vietnamese middle names ``Thị`` (female) and ``Văn`` (male).
    """

    lang: str
    honorifics: Tuple[Honorific, ...] = ()
    family_first: bool = False
    attached_prefixes: bool = False
    fold_marks: bool = False
    gender_markers: Mapping[str, Gender] = field(default_factory=dict)


@dataclass(frozen=True)
class PeeledPersonSpan:
    """A PERSON span split into verbatim honorific text and the bare name.

    ``prefix + name + suffix`` always reproduces the input span exactly, so
    separators and surrounding whitespace survive replacement.

    Attributes:
        prefix: Leading honorific text including its trailing separator.
        name: The bare name to replace. Empty for an honorific-only span.
        suffix: Trailing honorific text including its leading separator.
        gender: Gender cue taken from the span's own honorific, or from a
            name-internal marker when no honorific marks one.
    """

    prefix: str
    name: str
    suffix: str
    gender: Optional[Gender] = None

    @property
    def has_honorific(self) -> bool:
        """Return whether an honorific was peeled from the span."""

        return bool(self.prefix.strip() or self.suffix.strip())


def _prefixes(gender: Optional[Gender], *texts: str) -> Tuple[Honorific, ...]:
    return tuple(Honorific(text, gender) for text in texts)


def _suffixes(gender: Optional[Gender], *texts: str) -> Tuple[Honorific, ...]:
    return tuple(Honorific(text, gender, position="suffix") for text in texts)


# Seed conventions. Every entry is common cultural vocabulary (titles and
# forms of address), not a list of people. Words that double as names or
# surnames in the language (Urdu ``سید``/``بیگم``, Indonesian ``H.``) are
# deliberately left out so they remain part of the name.
_SEED_CONVENTIONS: Final[Tuple[NameConvention, ...]] = (
    NameConvention(
        lang="th",
        honorifics=(
            *_prefixes(
                "female",
                "นางสาว",
                "นาง",
                "น.ส.",
                "เด็กหญิง",
                "ด.ญ.",
                "แพทย์หญิง",
                "พญ.",
            ),
            *_prefixes("male", "นาย", "เด็กชาย", "ด.ช.", "นายแพทย์", "นพ."),
            *_prefixes(None, "คุณ", "ดร."),
        ),
        attached_prefixes=True,
    ),
    NameConvention(
        lang="am",
        honorifics=(
            *_prefixes("female", "ወይዘሮ", "ወ/ሮ", "ወይዘሪት", "ወ/ሪት", "ወ/ት"),
            *_prefixes("male", "አቶ"),
            *_prefixes(None, "ዶ/ር", "ዶክተር", "ፕሮፌሰር", "ፕ/ር"),
        ),
    ),
    NameConvention(
        lang="sw",
        honorifics=(
            *_prefixes("female", "Bi.", "Bi", "Bibi", "Mama"),
            *_prefixes("male", "Bw.", "Bw", "Bwana", "Mzee"),
            *_prefixes(None, "Dkt.", "Dk.", "Daktari", "Mwalimu", "Mhe.", "Mheshimiwa"),
        ),
    ),
    NameConvention(
        lang="ar",
        honorifics=(
            *_prefixes("female", "السيدة", "الآنسة", "الدكتورة", "الأستاذة", "الحاجة"),
            *_prefixes("male", "السيد", "الدكتور", "الأستاذ", "الحاج", "الشيخ"),
            *_prefixes(None, "د."),
        ),
        fold_marks=True,
    ),
    NameConvention(
        lang="fa",
        honorifics=(
            *_prefixes("female", "خانم", "سرکار خانم"),
            *_prefixes("male", "آقای", "آقا", "جناب", "جناب آقای"),
            *_prefixes(None, "دکتر", "مهندس", "استاد"),
            *_suffixes("female", "خانم"),
            *_suffixes("male", "آقا"),
        ),
        fold_marks=True,
    ),
    NameConvention(
        lang="ur",
        honorifics=(
            *_prefixes("female", "محترمہ", "مسز", "مس"),
            *_prefixes("male", "جناب", "محترم", "مسٹر"),
            *_prefixes(None, "ڈاکٹر", "پروفیسر"),
            *_suffixes("female", "صاحبہ"),
            *_suffixes("male", "صاحب"),
        ),
        fold_marks=True,
    ),
    NameConvention(
        lang="hi",
        honorifics=(
            *_prefixes("female", "श्रीमती", "सुश्री", "कुमारी", "कु.", "सौ."),
            *_prefixes("male", "श्री", "श्रीमान", "श्री."),
            *_prefixes(None, "डॉ.", "डा.", "डॉक्टर", "प्रो."),
            *_suffixes(None, "जी"),
        ),
    ),
    NameConvention(
        lang="vi",
        honorifics=(
            *_prefixes("female", "Bà", "Cô", "Chị", "Quý bà", "Quý cô"),
            *_prefixes("male", "Ông", "Chú", "Cậu", "Quý ông"),
            # ``Anh`` is also a very common given-name syllable (``Anh Thư``),
            # so it only counts as a title before a full name.
            Honorific("Anh", "male", min_name_tokens=2),
            *_prefixes(None, "Bác", "Em", "Bác sĩ", "BS.", "TS."),
        ),
        family_first=True,
        gender_markers={"Thị": "female", "Văn": "male"},
    ),
    NameConvention(
        lang="id",
        honorifics=(
            *_prefixes(
                "female",
                "Ibu",
                "Bu",
                "Nyonya",
                "Ny.",
                "Nona",
                "Nn.",
                "Sdri.",
                "Hj.",
                "Hajjah",
            ),
            *_prefixes("male", "Bapak", "Pak", "Bpk.", "Tuan", "Tn.", "Sdr.", "Haji"),
            *_prefixes(None, "dr.", "Prof.", "Ir."),
        ),
    ),
    NameConvention(
        lang="yo",
        honorifics=(
            *_prefixes("female", "Arábìnrin", "Ìyáàfin", "Alhaja"),
            *_prefixes("male", "Ọ̀gbẹ́ni", "Alhaji"),
            *_prefixes(None, "Olóyè", "Dókítà", "Ọ̀jọ̀gbọ́n"),
        ),
        fold_marks=True,
    ),
    NameConvention(
        lang="ha",
        honorifics=(
            *_prefixes("female", "Malama", "Hajiya", "Hajia", "Uwargida"),
            *_prefixes("male", "Malam", "Mallam", "Alhaji"),
        ),
    ),
)


PERSON_NAME_CONVENTIONS: Dict[str, NameConvention] = {
    convention.lang: convention for convention in _SEED_CONVENTIONS
}
"""Mutable per-language registry, extended via :func:`register_person_honorific`."""


def _base_language(lang: Optional[str]) -> str:
    """Return the primary subtag of ``lang`` (``ar-DZ`` -> ``ar``)."""

    return re.split(r"[-_]", str(lang or "").strip(), maxsplit=1)[0].casefold()


def name_convention_for(lang: Optional[str]) -> Optional[NameConvention]:
    """Return the registered name convention for ``lang``, if any.

    Region-qualified codes resolve through their primary language, so
    ``ar-DZ`` uses the Arabic convention.
    """

    return PERSON_NAME_CONVENTIONS.get(_base_language(lang))


def is_family_first(lang: Optional[str]) -> bool:
    """Return whether ``lang`` writes full names family name first."""

    convention = name_convention_for(lang)
    return bool(convention and convention.family_first)


def register_person_honorific(
    lang: str,
    text: str,
    *,
    gender: Optional[Gender] = None,
    position: HonorificPosition = "prefix",
    min_name_tokens: int = 1,
) -> None:
    """Register an additional honorific for a non-CJK language.

    Args:
        lang: ISO 639-1 language code. A convention is created when the
            language has none yet.
        text: The honorific as written.
        gender: Grammatical gender the honorific marks, or ``None``.
        position: ``"prefix"`` or ``"suffix"``.
        min_name_tokens: Name tokens required next to the honorific before it
            is treated as a title.

    Raises:
        ValueError: If ``lang`` is empty or a CJK language (use
            :func:`openmed.core.name_order.register_honorific` for those), or
            the honorific is invalid. Re-registering an identical honorific is
            a no-op.
    """

    from .name_order import CJK_LANGUAGES

    code = _base_language(lang)
    if not code:
        raise ValueError("lang must be a non-empty language code")
    if code in CJK_LANGUAGES:
        raise ValueError("CJK honorifics are registered through name_order")
    honorific = Honorific(
        text.strip(),
        gender,
        position=position,
        min_name_tokens=min_name_tokens,
    )
    convention = PERSON_NAME_CONVENTIONS.get(code) or NameConvention(lang=code)
    if honorific in convention.honorifics:
        return
    PERSON_NAME_CONVENTIONS[code] = replace(
        convention,
        honorifics=(*convention.honorifics, honorific),
    )


def fold_name_token(text: str, *, fold_marks: bool = False) -> str:
    """Return a comparison key for a name token or honorific.

    The key is NFC-normalized and case-folded, with Arabic-script spelling
    variants unified. With ``fold_marks`` combining marks (tone marks,
    harakat) are dropped as well. Keys are only compared, never emitted.
    """

    folded = unicodedata.normalize("NFC", text).casefold().translate(_ARABIC_FOLD)
    if fold_marks:
        decomposed = unicodedata.normalize("NFD", folded)
        folded = unicodedata.normalize(
            "NFC",
            "".join(char for char in decomposed if not unicodedata.combining(char)),
        )
    return folded


def _honorific_key(honorific: Honorific, convention: NameConvention) -> tuple:
    return tuple(
        fold_name_token(part, fold_marks=convention.fold_marks)
        for part in honorific.text.split()
    )


def _ordered_honorifics(
    convention: NameConvention,
    position: HonorificPosition,
) -> Tuple[Tuple[Honorific, tuple], ...]:
    """Return ``(honorific, key)`` pairs, most specific first."""

    entries = [
        (honorific, _honorific_key(honorific, convention))
        for honorific in convention.honorifics
        if honorific.position == position
    ]
    entries.sort(
        key=lambda entry: (len(entry[1]), len("".join(entry[1]))), reverse=True
    )
    return tuple(entries)


def _tokens(text: str) -> list[re.Match[str]]:
    return list(_TOKEN_RE.finditer(text))


def _match_prefix(
    body: str,
    convention: NameConvention,
) -> Optional[Tuple[Honorific, int]]:
    """Return the leading honorific of ``body`` and where the name starts."""

    tokens = _tokens(body)
    if not tokens:
        return None
    keys = [
        fold_name_token(token.group(0), fold_marks=convention.fold_marks)
        for token in tokens
    ]
    for honorific, key in _ordered_honorifics(convention, "prefix"):
        width = len(key)
        if tuple(keys[:width]) != key:
            continue
        remaining = len(tokens) - width
        if remaining and remaining < honorific.min_name_tokens:
            continue
        if not remaining and honorific.min_name_tokens > 1:
            continue
        name_start = tokens[width].start() if remaining else len(body)
        return honorific, name_start

    if not convention.attached_prefixes:
        return None
    first = tokens[0].group(0)
    folded_first = fold_name_token(first, fold_marks=convention.fold_marks)
    if len(folded_first) != len(first):
        return None
    for honorific, key in _ordered_honorifics(convention, "prefix"):
        if len(key) != 1 or not folded_first.startswith(key[0]):
            continue
        rest = first[len(key[0]) :]
        if rest and _THAI_SYLLABLE_START_RE.match(rest):
            return honorific, tokens[0].start() + len(key[0])
    return None


def _match_suffix(
    body: str,
    convention: NameConvention,
) -> Optional[Tuple[Honorific, int]]:
    """Return the trailing honorific of ``body`` and where the name ends."""

    tokens = _tokens(body)
    if not tokens:
        return None
    keys = [
        fold_name_token(token.group(0), fold_marks=convention.fold_marks)
        for token in tokens
    ]
    for honorific, key in _ordered_honorifics(convention, "suffix"):
        width = len(key)
        if len(keys) < width or tuple(keys[-width:]) != key:
            continue
        remaining = len(tokens) - width
        if remaining and remaining < honorific.min_name_tokens:
            continue
        if not remaining and honorific.min_name_tokens > 1:
            continue
        name_end = tokens[remaining - 1].end() if remaining else 0
        return honorific, name_end
    return None


def _combined_gender(genders: list[Optional[Gender]]) -> Optional[Gender]:
    marked = {gender for gender in genders if gender is not None}
    return marked.pop() if len(marked) == 1 else None


def _marker_gender(name: str, convention: NameConvention) -> Optional[Gender]:
    """Return the gender of name-internal markers in medial position."""

    if not convention.gender_markers:
        return None
    markers = {
        fold_name_token(marker, fold_marks=convention.fold_marks): gender
        for marker, gender in convention.gender_markers.items()
    }
    tokens = name.split()
    medial = tokens[1:-1]
    return _combined_gender(
        [
            markers.get(fold_name_token(token, fold_marks=convention.fold_marks))
            for token in medial
        ]
    )


def peel_person_honorifics(span: str, lang: Optional[str]) -> PeeledPersonSpan:
    """Split honorific prefixes and suffixes off a non-CJK PERSON span.

    Up to three stacked titles are peeled from each side (``جناب آقای دکتر``).
    The returned parts satisfy ``prefix + name + suffix == span``.

    Args:
        span: The detected PERSON span text.
        lang: ISO 639-1 language code. Languages without a registered
            convention return the span unchanged as ``name``.

    Returns:
        The peeled span. ``gender`` is the honorifics' shared gender, ``None``
        when they disagree or are neutral, falling back to a medial
        grammatical marker (Vietnamese ``Thị``/``Văn``) when present.

    Example:
        >>> peeled = peel_person_honorifics("Bi. Amina Juma", "sw")
        >>> (peeled.prefix, peeled.name, peeled.gender)
        ('Bi. ', 'Amina Juma', 'female')
        >>> peel_person_honorifics("Amina Juma", "sw").has_honorific
        False
    """

    convention = name_convention_for(lang)
    if convention is None or not span or not span.strip():
        return PeeledPersonSpan("", span or "", "")

    lead_length = len(span) - len(span.lstrip())
    body_end = len(span.rstrip())
    body = span[lead_length:body_end]
    prefix = span[:lead_length]
    suffix = span[body_end:]
    genders: list[Optional[Gender]] = []

    for _ in range(_MAX_PEELED_HONORIFICS):
        matched = _match_prefix(body, convention)
        if matched is None:
            break
        honorific, name_start = matched
        prefix += body[:name_start]
        body = body[name_start:]
        genders.append(honorific.gender)
        if not body:
            break

    for _ in range(_MAX_PEELED_HONORIFICS):
        if not body:
            break
        matched = _match_suffix(body, convention)
        if matched is None:
            break
        honorific, name_end = matched
        suffix = body[name_end:] + suffix
        body = body[:name_end]
        genders.append(honorific.gender)

    gender = _combined_gender(genders)
    if gender is None:
        gender = _marker_gender(body, convention)
    return PeeledPersonSpan(prefix=prefix, name=body, suffix=suffix, gender=gender)
