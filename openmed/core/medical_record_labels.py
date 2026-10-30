"""Localized medical-record-number labels for deterministic PII detection.

:data:`MEDICAL_RECORD_LABELS` is data, not detector code. Each entry lists the
record, file, patient, hospital and admission-number labels that introduce a
patient's record number in one language. :func:`medical_record_label_patterns`
compiles an entry into context-gated ``medical_record_number`` patterns, and
:mod:`openmed.core.pii_i18n` adds them to the language pack, so the
deterministic detector and the safety sweep share one rule set.

A pattern fires only when a listed label sits next to the number, so
unlabelled numbers, doses, lab values and phone numbers are never relabelled.
The number alone is the detected span (the pattern's ``value`` group), which
keeps the label readable after masking. Labels tolerate NFC and NFD spellings,
flexible spacing, label punctuation (``:``, ``：``, ``፡``, ``#``), bidi
controls and, for right-to-left scripts, a number written before its label.
Numbers may use any Unicode decimal digits, so Arabic-Indic, Persian,
Devanagari and Thai digits match without translation.

Adding a language needs one :data:`MEDICAL_RECORD_LABELS` entry plus one
synthetic fixture row; no detector code changes. Institution-specific number
formats and checksums are deliberately out of scope.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, Final, Mapping

from .pii_entity_merger import PIIPattern

MEDICAL_RECORD_LABELS_VERSION: Final = "medical-record-labels-v1"
MEDICAL_RECORD_ENTITY_TYPE: Final = "medical_record_number"


@dataclass(frozen=True)
class MedicalRecordLabels:
    """Record-number labels written before a patient's number in one language.

    Attributes:
        labels: Case-insensitive words or phrases, such as ``"No. RM"`` or
            ``"מספר תיק"``. A space matches any run of horizontal whitespace
            (and is optional after a period), an apostrophe also matches
            ``’`` and the Hebrew geresh, and a zero-width non-joiner is
            optional.
        abbreviations: Short Latin abbreviations matched case-sensitively, such
            as the Thai ``"HN"`` (hospital number), so ordinary words like
            ``"an"`` never act as labels.
    """

    labels: tuple[str, ...]
    abbreviations: tuple[str, ...] = ()

    @property
    def all_labels(self) -> tuple[str, ...]:
        """Return every label and abbreviation in table order."""
        return (*self.labels, *self.abbreviations)


# Keep entries specific to one patient's record. Words that also mean a count
# (for example Hindi "रोगी संख्या", "number of patients") are left out so a
# census figure is never relabelled as an identifier.
MEDICAL_RECORD_LABELS: Final[Mapping[str, MedicalRecordLabels]] = MappingProxyType(
    {
        "am": MedicalRecordLabels(
            labels=(
                "የካርድ ቁጥር",
                "ካርድ ቁጥር",
                "የሕክምና ካርድ ቁጥር",
                "የህክምና ካርድ ቁጥር",
                "የታካሚ ቁጥር",
                "የታካሚ መለያ ቁጥር",
                "የመዝገብ ቁጥር",
                "የሕክምና መዝገብ ቁጥር",
            ),
        ),
        "ar": MedicalRecordLabels(
            labels=(
                "رقم الملف الطبي",
                "رقم الملف",
                "رقم ملف المريض",
                "رقم ملف",
                "رقم السجل الطبي",
                "الرقم الطبي",
                "رقم المريض",
                "رقم الدخول",
                "رقم التنويم",
            ),
        ),
        "fa": MedicalRecordLabels(
            labels=(
                "شماره پرونده",
                "شماره\u200cی پرونده",
                "شماره پرونده پزشکی",
                "شماره پرونده بیمار",
                "کد پرونده",
                "شماره بیمار",
                "شماره بستری",
                "شماره پذیرش",
            ),
        ),
        "ha": MedicalRecordLabels(
            labels=(
                "lambar fayil",
                "lambar fayil na asibiti",
                "lambar majinyaci",
                "lambar mara lafiya",
            ),
        ),
        "he": MedicalRecordLabels(
            labels=(
                "מספר תיק",
                "מס' תיק",
                "מספר תיק רפואי",
                "מספר רשומה רפואית",
                "מספר מטופל",
                "מס' מטופל",
                "מספר אשפוז",
                "מס' אשפוז",
            ),
        ),
        "hi": MedicalRecordLabels(
            labels=(
                "पंजीकरण संख्या",
                "पंजीकरण क्रमांक",
                "पंजीकरण नंबर",
                "पंजीकरण सं.",
                "रजिस्ट्रेशन नंबर",
                "मेडिकल रिकॉर्ड नंबर",
                "चिकित्सा अभिलेख संख्या",
                "रोगी क्रमांक",
                "भर्ती क्रमांक",
                "भर्ती नंबर",
                "ओपीडी नंबर",
                "ओपीडी क्रमांक",
                "आईपीडी नंबर",
                "आईपीडी क्रमांक",
                "एमआरएन",
                "यूएचआईडी",
                "UHID",
            ),
        ),
        "id": MedicalRecordLabels(
            labels=(
                "No. RM",
                "No RM",
                "Nomor RM",
                "No. Rekam Medis",
                "No. Rekam Medik",
                "Nomor Rekam Medis",
                "Nomor Rekam Medik",
                "No. CM",
                "Nomor CM",
            ),
        ),
        "sw": MedicalRecordLabels(
            labels=(
                "namba ya faili",
                "nambari ya faili",
                "namba ya jalada",
                "nambari ya jalada",
                "namba ya mgonjwa",
                "nambari ya mgonjwa",
                "namba ya kulazwa",
                "nambari ya kulazwa",
            ),
        ),
        "th": MedicalRecordLabels(
            labels=(
                "เลขที่ผู้ป่วย",
                "เลขที่ผู้ป่วยนอก",
                "เลขที่ผู้ป่วยใน",
                "เลขประจำตัวผู้ป่วย",
                "เลขเวชระเบียน",
                "เลขที่เวชระเบียน",
                "รหัสผู้ป่วย",
            ),
            # HN = hospital number, AN = admission number.
            abbreviations=("HN", "H.N.", "AN", "A.N."),
        ),
        "ur": MedicalRecordLabels(
            labels=(
                "ایم آر نمبر",
                "ایم آر این",
                "ایم آر این نمبر",
                "فائل نمبر",
                "میڈیکل ریکارڈ نمبر",
                "مریض نمبر",
                "داخلہ نمبر",
                "رجسٹریشن نمبر",
            ),
        ),
        "vi": MedicalRecordLabels(
            labels=(
                "Mã bệnh nhân",
                "Mã BN",
                "Mã người bệnh",
                "Mã bệnh án",
                "Số bệnh án",
                "Số hồ sơ bệnh án",
                "Số hồ sơ",
                "Số lưu trữ",
                "Số vào viện",
            ),
        ),
        # Chinese-numeral record numbers keep their dedicated pii_i18n rule;
        # this entry covers the same labels followed by decimal digits.
        "zh": MedicalRecordLabels(
            labels=(
                "病历号",
                "病历号码",
                "病案号",
                "住院号",
                "住院号码",
                "门诊号",
                "患者编号",
            ),
        ),
    }
)

# English keeps its built-in ``MRN`` rule in ``pii_entity_merger.PII_PATTERNS``.
_BUILTIN_LABEL_LANGUAGES: Final = frozenset({"en"})

# Four to twenty decimal digits in any script, optionally split by single
# hyphens, slashes or dots (for example Thai ``65-012345`` or ``12345/65``).
_RECORD_NUMBER = r"\d(?:[-/.]?\d){3,19}"
_RECORD_NUMBER_END = r"(?![-/.]?\d)(?![A-Za-z])"
_RECORD_NUMBER_START = r"(?<!\d)(?<!\d[-/.])"
# Horizontal whitespace, and LRM, RLM, ALM and directional isolates.
_SPACE_CHARS = r" \t\u00a0\u202f"
_BIDI_MARK_CHARS = r"\u200e\u200f\u061c\u2066-\u2069"
_GAP = rf"[{_SPACE_CHARS}{_BIDI_MARK_CHARS}]*"
_LABEL_PUNCTUATION = r"[:：፡፥፦#=.\-–]"
_LEADING_SEPARATOR = rf"{_GAP}(?:{_LABEL_PUNCTUATION}{_GAP}){{0,2}}"
# A number written before its label (visual-order right-to-left text) needs a
# colon attached to the label, as in ``1234567 :label``. A colon that ends a
# clause (``0501234567: label``) is never read as a label separator.
_TRAILING_SEPARATOR = rf"{_GAP}[:：][{_BIDI_MARK_CHARS}]*"
_LATIN_LEFT_BOUNDARY = r"(?<![A-Za-z0-9\u00c0-\u024f\u1e00-\u1eff])"
_LABEL_SPACE = rf"[{_SPACE_CHARS}]+"
_LABEL_SPACE_AFTER_PERIOD = rf"[{_SPACE_CHARS}]*"
_APOSTROPHES = "'’׳"
_ZWNJ = "\u200c"


@dataclass(frozen=True)
class MedicalRecordLabelMatch:
    """Raw-text-free record of one labelled medical record number.

    Attributes:
        start: Start offset of the number in the source text.
        end: End offset of the number in the source text.
        label: Canonical table label that introduced the number.
        language: Language whose table entry matched.
    """

    start: int
    end: int
    label: str
    language: str

    def to_dict(self) -> dict[str, Any]:
        """Return the entity type, table label and offsets only."""
        return {
            "entity_type": MEDICAL_RECORD_ENTITY_TYPE,
            "label": self.label,
            "language": self.language,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class MedicalRecordLabelCoverage:
    """Which registered language packs carry record-number labels.

    Attributes:
        label_counts: Number of labels and abbreviations per table language.
        registered_without_labels: Registered non-English packs that still
            rely on a model alone for record numbers.
    """

    label_counts: Mapping[str, int]
    registered_without_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return language codes and counts, never vocabulary or source text."""
        return {
            "version": MEDICAL_RECORD_LABELS_VERSION,
            "languages": sorted(self.label_counts),
            "label_counts": dict(sorted(self.label_counts.items())),
            "label_total": sum(self.label_counts.values()),
            "registered_without_labels": list(self.registered_without_labels),
        }


def _is_latin_label(label: str) -> bool:
    letters = [char for char in label if char.isalpha()]
    return bool(letters) and all(
        unicodedata.name(char, "").startswith("LATIN") for char in letters
    )


def _is_right_to_left(label: str) -> bool:
    return any(unicodedata.bidirectional(char) in {"R", "AL"} for char in label)


def _spelling_variants(label: str) -> tuple[str, ...]:
    """Return NFC and NFD spellings, plus an unmarked one for Latin labels."""
    nfc = unicodedata.normalize("NFC", label)
    nfd = unicodedata.normalize("NFD", nfc)
    variants = [nfc, nfd]
    if _is_latin_label(nfc):
        unmarked = "".join(
            char for char in nfd if not unicodedata.category(char).startswith("M")
        )
        variants.append(unmarked)
    return tuple(dict.fromkeys(variants))


def _label_expression(spelling: str) -> str:
    """Escape one label spelling while relaxing spacing and punctuation."""
    parts: list[str] = []
    for index, char in enumerate(spelling):
        if char == " ":
            after_period = index > 0 and spelling[index - 1] == "."
            parts.append(_LABEL_SPACE_AFTER_PERIOD if after_period else _LABEL_SPACE)
        elif char in _APOSTROPHES:
            parts.append(f"[{_APOSTROPHES}]")
        elif char == _ZWNJ:
            parts.append(f"{re.escape(_ZWNJ)}?")
        else:
            parts.append(re.escape(char))
    return "".join(parts)


def _alternation(labels: tuple[str, ...], *, case_sensitive: bool) -> str:
    spellings = {spelling for label in labels for spelling in _spelling_variants(label)}
    latin: list[str] = []
    other: list[str] = []
    # Longest spellings first so the longest label wins at a shared start.
    for spelling in sorted(spellings, key=lambda item: (-len(item), item)):
        target = latin if _is_latin_label(spelling) else other
        target.append(_label_expression(spelling))
    # Latin labels share one left boundary so they never start inside a word.
    branches = [f"{_LATIN_LEFT_BOUNDARY}(?:{'|'.join(latin)})"] if latin else []
    branches.extend(other)
    if not branches:
        return ""
    expression = "|".join(branches)
    return f"(?-i:{expression})" if case_sensitive else expression


def _label_group(entry: MedicalRecordLabels) -> str:
    alternatives = [
        alternation
        for alternation in (
            _alternation(entry.labels, case_sensitive=False),
            _alternation(entry.abbreviations, case_sensitive=True),
        )
        if alternation
    ]
    return "(?P<label>" + "|".join(alternatives) + ")"


def _context_words(entry: MedicalRecordLabels) -> list[str]:
    """Return label words for the context gate.

    The regex already requires a whole label, so the gate only needs one of
    its words to survive spacing, apostrophe and joiner variants.
    """
    words = {
        word
        for label in entry.all_labels
        for spelling in _spelling_variants(label)
        for word in spelling.replace(_ZWNJ, " ").casefold().split()
    }
    # Keep the shared English cue so split ``MRN`` sweeps keep their score.
    words.add("mrn")
    return sorted(words)


def _label_flags(entry: MedicalRecordLabels) -> int:
    """Request case folding only when a case-insensitive label has case."""
    cased = any(char.islower() or char.isupper() for char in "".join(entry.labels))
    return re.IGNORECASE if cased else 0


def _pattern(expression: str, entry: MedicalRecordLabels) -> PIIPattern:
    return PIIPattern(
        expression,
        MEDICAL_RECORD_ENTITY_TYPE,
        # An explicit record label outranks shape-only date, phone, postcode,
        # permit and national-ID guesses for the same digits.
        priority=16,
        flags=_label_flags(entry),
        base_score=0.3,
        context_words=_context_words(entry),
        context_boost=0.6,
        context_required=True,
        safety_sweep_requires_context=True,
    )


def _language_key(lang: str) -> str:
    return str(lang).strip().replace("-", "_").split("_", 1)[0].casefold()


@lru_cache(maxsize=None)
def _compiled_patterns(language: str) -> tuple[PIIPattern, ...]:
    entry = MEDICAL_RECORD_LABELS.get(language)
    if entry is None or not entry.all_labels:
        return ()
    label_group = _label_group(entry)
    value = f"(?P<value>{_RECORD_NUMBER}){_RECORD_NUMBER_END}"
    patterns = [_pattern(f"{label_group}{_LEADING_SEPARATOR}{value}", entry)]
    if any(_is_right_to_left(label) for label in entry.all_labels):
        trailing = f"{_RECORD_NUMBER_START}{value}{_TRAILING_SEPARATOR}{label_group}"
        patterns.append(_pattern(trailing, entry))
    return tuple(patterns)


def medical_record_label_patterns(lang: str) -> list[PIIPattern]:
    """Return context-gated record-number patterns for one language.

    Each pattern names the record number ``value`` so detectors return only
    the number, never its label. A right-to-left language also receives a
    pattern for a number written before its label.

    Args:
        lang: ISO 639-1 language code, optionally with a region suffix.

    Returns:
        A new list of :class:`~openmed.core.pii_entity_merger.PIIPattern`
        objects, empty when the language has no table entry.
    """
    return list(_compiled_patterns(_language_key(lang)))


def _perso_arabic_fold(language: str) -> Callable[[str], str]:
    if language not in {"fa", "ur"}:
        return lambda value: value
    from .script_detect import fold_perso_arabic_letters

    return lambda value: fold_perso_arabic_letters(value, lang=language)


@lru_cache(maxsize=None)
def _canonical_label_patterns(language: str) -> tuple[tuple[str, re.Pattern], ...]:
    entry = MEDICAL_RECORD_LABELS[language]
    fold = _perso_arabic_fold(language)
    return tuple(
        (
            label,
            re.compile(
                fold(
                    _alternation((label,), case_sensitive=label in entry.abbreviations)
                ),
                re.IGNORECASE,
            ),
        )
        for label in entry.all_labels
    )


def medical_record_label_matches(
    text: str, lang: str
) -> tuple[MedicalRecordLabelMatch, ...]:
    """Locate labelled record numbers, reporting offsets and labels only.

    Args:
        text: Source text to scan.
        lang: ISO 639-1 language code, optionally with a region suffix.

    Returns:
        Non-overlapping matches ordered by offset. Each records the number's
        offsets and the canonical table label, never the number itself.
    """
    language = _language_key(lang)
    if language not in MEDICAL_RECORD_LABELS:
        return ()
    fold = _perso_arabic_fold(language)
    canonical = _canonical_label_patterns(language)
    matches: list[MedicalRecordLabelMatch] = []
    for pattern in _compiled_patterns(language):
        for match in re.finditer(fold(pattern.pattern), text, pattern.flags):
            start, end = match.span("value")
            if any(start < kept.end and end > kept.start for kept in matches):
                continue
            surface = match.group("label")
            label = next(name for name, regex in canonical if regex.fullmatch(surface))
            matches.append(
                MedicalRecordLabelMatch(
                    start=start, end=end, label=label, language=language
                )
            )
    return tuple(sorted(matches, key=lambda item: (item.start, item.end)))


def registered_label_languages() -> frozenset[str]:
    """Return the registered language packs that route deterministic patterns.

    Returns:
        Model-backed, national-ID-only, Indic and user-supplied-model language
        codes accepted by :func:`openmed.core.pii_i18n.get_patterns_for_language`.
    """
    from .pii_i18n import (
        INDIC_NER_LANGUAGES,
        NATIONAL_ID_ONLY_LANGUAGES,
        SUPPORTED_LANGUAGES,
        USER_SUPPLIED_MODEL_LANGUAGES,
    )

    return frozenset(
        SUPPORTED_LANGUAGES
        | NATIONAL_ID_ONLY_LANGUAGES
        | INDIC_NER_LANGUAGES
        | USER_SUPPLIED_MODEL_LANGUAGES
    )


def medical_record_label_coverage() -> MedicalRecordLabelCoverage:
    """Report which registered packs have record-number labels.

    Returns:
        Per-language label counts and the sorted registered non-English packs
        without a table entry. English is covered by the built-in ``MRN``
        rule, so it is never listed as missing.
    """
    registered = registered_label_languages()
    return MedicalRecordLabelCoverage(
        label_counts=MappingProxyType(
            {
                language: len(entry.all_labels)
                for language, entry in sorted(MEDICAL_RECORD_LABELS.items())
            }
        ),
        registered_without_labels=tuple(
            sorted(registered - set(MEDICAL_RECORD_LABELS) - _BUILTIN_LABEL_LANGUAGES)
        ),
    )


__all__ = [
    "MEDICAL_RECORD_ENTITY_TYPE",
    "MEDICAL_RECORD_LABELS",
    "MEDICAL_RECORD_LABELS_VERSION",
    "MedicalRecordLabelCoverage",
    "MedicalRecordLabelMatch",
    "MedicalRecordLabels",
    "medical_record_label_coverage",
    "medical_record_label_matches",
    "medical_record_label_patterns",
    "registered_label_languages",
]
