"""Context-anchored identifiers in German and English clinical headers.

These rules supplement model candidates. They do not establish language or
clinical qualification and never treat a capitalized word alone as a name.
"""

from __future__ import annotations

import re
from datetime import date

from openmed.processing.outputs import EntityPrediction

from .labels import DATE_OF_BIRTH, ID_NUM, PERSON, PHONE

CLINICAL_IDENTIFIER_VERSION = "clinical-identifiers-de-en-v1"
_WORD = r"[A-ZÀ-ÖØ-Þ][^\W\d_]*(?:[\u0300-\u036f][^\W\d_]*)*(?:[’'-][^\W\d_]+)*"
_INITIAL = r"[A-ZÀ-ÖØ-Þ]\."
_PART = rf"(?:{_INITIAL}|{_WORD})"
_NAME = rf"{_PART}(?:[ \t]+(?:(?:von|van|de|der|den|zu|zur)[ \t]+)?{_PART}){{0,5}}"
_TITLE = r"(?:(?i:dr|prof)\.[ \t]*(?:(?i:med)\.[ \t]*)?)"
_PATIENT = re.compile(
    rf"\b(?i:patient(?:in)?(?:[ \t]+name)?|patientenname|name|vorname|nachname)"
    rf"[ \t]*:[ \t]*(?:{_TITLE})?(?P<value>{_NAME})"
)
_CLINICIAN = re.compile(
    rf"\b(?i:arzt|ärztin|behandelnder[ \t]+arzt|behandelnde[ \t]+ärztin|"
    rf"doctor|physician|clinician|surgeon)[ \t]*:[ \t]*(?:{_TITLE})?"
    rf"(?P<value>{_NAME})"
)
_DOCTOR_TITLE = re.compile(rf"\b{_TITLE}(?P<value>{_NAME})")
_DOB = re.compile(
    r"\b(?i:geboren(?:[ \t]+am)?|geburtsdatum|geb\.|date[ \t]+of[ \t]+birth|"
    r"dob|born(?:[ \t]+on)?)[ \t]*:?[ \t]*"
    r"(?P<value>\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{4})\b"
)
_ID = re.compile(
    r"\b(?i:patienten[- ]?(?:id|nummer)|fallnummer|versichertennummer|"
    r"patient[ \t]+id|medical[ \t]+record[ \t]+(?:number|no\.)|mrn)"
    r"[ \t]*[:#][ \t]*(?P<value>[A-Za-z0-9][A-Za-z0-9/-]{2,63})\b"
)
_PHONE = re.compile(
    r"\b(?i:phone|telephone|telefon|tel\.|fax|mobile|mobil|handy)"
    r"[ \t]*:[ \t]*(?P<value>\+?\d[\d ()/.-]{5,30}\d)(?!\d)"
)


def personal_name_spans(text: str) -> tuple[tuple[int, int, str], ...]:
    """Return name-field/title spans with explicit patient or clinician roles."""
    found: dict[tuple[int, int], str] = {}
    # Header context wins over an embedded title, e.g. Patient: Dr. Parkinson.
    for pattern, role in ((_PATIENT, "patient"), (_CLINICIAN, "clinician")):
        for match in pattern.finditer(text):
            found[match.span("value")] = role
    for match in _DOCTOR_TITLE.finditer(text):
        bounds = match.span("value")
        if not any(start <= bounds[0] and bounds[1] <= end for start, end in found):
            found[bounds] = "clinician"
    return tuple((start, end, found[start, end]) for start, end in sorted(found))


def personal_name_role(text: str, start: int, end: int) -> str | None:
    """Resolve a contained name fragment without guessing a role from its name."""
    for name_start, name_end, role in personal_name_spans(text):
        if name_start <= start < end <= name_end:
            return role
    return None


def detect_clinical_identifiers(text: str, *, language: str) -> list[EntityPrediction]:
    """Return complete anchored names, IDs and validated numeric birth dates.

    All offsets address the supplied text. The caller must remap offsets if
    it applies normalization. Unsupported language requests fail explicitly.
    """
    if language not in {"de", "en"}:
        raise ValueError("clinical context identifiers support de and en")
    entities = [
        _entity(text, start, end, PERSON, role, "name_context")
        for start, end, role in personal_name_spans(text)
    ]
    for pattern, label, rule in (
        (_DOB, DATE_OF_BIRTH, "birth_context"),
        (_ID, ID_NUM, "id_context"),
        (_PHONE, PHONE, "phone_context"),
    ):
        for match in pattern.finditer(text):
            start, end = match.span("value")
            if label == DATE_OF_BIRTH and not _valid_date(text[start:end], language):
                continue
            if (
                label == PHONE
                and not 7 <= sum(char.isdigit() for char in text[start:end]) <= 15
            ):
                continue
            entities.append(_entity(text, start, end, label, "patient", rule))
    return sorted(entities, key=lambda entity: (entity.start, entity.end))


def _entity(
    text: str, start: int, end: int, label: str, role: str, rule: str
) -> EntityPrediction:
    return EntityPrediction(
        text=text[start:end],
        label=label,
        confidence=1.0,
        start=start,
        end=end,
        metadata={
            "detector": "rules:clinical_context",
            "clinical_role": role,
            "rule": rule,
            "version": CLINICAL_IDENTIFIER_VERSION,
        },
    )


def _valid_date(value: str, language: str) -> bool:
    parts = re.split(r"[./-]", value)
    if len(parts[0]) == 4:
        year, month, day = map(int, parts)
    elif language == "de" or "." in value:
        day, month, year = map(int, parts)
    else:
        # The English route does not assume a US date order. Both calendar
        # interpretations identify the entire DOB span and retain ambiguity.
        first, second, year = map(int, parts)
        month, day = (first, second) if first <= 12 else (second, first)
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


__all__ = [
    "CLINICAL_IDENTIFIER_VERSION",
    "detect_clinical_identifiers",
    "personal_name_role",
    "personal_name_spans",
]
