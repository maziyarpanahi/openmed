"""Deterministic key-value and form-field extraction from clinical text.

Intake forms and structured notes often flatten into lines such as
``Name: Example`` or ``DOB    1980-01-01``.  This module recovers those
field/value pairs without a model or network call.  The returned offsets are
half-open character ranges into the original source text, so a caller can
target a value for redaction or structured capture without re-matching it.

The label classification is deliberately routed through the canonical label
policy spine.  A recognized privacy label is therefore reported with its
canonical label and policy class; an unknown field remains unclassified
instead of being guessed as a clinical or privacy concept.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal, TypedDict

from openmed.core.labels import (
    DATE,
    DATE_OF_BIRTH,
    DIRECT_IDENTIFIER,
    EMAIL,
    ID_NUM,
    PERSON,
    PHONE,
    QUASI_IDENTIFIER,
    SENSITIVE_ATTRIBUTE,
    STREET_ADDRESS,
    normalize_label,
    policy_label_for,
)

KEY_VALUE_ADVISORY: Final = (
    "Key-value extraction uses deterministic line and delimiter heuristics; "
    "value offsets preserve source provenance, and PHI flags come from the "
    "canonical label policy spine. It is intake-structuring support, not a "
    "clinical decision."
)

# The alias map covers common form vocabulary that is more specific than the
# general label normalizer (for example, ``Patient Name``). Values are still
# canonical labels from ``openmed.core.labels``.
_FORM_LABEL_ALIASES: Final[dict[str, str]] = {
    "patientname": PERSON,
    "recordnumber": ID_NUM,
    "recordno": ID_NUM,
    "patientid": ID_NUM,
    "memberid": ID_NUM,
    "insuranceid": ID_NUM,
    "birthdate": DATE_OF_BIRTH,
    "visitdate": DATE,
    "admissiondate": DATE,
    "emailaddress": EMAIL,
    "homenumber": PHONE,
    "phonenumber": PHONE,
    "homeaddress": STREET_ADDRESS,
    "streetaddress": STREET_ADDRESS,
}

_PHI_POLICY_LABELS: Final[frozenset[str]] = frozenset(
    {DIRECT_IDENTIFIER, QUASI_IDENTIFIER, SENSITIVE_ATTRIBUTE}
)
_DELIMITER_CHARS: Final = frozenset({":", "：", "﹕", "꞉"})
_KEY_ALLOWED_PUNCTUATION: Final = frozenset(" \t/_#-'’")
_FIELD_PREFIX_RE = re.compile(r"^[ \t]*(?:(?:[-*•·])|(?:\d+[.)]))[ \t]+")
_WORD_RE = re.compile(r"\S+")


class KeyValuePair(TypedDict):
    """A source-preserving key/value field with policy metadata."""

    key: str
    value: str
    key_start: int
    key_end: int
    value_start: int
    value_end: int
    delimiter: Literal["colon", "whitespace"]
    confidence: float
    canonical_label: str | None
    policy_label: str
    is_phi: bool


class FormField(KeyValuePair):
    """Typed form-field result emitted for one recovered field."""


@dataclass(frozen=True)
class _Line:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _Candidate:
    key: str
    key_start: int
    key_end: int
    value_start: int | None
    value_end: int | None
    delimiter: Literal["colon", "whitespace"]
    confidence: float
    indent: int


def extract_key_value_pairs(
    text: str,
    *,
    lang: str = "en",
) -> list[KeyValuePair]:
    """Extract source-preserving key/value pairs from form-like text.

    Colon-delimited fields (including full-width colons), whitespace-column
    fields, and a value on the following line are recognized.  A following
    indented line is also treated as a continuation of an inline value.  Empty
    fields and ordinary lines without a field-shaped key are skipped.

    Args:
        text: Original form or structured-note text.
        lang: Language hint passed to the canonical label policy spine.

    Returns:
        Fields in source order. ``text[start:end]`` round-trips to each
        returned ``value`` through its ``value_start``/``value_end`` offsets.

    Raises:
        TypeError: If ``text`` is not a string.
    """

    return _extract_fields(text, lang=lang)


def extract_form_fields(
    text: str,
    *,
    lang: str = "en",
) -> list[FormField]:
    """Extract typed form fields and classify them using label policy.

    This is the form-oriented name for :func:`extract_key_value_pairs`.  Each
    result includes the raw key/value pair, exact source offsets, confidence,
    canonical label, policy class, and ``is_phi``.  ``is_phi`` is true for the
    policy classes that require privacy handling (direct identifiers, quasi
    identifiers, and sensitive attributes); clinical concepts remain false.

    Args:
        text: Original form or structured-note text.
        lang: Language hint passed to the canonical label policy spine.

    Returns:
        Typed form fields in source order.
    """

    return _extract_fields(text, lang=lang)


def structure_form_fields(
    text: str,
    *,
    lang: str = "en",
) -> list[FormField]:
    """Alias for :func:`extract_form_fields` used by structured pipelines."""

    return extract_form_fields(text, lang=lang)


def _extract_fields(text: str, *, lang: str) -> list[FormField]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    lines = _lines(text)
    fields: list[FormField] = []
    index = 0
    while index < len(lines):
        candidate = _parse_line(lines[index], lang=lang)
        if candidate is None:
            index += 1
            continue

        candidate, consumed_until = _resolve_value(
            lines,
            index,
            candidate,
            lang=lang,
        )
        if candidate.value_start is not None and candidate.value_end is not None:
            if candidate.value_start < candidate.value_end:
                fields.append(_to_form_field(text, candidate, lang=lang))
        index = consumed_until + 1

    return fields


def _lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        end = offset + len(line)
        lines.append(_Line(start=offset, end=end, text=line))
        offset += len(raw_line)

    return lines


def _parse_line(line: _Line, *, lang: str) -> _Candidate | None:
    prefix_match = _FIELD_PREFIX_RE.match(line.text)
    prefix_end = prefix_match.end() if prefix_match else 0
    body = line.text[prefix_end:]
    body_leading = len(body) - len(body.lstrip(" \t"))
    body = body[body_leading:]
    body_start = line.start + prefix_end + body_leading
    if not body:
        return None

    delimiter_index = next(
        (index for index, char in enumerate(body) if char in _DELIMITER_CHARS),
        None,
    )
    if delimiter_index is not None:
        key = body[:delimiter_index].strip()
        if not _looks_like_key(key):
            return None
        key_start = (
            body_start
            + len(body[:delimiter_index])
            - len(body[:delimiter_index].lstrip())
        )
        key_end = key_start + len(key)
        value_text = body[delimiter_index + 1 :]
        value_start, value_end = _value_offsets(
            value_text,
            body_start + delimiter_index + 1,
        )
        canonical = _canonical_label_for_key(key, lang=lang)
        confidence = 0.98 if canonical is not None else 0.82
        return _Candidate(
            key=key,
            key_start=key_start,
            key_end=key_end,
            value_start=value_start,
            value_end=value_end,
            delimiter="colon",
            confidence=confidence,
            indent=body_start - line.start,
        )

    return _parse_whitespace_line(line, body, body_start, lang=lang)


def _parse_whitespace_line(
    line: _Line,
    body: str,
    body_start: int,
    *,
    lang: str,
) -> _Candidate | None:
    tokens = list(_WORD_RE.finditer(body))
    best: _Candidate | None = None
    for token in tokens:
        key = body[: token.end()].strip()
        remainder = body[token.end() :]
        if not remainder or not remainder[0].isspace():
            continue
        canonical = _canonical_label_for_key(key, lang=lang)
        if canonical is None:
            continue
        value_start, value_end = _value_offsets(
            remainder,
            body_start + token.end(),
        )
        if value_start is None or value_end is None:
            continue
        best = _Candidate(
            key=key,
            key_start=body_start,
            key_end=body_start + token.end(),
            value_start=value_start,
            value_end=value_end,
            delimiter="whitespace",
            confidence=0.90,
            indent=body_start - line.start,
        )

    if best is not None:
        return best

    # Unknown labels are allowed when layout makes the split unambiguous: a
    # tab or a two-space column gap is a stronger signal than a single space.
    separator = re.search(r"(?:\t+| {2,})", body)
    if separator is None:
        return None
    key = body[: separator.start()].strip()
    if not _looks_like_key(key):
        return None
    value_start, value_end = _value_offsets(
        body[separator.end() :],
        body_start + separator.end(),
    )
    if value_start is None or value_end is None:
        return None
    key_start = (
        body_start
        + len(body[: separator.start()])
        - len(body[: separator.start()].lstrip())
    )
    return _Candidate(
        key=key,
        key_start=key_start,
        key_end=key_start + len(key),
        value_start=value_start,
        value_end=value_end,
        delimiter="whitespace",
        confidence=0.74,
        indent=body_start - line.start,
    )


def _resolve_value(
    lines: Sequence[_Line],
    index: int,
    candidate: _Candidate,
    *,
    lang: str,
) -> tuple[_Candidate, int]:
    if candidate.value_start is None:
        first_value: int | None = None
        last_value: int | None = None
        next_index = index + 1
        while next_index < len(lines):
            following = lines[next_index]
            if _parse_line(following, lang=lang) is not None:
                break
            start, end = _value_offsets(following.text, following.start)
            if start is not None and end is not None:
                first_value = start if first_value is None else first_value
                last_value = end
            next_index += 1
        if first_value is None or last_value is None:
            return candidate, index
        return (
            replace(
                candidate,
                value_start=first_value,
                value_end=last_value,
                confidence=max(0.0, candidate.confidence - 0.04),
            ),
            next_index - 1,
        )

    next_index = index + 1
    last_value = candidate.value_end
    while next_index < len(lines):
        following = lines[next_index]
        if _parse_line(following, lang=lang) is not None:
            break
        if not following.text.strip():
            break
        leading = len(following.text) - len(following.text.lstrip(" \t"))
        if leading <= candidate.indent:
            break
        _, end = _value_offsets(following.text, following.start)
        if end is None:
            break
        last_value = end
        next_index += 1

    return replace(
        candidate, value_end=last_value
    ), next_index - 1 if next_index > index + 1 else index


def _value_offsets(
    value_text: str, absolute_start: int
) -> tuple[int | None, int | None]:
    stripped = value_text.strip()
    if not stripped:
        return None, None
    leading = len(value_text) - len(value_text.lstrip())
    trailing = len(value_text.rstrip())
    return absolute_start + leading, absolute_start + trailing


def _looks_like_key(key: str) -> bool:
    if not key or len(key) > 80 or len(key.split()) > 8:
        return False
    if key[0].isdigit() or key.casefold() in {"http", "https", "ftp"}:
        return False
    if key.endswith((".", "!", "?", ";")):
        return False
    return all(char.isalnum() or char in _KEY_ALLOWED_PUNCTUATION for char in key)


def _canonical_label_for_key(key: str, *, lang: str) -> str | None:
    canonical = normalize_label(key, lang=lang)
    if canonical != "OTHER":
        return canonical
    return _FORM_LABEL_ALIASES.get(_field_key(key))


def _field_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _to_form_field(text: str, candidate: _Candidate, *, lang: str) -> FormField:
    canonical = _canonical_label_for_key(candidate.key, lang=lang)
    policy_label = policy_label_for(canonical or candidate.key, lang=lang)
    return FormField(
        key=text[candidate.key_start : candidate.key_end],
        value=text[candidate.value_start : candidate.value_end],
        key_start=candidate.key_start,
        key_end=candidate.key_end,
        value_start=candidate.value_start,
        value_end=candidate.value_end,
        delimiter=candidate.delimiter,
        confidence=round(candidate.confidence, 3),
        canonical_label=canonical,
        policy_label=policy_label,
        is_phi=policy_label in _PHI_POLICY_LABELS,
    )


__all__ = [
    "FormField",
    "KEY_VALUE_ADVISORY",
    "KeyValuePair",
    "extract_form_fields",
    "extract_key_value_pairs",
    "structure_form_fields",
]
