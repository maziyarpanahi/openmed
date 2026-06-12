"""Residual re-identification risk scoring for text and table records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openmed.core.decoding.spans import trim_span_whitespace
from openmed.core.labels import (
    ACCOUNT_NUMBER,
    AGE,
    API_KEY,
    BIC,
    BITCOIN_ADDRESS,
    BUILDING_NUMBER,
    CREDIT_CARD,
    CVV,
    DATE,
    DATE_OF_BIRTH,
    EMAIL,
    ETHEREUM_ADDRESS,
    FIRST_NAME,
    GPS_COORDINATES,
    IBAN,
    ID_NUM,
    IMEI,
    IP_ADDRESS,
    LAST_NAME,
    LITECOIN_ADDRESS,
    LOCATION,
    MAC_ADDRESS,
    MIDDLE_NAME,
    ORGANIZATION,
    PASSWORD,
    PERSON,
    PHONE,
    PIN,
    PREFIX,
    SSN,
    STREET_ADDRESS,
    URL,
    USERNAME,
    VEHICLE_REGISTRATION,
    VIN,
    ZIPCODE,
    normalize_label,
)


_TEXT_KEYS = (
    "text",
    "note",
    "content",
    "document",
    "deidentified_text",
    "original_text",
)
_SPAN_KEYS = ("entities", "spans", "pii", "predictions")
_CONTAINER_KEYS = ("records", "rows", "items", "documents")
_ID_KEYS = ("id", "record_id", "doc_id", "document_id")
_RESERVED_KEYS = set(_TEXT_KEYS + _SPAN_KEYS + _CONTAINER_KEYS + _ID_KEYS)

_DIRECT_LABELS = {
    ACCOUNT_NUMBER,
    API_KEY,
    BIC,
    BITCOIN_ADDRESS,
    BUILDING_NUMBER,
    CREDIT_CARD,
    CVV,
    DATE_OF_BIRTH,
    EMAIL,
    ETHEREUM_ADDRESS,
    FIRST_NAME,
    GPS_COORDINATES,
    IBAN,
    ID_NUM,
    IMEI,
    IP_ADDRESS,
    LAST_NAME,
    LITECOIN_ADDRESS,
    LOCATION,
    MAC_ADDRESS,
    MIDDLE_NAME,
    PASSWORD,
    PERSON,
    PHONE,
    PIN,
    PREFIX,
    SSN,
    STREET_ADDRESS,
    URL,
    USERNAME,
    VEHICLE_REGISTRATION,
    VIN,
    ZIPCODE,
}

_AGE_PATTERN = re.compile(
    r"\b(?:age(?:d)?\s*)?((?:1[01]\d)|(?:[1-9]?\d))\s*"
    r"(?:years?\s+old|year-old|y/o|yo)\b",
    re.IGNORECASE,
)
_DATE_PATTERN = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"
    r"\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_INSTITUTION_PATTERN = re.compile(
    r"\b(?:Dr\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?|"
    r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\s+"
    r"(?:Hospital|Clinic|Medical Center|Health System|University Hospital|Practice))\b"
)
_GEOGRAPHY_PATTERN = re.compile(
    r"\b(?:in|from|resident of|lives in|located in)\s+"
    r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}"
    r"(?:,?\s+(?:[A-Z]{2}|\d{5}))?)\b"
)
_RARE_CONDITION_PATTERN = re.compile(
    r"[\[<{(]?\s*(?:rare[_\s-]?)?(?:condition|diagnosis|disease)\s*[\]>})]?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Record:
    index: int
    record_id: str | None
    text: str
    fields: Mapping[str, Any]
    spans: tuple[Mapping[str, Any], ...]
    source: str


@dataclass(frozen=True)
class _QuasiIdentifier:
    record_index: int
    record_id: str | None
    category: str
    value: str
    normalized_value: str
    source: str
    start: int | None = None
    end: int | None = None
    section: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_index": self.record_index,
            "record_id": self.record_id,
            "category": self.category,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "section": self.section,
        }


@dataclass(frozen=True)
class _Profile:
    record: _Record
    quasi_identifiers: tuple[_QuasiIdentifier, ...]
    key: tuple[tuple[str, tuple[str, ...]], ...]


def risk_report(
    deidentified: Any,
    original: Any | None = None,
    aux: Any | None = None,
) -> dict[str, Any]:
    """Score residual re-identification risk for text or table records.

    Inputs may be a string, a prediction-result-like mapping with ``text`` and
    span dictionaries, a row mapping, a sequence of records, or a
    DataFrame-like object exposing ``to_dict("records")``. Existing OpenMed
    span offsets are preferred: when a span provides ``start`` and ``end``,
    the quasi-identifier value is sliced from the source text and its section
    metadata is carried into the report. Regex and column-name extraction are
    deliberately small hooks until grounding-driven QI classification lands.
    """

    deidentified_records = _coerce_records(deidentified, source="deidentified")
    original_records = _coerce_records(original, source="original") if original is not None else []
    aux_records = _coerce_records(aux, source="aux") if aux is not None else []

    deidentified_profiles = [_profile_record(record) for record in deidentified_records]
    original_profiles = [_profile_record(record) for record in original_records]
    aux_profiles = [_profile_record(record) for record in aux_records]

    class_counts = Counter(profile.key for profile in deidentified_profiles)
    k_values = [class_counts[profile.key] for profile in deidentified_profiles]
    k_min = min(k_values) if k_values else 0

    singleton_records = [
        _singleton_record(profile, class_counts[profile.key])
        for profile in deidentified_profiles
        if class_counts[profile.key] == 1
    ]
    quasi_identifiers = [
        qi.to_dict()
        for profile in deidentified_profiles
        for qi in profile.quasi_identifiers
    ]

    return {
        "leakage_rate": _leakage_rate(deidentified_records, original_records),
        "reid_rate": _reid_rate(deidentified_profiles, original_profiles, aux_profiles),
        "k_min": k_min,
        "singleton_records": singleton_records,
        "quasi_identifiers": quasi_identifiers,
    }


def _coerce_records(data: Any, *, source: str) -> list[_Record]:
    if data is None:
        return []

    dataframe_records = _maybe_dataframe_records(data)
    if dataframe_records is not None:
        data = dataframe_records

    if isinstance(data, str):
        return [_Record(0, None, data, {}, (), source)]

    if isinstance(data, Mapping):
        container = _first_container(data)
        if container is not None and not _looks_like_single_record(data):
            return _coerce_records(container, source=source)
        return [_record_from_mapping(data, 0, source)]

    if _is_sequence(data):
        records: list[_Record] = []
        for item in data:
            records.extend(_coerce_records(item, source=source))
        return [
            _Record(index, record.record_id, record.text, record.fields, record.spans, record.source)
            for index, record in enumerate(records)
        ]

    return [_Record(0, None, str(data), {}, (), source)]


def _maybe_dataframe_records(data: Any) -> list[Mapping[str, Any]] | None:
    to_dict = getattr(data, "to_dict", None)
    if to_dict is None or isinstance(data, Mapping):
        return None
    try:
        records = to_dict("records")
    except TypeError:
        return None
    if isinstance(records, list) and all(isinstance(item, Mapping) for item in records):
        return records
    return None


def _first_container(data: Mapping[str, Any]) -> Any | None:
    for key in _CONTAINER_KEYS:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _looks_like_single_record(data: Mapping[str, Any]) -> bool:
    return any(key in data for key in _TEXT_KEYS + _SPAN_KEYS) or any(
        key in data for key in _ID_KEYS
    )


def _record_from_mapping(data: Mapping[str, Any], index: int, source: str) -> _Record:
    record_id = _record_id(data)
    text = _record_text(data)
    spans = tuple(_span for key in _SPAN_KEYS for _span in _coerce_spans(data.get(key)))
    fields = {
        str(key): value
        for key, value in data.items()
        if key not in _RESERVED_KEYS and _is_scalar(value)
    }
    if not text:
        text = " ".join(str(value) for value in fields.values() if value is not None)
    return _Record(index, record_id, text, fields, spans, source)


def _record_id(data: Mapping[str, Any]) -> str | None:
    for key in _ID_KEYS:
        value = data.get(key)
        if value is not None:
            return str(value)
    return None


def _record_text(data: Mapping[str, Any]) -> str:
    for key in _TEXT_KEYS:
        value = data.get(key)
        if isinstance(value, str):
            return value
    return ""


def _coerce_spans(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if _is_sequence(value):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _profile_record(record: _Record) -> _Profile:
    qis = _dedupe_qis(
        [
            *_span_quasi_identifiers(record),
            *_field_quasi_identifiers(record),
            *_regex_quasi_identifiers(record),
        ]
    )
    return _Profile(record, tuple(qis), _profile_key(qis))


def _span_quasi_identifiers(record: _Record) -> list[_QuasiIdentifier]:
    qis: list[_QuasiIdentifier] = []
    for span in record.spans:
        category = _span_category(span)
        if category is None:
            continue
        value, start, end = _span_value(record, span)
        qi = _build_qi(
            record,
            category,
            value,
            source="span",
            start=start,
            end=end,
            section=_span_section(span),
        )
        if qi is not None:
            qis.append(qi)
    return qis


def _span_category(span: Mapping[str, Any]) -> str | None:
    label = _span_label(span)
    if not label:
        return None
    canonical = normalize_label(label)
    if canonical == AGE:
        return "age"
    if canonical in {DATE, DATE_OF_BIRTH}:
        return "date"
    if canonical in {LOCATION, ZIPCODE, GPS_COORDINATES, STREET_ADDRESS}:
        return "geography"
    if canonical == ORGANIZATION:
        return "provider_institution"

    normalized = _name_key(label)
    if any(hint in normalized for hint in ("age", "date", "dob", "birthdate")):
        return "age" if "age" in normalized else "date"
    if any(hint in normalized for hint in ("city", "state", "county", "zip", "postal", "location", "geography")):
        return "geography"
    if any(hint in normalized for hint in ("provider", "doctor", "physician", "hospital", "clinic", "facility", "institution")):
        return "provider_institution"
    if any(hint in normalized for hint in ("rare", "condition", "diagnosis", "disease")):
        return "rare_condition"
    return None


def _span_label(span: Mapping[str, Any]) -> str:
    for key in ("canonical_label", "policy_label", "label", "entity_group", "entity", "entity_type", "type"):
        value = span.get(key)
        if value:
            return str(value)
    return ""


def _span_value(record: _Record, span: Mapping[str, Any]) -> tuple[str, int | None, int | None]:
    start = _optional_int(span.get("start"))
    end = _optional_int(span.get("end"))
    if start is not None and end is not None and record.text:
        start = max(0, min(start, len(record.text)))
        end = max(start, min(end, len(record.text)))
        start, end = trim_span_whitespace(start, end, record.text)
        if end > start:
            return record.text[start:end], start, end

    for key in ("text", "word", "value", "surface", "replacement"):
        value = span.get(key)
        if value is not None:
            return str(value), start, end
    return "", start, end


def _span_section(span: Mapping[str, Any]) -> str | None:
    section = span.get("section")
    if section is not None:
        return str(section)
    metadata = span.get("metadata")
    if isinstance(metadata, Mapping):
        section = metadata.get("section")
        if section is not None:
            return str(section)
    return None


def _field_quasi_identifiers(record: _Record) -> list[_QuasiIdentifier]:
    qis: list[_QuasiIdentifier] = []
    for name, value in record.fields.items():
        category = _field_category(name)
        if category is None or value is None:
            continue
        qi = _build_qi(record, category, str(value), source="field")
        if qi is not None:
            qis.append(qi)
    return qis


def _field_category(name: str) -> str | None:
    normalized = _name_key(name)
    if "age" in normalized:
        return "age"
    if any(hint in normalized for hint in ("date", "dob", "birth", "visit", "admission", "discharge")):
        return "date"
    if any(hint in normalized for hint in ("city", "state", "county", "zip", "postal", "country", "region", "location", "geography")):
        return "geography"
    if any(hint in normalized for hint in ("provider", "doctor", "physician", "hospital", "clinic", "facility", "institution", "organization")):
        return "provider_institution"
    if any(hint in normalized for hint in ("rare", "condition", "diagnosis", "disease")):
        return "rare_condition"
    return None


def _regex_quasi_identifiers(record: _Record) -> list[_QuasiIdentifier]:
    if not record.text:
        return []
    qis: list[_QuasiIdentifier] = []
    for category, pattern in (
        ("age", _AGE_PATTERN),
        ("date", _DATE_PATTERN),
        ("geography", _GEOGRAPHY_PATTERN),
        ("provider_institution", _INSTITUTION_PATTERN),
        ("rare_condition", _RARE_CONDITION_PATTERN),
    ):
        for match in pattern.finditer(record.text):
            start = match.start(1) if category == "geography" else match.start()
            end = match.end(1) if category == "geography" else match.end()
            qi = _build_qi(
                record,
                category,
                record.text[start:end],
                source="text",
                start=start,
                end=end,
            )
            if qi is not None:
                qis.append(qi)
    return qis


def _build_qi(
    record: _Record,
    category: str,
    value: str,
    *,
    source: str,
    start: int | None = None,
    end: int | None = None,
    section: str | None = None,
) -> _QuasiIdentifier | None:
    normalized = _normalize_qi_value(category, value)
    if not normalized:
        return None
    if category != "rare_condition" and _is_generic_placeholder(value):
        return None
    return _QuasiIdentifier(
        record_index=record.index,
        record_id=record.record_id,
        category=category,
        value=str(value).strip(),
        normalized_value=normalized,
        source=source,
        start=start,
        end=end,
        section=section,
    )


def _normalize_qi_value(category: str, value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text:
        return ""

    if category == "age":
        match = re.search(r"\b(?:1[01]\d|[1-9]?\d)\b", text)
        return match.group(0) if match else ""

    normalized = text.casefold()
    normalized = re.sub(r"^[\[{(<]\s*|\s*[\]})>]$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" .,:;")


def _is_generic_placeholder(value: Any) -> bool:
    text = str(value).strip()
    return bool(re.fullmatch(r"[\[{(<]?\s*[A-Z][A-Z0-9_\s-]{1,}\s*[\]})>]?", text))


def _dedupe_qis(qis: list[_QuasiIdentifier]) -> list[_QuasiIdentifier]:
    seen: set[tuple[str, str, int | None, int | None]] = set()
    deduped: list[_QuasiIdentifier] = []
    for qi in qis:
        key = (qi.category, qi.normalized_value, qi.start, qi.end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(qi)
    return deduped


def _profile_key(qis: list[_QuasiIdentifier]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    by_category: dict[str, set[str]] = {}
    for qi in qis:
        by_category.setdefault(qi.category, set()).add(qi.normalized_value)
    return tuple(
        (category, tuple(sorted(values)))
        for category, values in sorted(by_category.items())
        if values
    )


def _singleton_record(profile: _Profile, effective_k: int) -> dict[str, Any]:
    return {
        "record_index": profile.record.index,
        "record_id": profile.record.record_id,
        "effective_k": effective_k,
        "quasi_identifier_key": [
            {"category": category, "values": list(values)}
            for category, values in profile.key
        ],
    }


def _reid_rate(
    deidentified_profiles: list[_Profile],
    original_profiles: list[_Profile],
    aux_profiles: list[_Profile],
) -> float:
    if not deidentified_profiles:
        return 0.0

    original_counts = Counter(profile.key for profile in original_profiles if profile.key)
    aux_counts = Counter(profile.key for profile in aux_profiles if profile.key)

    linked = 0
    for profile in deidentified_profiles:
        if not profile.key:
            continue
        if original_counts[profile.key] == 1 or aux_counts[profile.key] == 1:
            linked += 1
    return linked / len(deidentified_profiles)


def _leakage_rate(deidentified_records: list[_Record], original_records: list[_Record]) -> float:
    if not deidentified_records:
        return 0.0

    original_values = {
        _normalize_direct_value(value)
        for record in original_records
        for value in _direct_identifier_values(record)
    }
    original_values.discard("")

    leaked_records = 0
    for record in deidentified_records:
        direct_values = {
            _normalize_direct_value(value)
            for value in _direct_identifier_values(record)
            if not _is_generic_placeholder(value)
        }
        direct_values.discard("")

        if direct_values:
            leaked_records += 1
            continue

        blob = _normalize_direct_value(" ".join([record.text, *map(str, record.fields.values())]))
        if original_values and any(value in blob for value in original_values if len(value) > 1):
            leaked_records += 1

    return leaked_records / len(deidentified_records)


def _direct_identifier_values(record: _Record) -> list[str]:
    values: list[str] = []
    for span in record.spans:
        if _span_is_direct_identifier(span):
            value, _, _ = _span_value(record, span)
            if value:
                values.append(value)

    for name, value in record.fields.items():
        if value is not None and _field_is_direct_identifier(name):
            values.append(str(value))
    return values


def _span_is_direct_identifier(span: Mapping[str, Any]) -> bool:
    label = _span_label(span)
    if not label:
        return False
    if normalize_label(label) in _DIRECT_LABELS:
        return True
    normalized = _name_key(label)
    return any(
        hint in normalized
        for hint in (
            "name",
            "email",
            "phone",
            "ssn",
            "mrn",
            "id",
            "address",
            "account",
            "password",
            "apikey",
        )
    )


def _field_is_direct_identifier(name: str) -> bool:
    normalized = _name_key(name)
    if any(safe_hint in normalized for safe_hint in ("date", "age", "diagnosis", "condition")):
        return False
    return any(
        hint in normalized
        for hint in (
            "name",
            "email",
            "phone",
            "ssn",
            "mrn",
            "id",
            "address",
            "account",
            "password",
            "apikey",
        )
    )


def _normalize_direct_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


__all__ = ["risk_report"]
