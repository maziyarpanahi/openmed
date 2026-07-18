"""Residual re-identification risk scoring for text and table records."""

from __future__ import annotations

import hashlib
import hmac
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

_DEFAULT_LONGITUDINAL_HMAC_KEY = "openmed-longitudinal-linkage-local-key"
_PATIENT_KEY_FIELDS = (
    "patient_id",
    "patient_key",
    "source_patient_id",
    "source_patient_key",
    "subject_id",
    "person_id",
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
_LONGITUDINAL_CONTAINER_KEYS = _CONTAINER_KEYS + ("notes", "encounters")
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


@dataclass(frozen=True)
class LongitudinalEvidence:
    """Hashed cross-document linkage evidence for one note.

    ``value_hash`` is an HMAC digest of the normalized quasi-identifier or
    surrogate value. It intentionally does not expose the raw value.
    """

    note_index: int
    note_hash: str
    category: str
    value_hash: str
    source: str
    start: int | None = None
    end: int | None = None
    section: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "note_index": self.note_index,
            "note_hash": self.note_hash,
            "category": self.category,
            "value_hash": self.value_hash,
            "source": self.source,
            "start": self.start,
            "end": self.end,
        }
        if self.section is not None:
            payload["section"] = self.section
        return payload


@dataclass(frozen=True)
class LongitudinalNote:
    """One de-identified note inside a longitudinal corpus."""

    note_index: int
    note_hash: str
    patient_pseudonym: str
    evidence: tuple[LongitudinalEvidence, ...]
    direct_identifier_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_index": self.note_index,
            "note_hash": self.note_hash,
            "patient_pseudonym": self.patient_pseudonym,
            "direct_identifier_count": int(self.direct_identifier_count),
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class LongitudinalPatient:
    """A privacy-safe patient cluster built from hashed source keys."""

    patient_pseudonym: str
    notes: tuple[LongitudinalNote, ...]

    @property
    def document_count(self) -> int:
        return len(self.notes)

    @property
    def evidence(self) -> tuple[LongitudinalEvidence, ...]:
        return tuple(item for note in self.notes for item in note.evidence)

    @property
    def direct_identifier_count(self) -> int:
        return sum(note.direct_identifier_count for note in self.notes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_pseudonym": self.patient_pseudonym,
            "document_count": self.document_count,
            "direct_identifier_count": self.direct_identifier_count,
            "notes": [note.to_dict() for note in self.notes],
        }


@dataclass(frozen=True)
class LongitudinalCorpus:
    """Privacy-safe longitudinal corpus for cross-document risk scoring."""

    patients: tuple[LongitudinalPatient, ...]

    @property
    def patient_count(self) -> int:
        return len(self.patients)

    @property
    def document_count(self) -> int:
        return sum(patient.document_count for patient in self.patients)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_count": self.patient_count,
            "document_count": self.document_count,
            "patients": [patient.to_dict() for patient in self.patients],
        }


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
    original_records = (
        _coerce_records(original, source="original") if original is not None else []
    )
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


def build_longitudinal_corpus(
    records: Any,
    *,
    hmac_key: bytes | str = _DEFAULT_LONGITUDINAL_HMAC_KEY,
    patient_key_fields: Sequence[str] = _PATIENT_KEY_FIELDS,
) -> LongitudinalCorpus:
    """Build a privacy-safe longitudinal corpus from de-identified notes.

    Patient source keys are never stored directly: each key is converted into
    an HMAC pseudonym, and note ids plus quasi-identifier evidence values are
    stored as HMAC digests. The resulting object is safe to serialize for audit
    evidence because it contains hashes, offsets, categories, and counts only.
    """

    grouped_notes: dict[str, list[LongitudinalNote]] = {}
    note_index = 0
    for item in _flatten_longitudinal_items(records, patient_key_fields):
        mapping = item if isinstance(item, Mapping) else None
        source_patient_key = _source_patient_key(mapping, patient_key_fields)
        sanitized = (
            _strip_patient_keys(mapping, patient_key_fields) if mapping else item
        )

        for record in _coerce_records(sanitized, source="deidentified"):
            record = _Record(
                note_index,
                record.record_id,
                record.text,
                record.fields,
                record.spans,
                record.source,
            )
            source_key = source_patient_key or record.record_id or f"note:{note_index}"
            patient_pseudonym = _hmac_digest(hmac_key, f"patient:{source_key}")
            note_hash = _hmac_digest(
                hmac_key,
                f"note:{record.record_id or note_index}",
            )
            evidence = tuple(_longitudinal_evidence(record, note_hash, hmac_key))
            note = LongitudinalNote(
                note_index=note_index,
                note_hash=note_hash,
                patient_pseudonym=patient_pseudonym,
                evidence=evidence,
                direct_identifier_count=_direct_identifier_count(record),
            )
            grouped_notes.setdefault(patient_pseudonym, []).append(note)
            note_index += 1

    patients = tuple(
        LongitudinalPatient(patient_pseudonym, tuple(notes))
        for patient_pseudonym, notes in sorted(grouped_notes.items())
    )
    return LongitudinalCorpus(patients)


def longitudinal_risk_report(
    records: Any,
    *,
    hmac_key: bytes | str = _DEFAULT_LONGITUDINAL_HMAC_KEY,
    patient_key_fields: Sequence[str] = _PATIENT_KEY_FIELDS,
) -> dict[str, Any]:
    """Score same-patient linkage risk across a longitudinal note corpus.

    The report's ``linkage_success_upper_bound`` is a conservative patient-level
    upper bound: if any patient has a stable longitudinal attack fingerprint,
    the highest-risk patient bound is 1.0. The realized attack rate produced by
    :func:`openmed.eval.attacks.longitudinal_linkage_attack` is therefore never
    higher than this bound, while the per-patient breakdown explains which
    hashed features created the bound.
    """

    corpus = build_longitudinal_corpus(
        records,
        hmac_key=hmac_key,
        patient_key_fields=patient_key_fields,
    )
    patient_risks = [
        _longitudinal_patient_breakdown(patient) for patient in corpus.patients
    ]
    patient_count = corpus.patient_count
    document_count = corpus.document_count
    direct_note_count = sum(
        1
        for patient in corpus.patients
        for note in patient.notes
        if note.direct_identifier_count > 0
    )
    direct_identifier_count = sum(
        patient.direct_identifier_count for patient in corpus.patients
    )
    upper_bound = max(
        (float(patient["linkage_upper_bound"]) for patient in patient_risks),
        default=0.0,
    )
    mean_bound = _rate(
        sum(float(patient["linkage_upper_bound"]) for patient in patient_risks),
        patient_count,
    )
    linkable_patient_count = sum(
        1 for patient in patient_risks if patient["linkage_upper_bound"] > 0.0
    )

    return {
        "schema_version": 1,
        "patient_count": patient_count,
        "document_count": document_count,
        "linkage_success_upper_bound": upper_bound,
        "mean_patient_linkage_upper_bound": mean_bound,
        "linkable_patient_count": linkable_patient_count,
        "residual_direct_identifier_leakage": _rate(
            direct_note_count,
            document_count,
        ),
        "residual_direct_identifier_leakage_count": direct_identifier_count,
        "patient_risks": patient_risks,
        "high_risk_patients": [
            patient
            for patient in patient_risks
            if upper_bound > 0.0
            and float(patient["linkage_upper_bound"]) == upper_bound
        ],
    }


def longitudinal_attack_fingerprint(
    patient: LongitudinalPatient,
) -> tuple[tuple[str, str], ...]:
    """Return the hashed evidence an adversary can use to cluster notes."""

    if patient.document_count < 2:
        return ()

    evidence = patient.evidence
    fingerprint: list[tuple[str, str]] = []

    surrogate_notes: dict[str, set[int]] = {}
    for item in evidence:
        if item.category != "stable_surrogate":
            continue
        surrogate_notes.setdefault(item.value_hash, set()).add(item.note_index)
    for value_hash, note_indexes in sorted(surrogate_notes.items()):
        if len(note_indexes) >= 2:
            fingerprint.append(("stable_surrogate", value_hash))

    age_notes = {item.note_index for item in evidence if item.category == "age"}
    if len(age_notes) >= 2:
        for value_hash in sorted(
            {item.value_hash for item in evidence if item.category == "age"}
        ):
            fingerprint.append(("age_trajectory", value_hash))

    rare_hashes = sorted(
        {item.value_hash for item in evidence if item.category == "rare_condition"}
    )
    for value_hash in rare_hashes:
        fingerprint.append(("rare_attribute", value_hash))
    if len(rare_hashes) >= 2:
        fingerprint.append(
            ("rare_attribute_cooccurrence", _fingerprint_hash(rare_hashes))
        )

    return tuple(fingerprint)


def _flatten_longitudinal_items(
    data: Any,
    patient_key_fields: Sequence[str],
    inherited_patient_key: str | None = None,
) -> list[Any]:
    dataframe_records = _maybe_dataframe_records(data)
    if dataframe_records is not None:
        data = dataframe_records

    if isinstance(data, Mapping):
        source_patient_key = (
            _source_patient_key(data, patient_key_fields) or inherited_patient_key
        )
        container = _first_longitudinal_container(data)
        if container is not None and not _looks_like_single_record(data):
            return _flatten_longitudinal_items(
                container,
                patient_key_fields,
                source_patient_key,
            )
        if (
            source_patient_key is not None
            and patient_key_fields
            and not any(key in data for key in patient_key_fields)
        ):
            item = dict(data)
            item[str(patient_key_fields[0])] = source_patient_key
            return [item]
        return [data]

    if _is_sequence(data):
        items: list[Any] = []
        for item in data:
            items.extend(
                _flatten_longitudinal_items(
                    item,
                    patient_key_fields,
                    inherited_patient_key,
                )
            )
        return items

    return [data]


def _first_longitudinal_container(data: Mapping[str, Any]) -> Any | None:
    for key in _LONGITUDINAL_CONTAINER_KEYS:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _source_patient_key(
    data: Mapping[str, Any] | None,
    patient_key_fields: Sequence[str],
) -> str | None:
    if data is None:
        return None
    for key in patient_key_fields:
        value = data.get(key)
        if value is not None:
            return str(value)
    return None


def _strip_patient_keys(
    data: Mapping[str, Any],
    patient_key_fields: Sequence[str],
) -> dict[str, Any]:
    blocked = set(patient_key_fields)
    sanitized = {key: value for key, value in data.items() if key not in blocked}
    audit_spans = data.get("audit_spans")
    if audit_spans is None:
        return sanitized

    existing_spans: list[Mapping[str, Any]] = []
    for key in _SPAN_KEYS:
        existing_spans.extend(_coerce_spans(sanitized.get(key)))
    audit_span_items = list(_coerce_spans(audit_spans))
    if audit_span_items:
        sanitized["spans"] = [*existing_spans, *audit_span_items]
    return sanitized


def _longitudinal_evidence(
    record: _Record,
    note_hash: str,
    hmac_key: bytes | str,
) -> list[LongitudinalEvidence]:
    evidence: list[LongitudinalEvidence] = []
    for qi in _profile_record(record).quasi_identifiers:
        evidence.append(
            LongitudinalEvidence(
                note_index=record.index,
                note_hash=note_hash,
                category=qi.category,
                value_hash=_hmac_digest(
                    hmac_key,
                    f"qi:{qi.category}:{qi.normalized_value}",
                ),
                source=f"quasi_identifier:{qi.source}",
                start=qi.start,
                end=qi.end,
                section=qi.section,
            )
        )
    evidence.extend(_stable_surrogate_evidence(record, note_hash, hmac_key))
    return _dedupe_longitudinal_evidence(evidence)


def _stable_surrogate_evidence(
    record: _Record,
    note_hash: str,
    hmac_key: bytes | str,
) -> list[LongitudinalEvidence]:
    evidence: list[LongitudinalEvidence] = []
    for span in record.spans:
        surrogate = span.get("surrogate")
        if surrogate is None:
            continue
        normalized = _normalize_qi_value("surrogate", surrogate)
        if not normalized or _is_generic_placeholder(surrogate):
            continue
        evidence.append(
            LongitudinalEvidence(
                note_index=record.index,
                note_hash=note_hash,
                category="stable_surrogate",
                value_hash=_hmac_digest(
                    hmac_key,
                    f"surrogate:{_span_label(span)}:{normalized}",
                ),
                source="surrogate:span",
                start=_optional_int(span.get("start")),
                end=_optional_int(span.get("end")),
                section=_span_section(span),
            )
        )

    for name, value in record.fields.items():
        if "surrogate" not in _name_key(name) or value is None:
            continue
        normalized = _normalize_qi_value("surrogate", value)
        if not normalized or _is_generic_placeholder(value):
            continue
        evidence.append(
            LongitudinalEvidence(
                note_index=record.index,
                note_hash=note_hash,
                category="stable_surrogate",
                value_hash=_hmac_digest(hmac_key, f"surrogate:{name}:{normalized}"),
                source="surrogate:field",
            )
        )
    return evidence


def _dedupe_longitudinal_evidence(
    evidence: list[LongitudinalEvidence],
) -> list[LongitudinalEvidence]:
    seen: set[tuple[str, str, int, int | None, int | None]] = set()
    deduped: list[LongitudinalEvidence] = []
    for item in evidence:
        key = (
            item.category,
            item.value_hash,
            item.note_index,
            item.start,
            item.end,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _direct_identifier_count(record: _Record) -> int:
    return sum(
        1
        for value in _direct_identifier_values(record)
        if value and not _is_generic_placeholder(value)
    )


def _longitudinal_patient_breakdown(
    patient: LongitudinalPatient,
) -> dict[str, Any]:
    evidence = patient.evidence
    by_category = Counter(item.category for item in evidence)
    unique_hashes_by_category: dict[str, set[str]] = {}
    for item in evidence:
        unique_hashes_by_category.setdefault(item.category, set()).add(item.value_hash)

    surrogate_reuse = _reused_value_count(evidence, "stable_surrogate")
    age_note_count = len(
        {item.note_index for item in evidence if item.category == "age"}
    )
    rare_attribute_count = len(unique_hashes_by_category.get("rare_condition", set()))
    fingerprint = longitudinal_attack_fingerprint(patient)
    upper_bound = 1.0 if fingerprint else 0.0

    return {
        "patient_pseudonym": patient.patient_pseudonym,
        "document_count": patient.document_count,
        "evidence_count": len(evidence),
        "direct_identifier_count": patient.direct_identifier_count,
        "linkage_upper_bound": upper_bound,
        "stable_surrogate_reuse_count": surrogate_reuse,
        "age_observation_count": age_note_count,
        "rare_attribute_count": rare_attribute_count,
        "categories": dict(sorted(by_category.items())),
        "attack_fingerprint": [
            {"category": category, "value_hash": value_hash}
            for category, value_hash in fingerprint
        ],
        "evidence": [item.to_dict() for item in evidence],
    }


def _reused_value_count(
    evidence: tuple[LongitudinalEvidence, ...],
    category: str,
) -> int:
    note_indexes: dict[str, set[int]] = {}
    for item in evidence:
        if item.category == category:
            note_indexes.setdefault(item.value_hash, set()).add(item.note_index)
    return sum(1 for indexes in note_indexes.values() if len(indexes) >= 2)


def _fingerprint_hash(values: Sequence[str]) -> str:
    payload = "\0".join(sorted(values)).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _hmac_digest(key: bytes | str, value: str) -> str:
    key_bytes = key if isinstance(key, bytes) else str(key).encode("utf-8")
    digest = hmac.new(key_bytes, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"hmac-sha256:{digest}"


def _rate(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


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
            _Record(
                index,
                record.record_id,
                record.text,
                record.fields,
                record.spans,
                record.source,
            )
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
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


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
    if any(
        hint in normalized
        for hint in (
            "city",
            "state",
            "county",
            "zip",
            "postal",
            "location",
            "geography",
        )
    ):
        return "geography"
    if any(
        hint in normalized
        for hint in (
            "provider",
            "doctor",
            "physician",
            "hospital",
            "clinic",
            "facility",
            "institution",
        )
    ):
        return "provider_institution"
    if any(
        hint in normalized for hint in ("rare", "condition", "diagnosis", "disease")
    ):
        return "rare_condition"
    return None


def _span_label(span: Mapping[str, Any]) -> str:
    for key in (
        "canonical_label",
        "policy_label",
        "label",
        "entity_group",
        "entity",
        "entity_type",
        "type",
    ):
        value = span.get(key)
        if value:
            return str(value)
    return ""


def _span_value(
    record: _Record, span: Mapping[str, Any]
) -> tuple[str, int | None, int | None]:
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
    if any(
        hint in normalized
        for hint in ("date", "dob", "birth", "visit", "admission", "discharge")
    ):
        return "date"
    if any(
        hint in normalized
        for hint in (
            "city",
            "state",
            "county",
            "zip",
            "postal",
            "country",
            "region",
            "location",
            "geography",
        )
    ):
        return "geography"
    if any(
        hint in normalized
        for hint in (
            "provider",
            "doctor",
            "physician",
            "hospital",
            "clinic",
            "facility",
            "institution",
            "organization",
        )
    ):
        return "provider_institution"
    if any(
        hint in normalized for hint in ("rare", "condition", "diagnosis", "disease")
    ):
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


def _profile_key(
    qis: list[_QuasiIdentifier],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
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

    original_counts = Counter(
        profile.key for profile in original_profiles if profile.key
    )
    aux_counts = Counter(profile.key for profile in aux_profiles if profile.key)

    linked = 0
    for profile in deidentified_profiles:
        if not profile.key:
            continue
        if original_counts[profile.key] == 1 or aux_counts[profile.key] == 1:
            linked += 1
    return linked / len(deidentified_profiles)


def _leakage_rate(
    deidentified_records: list[_Record], original_records: list[_Record]
) -> float:
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

        blob = _normalize_direct_value(
            " ".join([record.text, *map(str, record.fields.values())])
        )
        if original_values and any(
            value in blob for value in original_values if len(value) > 1
        ):
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
    if any(
        safe_hint in normalized
        for safe_hint in ("date", "age", "diagnosis", "condition")
    ):
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


__all__ = [
    "LongitudinalCorpus",
    "LongitudinalEvidence",
    "LongitudinalNote",
    "LongitudinalPatient",
    "build_longitudinal_corpus",
    "longitudinal_attack_fingerprint",
    "longitudinal_risk_report",
    "risk_report",
]
