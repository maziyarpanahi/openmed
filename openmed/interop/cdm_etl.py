"""Bidirectional ETL primitives for clinical notes and CDM-style tables.

This first slice covers deterministic note-to-CDM extraction. It accepts
synthetic or caller-supplied clinical entities, resolves concepts only from an
optional user-provided Athena vocabulary index, and emits aggregate summaries
that do not include note text or source identifiers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .athena import AthenaVocabularyIndex, ConceptRecord

UNMAPPED_CONCEPT_ID = 0
UNMAPPED_CONCEPT_NAME = "Unmapped concept"
UNMAPPED_VOCABULARY_ID = "UNMAPPED"

CdmDomain = Literal["condition", "drug", "measurement"]

_MISSING = object()

_ATHENA_DOMAIN_BY_CDM_DOMAIN: dict[CdmDomain, str] = {
    "condition": "Condition",
    "drug": "Drug",
    "measurement": "Measurement",
}

_DOMAIN_ALIASES: dict[str, CdmDomain] = {
    "condition": "condition",
    "problem": "condition",
    "diagnosis": "condition",
    "disease": "condition",
    "disorder": "condition",
    "symptom": "condition",
    "drug": "drug",
    "medication": "drug",
    "medicine": "drug",
    "rx": "drug",
    "treatment": "drug",
    "measurement": "measurement",
    "observation": "measurement",
    "lab": "measurement",
    "lab_value": "measurement",
    "laboratory": "measurement",
    "vital": "measurement",
    "vital_sign": "measurement",
}

_TEXT_FIELDS = (
    "normalized_text",
    "text",
    "entity_text",
    "word",
    "surface",
    "source_value",
)
_LABEL_FIELDS = ("domain", "entity_label", "label", "entity_type", "entity_group")
_CODE_FIELDS = ("concept_code", "code", "code_value", "coding_code")
_VOCABULARY_FIELDS = (
    "vocabulary_id",
    "source_vocabulary_id",
    "system",
    "code_system",
    "coding_system",
)
_ENTITY_ID_FIELDS = ("source_entity_id", "entity_id", "id", "span_id")
_METADATA_FIELDS = ("metadata", "meta")
_CODING_FIELDS = ("coding", "codings", "code", "codeable_concept")


@dataclass(frozen=True)
class ClinicalEntity:
    """Clinical entity span accepted by the note-to-CDM transform."""

    label: str
    text: str
    start: int | None = None
    end: int | None = None
    source_entity_id: str | None = None
    code: str | None = None
    vocabulary_id: str | None = None
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class ClinicalNote:
    """Source note wrapper for deterministic CDM extraction."""

    document_id: str
    patient_id: str
    entities: Sequence[Any] = field(default_factory=tuple)
    visit_id: str | None = None
    document_version: str | int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class ConceptMapping:
    """Resolved CDM concept metadata for one source entity."""

    concept_id: int
    concept_name: str
    vocabulary_id: str
    concept_code: str
    mapped: bool
    source_value: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class PersonRow:
    """CDM-style ``person`` row with a deterministic surrogate key."""

    person_id: int
    person_source_value: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class VisitOccurrenceRow:
    """CDM-style ``visit_occurrence`` row with a deterministic key."""

    visit_occurrence_id: int
    person_id: int
    visit_source_value: str
    source_document_id: str
    source_document_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class ConditionOccurrenceRow:
    """CDM-style ``condition_occurrence`` row."""

    condition_occurrence_id: int
    person_id: int
    visit_occurrence_id: int
    condition_concept_id: int
    condition_source_value: str
    condition_source_concept_id: int
    source_vocabulary_id: str
    source_concept_code: str
    source_document_id: str
    source_document_version: str
    source_entity_id: str
    start: int | None = None
    end: int | None = None
    mapped: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class DrugExposureRow:
    """CDM-style ``drug_exposure`` row."""

    drug_exposure_id: int
    person_id: int
    visit_occurrence_id: int
    drug_concept_id: int
    drug_source_value: str
    drug_source_concept_id: int
    source_vocabulary_id: str
    source_concept_code: str
    source_document_id: str
    source_document_version: str
    source_entity_id: str
    start: int | None = None
    end: int | None = None
    mapped: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class MeasurementRow:
    """CDM-style ``measurement`` row."""

    measurement_id: int
    person_id: int
    visit_occurrence_id: int
    measurement_concept_id: int
    measurement_source_value: str
    measurement_source_concept_id: int
    source_vocabulary_id: str
    source_concept_code: str
    source_document_id: str
    source_document_version: str
    source_entity_id: str
    start: int | None = None
    end: int | None = None
    mapped: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class CdmEtlSummary:
    """PHI-free aggregate summary for a note-to-CDM run."""

    row_counts: Mapping[str, int]
    concept_counts: Mapping[str, Mapping[str, int]]
    concept_rates: Mapping[str, Mapping[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable aggregate summary."""

        return {
            "row_counts": dict(self.row_counts),
            "concept_counts": {
                domain: dict(counts) for domain, counts in self.concept_counts.items()
            },
            "concept_rates": {
                domain: dict(rates) for domain, rates in self.concept_rates.items()
            },
        }


@dataclass(frozen=True)
class CdmTables:
    """In-memory CDM-style tables emitted by note-to-CDM extraction."""

    person: tuple[PersonRow, ...]
    visit_occurrence: tuple[VisitOccurrenceRow, ...]
    condition_occurrence: tuple[ConditionOccurrenceRow, ...]
    drug_exposure: tuple[DrugExposureRow, ...]
    measurement: tuple[MeasurementRow, ...]
    summary: CdmEtlSummary

    def to_dict(self) -> dict[str, Any]:
        """Return all tables and the aggregate summary as plain mappings."""

        return {
            "person": [row.to_dict() for row in self.person],
            "visit_occurrence": [row.to_dict() for row in self.visit_occurrence],
            "condition_occurrence": [
                row.to_dict() for row in self.condition_occurrence
            ],
            "drug_exposure": [row.to_dict() for row in self.drug_exposure],
            "measurement": [row.to_dict() for row in self.measurement],
            "summary": self.summary.to_dict(),
        }


class AthenaConceptResolver:
    """Resolve concepts by exact code or alias from a user-supplied index."""

    def __init__(self, vocabulary_index: AthenaVocabularyIndex | None = None) -> None:
        self._by_code: dict[tuple[str, str], ConceptRecord] = {}
        self._by_any_code: dict[str, list[ConceptRecord]] = {}
        self._by_alias: dict[str, list[ConceptRecord]] = {}
        if vocabulary_index:
            self._index(vocabulary_index)

    @property
    def has_vocabulary(self) -> bool:
        """Return whether this resolver has user-supplied vocabulary records."""

        return bool(self._by_code or self._by_alias)

    def resolve(self, entity: Any, *, domain: CdmDomain) -> ConceptMapping:
        """Resolve one source entity to a concept mapping.

        Resolution is deliberately deterministic and conservative: exact
        vocabulary/code match first, exact code across loaded vocabularies
        second, and exact alias/name match last. No code is invented when the
        vocabulary is absent or a lookup misses.
        """

        source_value = _entity_text(entity)
        if not self.has_vocabulary:
            return _unmapped(source_value)

        vocabulary_id = _entity_vocabulary_id(entity)
        concept_code = _entity_code(entity)
        if concept_code:
            record = self._record_by_code(vocabulary_id, concept_code, domain)
            if record is not None:
                return _mapped(record, source_value)

        if source_value:
            record = self._record_by_alias(source_value, domain)
            if record is not None:
                return _mapped(record, source_value)

        return _unmapped(source_value)

    def _index(self, vocabulary_index: AthenaVocabularyIndex) -> None:
        for vocabulary_id, concepts in vocabulary_index.items():
            if vocabulary_id == "_meta" or not isinstance(concepts, Mapping):
                continue
            for concept_code, record in concepts.items():
                if not isinstance(record, Mapping):
                    continue
                typed_record = dict(record)
                vocab_key = _norm(vocabulary_id)
                code_key = _norm(concept_code)
                self._by_code[(vocab_key, code_key)] = typed_record
                self._by_any_code.setdefault(code_key, []).append(typed_record)

                aliases = typed_record.get("aliases") or (
                    typed_record.get("concept_name"),
                )
                for alias in _as_sequence(aliases):
                    alias_key = _norm(alias)
                    if alias_key:
                        self._by_alias.setdefault(alias_key, []).append(typed_record)

        for records in (*self._by_any_code.values(), *self._by_alias.values()):
            records.sort(key=_concept_sort_key)

    def _record_by_code(
        self,
        vocabulary_id: str,
        concept_code: str,
        domain: CdmDomain,
    ) -> ConceptRecord | None:
        if vocabulary_id:
            record = self._by_code.get((_norm(vocabulary_id), _norm(concept_code)))
            if record is not None and _record_matches_domain(record, domain):
                return record
            return None

        return _first_domain_record(
            self._by_any_code.get(_norm(concept_code), ()), domain
        )

    def _record_by_alias(
        self,
        source_value: str,
        domain: CdmDomain,
    ) -> ConceptRecord | None:
        return _first_domain_record(self._by_alias.get(_norm(source_value), ()), domain)


def notes_to_cdm(
    notes: Iterable[Any],
    *,
    vocabulary_index: AthenaVocabularyIndex | None = None,
    concept_resolver: AthenaConceptResolver | None = None,
    quality_floor: float | None = None,
) -> CdmTables:
    """Transform clinical note entities into deterministic CDM-style tables.

    Args:
        notes: Iterable of :class:`ClinicalNote` instances, note mappings, or
            simple objects with ``document_id``, ``patient_id``, and
            ``entities`` fields.
        vocabulary_index: Optional index returned by
            :func:`openmed.interop.athena.load_athena_vocab`. When omitted,
            every clinical entity maps to the placeholder concept.
        concept_resolver: Optional resolver instance for callers that cache
            vocabulary indexes across runs.
        quality_floor: Optional completeness floor. When set, the extracted
            batch is profiled before ETL and low-completeness batches are
            rejected without building CDM rows.

    Returns:
        CDM-style tables plus an aggregate, PHI-free summary.
    """

    if quality_floor is not None:
        records = tuple(notes)
        from openmed.structured.quality import (
            assert_profile_gate,
            profile_extracted_batch,
        )

        assert_profile_gate(
            profile_extracted_batch(
                records,
                completeness_floor=quality_floor,
                athena_index=vocabulary_index,
            )
        )
        notes = records

    resolver = concept_resolver or AthenaConceptResolver(vocabulary_index)
    people: dict[int, PersonRow] = {}
    visits: dict[int, VisitOccurrenceRow] = {}
    conditions: list[ConditionOccurrenceRow] = []
    drugs: list[DrugExposureRow] = []
    measurements: list[MeasurementRow] = []

    for note_index, note in enumerate(notes):
        document_id = _required_text(
            _first_value((note,), ("document_id", "doc_id", "source_document_id")),
            f"note at index {note_index} is missing document_id",
        )
        patient_id = _required_text(
            _first_value((note,), ("patient_id", "person_source_value", "subject_id")),
            f"note {document_id!r} is missing patient_id",
        )
        visit_id = _first_text(
            (note,), ("visit_id", "encounter_id", "visit_source_value")
        )
        if not visit_id:
            visit_id = f"document:{document_id}"
        document_version = _first_text(
            (note,),
            ("document_version", "version", "source_document_version"),
        )

        person_id = deterministic_surrogate_key("person", patient_id)
        visit_occurrence_id = deterministic_surrogate_key(
            "visit",
            patient_id,
            visit_id,
        )
        people.setdefault(
            person_id,
            PersonRow(person_id=person_id, person_source_value=patient_id),
        )
        visits.setdefault(
            visit_occurrence_id,
            VisitOccurrenceRow(
                visit_occurrence_id=visit_occurrence_id,
                person_id=person_id,
                visit_source_value=visit_id,
                source_document_id=document_id,
                source_document_version=document_version,
            ),
        )

        for entity_index, entity in enumerate(_note_entities(note)):
            domain = _entity_domain(entity)
            if domain is None:
                continue

            source_entity_id = _source_entity_id(
                entity,
                document_id=document_id,
                domain=domain,
                index=entity_index,
            )
            concept = resolver.resolve(entity, domain=domain)
            start = _optional_int(_first_value(_entity_sources(entity), ("start",)))
            end = _optional_int(_first_value(_entity_sources(entity), ("end",)))
            row_id = deterministic_surrogate_key(
                domain,
                document_id,
                source_entity_id,
            )

            if domain == "condition":
                conditions.append(
                    ConditionOccurrenceRow(
                        condition_occurrence_id=row_id,
                        person_id=person_id,
                        visit_occurrence_id=visit_occurrence_id,
                        condition_concept_id=concept.concept_id,
                        condition_source_value=concept.source_value,
                        condition_source_concept_id=concept.concept_id,
                        source_vocabulary_id=concept.vocabulary_id,
                        source_concept_code=concept.concept_code,
                        source_document_id=document_id,
                        source_document_version=document_version,
                        source_entity_id=source_entity_id,
                        start=start,
                        end=end,
                        mapped=concept.mapped,
                    )
                )
            elif domain == "drug":
                drugs.append(
                    DrugExposureRow(
                        drug_exposure_id=row_id,
                        person_id=person_id,
                        visit_occurrence_id=visit_occurrence_id,
                        drug_concept_id=concept.concept_id,
                        drug_source_value=concept.source_value,
                        drug_source_concept_id=concept.concept_id,
                        source_vocabulary_id=concept.vocabulary_id,
                        source_concept_code=concept.concept_code,
                        source_document_id=document_id,
                        source_document_version=document_version,
                        source_entity_id=source_entity_id,
                        start=start,
                        end=end,
                        mapped=concept.mapped,
                    )
                )
            elif domain == "measurement":
                measurements.append(
                    MeasurementRow(
                        measurement_id=row_id,
                        person_id=person_id,
                        visit_occurrence_id=visit_occurrence_id,
                        measurement_concept_id=concept.concept_id,
                        measurement_source_value=concept.source_value,
                        measurement_source_concept_id=concept.concept_id,
                        source_vocabulary_id=concept.vocabulary_id,
                        source_concept_code=concept.concept_code,
                        source_document_id=document_id,
                        source_document_version=document_version,
                        source_entity_id=source_entity_id,
                        start=start,
                        end=end,
                        mapped=concept.mapped,
                    )
                )

    person_rows = tuple(sorted(people.values(), key=lambda row: row.person_id))
    visit_rows = tuple(sorted(visits.values(), key=lambda row: row.visit_occurrence_id))
    condition_rows = tuple(
        sorted(conditions, key=lambda row: row.condition_occurrence_id)
    )
    drug_rows = tuple(sorted(drugs, key=lambda row: row.drug_exposure_id))
    measurement_rows = tuple(sorted(measurements, key=lambda row: row.measurement_id))

    return CdmTables(
        person=person_rows,
        visit_occurrence=visit_rows,
        condition_occurrence=condition_rows,
        drug_exposure=drug_rows,
        measurement=measurement_rows,
        summary=_summary(
            person_rows=person_rows,
            visit_rows=visit_rows,
            condition_rows=condition_rows,
            drug_rows=drug_rows,
            measurement_rows=measurement_rows,
        ),
    )


def deterministic_surrogate_key(*parts: Any) -> int:
    """Return a stable positive integer key derived from source identifiers."""

    if not parts:
        raise ValueError("at least one key part is required")
    canonical = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


def _summary(
    *,
    person_rows: Sequence[PersonRow],
    visit_rows: Sequence[VisitOccurrenceRow],
    condition_rows: Sequence[ConditionOccurrenceRow],
    drug_rows: Sequence[DrugExposureRow],
    measurement_rows: Sequence[MeasurementRow],
) -> CdmEtlSummary:
    row_counts = {
        "person": len(person_rows),
        "visit_occurrence": len(visit_rows),
        "condition_occurrence": len(condition_rows),
        "drug_exposure": len(drug_rows),
        "measurement": len(measurement_rows),
    }

    concept_counts: dict[str, dict[str, int]] = {}
    concept_rates: dict[str, dict[str, float]] = {}
    for domain, rows in (
        ("condition_occurrence", condition_rows),
        ("drug_exposure", drug_rows),
        ("measurement", measurement_rows),
    ):
        mapped = sum(1 for row in rows if row.mapped)
        unmapped = len(rows) - mapped
        concept_counts[domain] = {"mapped": mapped, "unmapped": unmapped}
        if rows:
            concept_rates[domain] = {
                "mapped": mapped / len(rows),
                "unmapped": unmapped / len(rows),
            }
        else:
            concept_rates[domain] = {"mapped": 0.0, "unmapped": 0.0}

    return CdmEtlSummary(
        row_counts=row_counts,
        concept_counts=concept_counts,
        concept_rates=concept_rates,
    )


def _mapped(record: Mapping[str, Any], source_value: str) -> ConceptMapping:
    return ConceptMapping(
        concept_id=int(record.get("concept_id") or UNMAPPED_CONCEPT_ID),
        concept_name=str(record.get("concept_name") or ""),
        vocabulary_id=str(record.get("vocabulary_id") or ""),
        concept_code=str(record.get("concept_code") or ""),
        mapped=True,
        source_value=source_value,
    )


def _unmapped(source_value: str) -> ConceptMapping:
    return ConceptMapping(
        concept_id=UNMAPPED_CONCEPT_ID,
        concept_name=UNMAPPED_CONCEPT_NAME,
        vocabulary_id=UNMAPPED_VOCABULARY_ID,
        concept_code="",
        mapped=False,
        source_value=source_value,
    )


def _note_entities(note: Any) -> tuple[Any, ...]:
    entities = _first_value((note,), ("entities", "clinical_entities", "spans"))
    if entities is _MISSING or entities is None:
        return ()
    return tuple(_as_sequence(entities))


def _entity_domain(entity: Any) -> CdmDomain | None:
    label = _first_text(_entity_sources(entity), _LABEL_FIELDS)
    normalized = _norm(label).replace("-", "_")
    return _DOMAIN_ALIASES.get(normalized)


def _source_entity_id(
    entity: Any,
    *,
    document_id: str,
    domain: CdmDomain,
    index: int,
) -> str:
    explicit = _first_text(_entity_sources(entity), _ENTITY_ID_FIELDS)
    if explicit:
        return explicit

    start = _first_text(_entity_sources(entity), ("start",))
    end = _first_text(_entity_sources(entity), ("end",))
    text_hash = hashlib.sha256(_entity_text(entity).encode("utf-8")).hexdigest()[:12]
    return f"{document_id}:{domain}:{start}:{end}:{index}:{text_hash}"


def _entity_text(entity: Any) -> str:
    return _first_text(_entity_sources(entity), _TEXT_FIELDS)


def _entity_code(entity: Any) -> str:
    return _first_text(_entity_and_coding_sources(entity), _CODE_FIELDS)


def _entity_vocabulary_id(entity: Any) -> str:
    vocabulary_id = _first_text(_entity_and_coding_sources(entity), _VOCABULARY_FIELDS)
    if vocabulary_id.startswith("http://") or vocabulary_id.startswith("https://"):
        return vocabulary_id.rstrip("/").rsplit("/", maxsplit=1)[-1]
    return vocabulary_id


def _entity_sources(entity: Any) -> tuple[Any, ...]:
    sources: list[Any] = [entity]
    for source in tuple(sources):
        for name in _METADATA_FIELDS:
            metadata = _value(source, name)
            if metadata is not _MISSING and metadata is not None:
                sources.append(metadata)
    return tuple(sources)


def _entity_and_coding_sources(entity: Any) -> tuple[Any, ...]:
    sources = list(_entity_sources(entity))
    for source in tuple(sources):
        for name in _CODING_FIELDS:
            coding = _coerce_coding(_value(source, name))
            if coding is not None:
                sources.insert(0, coding)
    return tuple(sources)


def _coerce_coding(value: Any) -> Mapping[str, Any] | None:
    if value is _MISSING or value is None or isinstance(value, (str, bytes)):
        return None
    if isinstance(value, Mapping):
        nested = _coerce_coding(_value(value, "coding"))
        if nested is not None:
            return nested
        return value
    if isinstance(value, Sequence):
        for item in value:
            coding = _coerce_coding(item)
            if coding is not None:
                return coding
    return None


def _first_value(sources: Iterable[Any], names: Iterable[str]) -> Any:
    for source in sources:
        for name in names:
            value = _value(source, name)
            if value is not _MISSING and value is not None:
                return value
    return _MISSING


def _first_text(sources: Iterable[Any], names: Iterable[str]) -> str:
    value = _first_value(sources, names)
    if value is _MISSING or value is None:
        return ""
    return str(value).strip()


def _required_text(value: Any, message: str) -> str:
    if value is _MISSING or value is None or not str(value).strip():
        raise ValueError(message)
    return str(value).strip()


def _value(item: Any, name: str) -> Any:
    if isinstance(item, Mapping) and name in item:
        return item[name]
    if hasattr(item, name):
        return getattr(item, name)
    return _MISSING


def _as_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return value
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _optional_int(value: Any) -> int | None:
    if value is _MISSING or value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def _concept_sort_key(record: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(record.get("vocabulary_id") or ""),
        int(record.get("concept_id") or 0),
        str(record.get("concept_code") or ""),
    )


def _record_matches_domain(record: Mapping[str, Any], domain: CdmDomain) -> bool:
    record_domain = str(record.get("domain_id") or "")
    expected = _ATHENA_DOMAIN_BY_CDM_DOMAIN[domain]
    return not record_domain or record_domain.casefold() == expected.casefold()


def _first_domain_record(
    records: Iterable[Mapping[str, Any]],
    domain: CdmDomain,
) -> ConceptRecord | None:
    for record in records:
        if _record_matches_domain(record, domain):
            return dict(record)
    return None


__all__ = [
    "AthenaConceptResolver",
    "CdmDomain",
    "CdmEtlSummary",
    "CdmTables",
    "ClinicalEntity",
    "ClinicalNote",
    "ConceptMapping",
    "ConditionOccurrenceRow",
    "DrugExposureRow",
    "MeasurementRow",
    "PersonRow",
    "UNMAPPED_CONCEPT_ID",
    "UNMAPPED_CONCEPT_NAME",
    "UNMAPPED_VOCABULARY_ID",
    "VisitOccurrenceRow",
    "deterministic_surrogate_key",
    "notes_to_cdm",
]
