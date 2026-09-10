"""Local-first FHIR R4 to OMOP CDM v5.4 conformance helpers.

The bridge intentionally supports a small, explicit FHIR subset.  It accepts
only caller-supplied vocabulary mappings, keeps source identifiers as hashes,
and emits the existing in-memory OMOP tables together with a de-identified
provenance sidecar.  No terminology package is bundled and a FHIR coding is
never treated as an OMOP concept identifier by inference.

The sidecar is deliberately separate from the existing OMOP loader schema:
the standard CDM rows remain usable by existing writers while the
``FhirOmopTables`` wrapper carries the FHIR element path, coded source value,
vocabulary snapshot, and NOTE_NLP offset required for conformance inspection.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from .omop.cdm_loader import (
    UNMAPPED_CONCEPT_ID,
    UNMAPPED_CONCEPT_NAME,
    UNMAPPED_VOCABULARY_ID,
    OmopCdmTables,
    OmopLoadSummary,
    deterministic_omop_id,
    load_grounded_notes,
    validate_omop_tables,
    write_omop_sqlite,
)

__all__ = [
    "FABRICATED_CONCEPT_FIELDS",
    "FHIR_RESOURCE_TYPES",
    "FhirOmopConformanceError",
    "FhirOmopConformanceReport",
    "FhirOmopError",
    "FhirOmopInformationLoss",
    "FhirOmopLoadSummary",
    "FhirOmopMappingError",
    "FhirOmopTables",
    "FhirReverseInspection",
    "SUPPORTED_FHIR_ELEMENTS",
    "UNSUPPORTED_FHIR_ELEMENTS",
    "assert_fhir_omop_conformant",
    "build_conformance_report",
    "fhir_to_omop",
    "inspect_fhir_from_omop",
    "load_fhir_bundle",
    "load_fhir_to_omop",
    "reverse_inspect_fhir",
    "run_fhir_omop_conformance",
    "write_fhir_omop_sqlite",
]

VocabularyResolver: TypeAlias = Callable[[str, str], Any]
VocabularyInput: TypeAlias = Mapping[Any, Any] | VocabularyResolver | Any
LoadMode = Literal["append"]

FHIR_RESOURCE_TYPES: tuple[str, ...] = (
    "Patient",
    "Encounter",
    "Condition",
    "Observation",
    "Procedure",
    "MedicationStatement",
    "MedicationRequest",
)

_RESOURCE_DOMAIN: Mapping[str, str | None] = {
    "Patient": None,
    "Encounter": None,
    "Condition": "Condition",
    "Observation": "Measurement",
    "Procedure": "Procedure",
    "MedicationStatement": "Drug",
    "MedicationRequest": "Drug",
}

_RESOURCE_CODE_PATH: Mapping[str, str | None] = {
    "Patient": None,
    "Encounter": "Encounter.class",
    "Condition": "Condition.code",
    "Observation": "Observation.code",
    "Procedure": "Procedure.code",
    "MedicationStatement": "MedicationStatement.medicationCodeableConcept",
    "MedicationRequest": "MedicationRequest.medicationCodeableConcept",
}

SUPPORTED_FHIR_ELEMENTS: Mapping[str, tuple[str, ...]] = {
    "Patient": ("resourceType", "id"),
    "Encounter": (
        "resourceType",
        "id",
        "subject.reference",
        "status",
        "class",
        "period.start",
        "period.end",
    ),
    "Condition": (
        "resourceType",
        "id",
        "subject.reference",
        "encounter.reference",
        "clinicalStatus",
        "verificationStatus",
        "code",
        "onsetDateTime",
        "recordedDate",
    ),
    "Observation": (
        "resourceType",
        "id",
        "subject.reference",
        "encounter.reference",
        "status",
        "code",
        "effectiveDateTime",
        "valueQuantity",
        "valueInteger",
        "valueDecimal",
        "valueBoolean",
        "valueDateTime",
        "valueCodeableConcept",
    ),
    "Procedure": (
        "resourceType",
        "id",
        "subject.reference",
        "encounter.reference",
        "status",
        "code",
        "performedDateTime",
    ),
    "MedicationStatement": (
        "resourceType",
        "id",
        "subject.reference",
        "encounter.reference",
        "status",
        "medicationCodeableConcept",
        "effectiveDateTime",
        "dateAsserted",
    ),
    "MedicationRequest": (
        "resourceType",
        "id",
        "subject.reference",
        "encounter.reference",
        "status",
        "intent",
        "medicationCodeableConcept",
        "authoredOn",
    ),
}

UNSUPPORTED_FHIR_ELEMENTS: Mapping[str, tuple[str, ...]] = {
    "Patient": ("name", "identifier", "telecom", "gender", "birthDate", "address"),
    "Encounter": ("participant", "location", "serviceProvider", "diagnosis"),
    "Condition": ("abatement[x]", "recorder", "asserter", "evidence", "note"),
    "Observation": (
        "category",
        "performer",
        "interpretation",
        "bodySite",
        "method",
        "specimen",
        "device",
        "referenceRange",
        "hasMember",
        "derivedFrom",
        "component",
        "valueString",
    ),
    "Procedure": ("performedPeriod", "performer", "location", "reason", "outcome"),
    "MedicationStatement": (
        "effectivePeriod",
        "informationSource",
        "dosage",
        "reasonCode",
        "note",
    ),
    "MedicationRequest": (
        "requester",
        "dosageInstruction",
        "dispenseRequest",
        "substitution",
        "priorPrescription",
        "reasonCode",
        "note",
    ),
}

FABRICATED_CONCEPT_FIELDS: frozenset[str] = frozenset(
    {
        "concept_id",
        "source_concept_id",
        "target_concept_id",
        "standard_concept_id",
        "omop_concept_id",
        "omopconceptid",
    }
)

_UNMAPPED_NO_MAPPING = "no_user_supplied_mapping"
_UNMAPPED_MISSING_CODE = "missing_source_code"
_UNMAPPED_DECLARED = "mapping_declared_unmapped"
_UNMAPPED_INVALID = "invalid_user_mapping"
_FHIR_SOURCE_TABLES = (
    "fhir_resource",
    "fhir_code",
    "fhir_value",
    "fhir_provenance",
    "fhir_information_loss",
)


class FhirOmopError(ValueError):
    """Base error for invalid or non-conformant FHIR-to-OMOP input."""


class FhirOmopMappingError(FhirOmopError):
    """Raised when an explicit vocabulary mapping is invalid."""


class FhirOmopConformanceError(FhirOmopError):
    """Raised when a bundle fails the local conformance checks."""


@dataclass(frozen=True)
class FhirOmopInformationLoss:
    """One de-identified, path-only information-loss finding."""

    source_resource_hash: str
    resource_type: str
    element_path: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe loss record without source values."""

        return asdict(self)


@dataclass(frozen=True)
class FhirOmopLoadSummary:
    """Aggregate load facts that contain hashes, counts, and paths only."""

    row_counts: Mapping[str, int]
    resource_counts: Mapping[str, int]
    mapped_codes: int
    unmapped_codes: int
    unsupported_elements: tuple[str, ...]
    information_loss_count: int
    source_resource_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable PHI-free summary."""

        return {
            "row_counts": dict(self.row_counts),
            "resource_counts": dict(self.resource_counts),
            "mapped_codes": self.mapped_codes,
            "unmapped_codes": self.unmapped_codes,
            "unsupported_elements": list(self.unsupported_elements),
            "information_loss_count": self.information_loss_count,
            "source_resource_hashes": list(self.source_resource_hashes),
        }


@dataclass(frozen=True)
class FhirOmopTables:
    """OMOP CDM rows plus the FHIR provenance and inspection sidecars."""

    omop: OmopCdmTables
    resources: tuple[dict[str, Any], ...]
    codes: tuple[dict[str, Any], ...]
    values: tuple[dict[str, Any], ...]
    provenance: tuple[dict[str, Any], ...]
    information_loss: tuple[FhirOmopInformationLoss, ...]
    summary: FhirOmopLoadSummary

    @property
    def tables(self) -> Mapping[str, tuple[dict[str, Any], ...]]:
        """Return standard OMOP and FHIR sidecar rows by table name."""

        return {
            **self.omop.tables,
            "fhir_resource": self.resources,
            "fhir_code": self.codes,
            "fhir_value": self.values,
            "fhir_provenance": self.provenance,
            "fhir_information_loss": tuple(
                finding.to_dict() for finding in self.information_loss
            ),
        }

    @property
    def row_counts(self) -> Mapping[str, int]:
        """Return counts for the standard OMOP tables."""

        return self.summary.row_counts

    @property
    def keys(self) -> Mapping[str, tuple[Any, ...]]:
        """Return deterministic primary-key values for standard OMOP tables."""

        primary_keys = {
            "concept": "concept_id",
            "person": "person_id",
            "visit_occurrence": "visit_occurrence_id",
            "note": "note_id",
            "note_nlp": "note_nlp_id",
            "condition_occurrence": "condition_occurrence_id",
            "drug_exposure": "drug_exposure_id",
            "measurement": "measurement_id",
            "procedure_occurrence": "procedure_occurrence_id",
            "observation": "observation_id",
            "source_to_concept_map": "source_to_concept_map_id",
        }
        return {
            table: tuple(row[key] for row in self.omop.table(table))
            for table, key in primary_keys.items()
        }

    def table(self, name: str) -> tuple[dict[str, Any], ...]:
        """Return rows for a standard OMOP or FHIR sidecar table."""

        return self.tables.get(name, ())

    def to_dict(self, *, include_note_text: bool = False) -> dict[str, Any]:
        """Return a stable representation with note text redacted by default."""

        omop_tables: dict[str, list[dict[str, Any]]] = {}
        for table, rows in self.omop.tables.items():
            rendered = []
            for row in rows:
                item = dict(row)
                if not include_note_text and table == "note":
                    item["note_text"] = None
                rendered.append(item)
            omop_tables[table] = rendered
        return {
            "tables": {
                **omop_tables,
                "fhir_resource": list(self.resources),
                "fhir_code": list(self.codes),
                "fhir_value": list(self.values),
                "fhir_provenance": list(self.provenance),
                "fhir_information_loss": [
                    finding.to_dict() for finding in self.information_loss
                ],
            },
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True)
class FhirReverseInspection:
    """A de-identified FHIR Bundle reconstructed from bridge sidecars."""

    bundle: Mapping[str, Any]
    information_loss: tuple[FhirOmopInformationLoss, ...]
    coded_content_digest: str

    @property
    def resources(self) -> tuple[Mapping[str, Any], ...]:
        """Return reconstructed resources in deterministic order."""

        return tuple(
            entry["resource"]
            for entry in self.bundle.get("entry", ())
            if isinstance(entry, Mapping) and isinstance(entry.get("resource"), Mapping)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the reconstructed Bundle and path-only loss records."""

        return {
            "bundle": dict(self.bundle),
            "information_loss": [
                finding.to_dict() for finding in self.information_loss
            ],
            "coded_content_digest": self.coded_content_digest,
        }


@dataclass(frozen=True)
class FhirOmopConformanceReport:
    """Offline conformance result for one synthetic or caller-supplied bundle."""

    passed: bool
    checks: tuple[dict[str, Any], ...]
    row_counts: Mapping[str, int]
    unsupported_elements: tuple[str, ...]
    information_loss: tuple[FhirOmopInformationLoss, ...]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a report containing no source resource values."""

        return {
            "passed": self.passed,
            "checks": [dict(check) for check in self.checks],
            "row_counts": dict(self.row_counts),
            "unsupported_elements": list(self.unsupported_elements),
            "information_loss": [
                finding.to_dict() for finding in self.information_loss
            ],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class _ResolvedConcept:
    """Internal explicit mapping decision for one FHIR Coding."""

    source_code: str
    source_vocabulary_id: str
    source_concept_id: int
    source_concept_name: str | None
    source_standard_concept: str | None
    target_concept_id: int
    target_concept_name: str
    target_vocabulary_id: str
    target_standard_concept: str | None
    target_concept_code: str
    unmapped_reason: str | None
    vocabulary_snapshot: str | None

    @property
    def mapped(self) -> bool:
        """Return whether the user supplied a target standard concept."""

        return self.target_concept_id != UNMAPPED_CONCEPT_ID


@dataclass(frozen=True)
class _PreparedResource:
    resource: Mapping[str, Any]
    resource_type: str
    source_resource_hash: str
    source_id_hash: str
    deidentified_id: str
    person_source_hash: str
    visit_source_hash: str
    subject_reference: str | None
    encounter_reference: str | None
    status: str | None
    intent: str | None
    date_path: str | None
    date_value: str | None
    period_start: str | None
    period_end: str | None
    clinical_status: str | None
    verification_status: str | None


def load_fhir_bundle(
    bundle: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    vocabulary: VocabularyInput | None = None,
    vocabulary_snapshot: str | None = None,
    vocabulary_version: str | None = None,
    source_system: str = "FHIR",
    source_version: str = "R4",
    mode: LoadMode = "append",
) -> FhirOmopTables:
    """Load a supported FHIR R4 bundle into OMOP CDM v5.4 tables.

    Args:
        bundle: A FHIR ``Bundle`` or an iterable of standalone resources.
        vocabulary: A caller-supplied mapping, ``VocabularyRouter``, or
            ``(system, code)`` resolver.  Missing mappings remain concept 0.
        vocabulary_snapshot: Version or release label for the supplied
            vocabulary.  It is copied into every provenance record.
        vocabulary_version: Compatibility alias for ``vocabulary_snapshot``.
        source_system: Stable source-system label, never a source identifier.
        source_version: Source-system/FHIR release label.
        mode: Only deterministic append/upsert semantics are supported.

    Returns:
        ``FhirOmopTables`` containing standard OMOP rows and hash-only FHIR
        provenance sidecars.

    Raises:
        FhirOmopMappingError: If an input resource embeds an OMOP concept ID
            or the explicit mapping is malformed.
        FhirOmopError: If the FHIR payload or source metadata is invalid.
    """

    if mode != "append":
        raise ValueError("FHIR-to-OMOP loader currently supports append mode only")
    source_system = _required_label(source_system, "source_system")
    source_version = _required_label(source_version, "source_version")
    if vocabulary_snapshot and vocabulary_version:
        if vocabulary_snapshot != vocabulary_version:
            raise ValueError(
                "vocabulary_snapshot and vocabulary_version must match when both "
                "are provided"
            )
    snapshot = vocabulary_snapshot or vocabulary_version or "unspecified"

    resource_inputs = _coerce_resources(bundle)
    if not resource_inputs:
        raise FhirOmopError("FHIR bundle contains no resources")
    for resource, _ in resource_inputs:
        _reject_fabricated_concept_fields(resource)

    prepared = _prepare_resources(resource_inputs, source_system, source_version)
    notes: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    code_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    losses: list[FhirOmopInformationLoss] = []

    for item in prepared:
        primary_path = _RESOURCE_CODE_PATH[item.resource_type]
        candidates = _code_candidates(item.resource, item.resource_type, primary_path)
        note_segments: list[str] = []
        entities: list[dict[str, Any]] = []
        candidate_details: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            path = str(candidate["element_path"])
            code = str(candidate.get("code") or "").strip()
            display = str(candidate.get("display") or "").strip()
            segment_value = display or code or str(candidate.get("text") or "")
            segment = f"{path}[{candidate_index}]: {segment_value}".strip()
            start = sum(len(previous) + 1 for previous in note_segments)
            note_segments.append(segment)
            end = start + len(segment)
            resolver_code = code or str(candidate.get("text") or "").strip()
            resolved = _resolve_concept(
                resolver_code,
                str(candidate.get("system") or ""),
                item.resource_type,
                vocabulary,
                snapshot,
            )
            if not code:
                resolved = _unmapped_concept(
                    resolver_code,
                    str(candidate.get("system") or ""),
                    snapshot,
                    _UNMAPPED_MISSING_CODE,
                )
            code_row = _code_row(
                item,
                candidate,
                resolved,
                role="primary" if path == primary_path else "additional",
                omop_table=_resource_omop_table(item.resource_type),
                omop_row_id=None,
            )
            code_rows.append(code_row)
            if (
                path == primary_path
                and _RESOURCE_DOMAIN[item.resource_type] is not None
            ):
                entity = {
                    "text": segment,
                    "domain_id": _RESOURCE_DOMAIN[item.resource_type],
                    "start": start,
                    "end": end,
                    "concept_id": resolved.target_concept_id or None,
                    "source_concept_id": resolved.source_concept_id or None,
                    "code": resolved.source_code,
                    "vocabulary_id": resolved.target_vocabulary_id
                    if resolved.mapped
                    else resolved.source_vocabulary_id,
                    "concept_name": resolved.target_concept_name,
                    "standard_concept": resolved.target_standard_concept,
                }
                entities.append(entity)
                candidate_detail = {
                    "resource": item,
                    "candidate": candidate,
                    "resolved": resolved,
                    "start": start,
                    "end": end,
                }
                candidate_details.append(candidate_detail)

        for extra_candidate in _additional_code_candidates(
            item.resource, item.resource_type, primary_path
        ):
            code = str(extra_candidate.get("code") or "").strip()
            resolver_code = code or str(extra_candidate.get("text") or "").strip()
            resolved = _resolve_concept(
                resolver_code,
                str(extra_candidate.get("system") or ""),
                item.resource_type,
                vocabulary,
                snapshot,
            )
            if not code:
                resolved = _unmapped_concept(
                    resolver_code,
                    str(extra_candidate.get("system") or ""),
                    snapshot,
                    _UNMAPPED_MISSING_CODE,
                )
            code_rows.append(
                _code_row(
                    item,
                    extra_candidate,
                    resolved,
                    role="additional",
                    omop_table=_resource_omop_table(item.resource_type),
                    omop_row_id=None,
                )
            )

        note_text = "\n".join(note_segments) or f"{item.resource_type} resource"
        note = {
            "document_id": item.source_resource_hash,
            "person_id": item.person_source_hash,
            "visit_id": item.visit_source_hash,
            "note_date": item.date_value,
            "note_text": note_text,
            "source_note_hash": item.source_resource_hash,
            "entities": entities,
        }
        notes.append(note)
        resource_rows.append(_resource_row(item, note))
        for candidate_detail in candidate_details:
            candidate_detail["note_text"] = note_text
            candidate_rows.append(candidate_detail)
        value_rows.extend(_value_rows(item))
        losses.extend(_resource_information_loss(item))

    omop = load_grounded_notes(notes, vocabulary_version=snapshot, mode=mode)
    omop, provenance = _finalize_omop_rows(
        omop,
        candidate_rows,
        code_rows,
        snapshot,
        source_system,
        source_version,
    )
    omop = _refresh_omop_summary(omop)

    # The row finalizer attaches deterministic OMOP IDs to code rows after the
    # existing loader has created NOTE_NLP and domain rows.
    code_rows = _attach_code_row_ids(code_rows, provenance)
    resource_rows = _attach_resource_links(resource_rows, omop)
    losses = _deduplicate_losses(losses)
    resource_rows.sort(key=lambda row: row["source_resource_hash"])
    code_rows.sort(key=_sidecar_sort_key)
    value_rows.sort(key=_sidecar_sort_key)
    provenance.sort(key=lambda row: int(row["provenance_id"]))

    resource_counts = Counter(row["resource_type"] for row in resource_rows)
    mapped_codes = sum(int(row["target_concept_id"]) != 0 for row in code_rows)
    unmapped_codes = len(code_rows) - mapped_codes
    unsupported = tuple(
        sorted(
            {
                finding.element_path
                for finding in losses
                if finding.reason == "unsupported_element"
            }
        )
    )
    summary = FhirOmopLoadSummary(
        row_counts=dict(omop.row_counts),
        resource_counts=dict(sorted(resource_counts.items())),
        mapped_codes=mapped_codes,
        unmapped_codes=unmapped_codes,
        unsupported_elements=unsupported,
        information_loss_count=len(losses),
        source_resource_hashes=tuple(
            sorted(row["source_resource_hash"] for row in resource_rows)
        ),
    )
    return FhirOmopTables(
        omop=omop,
        resources=tuple(resource_rows),
        codes=tuple(code_rows),
        values=tuple(value_rows),
        provenance=tuple(provenance),
        information_loss=tuple(losses),
        summary=summary,
    )


def load_fhir_to_omop(
    bundle: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> FhirOmopTables:
    """Compatibility alias for :func:`load_fhir_bundle`."""

    return load_fhir_bundle(bundle, **kwargs)


def fhir_to_omop(
    bundle: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> FhirOmopTables:
    """Short alias for :func:`load_fhir_bundle`."""

    return load_fhir_bundle(bundle, **kwargs)


def inspect_fhir_from_omop(tables: FhirOmopTables) -> FhirReverseInspection:
    """Reconstruct a de-identified supported FHIR Bundle from sidecars."""

    if not isinstance(tables, FhirOmopTables):
        raise TypeError("inspect_fhir_from_omop expects FhirOmopTables")
    codes_by_resource: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tables.codes:
        codes_by_resource[str(row["source_resource_hash"])].append(row)
    values_by_resource: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tables.values:
        values_by_resource[str(row["source_resource_hash"])].append(row)

    resources: list[dict[str, Any]] = []
    for metadata in sorted(
        tables.resources, key=lambda row: row["source_resource_hash"]
    ):
        resource_type = str(metadata["resource_type"])
        resource: dict[str, Any] = {
            "resourceType": resource_type,
            "id": str(metadata["deidentified_id"]),
        }
        _set_reverse_metadata(resource, metadata)
        grouped_codes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sorted(
            codes_by_resource.get(str(metadata["source_resource_hash"]), []),
            key=lambda item: (str(item["element_path"]), int(item["coding_index"])),
        ):
            grouped_codes[str(row["element_path"])].append(row)
        for path, rows in grouped_codes.items():
            _set_reverse_codeable(resource, path, rows)
        for value in sorted(
            values_by_resource.get(str(metadata["source_resource_hash"]), []),
            key=lambda item: str(item["element_path"]),
        ):
            _set_reverse_value(resource, value)
        resources.append(resource)

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"fullUrl": f"urn:sha256:{_hash_json(resource)}", "resource": resource}
            for resource in resources
        ],
    }
    digest = _coded_content_digest(bundle)
    return FhirReverseInspection(
        bundle=bundle,
        information_loss=tables.information_loss,
        coded_content_digest=digest,
    )


def reverse_inspect_fhir(tables: FhirOmopTables) -> FhirReverseInspection:
    """Compatibility alias for :func:`inspect_fhir_from_omop`."""

    return inspect_fhir_from_omop(tables)


def build_conformance_report(
    bundle: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    vocabulary: VocabularyInput | None = None,
    vocabulary_snapshot: str | None = None,
    vocabulary_version: str | None = None,
    source_system: str = "FHIR",
    source_version: str = "R4",
) -> FhirOmopConformanceReport:
    """Run deterministic, offline conformance checks over a FHIR payload."""

    bundle_payload: Mapping[str, Any] | list[Mapping[str, Any]]
    if isinstance(bundle, Mapping):
        bundle_payload = bundle
    else:
        bundle_payload = list(bundle)
    try:
        first = load_fhir_bundle(
            bundle_payload,
            vocabulary=vocabulary,
            vocabulary_snapshot=vocabulary_snapshot,
            vocabulary_version=vocabulary_version,
            source_system=source_system,
            source_version=source_version,
        )
        second = load_fhir_bundle(
            bundle_payload,
            vocabulary=vocabulary,
            vocabulary_snapshot=vocabulary_snapshot,
            vocabulary_version=vocabulary_version,
            source_system=source_system,
            source_version=source_version,
        )
        reverse = inspect_fhir_from_omop(first)
        violations = validate_omop_tables(first.omop)
        first_keys = first.keys
        second_keys = second.keys
        keys_match = first_keys == second_keys
        concept_ids = {int(row["concept_id"]) for row in first.omop.table("concept")}
        fk_values = [
            int(row[column])
            for table in first.omop.tables.values()
            for row in table
            for column, value in row.items()
            if column.endswith("_concept_id")
            or column in {"source_concept_id", "target_concept_id"}
            if value is not None
        ]
        foreign_keys_resolve = all(value in concept_ids for value in fk_values)
        traceable = all(
            row["source_resource_hash"]
            and row["element_path"]
            and row["vocabulary_snapshot"]
            and row["note_nlp_id"] is not None
            and int(row["offset"]) <= int(row["offset_end"])
            for row in first.provenance
            if int(row["target_concept_id"]) != UNMAPPED_CONCEPT_ID
        )
        unmapped_explained = all(
            int(row["target_concept_id"]) != UNMAPPED_CONCEPT_ID
            or bool(row["unmapped_reason"])
            for row in first.codes
        )
        coded_content_matches = (
            _coded_content_digest(bundle_payload) == reverse.coded_content_digest
        )
        checks = (
            {
                "name": "deterministic_reload_keys",
                "passed": keys_match,
            },
            {
                "name": "concept_foreign_keys_resolve",
                "passed": foreign_keys_resolve and not violations,
            },
            {
                "name": "source_traceability",
                "passed": traceable,
            },
            {
                "name": "explicit_unmapped_reasons",
                "passed": unmapped_explained,
            },
            {
                "name": "reverse_supported_coded_content",
                "passed": coded_content_matches,
            },
        )
        passed = all(bool(check["passed"]) for check in checks)
        errors = tuple(
            f"{violation.table}.{violation.column}:{violation.reason}"
            for violation in violations
        )
        return FhirOmopConformanceReport(
            passed=passed,
            checks=checks,
            row_counts=first.row_counts,
            unsupported_elements=first.summary.unsupported_elements,
            information_loss=first.information_loss,
            errors=errors,
        )
    except FhirOmopError as exc:
        reason = (
            "fabricated_concept_identifier"
            if isinstance(exc, FhirOmopMappingError)
            else "invalid_fhir_omop_input"
        )
        return FhirOmopConformanceReport(
            passed=False,
            checks=({"name": reason, "passed": False},),
            row_counts={},
            unsupported_elements=(),
            information_loss=(),
            errors=(reason,),
        )


def run_fhir_omop_conformance(
    bundle: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> FhirOmopConformanceReport:
    """Compatibility alias for :func:`build_conformance_report`."""

    return build_conformance_report(bundle, **kwargs)


def assert_fhir_omop_conformant(
    bundle: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> FhirOmopConformanceReport:
    """Return a passing report or raise a PHI-free conformance error."""

    report = build_conformance_report(bundle, **kwargs)
    if not report.passed:
        raise FhirOmopConformanceError("FHIR-to-OMOP conformance checks failed")
    return report


def write_fhir_omop_sqlite(
    tables: FhirOmopTables,
    target: str | Path | sqlite3.Connection = ":memory:",
    *,
    mode: LoadMode = "append",
) -> sqlite3.Connection:
    """Persist OMOP and FHIR sidecar rows in a local SQLite database.

    The operation uses deterministic primary keys and ``INSERT OR IGNORE`` so
    writing the same bundle twice is an idempotent reload.  No network or
    terminology lookup is performed.
    """

    if not isinstance(tables, FhirOmopTables):
        raise TypeError("write_fhir_omop_sqlite expects FhirOmopTables")
    if mode != "append":
        raise ValueError("FHIR-to-OMOP SQLite writer supports append mode only")
    con = write_omop_sqlite(tables.omop, target, mode=mode)
    _create_sidecar_schema(con)
    _insert_sidecar_rows(con, "fhir_resource", tables.resources)
    _insert_sidecar_rows(con, "fhir_code", tables.codes)
    _insert_sidecar_rows(con, "fhir_value", tables.values)
    _insert_sidecar_rows(con, "fhir_provenance", tables.provenance)
    _insert_sidecar_rows(
        con,
        "fhir_information_loss",
        tuple(finding.to_dict() for finding in tables.information_loss),
    )
    con.commit()
    return con


def _coerce_resources(
    payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], str]]:
    if isinstance(payload, Mapping):
        if payload.get("resourceType") == "Bundle":
            entries = payload.get("entry")
            if entries is None:
                return []
            if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
                raise FhirOmopError("FHIR Bundle.entry must be an array")
            result: list[tuple[Mapping[str, Any], str]] = []
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    raise FhirOmopError(f"FHIR Bundle.entry[{index}] must be an object")
                resource = entry.get("resource")
                if not isinstance(resource, Mapping):
                    raise FhirOmopError(
                        f"FHIR Bundle.entry[{index}].resource must be an object"
                    )
                result.append((resource, str(entry.get("fullUrl") or "")))
            return result
        return [(payload, "")]
    result = []
    for index, resource in enumerate(payload):
        if not isinstance(resource, Mapping):
            raise FhirOmopError(f"FHIR resource at index {index} must be an object")
        result.append((resource, ""))
    return result


def _prepare_resources(
    resource_inputs: Sequence[tuple[Mapping[str, Any], str]],
    source_system: str,
    source_version: str,
) -> list[_PreparedResource]:
    resource_hashes: dict[tuple[str, str], str] = {}
    for resource, full_url in resource_inputs:
        resource_type = _resource_type(resource)
        raw_id = _raw_resource_id(resource, full_url)
        resource_hash = _hash_json(resource)
        key = (resource_type, raw_id)
        prior_hash = resource_hashes.get(key)
        if prior_hash is not None and prior_hash != resource_hash:
            raise FhirOmopError(
                "FHIR bundle contains conflicting resources with the same type and id"
            )
        resource_hashes[key] = resource_hash

    id_hashes = {
        key: _hash_identifier(source_system, source_version, key[0], key[1])
        for key in resource_hashes
    }
    prepared: list[_PreparedResource] = []
    for resource, full_url in resource_inputs:
        resource_type = _resource_type(resource)
        resource_hash = _hash_json(resource)
        raw_id = _raw_resource_id(resource, full_url)
        source_id_hash = id_hashes[(resource_type, raw_id)]
        person_source_hash = _subject_source_hash(
            resource,
            resource_type,
            resource_hash,
            id_hashes,
            source_system,
            source_version,
        )
        visit_source_hash = _visit_source_hash(
            resource,
            resource_type,
            resource_hash,
            person_source_hash,
            id_hashes,
            source_system,
            source_version,
        )
        date_path, date_value = _resource_date(resource, resource_type)
        prepared.append(
            _PreparedResource(
                resource=resource,
                resource_type=resource_type,
                source_resource_hash=resource_hash,
                source_id_hash=source_id_hash,
                deidentified_id=f"{resource_type.lower()}-{source_id_hash[:16]}",
                person_source_hash=person_source_hash,
                visit_source_hash=visit_source_hash,
                subject_reference=_deidentified_reference(
                    resource.get("subject"),
                    "Patient",
                    id_hashes,
                    source_system,
                    source_version,
                ),
                encounter_reference=_deidentified_reference(
                    resource.get("encounter"),
                    "Encounter",
                    id_hashes,
                    source_system,
                    source_version,
                ),
                status=_status_value(resource.get("status")),
                intent=_status_value(resource.get("intent")),
                date_path=date_path,
                date_value=date_value,
                period_start=_period_value(resource, "start"),
                period_end=_period_value(resource, "end"),
                clinical_status=_codeable_code(resource.get("clinicalStatus")),
                verification_status=_codeable_code(resource.get("verificationStatus")),
            )
        )
    return sorted(
        prepared,
        key=lambda item: (
            item.resource_type,
            item.source_id_hash,
            item.source_resource_hash,
        ),
    )


def _resource_type(resource: Mapping[str, Any]) -> str:
    resource_type = resource.get("resourceType")
    if not isinstance(resource_type, str) or not resource_type.strip():
        raise FhirOmopError("FHIR resource is missing resourceType")
    resource_type = resource_type.strip()
    if resource_type not in FHIR_RESOURCE_TYPES:
        raise FhirOmopError(f"unsupported FHIR resource type: {resource_type}")
    return resource_type


def _raw_resource_id(resource: Mapping[str, Any], full_url: str) -> str:
    value = resource.get("id")
    if value is not None and str(value).strip():
        return str(value).strip()
    if full_url.strip():
        if full_url.startswith("urn:"):
            return full_url.rsplit(":", 1)[-1]
        return full_url.rstrip("/").rsplit("/", 1)[-1]
    return _hash_json(resource)


def _subject_source_hash(
    resource: Mapping[str, Any],
    resource_type: str,
    resource_hash: str,
    id_hashes: Mapping[tuple[str, str], str],
    source_system: str,
    source_version: str,
) -> str:
    if resource_type == "Patient":
        return _hash_identifier(
            source_system, source_version, "Patient", _raw_resource_id(resource, "")
        )
    subject = resource.get("subject")
    reference = _reference_value(subject)
    if reference:
        expected_type, raw_id = _split_reference(reference, "Patient")
        return id_hashes.get(
            (expected_type, raw_id),
            _hash_identifier(source_system, source_version, expected_type, raw_id),
        )
    if isinstance(subject, Mapping) and isinstance(subject.get("identifier"), Mapping):
        return _hash_identifier(
            source_system,
            source_version,
            "Patient.identifier",
            _hash_json(subject["identifier"]),
        )
    return _hash_identifier(source_system, source_version, "subject", resource_hash)


def _visit_source_hash(
    resource: Mapping[str, Any],
    resource_type: str,
    resource_hash: str,
    person_source_hash: str,
    id_hashes: Mapping[tuple[str, str], str],
    source_system: str,
    source_version: str,
) -> str:
    if resource_type == "Encounter":
        raw_id = _raw_resource_id(resource, "")
        return _hash_identifier(source_system, source_version, "Encounter", raw_id)
    reference = _reference_value(resource.get("encounter"))
    if reference:
        expected_type, raw_id = _split_reference(reference, "Encounter")
        return id_hashes.get(
            (expected_type, raw_id),
            _hash_identifier(source_system, source_version, expected_type, raw_id),
        )
    return _hash_identifier(source_system, source_version, "visit", person_source_hash)


def _deidentified_reference(
    value: Any,
    expected_type: str,
    id_hashes: Mapping[tuple[str, str], str],
    source_system: str,
    source_version: str,
) -> str | None:
    reference = _reference_value(value)
    if not reference:
        return None
    resource_type, raw_id = _split_reference(reference, expected_type)
    identifier_hash = id_hashes.get(
        (resource_type, raw_id),
        _hash_identifier(source_system, source_version, resource_type, raw_id),
    )
    return f"{resource_type}/pseudonym-{identifier_hash[:16]}"


def _split_reference(reference: str, expected_type: str) -> tuple[str, str]:
    value = reference.strip()
    if value.startswith("urn:"):
        return expected_type, value.rsplit(":", 1)[-1]
    parts = value.rstrip("/").split("/")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return expected_type, parts[-1]


def _reference_value(value: Any) -> str:
    if isinstance(value, Mapping):
        reference = value.get("reference")
        return str(reference).strip() if reference is not None else ""
    return ""


def _resource_date(
    resource: Mapping[str, Any], resource_type: str
) -> tuple[str | None, str | None]:
    fields: Mapping[str, tuple[str, ...]] = {
        "Condition": ("onsetDateTime", "recordedDate"),
        "Observation": ("effectiveDateTime",),
        "Procedure": ("performedDateTime",),
        "MedicationStatement": ("effectiveDateTime", "dateAsserted"),
        "MedicationRequest": ("authoredOn",),
        "Encounter": ("period.start",),
        "Patient": (),
    }
    for path in fields[resource_type]:
        value: Any = resource
        for part in path.split("."):
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        if value is not None and str(value).strip():
            return f"{resource_type}.{path}", str(value).strip()
    return None, None


def _period_value(resource: Mapping[str, Any], key: str) -> str | None:
    period = resource.get("period")
    if isinstance(period, Mapping) and period.get(key) is not None:
        value = str(period[key]).strip()
        return value or None
    return None


def _status_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("code") or value.get("display")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _codeable_code(value: Any) -> str | None:
    codings = _codeable_parts(value)
    if not codings:
        return None
    return str(codings[0].get("code") or codings[0].get("display") or "") or None


def _code_candidates(
    resource: Mapping[str, Any], resource_type: str, primary_path: str | None
) -> list[dict[str, Any]]:
    if primary_path is None:
        return []
    field = primary_path.split(".", 1)[1]
    value = resource.get(field)
    if resource_type == "Encounter":
        value = {"coding": [value]} if isinstance(value, Mapping) else value
    return _codeable_rows(value, primary_path)


def _additional_code_candidates(
    resource: Mapping[str, Any], resource_type: str, primary_path: str | None
) -> list[dict[str, Any]]:
    paths: tuple[str, ...]
    if resource_type == "Condition":
        paths = ("Condition.clinicalStatus", "Condition.verificationStatus")
    elif resource_type == "Observation":
        paths = ("Observation.valueCodeableConcept",)
    else:
        paths = ()
    result: list[dict[str, Any]] = []
    for path in paths:
        if path == primary_path:
            continue
        field = path.split(".", 1)[1]
        result.extend(_codeable_rows(resource.get(field), path))
    return result


def _codeable_rows(value: Any, element_path: str) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    raw_codings = value.get("coding")
    if isinstance(raw_codings, Mapping):
        raw_codings = [raw_codings]
    if not isinstance(raw_codings, Sequence) or isinstance(raw_codings, (str, bytes)):
        raw_codings = []
    safe_text = _safe_codeable_text(value, raw_codings)
    rows: list[dict[str, Any]] = []
    for index, coding in enumerate(raw_codings):
        if not isinstance(coding, Mapping):
            continue
        rows.append(
            {
                "element_path": element_path,
                "system": str(coding.get("system") or "").strip(),
                "code": str(coding.get("code") or "").strip(),
                "display": str(coding.get("display") or "").strip(),
                "text": safe_text,
                "coding_index": index,
            }
        )
    if not rows and safe_text:
        rows.append(
            {
                "element_path": element_path,
                "system": "",
                "code": "",
                "display": "",
                "text": safe_text,
                "coding_index": 0,
            }
        )
    return rows


def _codeable_parts(value: Any) -> list[dict[str, Any]]:
    return _codeable_rows(value, "CodeableConcept")


def _safe_codeable_text(value: Mapping[str, Any], raw_codings: Any) -> str | None:
    text = str(value.get("text") or "").strip()
    if not text:
        return None
    displays = {
        str(item.get("display") or "").strip()
        for item in raw_codings
        if isinstance(item, Mapping)
    }
    return text if text in displays or not displays else None


def _value_rows(item: _PreparedResource) -> list[dict[str, Any]]:
    if item.resource_type != "Observation":
        return []
    result: list[dict[str, Any]] = []
    for key in (
        "valueQuantity",
        "valueInteger",
        "valueDecimal",
        "valueBoolean",
        "valueDateTime",
    ):
        if key not in item.resource:
            continue
        value = item.resource[key]
        row: dict[str, Any] = {
            "value_id": deterministic_omop_id(
                "fhir_value", item.source_resource_hash, f"Observation.{key}"
            ),
            "source_resource_hash": item.source_resource_hash,
            "source_id_hash": item.source_id_hash,
            "resource_type": item.resource_type,
            "element_path": f"Observation.{key}",
            "value_kind": key.removeprefix("value"),
            "value_number": None,
            "value_boolean": None,
            "value_date": None,
            "unit": None,
            "system": None,
            "code": None,
        }
        if key == "valueQuantity":
            if not isinstance(value, Mapping):
                continue
            row["value_number"] = value.get("value")
            row["unit"] = str(value.get("unit") or "").strip() or None
            row["system"] = str(value.get("system") or "").strip() or None
            row["code"] = str(value.get("code") or "").strip() or None
        elif key == "valueBoolean":
            row["value_boolean"] = bool(value)
        elif key in {"valueInteger", "valueDecimal"}:
            row["value_number"] = value
        else:
            row["value_date"] = str(value).strip()
        result.append(row)
    return result


def _resource_row(item: _PreparedResource, note: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_resource_hash": item.source_resource_hash,
        "source_id_hash": item.source_id_hash,
        "resource_type": item.resource_type,
        "deidentified_id": item.deidentified_id,
        "person_source_hash": item.person_source_hash,
        "visit_source_hash": item.visit_source_hash,
        "subject_reference": item.subject_reference,
        "encounter_reference": item.encounter_reference,
        "status": item.status,
        "intent": item.intent,
        "date_path": item.date_path,
        "date_value": item.date_value,
        "period_start": item.period_start,
        "period_end": item.period_end,
        "clinical_status": item.clinical_status,
        "verification_status": item.verification_status,
        "note_id": None,
        "visit_occurrence_id": None,
    }


def _code_row(
    item: _PreparedResource,
    candidate: Mapping[str, Any],
    resolved: _ResolvedConcept,
    *,
    role: str,
    omop_table: str | None,
    omop_row_id: int | None,
) -> dict[str, Any]:
    element_path = str(candidate["element_path"])
    index = int(candidate.get("coding_index", 0))
    return {
        "code_row_id": deterministic_omop_id(
            "fhir_code", item.source_resource_hash, element_path, index
        ),
        "source_resource_hash": item.source_resource_hash,
        "source_id_hash": item.source_id_hash,
        "resource_type": item.resource_type,
        "element_path": element_path,
        "codeable_path": element_path,
        "code_role": role,
        "coding_index": index,
        "system": str(candidate.get("system") or "").strip() or None,
        "code": str(candidate.get("code") or "").strip() or None,
        "display": str(candidate.get("display") or "").strip() or None,
        "text": candidate.get("text"),
        "source_concept_id": resolved.source_concept_id,
        "target_concept_id": resolved.target_concept_id,
        "source_vocabulary_id": resolved.source_vocabulary_id or None,
        "target_vocabulary_id": resolved.target_vocabulary_id or None,
        "standard_concept": resolved.target_standard_concept,
        "vocabulary_snapshot": resolved.vocabulary_snapshot,
        "unmapped_reason": resolved.unmapped_reason,
        "omop_table": omop_table,
        "omop_row_id": omop_row_id,
    }


def _resolve_concept(
    source_code: str,
    source_vocabulary_id: str,
    resource_type: str,
    vocabulary: VocabularyInput | None,
    snapshot: str,
) -> _ResolvedConcept:
    source_code = str(source_code or "").strip()
    source_vocabulary_id = str(source_vocabulary_id or "").strip()
    if not source_code:
        return _unmapped_concept(
            source_code, source_vocabulary_id, snapshot, _UNMAPPED_MISSING_CODE
        )
    if vocabulary is None:
        return _unmapped_concept(
            source_code, source_vocabulary_id, snapshot, _UNMAPPED_NO_MAPPING
        )
    mapping = _lookup_mapping(vocabulary, source_vocabulary_id, source_code)
    if mapping is None:
        return _unmapped_concept(
            source_code, source_vocabulary_id, snapshot, _UNMAPPED_NO_MAPPING
        )
    return _mapping_record(
        mapping,
        source_code=source_code,
        source_vocabulary_id=source_vocabulary_id,
        resource_type=resource_type,
        snapshot=snapshot,
    )


def _lookup_mapping(
    vocabulary: VocabularyInput, source_vocabulary_id: str, source_code: str
) -> Any:
    route = getattr(vocabulary, "route", None)
    if callable(route):
        result = route(
            source_code,
            source_vocabulary_id=source_vocabulary_id or None,
        )
        if getattr(result, "target_concept_id", UNMAPPED_CONCEPT_ID) == 0:
            return None
        return result
    resolve = getattr(vocabulary, "resolve", None)
    if callable(resolve):
        return resolve(source_vocabulary_id, source_code)
    if callable(vocabulary):
        return vocabulary(source_vocabulary_id, source_code)
    if not isinstance(vocabulary, Mapping):
        raise FhirOmopMappingError(
            "vocabulary must be a mapping, resolver, or VocabularyRouter"
        )
    nested = vocabulary.get(source_vocabulary_id)
    if isinstance(nested, Mapping) and source_code in nested:
        return nested[source_code]
    for key in (
        (source_vocabulary_id, source_code),
        f"{source_vocabulary_id}|{source_code}",
        f"{source_vocabulary_id}:{source_code}",
        source_code,
    ):
        try:
            if key in vocabulary:
                return vocabulary[key]
        except TypeError:
            continue
    return None


def _mapping_record(
    mapping: Any,
    *,
    source_code: str,
    source_vocabulary_id: str,
    resource_type: str,
    snapshot: str,
) -> _ResolvedConcept:
    if hasattr(mapping, "target_concept_id"):
        target_id = int(getattr(mapping, "target_concept_id", 0) or 0)
        if target_id <= 0:
            return _unmapped_concept(
                source_code,
                source_vocabulary_id,
                snapshot,
                str(getattr(mapping, "unmapped_reason", None) or _UNMAPPED_DECLARED),
            )
        return _ResolvedConcept(
            source_code=source_code,
            source_vocabulary_id=str(
                getattr(mapping, "source_vocabulary_id", None) or source_vocabulary_id
            ),
            source_concept_id=int(getattr(mapping, "source_concept_id", 0) or 0),
            source_concept_name=None,
            source_standard_concept=None,
            target_concept_id=target_id,
            target_concept_name=str(
                getattr(mapping, "source_code_description", None) or source_code
            ),
            target_vocabulary_id=str(
                getattr(mapping, "target_vocabulary_id", None) or "USER_SUPPLIED"
            ),
            target_standard_concept=str(
                getattr(mapping, "standard_concept", None) or "S"
            ),
            target_concept_code=source_code,
            unmapped_reason=None,
            vocabulary_snapshot=str(
                getattr(mapping, "vocabulary_version", None) or snapshot
            ),
        )
    if isinstance(mapping, bool):
        raise FhirOmopMappingError("boolean vocabulary mappings are not concept IDs")
    if isinstance(mapping, int):
        if mapping <= 0:
            return _unmapped_concept(
                source_code,
                source_vocabulary_id,
                snapshot,
                _UNMAPPED_DECLARED,
            )
        record: Mapping[str, Any] = {"concept_id": mapping}
    elif isinstance(mapping, Mapping):
        record = mapping
    else:
        raise FhirOmopMappingError(
            "vocabulary mapping values must be an integer or an object"
        )

    target_id = _optional_int(record.get("target_concept_id", record.get("concept_id")))
    if target_id is None or target_id <= 0:
        return _unmapped_concept(
            source_code,
            str(record.get("source_vocabulary_id") or source_vocabulary_id),
            str(record.get("vocabulary_snapshot") or snapshot),
            str(
                record.get("unmapped_reason")
                or record.get("invalid_reason")
                or _UNMAPPED_DECLARED
            ),
        )
    target_vocabulary = str(
        record.get("target_vocabulary_id")
        or record.get("vocabulary_id")
        or "USER_SUPPLIED"
    ).strip()
    target_name = str(
        record.get("target_concept_name")
        or record.get("concept_name")
        or record.get("display")
        or source_code
    ).strip()
    standard = record.get(
        "target_standard_concept", record.get("standard_concept", "S")
    )
    standard_text = str(standard).strip() if standard is not None else None
    source_id = _optional_int(record.get("source_concept_id")) or 0
    source_name = str(record.get("source_concept_name") or "").strip() or None
    source_standard = record.get("source_standard_concept")
    source_standard_text = (
        str(source_standard).strip() if source_standard is not None else None
    )
    return _ResolvedConcept(
        source_code=source_code,
        source_vocabulary_id=str(
            record.get("source_vocabulary_id") or source_vocabulary_id
        ).strip(),
        source_concept_id=source_id,
        source_concept_name=source_name,
        source_standard_concept=source_standard_text,
        target_concept_id=target_id,
        target_concept_name=target_name,
        target_vocabulary_id=target_vocabulary,
        target_standard_concept=standard_text,
        target_concept_code=str(record.get("target_concept_code") or source_code),
        unmapped_reason=None,
        vocabulary_snapshot=str(record.get("vocabulary_snapshot") or snapshot),
    )


def _unmapped_concept(
    source_code: str,
    source_vocabulary_id: str,
    snapshot: str,
    reason: str,
) -> _ResolvedConcept:
    return _ResolvedConcept(
        source_code=source_code,
        source_vocabulary_id=source_vocabulary_id or UNMAPPED_VOCABULARY_ID,
        source_concept_id=UNMAPPED_CONCEPT_ID,
        source_concept_name=None,
        source_standard_concept=None,
        target_concept_id=UNMAPPED_CONCEPT_ID,
        target_concept_name=UNMAPPED_CONCEPT_NAME,
        target_vocabulary_id=UNMAPPED_VOCABULARY_ID,
        target_standard_concept=None,
        target_concept_code=source_code,
        unmapped_reason=reason or _UNMAPPED_NO_MAPPING,
        vocabulary_snapshot=snapshot,
    )


def _finalize_omop_rows(
    omop: OmopCdmTables,
    candidate_rows: Sequence[Mapping[str, Any]],
    code_rows: list[dict[str, Any]],
    snapshot: str,
    source_system: str,
    source_version: str,
) -> tuple[OmopCdmTables, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {
        table: [dict(row) for row in omop.table(table)] for table in omop.tables
    }
    rows["concept"] = [
        row for row in rows.get("concept", []) if int(row["concept_id"]) != 0
    ]
    rows["concept"].append(
        {
            "concept_id": UNMAPPED_CONCEPT_ID,
            "concept_name": UNMAPPED_CONCEPT_NAME,
            "domain_id": "",
            "vocabulary_id": UNMAPPED_VOCABULARY_ID,
            "concept_class_id": "",
            "standard_concept": None,
            "concept_code": "",
        }
    )
    concept_by_id = {int(row["concept_id"]): row for row in rows["concept"]}
    provenance: list[dict[str, Any]] = []
    for detail in candidate_rows:
        item = detail["resource"]
        candidate = detail["candidate"]
        resolved: _ResolvedConcept = detail["resolved"]
        start = int(detail["start"])
        end = int(detail["end"])
        person_id = deterministic_omop_id("person", item.person_source_hash)
        note_id = deterministic_omop_id(
            "note", item.person_source_hash, item.source_resource_hash
        )
        target_id = int(resolved.target_concept_id)
        idempotent_key = _idempotent_key(
            person_id,
            item.source_resource_hash,
            start,
            end,
            target_id,
        )
        note_nlp_id = deterministic_omop_id("note_nlp", idempotent_key)
        table_name = _resource_omop_table(item.resource_type)
        row_id = (
            deterministic_omop_id(table_name, idempotent_key)
            if table_name is not None
            else None
        )
        if row_id is not None:
            _patch_domain_row(
                rows,
                table_name,
                row_id,
                source_concept_id=resolved.source_concept_id,
                source_note_hash=item.source_resource_hash,
                note_nlp_id=note_nlp_id,
                idempotent_key=idempotent_key,
            )
        _patch_note_nlp(rows, note_nlp_id)
        _patch_source_map(
            rows,
            note_nlp_id,
            resolved,
            item.source_resource_hash,
            snapshot,
            str(candidate.get("display") or "").strip() or None,
        )
        _add_explicit_concept_rows(concept_by_id, resolved, item.resource_type)
        provenance_id = deterministic_omop_id(
            "fhir_provenance",
            item.source_resource_hash,
            str(candidate["element_path"]),
            int(candidate.get("coding_index", 0)),
        )
        provenance.append(
            {
                "provenance_id": provenance_id,
                "source_resource_hash": item.source_resource_hash,
                "source_id_hash": item.source_id_hash,
                "resource_type": item.resource_type,
                "element_path": str(candidate["element_path"]),
                "source_system": source_system,
                "source_version": source_version,
                "source_code": resolved.source_code or None,
                "source_vocabulary_id": resolved.source_vocabulary_id or None,
                "source_concept_id": resolved.source_concept_id,
                "target_concept_id": resolved.target_concept_id,
                "target_vocabulary_id": resolved.target_vocabulary_id,
                "standard_concept": resolved.target_standard_concept,
                "vocabulary_snapshot": resolved.vocabulary_snapshot or snapshot,
                "unmapped_reason": resolved.unmapped_reason,
                "note_id": note_id,
                "note_nlp_id": note_nlp_id,
                "offset": start,
                "offset_end": end,
                "omop_table": table_name,
                "omop_row_id": row_id,
                "idempotent_key": idempotent_key,
            }
        )
        for code_row in code_rows:
            if (
                code_row["source_resource_hash"] == item.source_resource_hash
                and code_row["element_path"] == candidate["element_path"]
                and int(code_row["coding_index"]) == int(candidate["coding_index"])
            ):
                code_row["omop_row_id"] = row_id
    for code_row in code_rows:
        _add_sidecar_concept_rows(concept_by_id, code_row)
    rows["concept"] = sorted(
        concept_by_id.values(), key=lambda row: int(row["concept_id"])
    )
    normalized_tables = {
        table: tuple(sorted(table_rows, key=_primary_row_id))
        for table, table_rows in rows.items()
    }
    return (
        OmopCdmTables(tables=normalized_tables, summary=omop.summary),
        provenance,
    )


def _refresh_omop_summary(omop: OmopCdmTables) -> OmopCdmTables:
    """Refresh row counts after explicit sidecar concepts are added."""

    summary = OmopLoadSummary(
        row_counts={table: len(omop.table(table)) for table in omop.tables},
        rejection_counts=omop.summary.rejection_counts,
        rejected_spans=omop.summary.rejected_spans,
        source_note_hashes=omop.summary.source_note_hashes,
    )
    return OmopCdmTables(tables=omop.tables, summary=summary)


def _idempotent_key(
    person_id: int, source_resource_hash: str, start: int, end: int, concept_id: int
) -> str:
    payload = f"{person_id}\x1f{source_resource_hash}\x1f{start}:{end}\x1f{concept_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _patch_domain_row(
    rows: Mapping[str, list[dict[str, Any]]],
    table_name: str,
    row_id: int,
    *,
    source_concept_id: int,
    source_note_hash: str,
    note_nlp_id: int,
    idempotent_key: str,
) -> None:
    primary_key = {
        "condition_occurrence": "condition_occurrence_id",
        "drug_exposure": "drug_exposure_id",
        "measurement": "measurement_id",
        "procedure_occurrence": "procedure_occurrence_id",
        "observation": "observation_id",
    }[table_name]
    for row in rows.get(table_name, []):
        if int(row[primary_key]) == row_id:
            source_column = {
                "condition_occurrence": "condition_source_concept_id",
                "drug_exposure": "drug_source_concept_id",
                "measurement": "measurement_source_concept_id",
                "procedure_occurrence": "procedure_source_concept_id",
                "observation": "observation_source_concept_id",
            }[table_name]
            row[source_column] = source_concept_id
            row["source_note_hash"] = source_note_hash
            row["note_nlp_id"] = note_nlp_id
            row["idempotent_key"] = idempotent_key
            return


def _patch_note_nlp(rows: Mapping[str, list[dict[str, Any]]], note_nlp_id: int) -> None:
    for row in rows.get("note_nlp", []):
        if int(row["note_nlp_id"]) == note_nlp_id:
            row["nlp_system"] = "openmed.interop.fhir_omop"
            return


def _patch_source_map(
    rows: Mapping[str, list[dict[str, Any]]],
    note_nlp_id: int,
    resolved: _ResolvedConcept,
    source_resource_hash: str,
    snapshot: str,
    source_code_description: str | None,
) -> None:
    for row in rows.get("source_to_concept_map", []):
        if int(row["note_nlp_id"]) != note_nlp_id:
            continue
        row.update(
            {
                "source_code": resolved.source_code or None,
                "source_concept_id": resolved.source_concept_id,
                "source_vocabulary_id": resolved.source_vocabulary_id,
                "source_code_description": source_code_description,
                "target_concept_id": resolved.target_concept_id,
                "target_vocabulary_id": resolved.target_vocabulary_id,
                "invalid_reason": resolved.unmapped_reason,
                "vocabulary_version": resolved.vocabulary_snapshot or snapshot,
                "source_note_hash": source_resource_hash,
            }
        )
        return


def _add_explicit_concept_rows(
    concept_by_id: dict[int, dict[str, Any]],
    resolved: _ResolvedConcept,
    domain: str,
) -> None:
    if resolved.target_concept_id:
        concept_by_id[resolved.target_concept_id] = {
            "concept_id": resolved.target_concept_id,
            "concept_name": resolved.target_concept_name or UNMAPPED_CONCEPT_NAME,
            "domain_id": _RESOURCE_DOMAIN.get(domain) or "",
            "vocabulary_id": resolved.target_vocabulary_id,
            "concept_class_id": "",
            "standard_concept": resolved.target_standard_concept,
            "concept_code": resolved.target_concept_code,
        }
    if resolved.source_concept_id:
        concept_by_id[resolved.source_concept_id] = {
            "concept_id": resolved.source_concept_id,
            "concept_name": resolved.source_concept_name or resolved.source_code,
            "domain_id": _RESOURCE_DOMAIN.get(domain) or "",
            "vocabulary_id": resolved.source_vocabulary_id,
            "concept_class_id": "",
            "standard_concept": resolved.source_standard_concept,
            "concept_code": resolved.source_code,
        }


def _add_sidecar_concept_rows(
    concept_by_id: dict[int, dict[str, Any]], row: Mapping[str, Any]
) -> None:
    """Keep concept IDs in coded sidecars resolvable as well as OMOP FKs."""

    domain = _RESOURCE_DOMAIN.get(str(row.get("resource_type"))) or ""
    target_id = int(row.get("target_concept_id") or 0)
    if target_id and target_id not in concept_by_id:
        concept_by_id[target_id] = {
            "concept_id": target_id,
            "concept_name": row.get("display")
            or row.get("code")
            or UNMAPPED_CONCEPT_NAME,
            "domain_id": domain,
            "vocabulary_id": row.get("target_vocabulary_id") or "USER_SUPPLIED",
            "concept_class_id": "",
            "standard_concept": row.get("standard_concept"),
            "concept_code": row.get("code") or "",
        }
    source_id = int(row.get("source_concept_id") or 0)
    if source_id and source_id not in concept_by_id:
        concept_by_id[source_id] = {
            "concept_id": source_id,
            "concept_name": row.get("display") or row.get("code") or "",
            "domain_id": domain,
            "vocabulary_id": row.get("source_vocabulary_id") or "",
            "concept_class_id": "",
            "standard_concept": None,
            "concept_code": row.get("code") or "",
        }


def _primary_row_id(row: Mapping[str, Any]) -> int:
    for key in (
        "concept_id",
        "person_id",
        "visit_occurrence_id",
        "note_id",
        "note_nlp_id",
        "condition_occurrence_id",
        "drug_exposure_id",
        "measurement_id",
        "procedure_occurrence_id",
        "observation_id",
        "source_to_concept_map_id",
    ):
        if key in row:
            return int(row[key])
    return 0


def _resource_omop_table(resource_type: str) -> str | None:
    return {
        "Condition": "condition_occurrence",
        "Observation": "measurement",
        "Procedure": "procedure_occurrence",
        "MedicationStatement": "drug_exposure",
        "MedicationRequest": "drug_exposure",
    }.get(resource_type)


def _attach_code_row_ids(
    code_rows: list[dict[str, Any]], provenance: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    del provenance
    return code_rows


def _attach_resource_links(
    resource_rows: list[dict[str, Any]], omop: OmopCdmTables
) -> list[dict[str, Any]]:
    notes = {row["source_note_hash"]: row for row in omop.table("note")}
    visits = {row["visit_source_value"]: row for row in omop.table("visit_occurrence")}
    for row in resource_rows:
        note = notes.get(row["source_resource_hash"])
        visit = visits.get(row["visit_source_hash"])
        row["note_id"] = note["note_id"] if note else None
        row["visit_occurrence_id"] = visit["visit_occurrence_id"] if visit else None
    return resource_rows


def _resource_information_loss(
    item: _PreparedResource,
) -> list[FhirOmopInformationLoss]:
    resource = item.resource
    resource_type = item.resource_type
    supported_roots = {
        path.split(".", 1)[0] for path in SUPPORTED_FHIR_ELEMENTS[resource_type]
    }
    findings: list[FhirOmopInformationLoss] = []
    if "id" in resource:
        findings.append(
            FhirOmopInformationLoss(
                item.source_resource_hash,
                resource_type,
                f"{resource_type}.id",
                "source identifier replaced with hash",
            )
        )
    for key in resource:
        if key in {"resourceType", "id"}:
            continue
        path = f"{resource_type}.{key}"
        if key not in supported_roots:
            findings.append(
                FhirOmopInformationLoss(
                    item.source_resource_hash,
                    resource_type,
                    path,
                    "unsupported_element",
                )
            )
            continue
        if key == "text":
            findings.append(
                FhirOmopInformationLoss(
                    item.source_resource_hash,
                    resource_type,
                    path,
                    "unsupported_element",
                )
            )
        elif key in {"subject", "encounter"}:
            value = resource.get(key)
            if isinstance(value, Mapping) and value.get("reference"):
                findings.append(
                    FhirOmopInformationLoss(
                        item.source_resource_hash,
                        resource_type,
                        f"{path}.reference",
                        "reference replaced with hash",
                    )
                )
            for nested in value if isinstance(value, Mapping) else ():
                if nested not in {"reference", "type"}:
                    findings.append(
                        FhirOmopInformationLoss(
                            item.source_resource_hash,
                            resource_type,
                            f"{path}.{nested}",
                            "unsupported_element",
                        )
                    )
        elif key in {
            "code",
            "class",
            "clinicalStatus",
            "verificationStatus",
            "valueCodeableConcept",
        }:
            findings.extend(_codeable_information_loss(item, key, resource.get(key)))
        elif key == "valueQuantity":
            value = resource.get(key)
            if isinstance(value, Mapping):
                for nested in value:
                    if nested not in {"value", "unit", "system", "code"}:
                        findings.append(
                            FhirOmopInformationLoss(
                                item.source_resource_hash,
                                resource_type,
                                f"{path}.{nested}",
                                "unsupported_element",
                            )
                        )
        elif key == "period":
            value = resource.get(key)
            if isinstance(value, Mapping):
                for nested in value:
                    if nested not in {"start", "end"}:
                        findings.append(
                            FhirOmopInformationLoss(
                                item.source_resource_hash,
                                resource_type,
                                f"{path}.{nested}",
                                "unsupported_element",
                            )
                        )
        elif key == "valueString":
            findings.append(
                FhirOmopInformationLoss(
                    item.source_resource_hash,
                    resource_type,
                    path,
                    "unsupported_element",
                )
            )
    return findings


def _codeable_information_loss(
    item: _PreparedResource, key: str, value: Any
) -> list[FhirOmopInformationLoss]:
    if not isinstance(value, Mapping):
        return []
    findings: list[FhirOmopInformationLoss] = []
    path = f"{item.resource_type}.{key}"
    for nested in value:
        if nested not in {"coding", "text"}:
            findings.append(
                FhirOmopInformationLoss(
                    item.source_resource_hash,
                    item.resource_type,
                    f"{path}.{nested}",
                    "unsupported_element",
                )
            )
    raw_codings = value.get("coding")
    if isinstance(raw_codings, Mapping):
        raw_codings = [raw_codings]
    if isinstance(raw_codings, Sequence) and not isinstance(raw_codings, (str, bytes)):
        for index, coding in enumerate(raw_codings):
            if not isinstance(coding, Mapping):
                continue
            for nested in coding:
                if nested not in {"system", "code", "display"}:
                    findings.append(
                        FhirOmopInformationLoss(
                            item.source_resource_hash,
                            item.resource_type,
                            f"{path}.coding[{index}].{nested}",
                            "unsupported_element",
                        )
                    )
    text = str(value.get("text") or "").strip()
    displays = {
        str(coding.get("display") or "").strip()
        for coding in raw_codings or ()
        if isinstance(coding, Mapping)
    }
    if text and displays and text not in displays:
        findings.append(
            FhirOmopInformationLoss(
                item.source_resource_hash,
                item.resource_type,
                f"{path}.text",
                "free text not retained",
            )
        )
    return findings


def _deduplicate_losses(
    losses: Iterable[FhirOmopInformationLoss],
) -> list[FhirOmopInformationLoss]:
    unique = {
        (
            finding.source_resource_hash,
            finding.resource_type,
            finding.element_path,
            finding.reason,
        ): finding
        for finding in losses
    }
    return [unique[key] for key in sorted(unique)]


def _set_reverse_metadata(
    resource: dict[str, Any], metadata: Mapping[str, Any]
) -> None:
    resource_type = str(metadata["resource_type"])
    if metadata.get("subject_reference") and resource_type != "Patient":
        resource["subject"] = {"reference": metadata["subject_reference"]}
    if metadata.get("encounter_reference") and resource_type not in {
        "Patient",
        "Encounter",
    }:
        resource["encounter"] = {"reference": metadata["encounter_reference"]}
    if metadata.get("status") is not None:
        resource["status"] = metadata["status"]
    if metadata.get("intent") is not None:
        resource["intent"] = metadata["intent"]
    if (
        metadata.get("date_path")
        and metadata.get("date_value") is not None
        and resource_type != "Encounter"
    ):
        field = str(metadata["date_path"]).split(".", 1)[1].split(".")[-1]
        resource[field] = metadata["date_value"]
    if resource_type == "Encounter" and (
        metadata.get("period_start") or metadata.get("period_end")
    ):
        resource["period"] = {
            output_key: metadata[input_key]
            for input_key, output_key in (
                ("period_start", "start"),
                ("period_end", "end"),
            )
            if metadata.get(input_key) is not None
        }
    if resource_type == "Condition":
        if metadata.get("clinical_status"):
            resource["clinicalStatus"] = {
                "coding": [{"code": metadata["clinical_status"]}]
            }
        if metadata.get("verification_status"):
            resource["verificationStatus"] = {
                "coding": [{"code": metadata["verification_status"]}]
            }


def _set_reverse_codeable(
    resource: dict[str, Any], path: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    field = path.split(".", 1)[1]
    if field == "class":
        row = rows[0]
        coding = _reverse_coding(row)
        if coding:
            resource[field] = coding
        return
    codings = [_reverse_coding(row) for row in rows]
    codings = [coding for coding in codings if coding]
    codeable: dict[str, Any] = {}
    if codings:
        codeable["coding"] = codings
    text = next((row.get("text") for row in rows if row.get("text")), None)
    if text:
        codeable["text"] = text
    if codeable:
        resource[field] = codeable


def _reverse_coding(row: Mapping[str, Any]) -> dict[str, Any] | None:
    coding: dict[str, Any] = {}
    if row.get("system"):
        coding["system"] = row["system"]
    if row.get("code"):
        coding["code"] = row["code"]
    if row.get("display"):
        coding["display"] = row["display"]
    return coding or None


def _set_reverse_value(resource: dict[str, Any], row: Mapping[str, Any]) -> None:
    field = str(row["element_path"]).split(".", 1)[1]
    kind = str(row["value_kind"])
    if kind == "Quantity":
        quantity: dict[str, Any] = {}
        if row.get("value_number") is not None:
            quantity["value"] = row["value_number"]
        for key in ("unit", "system", "code"):
            if row.get(key) is not None:
                quantity[key] = row[key]
        resource[field] = quantity
    elif kind in {"Integer", "Decimal"}:
        resource[field] = row.get("value_number")
    elif kind == "Boolean":
        resource[field] = bool(row.get("value_boolean"))
    elif kind == "DateTime":
        resource[field] = row.get("value_date")


def _coded_content_digest(
    payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> str:
    resources = _coerce_resources(payload)
    records: list[tuple[str, str, int, str, str, str]] = []
    for resource, _ in resources:
        resource_type = str(resource.get("resourceType") or "")
        if resource_type not in FHIR_RESOURCE_TYPES:
            continue
        primary = _RESOURCE_CODE_PATH.get(resource_type)
        paths = [path for path in (primary,) if path]
        paths.extend(
            path
            for path in (
                "Condition.clinicalStatus",
                "Condition.verificationStatus",
                "Observation.valueCodeableConcept",
            )
            if path.startswith(resource_type + ".") and path not in paths
        )
        for path in paths:
            field = path.split(".", 1)[1]
            value = resource.get(field)
            if (
                resource_type == "Encounter"
                and field == "class"
                and isinstance(value, Mapping)
            ):
                value = {"coding": [value]}
            for candidate in _codeable_rows(value, path):
                records.append(
                    (
                        resource_type,
                        path,
                        int(candidate["coding_index"]),
                        str(candidate.get("system") or ""),
                        str(candidate.get("code") or ""),
                        str(candidate.get("display") or ""),
                    )
                )
    return _hash_json(sorted(records))


def _reject_fabricated_concept_fields(value: Any, path: str = "resource") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).replace("-", "_").casefold()
            if normalized in FABRICATED_CONCEPT_FIELDS:
                raise FhirOmopMappingError(
                    f"FHIR input contains an embedded OMOP concept field at {path}.{key}"
                )
            _reject_fabricated_concept_fields(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_fabricated_concept_fields(nested, f"{path}[{index}]")


def _hash_identifier(
    source_system: str, source_version: str, resource_type: str, identifier: str
) -> str:
    return hashlib.sha256(
        "\x1f".join(
            (source_system, source_version, resource_type, str(identifier))
        ).encode("utf-8")
    ).hexdigest()


def _hash_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _required_label(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty label")
    return text


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FhirOmopMappingError(
            "concept IDs in supplied mappings must be integers"
        ) from exc


def _sidecar_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("source_resource_hash") or ""),
        str(row.get("element_path") or ""),
        int(
            row.get("coding_index", row.get("value_id", row.get("provenance_id", 0)))
            or 0
        ),
    )


def _create_sidecar_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS fhir_resource (
            source_resource_hash TEXT PRIMARY KEY,
            source_id_hash TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            deidentified_id TEXT NOT NULL,
            person_source_hash TEXT NOT NULL,
            visit_source_hash TEXT NOT NULL,
            subject_reference TEXT,
            encounter_reference TEXT,
            status TEXT,
            intent TEXT,
            date_path TEXT,
            date_value TEXT,
            period_start TEXT,
            period_end TEXT,
            clinical_status TEXT,
            verification_status TEXT,
            note_id INTEGER,
            visit_occurrence_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS fhir_code (
            code_row_id INTEGER PRIMARY KEY,
            source_resource_hash TEXT NOT NULL,
            source_id_hash TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            element_path TEXT NOT NULL,
            codeable_path TEXT NOT NULL,
            code_role TEXT NOT NULL,
            coding_index INTEGER NOT NULL,
            system TEXT,
            code TEXT,
            display TEXT,
            text TEXT,
            source_concept_id INTEGER NOT NULL,
            target_concept_id INTEGER NOT NULL,
            source_vocabulary_id TEXT,
            target_vocabulary_id TEXT,
            standard_concept TEXT,
            vocabulary_snapshot TEXT,
            unmapped_reason TEXT,
            omop_table TEXT,
            omop_row_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS fhir_value (
            value_id INTEGER PRIMARY KEY,
            source_resource_hash TEXT NOT NULL,
            source_id_hash TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            element_path TEXT NOT NULL,
            value_kind TEXT NOT NULL,
            value_number REAL,
            value_boolean INTEGER,
            value_date TEXT,
            unit TEXT,
            system TEXT,
            code TEXT
        );
        CREATE TABLE IF NOT EXISTS fhir_provenance (
            provenance_id INTEGER PRIMARY KEY,
            source_resource_hash TEXT NOT NULL,
            source_id_hash TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            element_path TEXT NOT NULL,
            source_system TEXT NOT NULL,
            source_version TEXT NOT NULL,
            source_code TEXT,
            source_vocabulary_id TEXT,
            source_concept_id INTEGER NOT NULL,
            target_concept_id INTEGER NOT NULL,
            target_vocabulary_id TEXT NOT NULL,
            standard_concept TEXT,
            vocabulary_snapshot TEXT NOT NULL,
            unmapped_reason TEXT,
            note_id INTEGER NOT NULL,
            note_nlp_id INTEGER NOT NULL,
            offset INTEGER NOT NULL,
            offset_end INTEGER NOT NULL,
            omop_table TEXT,
            omop_row_id INTEGER,
            idempotent_key TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fhir_information_loss (
            source_resource_hash TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            element_path TEXT NOT NULL,
            reason TEXT NOT NULL,
            PRIMARY KEY (source_resource_hash, element_path, reason)
        );
        """
    )


def _insert_sidecar_rows(
    con: sqlite3.Connection, table: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    if not rows:
        return
    columns = tuple(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    names = ", ".join(f'"{column}"' for column in columns)
    statement = f'INSERT OR IGNORE INTO "{table}" ({names}) VALUES ({placeholders})'
    values = [
        tuple(_sqlite_value(row.get(column)) for column in columns) for row in rows
    ]
    con.executemany(statement, values)


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value
