"""Loss-aware FHIR R4 exchange for immutable Journey records.

The standard FHIR fields make supported facts useful to ordinary R4 clients.
An explicit OpenMed extension also carries the canonical Journey record so an
OpenMed-to-OpenMed round trip is exact.  The extension is deliberately visible
and documented: it is not an encrypted envelope and a produced Bundle must be
protected like any other clinical record.

No network, terminology lookup, or implicit vocabulary download occurs here.
Unknown fact types and resources are returned as typed losses rather than
being silently discarded.
"""

from __future__ import annotations

import base64
import binascii
import copy
import json
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any, Final

from openmed.clinical.journey_contracts import (
    ClinicalFact,
    DatasetSnapshot,
    JourneyContractError,
    ResolutionEvent,
    canonical_digest,
    canonical_json,
)
from openmed.structured.store import StoreResult, StoreState

from .reference_integrity import reference_integrity_report
from .validation import validation_result

FHIR_JOURNEY_SCHEMA_VERSION: Final = "1.0.0"
FHIR_JOURNEY_COMPATIBILITY_POLICY: Final = "same_major"
FHIR_JOURNEY_RELEASE: Final = "4.0.1"
FHIR_JOURNEY_SCHEMA_NAME: Final = "fhir_journey_roundtrip"
FHIR_JOURNEY_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"

OPENMED_FACT_EXTENSION: Final = (
    "https://openmed.dev/fhir/StructureDefinition/journey-clinical-fact"
)
OPENMED_JOURNEY_EVENT_EXTENSION: Final = (
    "https://openmed.dev/fhir/StructureDefinition/journey-event"
)
OPENMED_RESOLUTION_EXTENSION: Final = (
    "https://openmed.dev/fhir/StructureDefinition/journey-resolution-event"
)
OPENMED_SNAPSHOT_EXTENSION: Final = (
    "https://openmed.dev/fhir/StructureDefinition/journey-dataset-snapshot"
)
OPENMED_CANONICAL_HASH_SYSTEM: Final = (
    "https://openmed.dev/fhir/sid/journey-canonical-hash"
)
OPENMED_FACT_ID_SYSTEM: Final = "https://openmed.dev/fhir/sid/journey-fact-id"
OPENMED_SUBJECT_ID_SYSTEM: Final = "https://openmed.dev/fhir/sid/journey-subject-id"
OPENMED_EVIDENCE_ID_SYSTEM: Final = "https://openmed.dev/fhir/sid/journey-evidence-id"
OPENMED_DERIVATION_HASH_SYSTEM: Final = (
    "https://openmed.dev/fhir/sid/journey-derivation-hash"
)

_BUNDLE_NAMESPACE: Final = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://openmed.dev/fhir/journey-roundtrip",
)
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_FHIR_ID_RE = re.compile(r"^[A-Za-z0-9\-.]{1,64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")

_RESOURCE_BY_FACT_TYPE: Final[Mapping[str, str]] = {
    "condition": "Condition",
    "diagnosis": "Condition",
    "problem": "Condition",
    "drug": "MedicationStatement",
    "medication": "MedicationStatement",
    "procedure": "Procedure",
    "laboratory": "Observation",
    "measurement": "Observation",
    "observation": "Observation",
    "social_determinant": "Observation",
    "vital": "Observation",
}
_FACT_RESOURCE_TYPES: Final = frozenset(_RESOURCE_BY_FACT_TYPE.values())
_OBSERVATION_STATUSES: Final = frozenset(
    {
        "registered",
        "preliminary",
        "final",
        "amended",
        "corrected",
        "cancelled",
        "entered-in-error",
        "unknown",
    }
)
_MEDICATION_STATUSES: Final = frozenset(
    {
        "active",
        "completed",
        "entered-in-error",
        "intended",
        "stopped",
        "on-hold",
        "unknown",
        "not-taken",
    }
)
_PROCEDURE_STATUSES: Final = frozenset(
    {
        "preparation",
        "in-progress",
        "not-done",
        "on-hold",
        "stopped",
        "completed",
        "entered-in-error",
        "unknown",
    }
)
_CONDITION_STATUSES: Final = frozenset(
    {"active", "recurrence", "relapse", "inactive", "remission", "resolved"}
)


class FHIRJourneyRoundTripError(ValueError):
    """Base error for malformed round-trip contracts."""


class FHIRJourneyConflictError(FHIRJourneyRoundTripError):
    """Raised when identities, snapshots, or canonical hashes disagree."""


class FHIRJourneyUnsupportedError(FHIRJourneyRoundTripError):
    """Raised when a requested exchange feature has no supported projection."""


@dataclass(frozen=True, slots=True)
class FHIRJourneyLoss:
    """Value-free record of one unsupported or lossy conversion surface."""

    record_id: str | None
    path: str
    reason_code: str

    def __post_init__(self) -> None:
        if self.record_id is not None:
            _opaque_id(self.record_id, "loss record_id")
        _path(self.path)
        _controlled(self.reason_code, "loss reason_code")

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-free loss payload."""

        return {
            "path": self.path,
            "reason_code": self.reason_code,
            "record_id": self.record_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FHIRJourneyLoss":
        """Build a strict loss record."""

        data = _mapping(value, "loss")
        _exact_keys(data, {"path", "reason_code", "record_id"}, "loss")
        record_id = data["record_id"]
        if record_id is not None and not isinstance(record_id, str):
            raise FHIRJourneyRoundTripError("loss record_id must be text or null")
        return cls(
            record_id=record_id,
            path=_text(data["path"], "loss path"),
            reason_code=_text(data["reason_code"], "loss reason_code"),
        )


@dataclass(frozen=True, slots=True)
class FHIREvidenceMapping:
    """Stable record-to-resource mapping without clinical values."""

    record_id: str
    target_reference: str
    evidence_ids: tuple[str, ...]
    canonical_hash: str

    def __post_init__(self) -> None:
        _opaque_id(self.record_id, "mapping record_id")
        _target_reference(self.target_reference)
        object.__setattr__(
            self,
            "evidence_ids",
            _opaque_ids(self.evidence_ids, "mapping evidence_ids"),
        )
        _digest(self.canonical_hash, "mapping canonical_hash")

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free evidence mapping."""

        return {
            "canonical_hash": self.canonical_hash,
            "evidence_ids": list(self.evidence_ids),
            "record_id": self.record_id,
            "target_reference": self.target_reference,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FHIREvidenceMapping":
        """Build a strict evidence mapping."""

        data = _mapping(value, "evidence mapping")
        _exact_keys(
            data,
            {"canonical_hash", "evidence_ids", "record_id", "target_reference"},
            "evidence mapping",
        )
        return cls(
            record_id=_text(data["record_id"], "mapping record_id"),
            target_reference=_text(
                data["target_reference"], "mapping target_reference"
            ),
            evidence_ids=_text_sequence(data["evidence_ids"], "mapping evidence_ids"),
            canonical_hash=_text(data["canonical_hash"], "mapping canonical_hash"),
        )


@dataclass(frozen=True, slots=True)
class FHIRJourneyEventPayload:
    """Exact serialized Journey event attached to its FHIR fact resource."""

    event_id: str
    fact_id: str
    payload: Mapping[str, Any]
    canonical_hash: str

    def __post_init__(self) -> None:
        _opaque_id(self.event_id, "event_id")
        _opaque_id(self.fact_id, "event fact_id")
        payload = json.loads(canonical_json(_mapping(self.payload, "event payload")))
        if payload.get("event_id") != self.event_id:
            raise FHIRJourneyConflictError("event payload identity differs")
        fact = payload.get("fact")
        if not isinstance(fact, Mapping) or fact.get("fact_id") != self.fact_id:
            raise FHIRJourneyConflictError("event payload fact identity differs")
        if canonical_digest(payload) != self.canonical_hash:
            raise FHIRJourneyConflictError("event payload hash differs")
        object.__setattr__(self, "payload", payload)
        _digest(self.canonical_hash, "event canonical_hash")

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "FHIRJourneyEventPayload":
        """Build a digest-bound event payload from canonical event JSON."""

        payload = _mapping(value, "event payload")
        event_id = _text(payload.get("event_id"), "event_id")
        fact = _mapping(payload.get("fact"), "event fact")
        fact_id = _text(fact.get("fact_id"), "event fact_id")
        return cls(
            event_id=event_id,
            fact_id=fact_id,
            payload=payload,
            canonical_hash=canonical_digest(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the exact event plus value-free custody fields."""

        return {
            "canonical_hash": self.canonical_hash,
            "event_id": self.event_id,
            "fact_id": self.fact_id,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class FHIRJourneyExport:
    """FHIR Bundle plus complete conversion custody and visible losses."""

    bundle: Mapping[str, Any]
    source_snapshot: DatasetSnapshot
    evidence_mappings: tuple[FHIREvidenceMapping, ...]
    losses: tuple[FHIRJourneyLoss, ...] = ()
    schema_version: str = FHIR_JOURNEY_SCHEMA_VERSION
    compatibility_policy: str = FHIR_JOURNEY_COMPATIBILITY_POLICY
    fhir_release: str = FHIR_JOURNEY_RELEASE

    def __post_init__(self) -> None:
        _contract_header(
            self.schema_version, self.compatibility_policy, self.fhir_release
        )
        if not isinstance(self.source_snapshot, DatasetSnapshot):
            raise TypeError("source_snapshot must be DatasetSnapshot")
        bundle = copy.deepcopy(dict(_mapping(self.bundle, "FHIR bundle")))
        if bundle.get("resourceType") != "Bundle":
            raise FHIRJourneyRoundTripError("FHIR export target must be a Bundle")
        object.__setattr__(self, "bundle", bundle)
        object.__setattr__(
            self,
            "evidence_mappings",
            tuple(sorted(self.evidence_mappings, key=lambda item: item.record_id)),
        )
        object.__setattr__(
            self,
            "losses",
            tuple(
                sorted(
                    self.losses,
                    key=lambda item: (
                        item.record_id or "",
                        item.path,
                        item.reason_code,
                    ),
                )
            ),
        )

    @property
    def lossless(self) -> bool:
        """Return whether every input record has an exact representation."""

        return not self.losses

    @property
    def canonical_hash(self) -> str:
        """Return the digest of the complete persisted contract."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned export contract."""

        return {
            "bundle": copy.deepcopy(dict(self.bundle)),
            "compatibility_policy": self.compatibility_policy,
            "evidence_mappings": [item.to_dict() for item in self.evidence_mappings],
            "fhir_release": self.fhir_release,
            "losses": [item.to_dict() for item in self.losses],
            "lossless": self.lossless,
            "schema_version": self.schema_version,
            "source_snapshot": self.source_snapshot.to_dict(),
        }

    def to_json(self) -> str:
        """Serialize the contract deterministically."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FHIRJourneyExport":
        """Build and validate a persisted export contract."""

        data = dict(_mapping(value, "FHIR journey export"))
        _exact_keys(
            data,
            {
                "bundle",
                "compatibility_policy",
                "evidence_mappings",
                "fhir_release",
                "losses",
                "lossless",
                "schema_version",
                "source_snapshot",
            },
            "FHIR journey export",
        )
        result = cls(
            bundle=_mapping(data["bundle"], "bundle"),
            source_snapshot=DatasetSnapshot.from_dict(
                _mapping(data["source_snapshot"], "source_snapshot")
            ),
            evidence_mappings=tuple(
                FHIREvidenceMapping.from_dict(_mapping(item, "evidence mapping"))
                for item in _sequence(data["evidence_mappings"], "evidence_mappings")
            ),
            losses=tuple(
                FHIRJourneyLoss.from_dict(_mapping(item, "loss"))
                for item in _sequence(data["losses"], "losses")
            ),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
            fhir_release=_text(data["fhir_release"], "fhir_release"),
        )
        if data["lossless"] is not result.lossless:
            raise FHIRJourneyConflictError("persisted lossless flag differs")
        return result


@dataclass(frozen=True, slots=True)
class FHIRJourneyImport:
    """Journey records reconstructed from an OpenMed FHIR Bundle."""

    facts: tuple[ClinicalFact, ...]
    events: tuple[FHIRJourneyEventPayload, ...]
    resolutions: tuple[ResolutionEvent, ...]
    source_snapshot: DatasetSnapshot | None
    evidence_mappings: tuple[FHIREvidenceMapping, ...]
    losses: tuple[FHIRJourneyLoss, ...] = ()
    schema_version: str = FHIR_JOURNEY_SCHEMA_VERSION
    compatibility_policy: str = FHIR_JOURNEY_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract_header(self.schema_version, self.compatibility_policy, None)
        object.__setattr__(self, "facts", tuple(self.facts))
        object.__setattr__(
            self, "events", tuple(sorted(self.events, key=lambda item: item.event_id))
        )
        object.__setattr__(self, "resolutions", tuple(self.resolutions))
        object.__setattr__(
            self,
            "evidence_mappings",
            tuple(sorted(self.evidence_mappings, key=lambda item: item.record_id)),
        )
        object.__setattr__(
            self,
            "losses",
            tuple(
                sorted(
                    self.losses,
                    key=lambda item: (
                        item.record_id or "",
                        item.path,
                        item.reason_code,
                    ),
                )
            ),
        )

    @property
    def lossless(self) -> bool:
        """Return whether the import observed no unsupported surface."""

        return not self.losses

    def to_dict(self) -> dict[str, Any]:
        """Return the imported records and conversion custody."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "evidence_mappings": [item.to_dict() for item in self.evidence_mappings],
            "events": [item.to_dict() for item in self.events],
            "facts": [item.to_dict() for item in self.facts],
            "losses": [item.to_dict() for item in self.losses],
            "lossless": self.lossless,
            "resolutions": [item.to_dict() for item in self.resolutions],
            "schema_version": self.schema_version,
            "source_snapshot": (
                None if self.source_snapshot is None else self.source_snapshot.to_dict()
            ),
        }


def export_journey_to_fhir(
    facts: Iterable[ClinicalFact],
    *,
    source_snapshot: DatasetSnapshot,
    resolutions: Iterable[ResolutionEvent] = (),
    events: Iterable[Any] = (),
    strict: bool = False,
) -> StoreResult[FHIRJourneyExport]:
    """Export Journey facts and resolutions as a deterministic FHIR R4 Bundle.

    ``strict=False`` returns a typed partial result with a populated loss report
    when at least one input is representable. ``strict=True`` changes that
    state to ``unsupported`` while retaining the report in ``value``.
    """

    materialized_facts = tuple(facts)
    materialized_resolutions = tuple(resolutions)
    materialized_events = tuple(events)
    if (
        not materialized_facts
        and not materialized_resolutions
        and not materialized_events
    ):
        return StoreResult.outcome(StoreState.UNKNOWN, "fhir_inputs_empty")
    try:
        from openmed.clinical.journey import JourneyEvent

        if not isinstance(source_snapshot, DatasetSnapshot):
            raise TypeError("source_snapshot must be DatasetSnapshot")
        if any(not isinstance(item, ClinicalFact) for item in materialized_facts):
            raise TypeError("facts must contain ClinicalFact records")
        if any(
            not isinstance(item, ResolutionEvent) for item in materialized_resolutions
        ):
            raise TypeError("resolutions must contain ResolutionEvent records")
        if any(not isinstance(item, JourneyEvent) for item in materialized_events):
            raise TypeError("events must contain JourneyEvent records")
        _unique_ids(item.event_id for item in materialized_events)
        facts_by_id = {item.fact_id: item for item in materialized_facts}
        if len(facts_by_id) != len(materialized_facts):
            raise FHIRJourneyConflictError("record identities must be unique")
        for event in materialized_events:
            existing_fact = facts_by_id.get(event.fact.fact_id)
            if (
                existing_fact is not None
                and existing_fact.canonical_hash != event.fact.canonical_hash
            ):
                raise FHIRJourneyConflictError("event fact differs from supplied fact")
            facts_by_id[event.fact.fact_id] = event.fact
        resolution_by_id = {
            item.resolution_id: item for item in materialized_resolutions
        }
        if len(resolution_by_id) != len(materialized_resolutions):
            raise FHIRJourneyConflictError("record identities must be unique")
        for event in materialized_events:
            for resolution in event.resolutions:
                existing_resolution = resolution_by_id.get(resolution.resolution_id)
                if (
                    existing_resolution is not None
                    and existing_resolution.canonical_hash != resolution.canonical_hash
                ):
                    raise FHIRJourneyConflictError(
                        "event resolution differs from supplied resolution"
                    )
                resolution_by_id[resolution.resolution_id] = resolution
        merged_facts = tuple(facts_by_id.values())
        merged_resolutions = tuple(resolution_by_id.values())
        _validate_snapshot_custody(merged_facts, source_snapshot)
        return _export(
            merged_facts,
            source_snapshot=source_snapshot,
            resolutions=merged_resolutions,
            events=materialized_events,
            strict=strict,
        )
    except FHIRJourneyConflictError:
        return StoreResult.outcome(StoreState.CONFLICT, "fhir_input_conflict")
    except FHIRJourneyUnsupportedError:
        return StoreResult.outcome(StoreState.UNSUPPORTED, "fhir_unsupported")
    except (FHIRJourneyRoundTripError, JourneyContractError, TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "fhir_export_invalid")


def import_journey_from_fhir(
    bundle: Mapping[str, Any],
    *,
    expected_snapshot: DatasetSnapshot | None = None,
    strict: bool = False,
) -> StoreResult[FHIRJourneyImport]:
    """Import exact Journey records from an OpenMed-extended FHIR R4 Bundle."""

    try:
        if not isinstance(bundle, Mapping) or bundle.get("resourceType") != "Bundle":
            raise FHIRJourneyRoundTripError("FHIR import requires a Bundle")
        if expected_snapshot is not None and not isinstance(
            expected_snapshot, DatasetSnapshot
        ):
            raise TypeError("expected_snapshot must be DatasetSnapshot")
        return _import(bundle, expected_snapshot=expected_snapshot, strict=strict)
    except FHIRJourneyConflictError:
        return StoreResult.outcome(StoreState.CONFLICT, "fhir_custody_conflict")
    except FHIRJourneyUnsupportedError:
        return StoreResult.outcome(StoreState.UNSUPPORTED, "fhir_unsupported")
    except (FHIRJourneyRoundTripError, JourneyContractError, TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "fhir_import_invalid")


def load_fhir_journey_schema() -> Mapping[str, Any]:
    """Load the bundled strict JSON Schema for persisted export contracts."""

    payload = resources.files(FHIR_JOURNEY_SCHEMA_PACKAGE).joinpath(
        f"{FHIR_JOURNEY_SCHEMA_NAME}.schema.json"
    )
    parsed = json.loads(payload.read_text(encoding="utf-8"))
    if parsed.get("schema_version") != FHIR_JOURNEY_SCHEMA_VERSION:
        raise RuntimeError("bundled FHIR journey schema is incompatible")
    return parsed


def _export(
    facts: tuple[ClinicalFact, ...],
    *,
    source_snapshot: DatasetSnapshot,
    resolutions: tuple[ResolutionEvent, ...],
    events: tuple[Any, ...],
    strict: bool,
) -> StoreResult[FHIRJourneyExport]:
    supported = [fact for fact in facts if fact.fact_type in _RESOURCE_BY_FACT_TYPE]
    losses = [
        FHIRJourneyLoss(fact.fact_id, "fact_type", "fact_type_unsupported")
        for fact in facts
        if fact.fact_type not in _RESOURCE_BY_FACT_TYPE
    ]
    losses.extend(
        FHIRJourneyLoss(
            event.event_id,
            "event.fact.fact_type",
            "event_fact_type_unsupported",
        )
        for event in events
        if event.fact.fact_type not in _RESOURCE_BY_FACT_TYPE
    )
    subject_ids = sorted({fact.subject_id for fact in supported})
    resource_by_fact_id: dict[str, dict[str, Any]] = {}
    resources_out: list[dict[str, Any]] = [
        _subject_resource(subject_id) for subject_id in subject_ids
    ]
    mappings: list[FHIREvidenceMapping] = []
    events_by_fact_id = {item.fact.fact_id: item for item in events}
    for fact in sorted(supported, key=lambda item: item.fact_id):
        event = events_by_fact_id.get(fact.fact_id)
        event_payload = None if event is None else event.to_dict()
        resource = _fact_resource(fact, event_payload=event_payload)
        resources_out.append(resource)
        resource_by_fact_id[fact.fact_id] = resource
        reference = f"{resource['resourceType']}/{resource['id']}"
        mappings.append(
            FHIREvidenceMapping(
                record_id=fact.fact_id,
                target_reference=reference,
                evidence_ids=fact.evidence_ids,
                canonical_hash=fact.canonical_hash,
            )
        )
        if event is not None:
            mappings.append(
                FHIREvidenceMapping(
                    record_id=event.event_id,
                    target_reference=reference,
                    evidence_ids=fact.evidence_ids,
                    canonical_hash=canonical_digest(event_payload),
                )
            )

    for resolution in sorted(resolutions, key=lambda item: item.resolution_id):
        referenced = tuple(
            dict.fromkeys(
                (*resolution.selected_fact_ids, *resolution.rejected_fact_ids)
            )
        )
        missing = tuple(item for item in referenced if item not in resource_by_fact_id)
        if missing:
            losses.append(
                FHIRJourneyLoss(
                    resolution.resolution_id,
                    "resolution.target",
                    "resolution_target_unsupported",
                )
            )
            continue
        resource = _resolution_resource(resolution, resource_by_fact_id)
        resources_out.append(resource)
        mappings.append(
            FHIREvidenceMapping(
                record_id=resolution.resolution_id,
                target_reference=f"Provenance/{resource['id']}",
                evidence_ids=(),
                canonical_hash=resolution.canonical_hash,
            )
        )

    bundle = _bundle(resources_out, source_snapshot)
    integrity = reference_integrity_report(bundle, fhir_version="R4")
    validation = validation_result(bundle, "R4")
    if not integrity.valid:
        raise FHIRJourneyConflictError("emitted Bundle has invalid references")
    if not validation.valid:
        raise FHIRJourneyRoundTripError("emitted Bundle failed local validation")
    result = FHIRJourneyExport(
        bundle=bundle,
        source_snapshot=source_snapshot,
        evidence_mappings=tuple(mappings),
        losses=tuple(losses),
    )
    if losses:
        state = StoreState.UNSUPPORTED if strict or not mappings else StoreState.PARTIAL
        return StoreResult.outcome(
            state,
            "fhir_projection_unsupported" if strict else "fhir_projection_partial",
            value=result,
        )
    return StoreResult.success(result, created=True)


def _import(
    bundle: Mapping[str, Any],
    *,
    expected_snapshot: DatasetSnapshot | None,
    strict: bool,
) -> StoreResult[FHIRJourneyImport]:
    integrity = reference_integrity_report(bundle, fhir_version="R4")
    if not integrity.valid:
        raise FHIRJourneyConflictError("FHIR Bundle has invalid references")
    validation = validation_result(bundle, "R4")
    if not validation.valid:
        raise FHIRJourneyRoundTripError("FHIR Bundle failed local validation")
    snapshot = _snapshot_from_bundle(bundle)
    if expected_snapshot is not None:
        if (
            snapshot is None
            or snapshot.canonical_hash != expected_snapshot.canonical_hash
        ):
            raise FHIRJourneyConflictError("FHIR source snapshot differs")

    entries = _sequence(bundle.get("entry", ()), "Bundle.entry")
    facts: list[ClinicalFact] = []
    events: list[FHIRJourneyEventPayload] = []
    resolutions: list[ResolutionEvent] = []
    resolution_resources: list[tuple[ResolutionEvent, Mapping[str, Any]]] = []
    fact_resources: dict[str, Mapping[str, Any]] = {}
    subject_resources: dict[str, Mapping[str, Any]] = {}
    mappings: list[FHIREvidenceMapping] = []
    losses: list[FHIRJourneyLoss] = (
        []
        if snapshot is not None
        else [FHIRJourneyLoss(None, "Bundle.extension", "source_snapshot_missing")]
    )
    seen_record_ids: set[str] = set()

    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, "Bundle.entry")
        resource = _mapping(entry.get("resource"), "Bundle.entry.resource")
        resource_type = resource.get("resourceType")
        path = f"Bundle.entry[{index}].resource"
        if resource_type == "Patient":
            subject_id = _subject_id_from_resource(resource)
            if subject_id is None:
                losses.append(
                    FHIRJourneyLoss(None, path, "patient_carrier_unsupported")
                )
                continue
            if subject_id in subject_resources:
                raise FHIRJourneyConflictError("subject identity is duplicated")
            if canonical_json(resource) != canonical_json(
                _subject_resource(subject_id)
            ):
                raise FHIRJourneyConflictError("subject carrier differs")
            subject_resources[subject_id] = resource
            continue
        if resource_type in _FACT_RESOURCE_TYPES:
            payload = _canonical_extension_payload(resource, OPENMED_FACT_EXTENSION)
            if payload is None:
                losses.append(FHIRJourneyLoss(None, path, "canonical_payload_missing"))
                continue
            fact = ClinicalFact.from_dict(_mapping(payload, "clinical fact payload"))
            _verify_record_extension(resource, fact.fact_id, fact.canonical_hash)
            if fact.fact_type not in _RESOURCE_BY_FACT_TYPE:
                raise FHIRJourneyConflictError("fact type is not supported")
            if _RESOURCE_BY_FACT_TYPE[fact.fact_type] != resource_type:
                raise FHIRJourneyConflictError("fact resource type differs")
            event_payload = _canonical_extension_payload(
                resource, OPENMED_JOURNEY_EVENT_EXTENSION
            )
            event = (
                None
                if event_payload is None
                else FHIRJourneyEventPayload.from_payload(event_payload)
            )
            if event is not None and event.fact_id != fact.fact_id:
                raise FHIRJourneyConflictError("event fact differs")
            if canonical_json(resource) != canonical_json(
                _fact_resource(fact, event_payload=event_payload)
            ):
                raise FHIRJourneyConflictError("FHIR fact projection differs")
            _new_record_id(fact.fact_id, seen_record_ids)
            facts.append(fact)
            fact_resources[fact.fact_id] = resource
            mappings.append(
                FHIREvidenceMapping(
                    record_id=fact.fact_id,
                    target_reference=f"{resource_type}/{resource['id']}",
                    evidence_ids=fact.evidence_ids,
                    canonical_hash=fact.canonical_hash,
                )
            )
            if event is not None:
                _new_record_id(event.event_id, seen_record_ids)
                events.append(event)
                mappings.append(
                    FHIREvidenceMapping(
                        record_id=event.event_id,
                        target_reference=f"{resource_type}/{resource['id']}",
                        evidence_ids=fact.evidence_ids,
                        canonical_hash=event.canonical_hash,
                    )
                )
            continue
        if resource_type == "Provenance":
            payload = _canonical_extension_payload(
                resource, OPENMED_RESOLUTION_EXTENSION
            )
            if payload is None:
                losses.append(FHIRJourneyLoss(None, path, "canonical_payload_missing"))
                continue
            resolution = ResolutionEvent.from_dict(
                _mapping(payload, "resolution event payload")
            )
            _verify_record_extension(
                resource, resolution.resolution_id, resolution.canonical_hash
            )
            _new_record_id(resolution.resolution_id, seen_record_ids)
            resolutions.append(resolution)
            resolution_resources.append((resolution, resource))
            mappings.append(
                FHIREvidenceMapping(
                    record_id=resolution.resolution_id,
                    target_reference=f"Provenance/{resource['id']}",
                    evidence_ids=(),
                    canonical_hash=resolution.canonical_hash,
                )
            )
            continue
        losses.append(FHIRJourneyLoss(None, path, "resource_type_unsupported"))

    for fact in facts:
        if fact.subject_id not in subject_resources:
            raise FHIRJourneyConflictError("fact subject carrier is absent")
    for resolution, resource in resolution_resources:
        referenced = {
            *resolution.selected_fact_ids,
            *resolution.rejected_fact_ids,
        }
        if not referenced.issubset(fact_resources):
            raise FHIRJourneyConflictError("resolution target record is absent")
        expected_resources = {
            fact_id: fact_resources[fact_id] for fact_id in referenced
        }
        if canonical_json(resource) != canonical_json(
            _resolution_resource(resolution, expected_resources)
        ):
            raise FHIRJourneyConflictError("FHIR resolution projection differs")

    result = FHIRJourneyImport(
        facts=tuple(facts),
        events=tuple(events),
        resolutions=tuple(resolutions),
        source_snapshot=snapshot,
        evidence_mappings=tuple(mappings),
        losses=tuple(losses),
    )
    if not facts and not resolutions:
        state = StoreState.UNSUPPORTED if losses else StoreState.UNKNOWN
        code = "fhir_resources_unsupported" if losses else "fhir_records_empty"
        return StoreResult.outcome(state, code, value=result)
    if losses:
        state = StoreState.UNSUPPORTED if strict else StoreState.PARTIAL
        return StoreResult.outcome(
            state,
            "fhir_import_unsupported" if strict else "fhir_import_partial",
            value=result,
        )
    return StoreResult.success(result)


def _fact_resource(
    fact: ClinicalFact, *, event_payload: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    resource_type = _RESOURCE_BY_FACT_TYPE[fact.fact_type]
    resource_id = _fhir_id("fact", fact.fact_id)
    subject_ref = f"Patient/{_fhir_id('subject', fact.subject_id)}"
    extension = _record_extension(OPENMED_FACT_EXTENSION, fact.to_dict())
    identifiers = [
        {"system": OPENMED_FACT_ID_SYSTEM, "value": fact.fact_id},
        {"system": OPENMED_CANONICAL_HASH_SYSTEM, "value": fact.canonical_hash},
        {"system": OPENMED_DERIVATION_HASH_SYSTEM, "value": fact.derivation_hash},
        *(
            {"system": OPENMED_EVIDENCE_ID_SYSTEM, "value": evidence_id}
            for evidence_id in fact.evidence_ids
        ),
    ]
    code = _codeable_concept(fact)
    extensions = [extension]
    if event_payload is not None:
        extensions.append(
            _record_extension(OPENMED_JOURNEY_EVENT_EXTENSION, event_payload)
        )
    common: dict[str, Any] = {
        "resourceType": resource_type,
        "id": resource_id,
        "identifier": identifiers,
        "extension": extensions,
        "subject": {"reference": subject_ref},
    }
    if resource_type == "Condition":
        common["code"] = code
        common["verificationStatus"] = _status_concept(
            "http://terminology.hl7.org/CodeSystem/condition-ver-status",
            "confirmed" if fact.status != "entered-in-error" else "entered-in-error",
        )
        if fact.status in _CONDITION_STATUSES:
            common["clinicalStatus"] = _status_concept(
                "http://terminology.hl7.org/CodeSystem/condition-clinical",
                fact.status,
            )
        _add_effective(common, fact.effective_time, "onset")
    elif resource_type == "Observation":
        common["status"] = (
            fact.status if fact.status in _OBSERVATION_STATUSES else "unknown"
        )
        common["code"] = code
        _add_observation_value(common, fact)
        _add_effective(common, fact.effective_time, "effective")
    elif resource_type == "MedicationStatement":
        common["status"] = (
            fact.status if fact.status in _MEDICATION_STATUSES else "unknown"
        )
        common["medicationCodeableConcept"] = code
        _add_effective(common, fact.effective_time, "effective")
    else:
        common["status"] = (
            fact.status if fact.status in _PROCEDURE_STATUSES else "unknown"
        )
        common["code"] = code
        _add_effective(common, fact.effective_time, "performed")
    return common


def _resolution_resource(
    resolution: ResolutionEvent,
    resources_by_fact_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    target_ids = tuple(
        dict.fromkeys((*resolution.selected_fact_ids, *resolution.rejected_fact_ids))
    )
    targets = [
        {
            "reference": (
                f"{resources_by_fact_id[fact_id]['resourceType']}/"
                f"{resources_by_fact_id[fact_id]['id']}"
            )
        }
        for fact_id in target_ids
    ]
    return {
        "resourceType": "Provenance",
        "id": _fhir_id("resolution", resolution.resolution_id),
        "extension": [
            _record_extension(OPENMED_RESOLUTION_EXTENSION, resolution.to_dict())
        ],
        "target": targets,
        "recorded": resolution.occurred_at,
        "activity": {
            "coding": [
                {
                    "system": "https://openmed.dev/fhir/CodeSystem/journey-resolution",
                    "code": resolution.action,
                }
            ]
        },
        "agent": [
            {
                "type": {
                    "coding": [
                        {
                            "system": (
                                "http://terminology.hl7.org/CodeSystem/"
                                "provenance-participant-type"
                            ),
                            "code": "author",
                        }
                    ]
                },
                "who": {
                    "identifier": {
                        "system": "https://openmed.dev/fhir/sid/software",
                        "value": "openmed",
                    }
                },
            }
        ],
    }


def _subject_resource(subject_id: str) -> dict[str, Any]:
    return {
        "resourceType": "Patient",
        "id": _fhir_id("subject", subject_id),
        "identifier": [{"system": OPENMED_SUBJECT_ID_SYSTEM, "value": subject_id}],
    }


def _subject_id_from_resource(resource: Mapping[str, Any]) -> str | None:
    identifiers = resource.get("identifier", ())
    if not isinstance(identifiers, Sequence) or isinstance(identifiers, (str, bytes)):
        raise FHIRJourneyRoundTripError("Patient.identifier must be an array")
    values = [
        item.get("value")
        for item in identifiers
        if isinstance(item, Mapping) and item.get("system") == OPENMED_SUBJECT_ID_SYSTEM
    ]
    if not values:
        return None
    if len(values) != 1 or not isinstance(values[0], str):
        raise FHIRJourneyConflictError("Patient subject identifier conflicts")
    _opaque_id(values[0], "Patient subject identifier")
    return values[0]


def _bundle(
    resources_out: Sequence[Mapping[str, Any]], source_snapshot: DatasetSnapshot
) -> dict[str, Any]:
    snapshot_extension = _record_extension(
        OPENMED_SNAPSHOT_EXTENSION, source_snapshot.to_dict()
    )
    seed = canonical_digest(
        {
            "resources": [
                [resource["resourceType"], resource["id"]] for resource in resources_out
            ],
            "snapshot": source_snapshot.canonical_hash,
        }
    )
    bundle_id = _fhir_id("bundle", seed)
    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "extension": [snapshot_extension],
        "entry": [
            {
                "fullUrl": _full_url(
                    str(resource["resourceType"]), str(resource["id"])
                ),
                "resource": copy.deepcopy(dict(resource)),
            }
            for resource in resources_out
        ],
    }


def _record_extension(url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    canonical = canonical_json(payload)
    encoded = base64.b64encode(canonical.encode("utf-8")).decode("ascii")
    return {
        "url": url,
        "extension": [
            {"url": "schemaVersion", "valueString": FHIR_JOURNEY_SCHEMA_VERSION},
            {
                "url": "compatibilityPolicy",
                "valueCode": FHIR_JOURNEY_COMPATIBILITY_POLICY,
            },
            {
                "url": "canonicalHash",
                "valueString": canonical_digest(payload),
            },
            {"url": "canonicalJson", "valueBase64Binary": encoded},
        ],
    }


def _canonical_extension_payload(
    resource: Mapping[str, Any], expected_url: str
) -> Mapping[str, Any] | None:
    extensions = resource.get("extension", ())
    if not isinstance(extensions, Sequence) or isinstance(extensions, (str, bytes)):
        raise FHIRJourneyRoundTripError("FHIR extension must be an array")
    candidates = [
        item
        for item in extensions
        if isinstance(item, Mapping) and item.get("url") == expected_url
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise FHIRJourneyConflictError("canonical extension is duplicated")
    parts = candidates[0].get("extension")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        raise FHIRJourneyRoundTripError("canonical extension parts are invalid")
    values: dict[str, Any] = {}
    for item in parts:
        if not isinstance(item, Mapping) or not isinstance(item.get("url"), str):
            raise FHIRJourneyRoundTripError("canonical extension part is invalid")
        url = str(item["url"])
        value_keys = [key for key in item if key.startswith("value")]
        if len(value_keys) != 1 or url in values:
            raise FHIRJourneyConflictError("canonical extension part conflicts")
        values[url] = item[value_keys[0]]
    if values.get("schemaVersion") != FHIR_JOURNEY_SCHEMA_VERSION:
        raise FHIRJourneyUnsupportedError("FHIR journey schema is unsupported")
    if values.get("compatibilityPolicy") != FHIR_JOURNEY_COMPATIBILITY_POLICY:
        raise FHIRJourneyUnsupportedError("FHIR journey policy is unsupported")
    encoded = values.get("canonicalJson")
    if not isinstance(encoded, str):
        raise FHIRJourneyRoundTripError("canonical payload is absent")
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
        payload = json.loads(decoded)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        raise FHIRJourneyRoundTripError("canonical payload is invalid") from None
    if not isinstance(payload, Mapping):
        raise FHIRJourneyRoundTripError("canonical payload must be an object")
    if values.get("canonicalHash") != canonical_digest(payload):
        raise FHIRJourneyConflictError("canonical payload hash differs")
    return payload


def _snapshot_from_bundle(bundle: Mapping[str, Any]) -> DatasetSnapshot | None:
    payload = _canonical_extension_payload(bundle, OPENMED_SNAPSHOT_EXTENSION)
    if payload is None:
        return None
    return DatasetSnapshot.from_dict(payload)


def _verify_record_extension(
    resource: Mapping[str, Any], record_id: str, record_hash: str
) -> None:
    if resource.get("resourceType") != "Provenance":
        identifiers = resource.get("identifier", ())
        if not isinstance(identifiers, Sequence) or isinstance(
            identifiers, (str, bytes)
        ):
            raise FHIRJourneyRoundTripError("FHIR identifiers must be an array")
        hashes = {
            item.get("value")
            for item in identifiers
            if isinstance(item, Mapping)
            and item.get("system") == OPENMED_CANONICAL_HASH_SYSTEM
        }
        if hashes != {record_hash}:
            raise FHIRJourneyConflictError("FHIR canonical identifier differs")
    expected_id = (
        _fhir_id("fact", record_id)
        if record_id.startswith("fact_")
        else _fhir_id("resolution", record_id)
    )
    if resource.get("id") != expected_id:
        raise FHIRJourneyConflictError("FHIR logical id differs")


def _codeable_concept(fact: ClinicalFact) -> dict[str, Any]:
    value = fact.value
    if isinstance(value, Mapping):
        code_value = value.get("code")
        system_value = value.get("system")
        display_value = value.get("display")
        text_value = value.get("text")
        code = str(code_value) if code_value not in (None, "") else fact.fact_type
        system = (
            str(system_value)
            if isinstance(system_value, str) and system_value
            else "https://openmed.dev/fhir/CodeSystem/journey-fact-type"
        )
        coding: dict[str, Any] = {"system": system, "code": code}
        if isinstance(display_value, str) and display_value:
            coding["display"] = display_value
        result: dict[str, Any] = {"coding": [coding]}
        if isinstance(text_value, str) and text_value:
            result["text"] = text_value
        elif isinstance(display_value, str) and display_value:
            result["text"] = display_value
        return result
    return {
        "coding": [
            {
                "system": "https://openmed.dev/fhir/CodeSystem/journey-fact-type",
                "code": fact.fact_type,
            }
        ]
    }


def _add_observation_value(resource: dict[str, Any], fact: ClinicalFact) -> None:
    value: Any = fact.value
    if isinstance(value, Mapping):
        if "value" not in value:
            return
        value = value["value"]
    if value is None:
        return
    if type(value) is bool:
        resource["valueBoolean"] = value
    elif type(value) is int and fact.unit is None:
        resource["valueInteger"] = value
    elif isinstance(value, (int, float)):
        quantity: dict[str, Any] = {"value": value}
        if fact.unit:
            quantity.update(
                {
                    "code": fact.unit,
                    "system": "http://unitsofmeasure.org",
                    "unit": fact.unit,
                }
            )
        resource["valueQuantity"] = quantity
    elif isinstance(value, str):
        resource["valueString"] = value


def _add_effective(
    resource: dict[str, Any], effective_time: Mapping[str, Any], prefix: str
) -> None:
    start = effective_time.get("start")
    end = effective_time.get("end")
    if isinstance(start, str) and isinstance(end, str):
        resource[f"{prefix}Period"] = {"start": start, "end": end}
    elif isinstance(start, str):
        resource[f"{prefix}DateTime"] = start


def _status_concept(system: str, code: str) -> dict[str, Any]:
    return {"coding": [{"system": system, "code": code}]}


def _fhir_id(kind: str, value: str) -> str:
    digest = uuid.uuid5(_BUNDLE_NAMESPACE, f"{kind}:{value}").hex
    result = f"{kind}-{digest[:32]}"
    if _FHIR_ID_RE.fullmatch(result) is None:  # pragma: no cover - invariant
        raise FHIRJourneyRoundTripError("derived FHIR id is invalid")
    return result


def _full_url(resource_type: str, resource_id: str) -> str:
    return f"urn:uuid:{uuid.uuid5(_BUNDLE_NAMESPACE, f'{resource_type}/{resource_id}')}"


def _contract_header(
    schema_version: str, compatibility_policy: str, fhir_release: str | None
) -> None:
    if schema_version != FHIR_JOURNEY_SCHEMA_VERSION:
        raise FHIRJourneyUnsupportedError("FHIR journey schema is unsupported")
    if compatibility_policy != FHIR_JOURNEY_COMPATIBILITY_POLICY:
        raise FHIRJourneyUnsupportedError("FHIR journey policy is unsupported")
    if fhir_release is not None and fhir_release != FHIR_JOURNEY_RELEASE:
        raise FHIRJourneyUnsupportedError("FHIR release is unsupported")


def _new_record_id(value: str, seen: set[str]) -> None:
    if value in seen:
        raise FHIRJourneyConflictError("record identity is duplicated")
    seen.add(value)


def _unique_ids(values: Iterable[str]) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise FHIRJourneyConflictError("record identities must be unique")


def _validate_snapshot_custody(
    facts: Sequence[ClinicalFact], snapshot: DatasetSnapshot
) -> None:
    fact_ids = {item.fact_id for item in facts}
    if not fact_ids.issubset(snapshot.source_fact_ids):
        raise FHIRJourneyConflictError("source snapshot does not cover every fact")
    splits_by_subject: dict[str, set[str]] = {}
    for fact in facts:
        split = fact.attributes.get("dataset_split")
        if split is None:
            continue
        if not isinstance(split, str) or split not in {
            "train",
            "validation",
            "holdout",
        }:
            raise FHIRJourneyUnsupportedError("dataset split is unsupported")
        if snapshot.split_hashes and split not in snapshot.split_hashes:
            raise FHIRJourneyConflictError("dataset split is absent from snapshot")
        splits_by_subject.setdefault(fact.subject_id, set()).add(split)
    if any(len(splits) > 1 for splits in splits_by_subject.values()):
        raise FHIRJourneyConflictError("subject appears in multiple dataset splits")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FHIRJourneyRoundTripError(f"{field_name} must be an object")
    return value


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FHIRJourneyRoundTripError(f"{field_name} must be an array")
    return value


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FHIRJourneyRoundTripError(f"{field_name} must be non-empty text")
    return value


def _text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    return tuple(_text(item, field_name) for item in _sequence(value, field_name))


def _exact_keys(value: Mapping[str, Any], expected: set[str], field_name: str) -> None:
    if set(value) != expected:
        raise FHIRJourneyRoundTripError(f"{field_name} fields are incompatible")


def _controlled(value: str, field_name: str) -> str:
    if _CONTROLLED_RE.fullmatch(value) is None:
        raise FHIRJourneyRoundTripError(f"{field_name} must be controlled")
    return value


def _opaque_id(value: str, field_name: str) -> str:
    if _OPAQUE_ID_RE.fullmatch(value) is None:
        raise FHIRJourneyRoundTripError(f"{field_name} must be opaque")
    return value


def _opaque_ids(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(result) != len(set(result)):
        raise FHIRJourneyConflictError(f"{field_name} must be unique")
    for value in result:
        _opaque_id(value, field_name)
    return result


def _digest(value: str, field_name: str) -> str:
    if _DIGEST_RE.fullmatch(value) is None:
        raise FHIRJourneyRoundTripError(f"{field_name} must be a digest")
    return value


def _path(value: str) -> str:
    if not value or len(value) > 256 or any(ord(char) < 32 for char in value):
        raise FHIRJourneyRoundTripError("loss path is invalid")
    return value


def _target_reference(value: str) -> str:
    if not value or "/" not in value or len(value) > 160:
        raise FHIRJourneyRoundTripError("target reference is invalid")
    return value


__all__ = [
    "FHIR_JOURNEY_COMPATIBILITY_POLICY",
    "FHIR_JOURNEY_RELEASE",
    "FHIR_JOURNEY_SCHEMA_NAME",
    "FHIR_JOURNEY_SCHEMA_VERSION",
    "FHIREvidenceMapping",
    "FHIRJourneyEventPayload",
    "FHIRJourneyConflictError",
    "FHIRJourneyExport",
    "FHIRJourneyImport",
    "FHIRJourneyLoss",
    "FHIRJourneyRoundTripError",
    "FHIRJourneyUnsupportedError",
    "OPENMED_CANONICAL_HASH_SYSTEM",
    "OPENMED_DERIVATION_HASH_SYSTEM",
    "OPENMED_EVIDENCE_ID_SYSTEM",
    "OPENMED_FACT_EXTENSION",
    "OPENMED_FACT_ID_SYSTEM",
    "OPENMED_JOURNEY_EVENT_EXTENSION",
    "OPENMED_RESOLUTION_EXTENSION",
    "OPENMED_SNAPSHOT_EXTENSION",
    "OPENMED_SUBJECT_ID_SYSTEM",
    "export_journey_to_fhir",
    "import_journey_from_fhir",
    "load_fhir_journey_schema",
]
