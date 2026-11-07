"""Versioned OMOP 5.4 projection for resolved Journey clinical facts.

The projector consumes immutable :class:`ClinicalFact` records and emits a
deterministic OMOP-shaped current view plus fact-to-row provenance and
append-only ETL lineage.  It never resolves a vocabulary over the network and
never bundles vocabulary records.  Restricted snapshots are accepted only as
explicitly user-supplied references.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from types import MappingProxyType
from typing import Any

from openmed.clinical.journey_contracts import (
    ClinicalFact,
    canonical_digest,
    canonical_json,
    derived_opaque_id,
)
from openmed.structured.store import StoreResult, StoreState

from .cdm_loader import (
    UNMAPPED_CONCEPT_ID,
    UNMAPPED_VOCABULARY_ID,
    deterministic_omop_id,
)

OMOP_FACT_PROJECTION_SCHEMA_VERSION = "1.0.0"
OMOP_FACT_PROJECTION_COMPATIBILITY_POLICY = "same_major"
OMOP_FACT_PROJECTION_SCHEMA_NAME = "omop_fact_projection"
OMOP_FACT_PROJECTION_SCHEMA_PACKAGE = "openmed.core.schemas.json"
OMOP_FACT_PROJECTION_CDM_VERSION = "5.4"
OMOP_FACT_PROJECTOR_VERSION = "1.0.0"

OMOP_FACT_PROJECTION_MODES = frozenset({"append", "replace_by_source"})
OMOP_MAPPING_STATES = frozenset({"mapped", "unmapped", "ambiguous", "rejected"})
OMOP_DATASET_SPLITS = frozenset({"train", "validation", "holdout"})
OMOP_FACT_TABLES = (
    "person",
    "visit_occurrence",
    "note",
    "condition_occurrence",
    "drug_exposure",
    "procedure_occurrence",
    "measurement",
    "observation",
    "source_to_concept_map",
)
OMOP_DOMAIN_TABLES = (
    "condition_occurrence",
    "drug_exposure",
    "procedure_occurrence",
    "measurement",
    "observation",
)
OMOP_PERMISSIVE_LICENSES = frozenset(
    {
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc-by-4.0",
        "cc0-1.0",
        "isc",
        "mit",
        "unlicense",
    }
)

_TABLE_BY_FACT_TYPE: Mapping[str, str] = MappingProxyType(
    {
        "condition": "condition_occurrence",
        "diagnosis": "condition_occurrence",
        "problem": "condition_occurrence",
        "drug": "drug_exposure",
        "medication": "drug_exposure",
        "procedure": "procedure_occurrence",
        "laboratory": "measurement",
        "measurement": "measurement",
        "vital": "measurement",
        "observation": "observation",
        "social_determinant": "observation",
    }
)
_PRIMARY_KEY_BY_TABLE: Mapping[str, str | None] = MappingProxyType(
    {
        "person": "person_id",
        "visit_occurrence": "visit_occurrence_id",
        "note": "note_id",
        "condition_occurrence": "condition_occurrence_id",
        "drug_exposure": "drug_exposure_id",
        "procedure_occurrence": "procedure_occurrence_id",
        "measurement": "measurement_id",
        "observation": "observation_id",
        "source_to_concept_map": None,
    }
)
_DOMAIN_SPEC: Mapping[str, tuple[str, str, str, str]] = MappingProxyType(
    {
        "condition_occurrence": (
            "condition_occurrence_id",
            "condition_concept_id",
            "condition_source_concept_id",
            "condition_start_date",
        ),
        "drug_exposure": (
            "drug_exposure_id",
            "drug_concept_id",
            "drug_source_concept_id",
            "drug_exposure_start_date",
        ),
        "procedure_occurrence": (
            "procedure_occurrence_id",
            "procedure_concept_id",
            "procedure_source_concept_id",
            "procedure_date",
        ),
        "measurement": (
            "measurement_id",
            "measurement_concept_id",
            "measurement_source_concept_id",
            "measurement_date",
        ),
        "observation": (
            "observation_id",
            "observation_concept_id",
            "observation_source_concept_id",
            "observation_date",
        ),
    }
)

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_VOCABULARY_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,19}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)


class OmopFactProjectionError(ValueError):
    """Base error for invalid fact-projection contracts."""


class OmopFactProjectionDeniedError(OmopFactProjectionError):
    """Raised when a vocabulary or privacy policy denies projection."""


class OmopFactProjectionConflictError(OmopFactProjectionError):
    """Raised when pinned inputs or incremental state conflict."""


class OmopFactProjectionUnsupportedError(OmopFactProjectionError):
    """Raised when a requested projection capability is unsupported."""


@dataclass(frozen=True, slots=True)
class OmopVocabularySnapshot:
    """Value-free reference to a caller-acquired vocabulary snapshot."""

    snapshot_id: str
    version: str
    digest: str
    license: str
    usage_lane: str
    bundled: bool = False

    def __post_init__(self) -> None:
        _controlled(self.snapshot_id, "snapshot_id")
        _bounded_text(self.version, "snapshot version", maximum=128)
        _digest(self.digest, "snapshot digest")
        _controlled(self.license, "snapshot license")
        if self.usage_lane not in {"redistributable", "user_supplied"}:
            raise OmopFactProjectionUnsupportedError(
                "vocabulary usage lane is unsupported"
            )
        if type(self.bundled) is not bool:
            raise OmopFactProjectionError("vocabulary bundled flag must be boolean")

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free vocabulary reference."""

        return {
            "bundled": self.bundled,
            "digest": self.digest,
            "license": self.license,
            "snapshot_id": self.snapshot_id,
            "usage_lane": self.usage_lane,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopVocabularySnapshot":
        """Build a validated vocabulary reference."""

        data = _mapping(value, "vocabulary_snapshot")
        _exact_keys(
            data,
            {"bundled", "digest", "license", "snapshot_id", "usage_lane", "version"},
            "vocabulary_snapshot",
        )
        return cls(
            snapshot_id=_text(data["snapshot_id"], "snapshot_id"),
            version=_text(data["version"], "snapshot version"),
            digest=_text(data["digest"], "snapshot digest"),
            license=_text(data["license"], "snapshot license"),
            usage_lane=_text(data["usage_lane"], "usage_lane"),
            bundled=_boolean(data["bundled"], "bundled"),
        )


@dataclass(frozen=True, slots=True)
class OmopConceptMapping:
    """Explicit source-to-standard concept decision for one clinical fact."""

    state: str
    source_system: str
    source_code: str
    source_concept_id: int
    standard_concept_id: int
    standard_vocabulary: str
    standard_code: str | None
    reason_code: str
    snapshot_digest: str
    valid_start_date: str = "1970-01-01"
    valid_end_date: str = "2099-12-31"

    def __post_init__(self) -> None:
        if self.state not in OMOP_MAPPING_STATES:
            raise OmopFactProjectionUnsupportedError("mapping state is unsupported")
        _vocabulary_id(self.source_system, "source_system")
        _bounded_text(self.source_code, "source_code", maximum=50)
        _integer(self.source_concept_id, "source_concept_id")
        target_id = _integer(self.standard_concept_id, "standard_concept_id")
        _vocabulary_id(self.standard_vocabulary, "standard_vocabulary")
        if self.standard_code is not None:
            _bounded_text(self.standard_code, "standard_code", maximum=256)
        _controlled(self.reason_code, "mapping reason_code")
        _digest(self.snapshot_digest, "mapping snapshot_digest")
        start = _date_text(self.valid_start_date, "mapping valid_start_date")
        end = _date_text(self.valid_end_date, "mapping valid_end_date")
        if end < start:
            raise OmopFactProjectionConflictError(
                "mapping validity interval is reversed"
            )
        if self.state == "mapped" and target_id == UNMAPPED_CONCEPT_ID:
            raise OmopFactProjectionConflictError(
                "mapped concept decisions require a positive standard concept"
            )
        if self.state != "mapped" and target_id != UNMAPPED_CONCEPT_ID:
            raise OmopFactProjectionConflictError(
                "non-mapped concept decisions must use concept 0"
            )
        if self.state != "mapped" and self.reason_code == "mapped":
            raise OmopFactProjectionConflictError(
                "non-mapped concept decisions require an explicit reason"
            )

    @property
    def outcome_id(self) -> str:
        """Return the canonical mapping-outcome digest."""

        return canonical_digest(self.to_dict())

    @property
    def requires_review(self) -> bool:
        """Return whether the concept decision must remain visible for review."""

        return self.state != "mapped"

    def to_dict(self) -> dict[str, Any]:
        """Return the complete concept decision without vocabulary content."""

        return {
            "reason_code": self.reason_code,
            "requires_review": self.requires_review,
            "snapshot_digest": self.snapshot_digest,
            "source_code": self.source_code,
            "source_concept_id": self.source_concept_id,
            "source_system": self.source_system,
            "standard_code": self.standard_code,
            "standard_concept_id": self.standard_concept_id,
            "standard_vocabulary": self.standard_vocabulary,
            "state": self.state,
            "valid_end_date": self.valid_end_date,
            "valid_start_date": self.valid_start_date,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopConceptMapping":
        """Build a concept decision from a public mapping payload."""

        data = _mapping(value, "concept_mapping")
        required = {
            "reason_code",
            "snapshot_digest",
            "source_code",
            "source_concept_id",
            "source_system",
            "standard_code",
            "standard_concept_id",
            "standard_vocabulary",
            "state",
            "valid_end_date",
            "valid_start_date",
        }
        if "requires_review" in data:
            expected = data["state"] != "mapped"
            if data["requires_review"] is not expected:
                raise OmopFactProjectionConflictError(
                    "serialized review flag differs from mapping state"
                )
            data = {key: item for key, item in data.items() if key != "requires_review"}
        _exact_keys(data, required, "concept_mapping")
        return cls(
            state=_text(data["state"], "mapping state"),
            source_system=_text(data["source_system"], "source_system"),
            source_code=_text(data["source_code"], "source_code"),
            source_concept_id=_integer(data["source_concept_id"], "source_concept_id"),
            standard_concept_id=_integer(
                data["standard_concept_id"], "standard_concept_id"
            ),
            standard_vocabulary=_text(
                data["standard_vocabulary"], "standard_vocabulary"
            ),
            standard_code=_optional_text(data["standard_code"], "standard_code"),
            reason_code=_text(data["reason_code"], "mapping reason_code"),
            snapshot_digest=_text(data["snapshot_digest"], "snapshot_digest"),
            valid_start_date=_text(data["valid_start_date"], "valid_start_date"),
            valid_end_date=_text(data["valid_end_date"], "valid_end_date"),
        )

    @classmethod
    def unmapped(
        cls,
        *,
        source_system: str,
        source_code: str,
        snapshot_digest: str,
        reason_code: str = "mapping_not_supplied",
        source_concept_id: int = 0,
    ) -> "OmopConceptMapping":
        """Build an explicit concept-0 mapping decision."""

        return cls(
            state="unmapped",
            source_system=source_system,
            source_code=source_code,
            source_concept_id=source_concept_id,
            standard_concept_id=0,
            standard_vocabulary=UNMAPPED_VOCABULARY_ID,
            standard_code=None,
            reason_code=reason_code,
            snapshot_digest=snapshot_digest,
        )


@dataclass(frozen=True, slots=True)
class OmopFactProjectionInput:
    """One fact plus value-free source, mapping, and optional split custody."""

    fact: ClinicalFact = field(repr=False)
    source_key: str
    source_revision: str
    mapping: OmopConceptMapping | None = None
    dataset_split: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fact, ClinicalFact):
            raise TypeError("projection input fact must be ClinicalFact")
        _opaque_id(self.source_key, "source_key")
        _digest(self.source_revision, "source_revision")
        if self.mapping is not None and not isinstance(
            self.mapping, OmopConceptMapping
        ):
            raise TypeError("projection input mapping must be OmopConceptMapping")
        if self.dataset_split is not None and self.dataset_split not in (
            OMOP_DATASET_SPLITS
        ):
            raise OmopFactProjectionUnsupportedError(
                "projection dataset split is unsupported"
            )

    def to_safe_dict(self) -> dict[str, Any]:
        """Return value-free input custody used for run identity."""

        return {
            "dataset_split": self.dataset_split,
            "fact_digest": self.fact.canonical_hash,
            "fact_id": self.fact.fact_id,
            "mapping": self.mapping.to_dict() if self.mapping is not None else None,
            "source_key": self.source_key,
            "source_revision": self.source_revision,
            "subject_id": self.fact.subject_id,
        }


@dataclass(frozen=True, slots=True)
class OmopProjectionLoss:
    """Value-free record of one field or fact not represented in OMOP."""

    fact_id: str
    source_key: str
    reason_code: str
    fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _opaque_id(self.fact_id, "loss fact_id")
        _opaque_id(self.source_key, "loss source_key")
        _controlled(self.reason_code, "loss reason_code")
        object.__setattr__(
            self,
            "fields",
            tuple(sorted({_controlled(item, "loss field") for item in self.fields})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free loss record."""

        return {
            "fact_id": self.fact_id,
            "fields": list(self.fields),
            "reason_code": self.reason_code,
            "source_key": self.source_key,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopProjectionLoss":
        """Build a projection-loss record from persisted data."""

        data = _mapping(value, "projection loss")
        _exact_keys(
            data, {"fact_id", "fields", "reason_code", "source_key"}, "projection loss"
        )
        return cls(
            fact_id=_text(data["fact_id"], "loss fact_id"),
            source_key=_text(data["source_key"], "loss source_key"),
            reason_code=_text(data["reason_code"], "loss reason_code"),
            fields=_text_sequence(data["fields"], "loss fields"),
        )


@dataclass(frozen=True, slots=True)
class OmopRowProvenance:
    """Fact-to-row custody for one current OMOP domain row."""

    provenance_id: str
    table: str
    row_id: int
    fact_id: str
    subject_id: str
    encounter_id: str | None
    source_key: str
    source_revision: str
    evidence_ids: tuple[str, ...]
    mapping_outcome_id: str
    etl_run_id: str
    correction_of: tuple[str, ...] = ()
    dataset_split: str | None = None

    def __post_init__(self) -> None:
        _opaque_id(self.provenance_id, "provenance_id")
        if self.table not in OMOP_DOMAIN_TABLES:
            raise OmopFactProjectionUnsupportedError("provenance table is unsupported")
        _positive_integer(self.row_id, "provenance row_id")
        _opaque_id(self.fact_id, "provenance fact_id")
        _opaque_id(self.subject_id, "provenance subject_id")
        if self.encounter_id is not None:
            _opaque_id(self.encounter_id, "provenance encounter_id")
        _opaque_id(self.source_key, "provenance source_key")
        _digest(self.source_revision, "provenance source_revision")
        _digest(self.mapping_outcome_id, "mapping_outcome_id")
        _opaque_id(self.etl_run_id, "provenance etl_run_id")
        evidence = _opaque_ids(self.evidence_ids, "provenance evidence_ids", minimum=1)
        corrections = _opaque_ids(self.correction_of, "correction_of")
        if self.dataset_split is not None and self.dataset_split not in (
            OMOP_DATASET_SPLITS
        ):
            raise OmopFactProjectionUnsupportedError(
                "provenance dataset split is unsupported"
            )
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "correction_of", corrections)

    @property
    def source_evidence_key(self) -> str:
        """Return the canonical evidence-set key for this OMOP row."""

        return canonical_digest(list(self.evidence_ids))

    def to_dict(self) -> dict[str, Any]:
        """Return complete value-free row provenance."""

        return {
            "correction_of": list(self.correction_of),
            "dataset_split": self.dataset_split,
            "encounter_id": self.encounter_id,
            "etl_run_id": self.etl_run_id,
            "evidence_ids": list(self.evidence_ids),
            "fact_id": self.fact_id,
            "mapping_outcome_id": self.mapping_outcome_id,
            "provenance_id": self.provenance_id,
            "row_id": self.row_id,
            "source_evidence_key": self.source_evidence_key,
            "source_key": self.source_key,
            "source_revision": self.source_revision,
            "subject_id": self.subject_id,
            "table": self.table,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopRowProvenance":
        """Build row provenance and verify its derived evidence key."""

        data = _mapping(value, "row provenance")
        _exact_keys(
            data,
            {
                "correction_of",
                "dataset_split",
                "encounter_id",
                "etl_run_id",
                "evidence_ids",
                "fact_id",
                "mapping_outcome_id",
                "provenance_id",
                "row_id",
                "source_evidence_key",
                "source_key",
                "source_revision",
                "subject_id",
                "table",
            },
            "row provenance",
        )
        result = cls(
            provenance_id=_text(data["provenance_id"], "provenance_id"),
            table=_text(data["table"], "provenance table"),
            row_id=_integer(data["row_id"], "provenance row_id"),
            fact_id=_text(data["fact_id"], "provenance fact_id"),
            subject_id=_text(data["subject_id"], "provenance subject_id"),
            encounter_id=_optional_text(data["encounter_id"], "encounter_id"),
            source_key=_text(data["source_key"], "provenance source_key"),
            source_revision=_text(data["source_revision"], "source_revision"),
            evidence_ids=_text_sequence(data["evidence_ids"], "evidence_ids"),
            mapping_outcome_id=_text(data["mapping_outcome_id"], "mapping_outcome_id"),
            etl_run_id=_text(data["etl_run_id"], "etl_run_id"),
            correction_of=_text_sequence(data["correction_of"], "correction_of"),
            dataset_split=_optional_text(data["dataset_split"], "dataset_split"),
        )
        if data["source_evidence_key"] != result.source_evidence_key:
            raise OmopFactProjectionConflictError(
                "persisted source evidence key differs from evidence IDs"
            )
        return result


@dataclass(frozen=True, slots=True)
class OmopMappingOutcome:
    """Visible mapping state for one projected fact and row."""

    fact_id: str
    source_key: str
    table: str
    row_id: int
    mapping: OmopConceptMapping

    def __post_init__(self) -> None:
        _opaque_id(self.fact_id, "mapping fact_id")
        _opaque_id(self.source_key, "mapping source_key")
        if self.table not in OMOP_DOMAIN_TABLES:
            raise OmopFactProjectionUnsupportedError(
                "mapping outcome table is unsupported"
            )
        _positive_integer(self.row_id, "mapping row_id")
        if not isinstance(self.mapping, OmopConceptMapping):
            raise TypeError("mapping outcome requires OmopConceptMapping")

    def to_dict(self) -> dict[str, Any]:
        """Return mapping state and concept relationship."""

        return {
            "fact_id": self.fact_id,
            "mapping": self.mapping.to_dict() | {"outcome_id": self.mapping.outcome_id},
            "row_id": self.row_id,
            "source_key": self.source_key,
            "table": self.table,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopMappingOutcome":
        """Build a visible mapping outcome from persisted data."""

        data = _mapping(value, "mapping outcome")
        _exact_keys(
            data,
            {"fact_id", "mapping", "row_id", "source_key", "table"},
            "mapping outcome",
        )
        mapping_data = dict(_mapping(data["mapping"], "mapping outcome mapping"))
        outcome_id = mapping_data.pop("outcome_id", None)
        mapping = OmopConceptMapping.from_dict(mapping_data)
        if outcome_id != mapping.outcome_id:
            raise OmopFactProjectionConflictError(
                "persisted mapping outcome digest differs"
            )
        return cls(
            fact_id=_text(data["fact_id"], "mapping fact_id"),
            source_key=_text(data["source_key"], "mapping source_key"),
            table=_text(data["table"], "mapping table"),
            row_id=_integer(data["row_id"], "mapping row_id"),
            mapping=mapping,
        )


@dataclass(frozen=True, slots=True)
class OmopEtlRun:
    """Append-only lineage for one deterministic projection application."""

    run_id: str
    etl_version: str
    occurred_at: str
    mode: str
    vocabulary_snapshot_digest: str
    input_digest: str
    output_digest: str
    source_keys: tuple[str, ...]
    input_fact_ids: tuple[str, ...]
    superseded_fact_ids: tuple[str, ...]
    row_counts: Mapping[str, int]
    cdm_version: str = OMOP_FACT_PROJECTION_CDM_VERSION
    schema_version: str = OMOP_FACT_PROJECTION_SCHEMA_VERSION
    compatibility_policy: str = OMOP_FACT_PROJECTION_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _opaque_id(self.run_id, "run_id")
        _version(self.etl_version, "etl_version")
        _timestamp(self.occurred_at, "occurred_at")
        if self.mode not in OMOP_FACT_PROJECTION_MODES:
            raise OmopFactProjectionUnsupportedError("ETL mode is unsupported")
        _digest(self.vocabulary_snapshot_digest, "vocabulary_snapshot_digest")
        _digest(self.input_digest, "input_digest")
        _digest(self.output_digest, "output_digest")
        source_keys = _opaque_ids(self.source_keys, "source_keys", minimum=1)
        facts = _opaque_ids(self.input_fact_ids, "input_fact_ids", minimum=1)
        superseded = _opaque_ids(self.superseded_fact_ids, "superseded_fact_ids")
        counts = _count_mapping(self.row_counts, "row_counts")
        if set(counts) != set(OMOP_FACT_TABLES):
            raise OmopFactProjectionError("ETL row counts are incomplete")
        _contract(self.schema_version, self.compatibility_policy, self.cdm_version)
        object.__setattr__(self, "source_keys", source_keys)
        object.__setattr__(self, "input_fact_ids", facts)
        object.__setattr__(self, "superseded_fact_ids", superseded)
        object.__setattr__(self, "row_counts", counts)

    def to_dict(self) -> dict[str, Any]:
        """Return complete immutable ETL lineage."""

        return {
            "cdm_version": self.cdm_version,
            "compatibility_policy": self.compatibility_policy,
            "etl_version": self.etl_version,
            "input_digest": self.input_digest,
            "input_fact_ids": list(self.input_fact_ids),
            "mode": self.mode,
            "occurred_at": self.occurred_at,
            "output_digest": self.output_digest,
            "row_counts": dict(self.row_counts),
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "source_keys": list(self.source_keys),
            "superseded_fact_ids": list(self.superseded_fact_ids),
            "vocabulary_snapshot_digest": self.vocabulary_snapshot_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopEtlRun":
        """Build immutable ETL lineage from persisted data."""

        data = _mapping(value, "ETL run")
        _exact_keys(
            data,
            {
                "cdm_version",
                "compatibility_policy",
                "etl_version",
                "input_digest",
                "input_fact_ids",
                "mode",
                "occurred_at",
                "output_digest",
                "row_counts",
                "run_id",
                "schema_version",
                "source_keys",
                "superseded_fact_ids",
                "vocabulary_snapshot_digest",
            },
            "ETL run",
        )
        return cls(
            run_id=_text(data["run_id"], "run_id"),
            etl_version=_text(data["etl_version"], "etl_version"),
            occurred_at=_text(data["occurred_at"], "occurred_at"),
            mode=_text(data["mode"], "mode"),
            vocabulary_snapshot_digest=_text(
                data["vocabulary_snapshot_digest"], "vocabulary_snapshot_digest"
            ),
            input_digest=_text(data["input_digest"], "input_digest"),
            output_digest=_text(data["output_digest"], "output_digest"),
            source_keys=_text_sequence(data["source_keys"], "source_keys"),
            input_fact_ids=_text_sequence(data["input_fact_ids"], "input_fact_ids"),
            superseded_fact_ids=_text_sequence(
                data["superseded_fact_ids"], "superseded_fact_ids"
            ),
            row_counts=_mapping(data["row_counts"], "row_counts"),
            cdm_version=_text(data["cdm_version"], "cdm_version"),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )


@dataclass(frozen=True, slots=True)
class OmopProjectionSummary:
    """Count-only projection and mapping summary."""

    row_counts: Mapping[str, int]
    mapping_counts: Mapping[str, int]
    loss_counts: Mapping[str, int]
    current_fact_count: int
    etl_run_count: int

    def __post_init__(self) -> None:
        rows = _count_mapping(self.row_counts, "summary row_counts")
        mappings = _count_mapping(self.mapping_counts, "summary mapping_counts")
        losses = _count_mapping(self.loss_counts, "summary loss_counts")
        if set(rows) != set(OMOP_FACT_TABLES):
            raise OmopFactProjectionError("summary row counts are incomplete")
        if set(mappings) != set(OMOP_MAPPING_STATES):
            raise OmopFactProjectionError("summary mapping counts are incomplete")
        _integer(self.current_fact_count, "current_fact_count")
        _integer(self.etl_run_count, "etl_run_count")
        object.__setattr__(self, "row_counts", rows)
        object.__setattr__(self, "mapping_counts", mappings)
        object.__setattr__(self, "loss_counts", losses)

    def to_dict(self) -> dict[str, Any]:
        """Return privacy-safe aggregate counts."""

        return {
            "current_fact_count": self.current_fact_count,
            "etl_run_count": self.etl_run_count,
            "loss_counts": dict(self.loss_counts),
            "mapping_counts": dict(self.mapping_counts),
            "row_counts": dict(self.row_counts),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopProjectionSummary":
        """Build aggregate projection counts from persisted data."""

        data = _mapping(value, "projection summary")
        _exact_keys(
            data,
            {
                "current_fact_count",
                "etl_run_count",
                "loss_counts",
                "mapping_counts",
                "row_counts",
            },
            "projection summary",
        )
        return cls(
            row_counts=_mapping(data["row_counts"], "row_counts"),
            mapping_counts=_mapping(data["mapping_counts"], "mapping_counts"),
            loss_counts=_mapping(data["loss_counts"], "loss_counts"),
            current_fact_count=_integer(
                data["current_fact_count"], "current_fact_count"
            ),
            etl_run_count=_integer(data["etl_run_count"], "etl_run_count"),
        )


@dataclass(frozen=True, slots=True)
class OmopFactProjection:
    """Current OMOP 5.4 rows plus mapping, provenance, and ETL history."""

    projection_id: str
    vocabulary_snapshot: OmopVocabularySnapshot
    tables: Mapping[str, tuple[Mapping[str, Any], ...]]
    provenance: tuple[OmopRowProvenance, ...]
    mapping_outcomes: tuple[OmopMappingOutcome, ...]
    losses: tuple[OmopProjectionLoss, ...]
    etl_runs: tuple[OmopEtlRun, ...]
    summary: OmopProjectionSummary
    cdm_version: str = OMOP_FACT_PROJECTION_CDM_VERSION
    schema_version: str = OMOP_FACT_PROJECTION_SCHEMA_VERSION
    compatibility_policy: str = OMOP_FACT_PROJECTION_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _opaque_id(self.projection_id, "projection_id")
        if not isinstance(self.vocabulary_snapshot, OmopVocabularySnapshot):
            raise TypeError("projection snapshot must be OmopVocabularySnapshot")
        table_data = _mapping(self.tables, "projection tables")
        if set(table_data) != set(OMOP_FACT_TABLES):
            raise OmopFactProjectionError("projection tables are incomplete")
        frozen_tables: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for table in OMOP_FACT_TABLES:
            rows = tuple(table_data[table])
            frozen = tuple(_freeze_mapping(row, f"{table} row") for row in rows)
            try:
                row_ids = tuple(_table_row_id(table, row) for row in frozen)
            except KeyError as exc:
                raise OmopFactProjectionError(
                    f"{table} row is missing its primary key"
                ) from exc
            if len(row_ids) != len(set(row_ids)):
                raise OmopFactProjectionConflictError(
                    f"{table} primary keys must be unique"
                )
            frozen_tables[table] = tuple(
                sorted(frozen, key=lambda row: _table_row_id(table, row))
            )
        provenance = tuple(sorted(self.provenance, key=_provenance_sort_key))
        if any(not isinstance(item, OmopRowProvenance) for item in provenance):
            raise TypeError("projection provenance has invalid records")
        if len({item.provenance_id for item in provenance}) != len(provenance):
            raise OmopFactProjectionConflictError("provenance identifiers conflict")
        outcomes = tuple(
            sorted(self.mapping_outcomes, key=lambda item: (item.table, item.row_id))
        )
        if any(not isinstance(item, OmopMappingOutcome) for item in outcomes):
            raise TypeError("projection mapping outcomes are invalid")
        if len({item.fact_id for item in outcomes}) != len(outcomes):
            raise OmopFactProjectionConflictError("mapping outcomes duplicate facts")
        losses = tuple(
            sorted(
                self.losses,
                key=lambda item: (item.fact_id, item.reason_code, item.fields),
            )
        )
        if any(not isinstance(item, OmopProjectionLoss) for item in losses):
            raise TypeError("projection losses are invalid")
        runs = tuple(
            sorted(self.etl_runs, key=lambda item: (item.occurred_at, item.run_id))
        )
        if any(not isinstance(item, OmopEtlRun) for item in runs):
            raise TypeError("projection ETL lineage is invalid")
        if len({item.run_id for item in runs}) != len(runs):
            raise OmopFactProjectionConflictError("ETL run identifiers conflict")
        if not isinstance(self.summary, OmopProjectionSummary):
            raise TypeError("projection summary is invalid")
        if dict(self.summary.row_counts) != {
            table: len(frozen_tables[table]) for table in OMOP_FACT_TABLES
        }:
            raise OmopFactProjectionConflictError(
                "projection summary row counts disagree"
            )
        expected_mapping_counts = Counter(item.mapping.state for item in outcomes)
        if dict(self.summary.mapping_counts) != {
            state: expected_mapping_counts.get(state, 0)
            for state in OMOP_MAPPING_STATES
        }:
            raise OmopFactProjectionConflictError(
                "projection summary mapping counts disagree"
            )
        if dict(self.summary.loss_counts) != dict(
            Counter(item.reason_code for item in losses)
        ):
            raise OmopFactProjectionConflictError(
                "projection summary loss counts disagree"
            )
        if self.summary.current_fact_count != len(provenance):
            raise OmopFactProjectionConflictError(
                "projection summary fact count disagrees"
            )
        if self.summary.etl_run_count != len(runs):
            raise OmopFactProjectionConflictError(
                "projection summary ETL run count disagrees"
            )
        if any(
            item.mapping.snapshot_digest != self.vocabulary_snapshot.digest
            for item in outcomes
        ) or any(
            item.vocabulary_snapshot_digest != self.vocabulary_snapshot.digest
            for item in runs
        ):
            raise OmopFactProjectionConflictError(
                "projection vocabulary snapshot lineage disagrees"
            )
        _contract(self.schema_version, self.compatibility_policy, self.cdm_version)
        object.__setattr__(self, "tables", MappingProxyType(frozen_tables))
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "mapping_outcomes", outcomes)
        object.__setattr__(self, "losses", losses)
        object.__setattr__(self, "etl_runs", runs)

    @property
    def digest(self) -> str:
        """Return the canonical current projection and lineage digest."""

        return canonical_digest(self.to_dict())

    @property
    def current_fact_ids(self) -> tuple[str, ...]:
        """Return the exact facts represented by current domain rows."""

        return tuple(sorted(item.fact_id for item in self.provenance))

    @property
    def current_source_keys(self) -> tuple[str, ...]:
        """Return source keys represented by current domain rows."""

        return tuple(sorted({item.source_key for item in self.provenance}))

    def table(self, name: str) -> tuple[Mapping[str, Any], ...]:
        """Return one named OMOP table."""

        if name not in OMOP_FACT_TABLES:
            raise OmopFactProjectionUnsupportedError("OMOP table is unsupported")
        return self.tables[name]

    def to_dict(self) -> dict[str, Any]:
        """Return the complete versioned projection record."""

        return {
            "cdm_version": self.cdm_version,
            "compatibility_policy": self.compatibility_policy,
            "etl_runs": [item.to_dict() for item in self.etl_runs],
            "losses": [item.to_dict() for item in self.losses],
            "mapping_outcomes": [item.to_dict() for item in self.mapping_outcomes],
            "projection_id": self.projection_id,
            "provenance": [item.to_dict() for item in self.provenance],
            "schema_version": self.schema_version,
            "summary": self.summary.to_dict(),
            "tables": {
                table: [_plain(row) for row in self.tables[table]]
                for table in OMOP_FACT_TABLES
            },
            "vocabulary_snapshot": self.vocabulary_snapshot.to_dict(),
        }

    def to_json(self) -> str:
        """Return deterministic compact JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, value: str) -> "OmopFactProjection":
        """Restore and validate a persisted JSON projection record."""

        if not isinstance(value, str):
            raise OmopFactProjectionError("OMOP fact projection JSON must be text")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise OmopFactProjectionError(
                "OMOP fact projection JSON is invalid"
            ) from exc
        return cls.from_dict(_mapping(decoded, "OMOP fact projection"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopFactProjection":
        """Restore and validate a persisted projection record."""

        data = _mapping(value, "OMOP fact projection")
        _exact_keys(
            data,
            {
                "cdm_version",
                "compatibility_policy",
                "etl_runs",
                "losses",
                "mapping_outcomes",
                "projection_id",
                "provenance",
                "schema_version",
                "summary",
                "tables",
                "vocabulary_snapshot",
            },
            "OMOP fact projection",
        )
        tables = _mapping(data["tables"], "projection tables")
        if set(tables) != set(OMOP_FACT_TABLES):
            raise OmopFactProjectionError("projection tables are incomplete")
        return cls(
            projection_id=_text(data["projection_id"], "projection_id"),
            vocabulary_snapshot=OmopVocabularySnapshot.from_dict(
                _mapping(data["vocabulary_snapshot"], "vocabulary_snapshot")
            ),
            tables={
                table: tuple(
                    _mapping(row, f"{table} row")
                    for row in _sequence(tables[table], table)
                )
                for table in OMOP_FACT_TABLES
            },
            provenance=tuple(
                OmopRowProvenance.from_dict(_mapping(item, "row provenance"))
                for item in _sequence(data["provenance"], "provenance")
            ),
            mapping_outcomes=tuple(
                OmopMappingOutcome.from_dict(_mapping(item, "mapping outcome"))
                for item in _sequence(data["mapping_outcomes"], "mapping_outcomes")
            ),
            losses=tuple(
                OmopProjectionLoss.from_dict(_mapping(item, "projection loss"))
                for item in _sequence(data["losses"], "losses")
            ),
            etl_runs=tuple(
                OmopEtlRun.from_dict(_mapping(item, "ETL run"))
                for item in _sequence(data["etl_runs"], "etl_runs")
            ),
            summary=OmopProjectionSummary.from_dict(
                _mapping(data["summary"], "projection summary")
            ),
            cdm_version=_text(data["cdm_version"], "cdm_version"),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )


@dataclass(frozen=True, slots=True)
class OmopProjectionViolation:
    """Value-free referential or mapping validation finding."""

    table: str
    row_id: int | None
    reason_code: str

    def __post_init__(self) -> None:
        if self.table not in {*OMOP_FACT_TABLES, "provenance"}:
            raise OmopFactProjectionUnsupportedError("violation table is unsupported")
        if self.row_id is not None:
            _positive_integer(self.row_id, "violation row_id")
        _controlled(self.reason_code, "violation reason_code")

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free validation finding."""

        return {
            "reason_code": self.reason_code,
            "row_id": self.row_id,
            "table": self.table,
        }


@dataclass(frozen=True, slots=True)
class OmopFactRoundTripReport:
    """Fact-identity preservation report without protected fact values."""

    input_fact_ids: tuple[str, ...]
    projected_fact_ids: tuple[str, ...]
    loss_fact_ids: tuple[str, ...]
    missing_fact_ids: tuple[str, ...]
    unexpected_fact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "input_fact_ids",
            "projected_fact_ids",
            "loss_fact_ids",
            "missing_fact_ids",
            "unexpected_fact_ids",
        ):
            object.__setattr__(self, name, _opaque_ids(getattr(self, name), name))

    @property
    def lossless(self) -> bool:
        """Return whether every input fact is represented with no extra fact."""

        return (
            not self.loss_fact_ids
            and not self.missing_fact_ids
            and not self.unexpected_fact_ids
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the identity-only round-trip report."""

        return {
            "input_fact_ids": list(self.input_fact_ids),
            "loss_fact_ids": list(self.loss_fact_ids),
            "lossless": self.lossless,
            "missing_fact_ids": list(self.missing_fact_ids),
            "projected_fact_ids": list(self.projected_fact_ids),
            "unexpected_fact_ids": list(self.unexpected_fact_ids),
        }


def project_clinical_facts_to_omop(
    inputs: Iterable[OmopFactProjectionInput],
    *,
    vocabulary_snapshot: OmopVocabularySnapshot,
    etl_version: str,
    occurred_at: str,
    mode: str = "replace_by_source",
    previous: OmopFactProjection | None = None,
) -> StoreResult[OmopFactProjection]:
    """Project resolved facts into a deterministic correction-aware OMOP view."""

    materialized = tuple(inputs)
    if not materialized:
        return StoreResult.outcome(StoreState.UNKNOWN, "projection_inputs_empty")
    if any(not isinstance(item, OmopFactProjectionInput) for item in materialized):
        return StoreResult.outcome(StoreState.FAILURE, "projection_input_invalid")
    try:
        if not isinstance(vocabulary_snapshot, OmopVocabularySnapshot):
            raise TypeError("vocabulary_snapshot must be OmopVocabularySnapshot")
        if vocabulary_snapshot.bundled and (
            vocabulary_snapshot.usage_lane != "redistributable"
            or vocabulary_snapshot.license not in OMOP_PERMISSIVE_LICENSES
        ):
            raise OmopFactProjectionDeniedError(
                "bundled vocabulary references must be redistributable"
            )
        _version(etl_version, "etl_version")
        _timestamp(occurred_at, "occurred_at")
        if mode not in OMOP_FACT_PROJECTION_MODES:
            raise OmopFactProjectionUnsupportedError("projection mode is unsupported")
        if previous is not None:
            if not isinstance(previous, OmopFactProjection):
                raise TypeError("previous projection has an invalid type")
            if previous.vocabulary_snapshot.digest != vocabulary_snapshot.digest:
                raise OmopFactProjectionConflictError(
                    "incremental vocabulary snapshot changed"
                )
        fact_ids = tuple(item.fact.fact_id for item in materialized)
        if len(fact_ids) != len(set(fact_ids)):
            raise OmopFactProjectionConflictError("projection fact IDs must be unique")
        _validate_source_subjects(materialized)
        _validate_split_isolation(materialized, previous)
        return _project(
            materialized,
            vocabulary_snapshot=vocabulary_snapshot,
            etl_version=etl_version,
            occurred_at=occurred_at,
            mode=mode,
            previous=previous,
        )
    except OmopFactProjectionDeniedError:
        return StoreResult.outcome(StoreState.DENIED, "projection_policy_denied")
    except OmopFactProjectionConflictError:
        return StoreResult.outcome(StoreState.CONFLICT, "projection_input_conflict")
    except OmopFactProjectionUnsupportedError:
        return StoreResult.outcome(StoreState.UNSUPPORTED, "projection_unsupported")
    except (OmopFactProjectionError, TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "projection_invalid")


def validate_omop_fact_projection(
    projection: OmopFactProjection,
) -> tuple[OmopProjectionViolation, ...]:
    """Validate current table references, mapping reasons, and fact custody."""

    if not isinstance(projection, OmopFactProjection):
        raise TypeError("projection must be OmopFactProjection")
    violations: list[OmopProjectionViolation] = []
    people = {int(row["person_id"]) for row in projection.table("person")}
    visits = {
        int(row["visit_occurrence_id"]): int(row["person_id"])
        for row in projection.table("visit_occurrence")
    }
    notes = {
        int(row["note_id"]): (
            int(row["person_id"]),
            int(row["visit_occurrence_id"]),
        )
        for row in projection.table("note")
    }
    provenance_by_row = {
        (item.table, item.row_id): item for item in projection.provenance
    }
    outcomes_by_row = {
        (item.table, item.row_id): item for item in projection.mapping_outcomes
    }
    etl_run_ids = {item.run_id for item in projection.etl_runs}
    source_maps = {
        (
            str(row["source_code"]),
            int(row["source_concept_id"]),
            str(row["source_vocabulary_id"]),
            int(row["target_concept_id"]),
            str(row["target_vocabulary_id"]),
            str(row["valid_start_date"]),
            str(row["valid_end_date"]),
            row["invalid_reason"],
        )
        for row in projection.table("source_to_concept_map")
    }

    for row_id, person_id in visits.items():
        if person_id not in people:
            violations.append(
                OmopProjectionViolation(
                    "visit_occurrence", row_id, "person_reference_missing"
                )
            )
    for note_id, (person_id, visit_id) in notes.items():
        if person_id not in people:
            violations.append(
                OmopProjectionViolation("note", note_id, "person_reference_missing")
            )
        if visit_id not in visits:
            violations.append(
                OmopProjectionViolation("note", note_id, "visit_reference_missing")
            )

    for table in OMOP_DOMAIN_TABLES:
        primary_key, concept_column, source_concept_column, _ = _DOMAIN_SPEC[table]
        for row in projection.table(table):
            row_id = int(row[primary_key])
            person_id = int(row["person_id"])
            visit_id = int(row["visit_occurrence_id"])
            if person_id not in people:
                violations.append(
                    OmopProjectionViolation(table, row_id, "person_reference_missing")
                )
            if visit_id not in visits:
                violations.append(
                    OmopProjectionViolation(table, row_id, "visit_reference_missing")
                )
            if (table, row_id) not in provenance_by_row:
                violations.append(
                    OmopProjectionViolation(table, row_id, "provenance_missing")
                )
            else:
                provenance = provenance_by_row[(table, row_id)]
                note_id = deterministic_omop_id(
                    "fact-note", provenance.source_key, provenance.subject_id
                )
                if note_id not in notes:
                    violations.append(
                        OmopProjectionViolation(table, row_id, "note_reference_missing")
                    )
                if provenance.etl_run_id not in etl_run_ids:
                    violations.append(
                        OmopProjectionViolation(table, row_id, "etl_run_missing")
                    )
            outcome = outcomes_by_row.get((table, row_id))
            if outcome is None:
                violations.append(
                    OmopProjectionViolation(table, row_id, "mapping_outcome_missing")
                )
            else:
                target_id = int(row[concept_column])
                source_id = int(row[source_concept_column])
                if target_id != outcome.mapping.standard_concept_id:
                    violations.append(
                        OmopProjectionViolation(
                            table, row_id, "standard_concept_mismatch"
                        )
                    )
                if source_id != outcome.mapping.source_concept_id:
                    violations.append(
                        OmopProjectionViolation(
                            table, row_id, "source_concept_mismatch"
                        )
                    )
                provenance = provenance_by_row.get((table, row_id))
                if provenance is not None and (
                    provenance.fact_id != outcome.fact_id
                    or provenance.source_key != outcome.source_key
                    or provenance.mapping_outcome_id != outcome.mapping.outcome_id
                ):
                    violations.append(
                        OmopProjectionViolation(
                            table, row_id, "mapping_provenance_mismatch"
                        )
                    )
                if target_id == 0 and not outcome.mapping.reason_code:
                    violations.append(
                        OmopProjectionViolation(
                            table, row_id, "unmapped_reason_missing"
                        )
                    )
                mapping = outcome.mapping
                expected_map = (
                    mapping.source_code,
                    mapping.source_concept_id,
                    mapping.source_system,
                    mapping.standard_concept_id,
                    mapping.standard_vocabulary,
                    mapping.valid_start_date,
                    mapping.valid_end_date,
                    None,
                )
                if expected_map not in source_maps:
                    violations.append(
                        OmopProjectionViolation(
                            table, row_id, "source_mapping_row_missing"
                        )
                    )
    return tuple(
        sorted(
            violations,
            key=lambda item: (item.table, item.row_id or 0, item.reason_code),
        )
    )


def assess_omop_fact_round_trip(
    inputs: Iterable[OmopFactProjectionInput],
    projection: OmopFactProjection,
) -> OmopFactRoundTripReport:
    """Compare input fact identities with projected and explicitly lost facts."""

    materialized = tuple(inputs)
    if any(not isinstance(item, OmopFactProjectionInput) for item in materialized):
        raise TypeError("round-trip inputs must be OmopFactProjectionInput values")
    input_ids = {item.fact.fact_id for item in materialized}
    projected = set(projection.current_fact_ids)
    loss_ids = {item.fact_id for item in projection.losses}
    represented = projected | loss_ids
    return OmopFactRoundTripReport(
        input_fact_ids=tuple(input_ids),
        projected_fact_ids=tuple(projected),
        loss_fact_ids=tuple(loss_ids),
        missing_fact_ids=tuple(input_ids - represented),
        unexpected_fact_ids=tuple(represented - input_ids),
    )


def load_omop_fact_projection_schema() -> dict[str, Any]:
    """Load the bundled OMOP fact-projection JSON Schema."""

    resource = resources.files(OMOP_FACT_PROJECTION_SCHEMA_PACKAGE).joinpath(
        f"{OMOP_FACT_PROJECTION_SCHEMA_NAME}.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _project(
    inputs: tuple[OmopFactProjectionInput, ...],
    *,
    vocabulary_snapshot: OmopVocabularySnapshot,
    etl_version: str,
    occurred_at: str,
    mode: str,
    previous: OmopFactProjection | None,
) -> StoreResult[OmopFactProjection]:
    input_digest = canonical_digest(
        [
            item.to_safe_dict()
            for item in sorted(inputs, key=lambda candidate: candidate.fact.fact_id)
        ]
    )
    run_id = derived_opaque_id(
        "omoprun",
        etl_version,
        occurred_at,
        mode,
        vocabulary_snapshot.digest,
        input_digest,
    )
    batch_tables: dict[str, dict[int, dict[str, Any]]] = {
        table: {} for table in OMOP_FACT_TABLES
    }
    provenance: list[OmopRowProvenance] = []
    outcomes: list[OmopMappingOutcome] = []
    losses: list[OmopProjectionLoss] = []
    source_context: dict[str, tuple[int, int, int]] = {}

    for item in sorted(inputs, key=lambda candidate: candidate.fact.fact_id):
        fact = item.fact
        table = _TABLE_BY_FACT_TYPE.get(fact.fact_type)
        if table is None:
            losses.append(
                OmopProjectionLoss(
                    fact_id=fact.fact_id,
                    source_key=item.source_key,
                    reason_code="fact_type_unsupported",
                    fields=("fact_type",),
                )
            )
            continue
        mapping = item.mapping or _mapping_from_fact(fact, vocabulary_snapshot)
        if mapping.snapshot_digest != vocabulary_snapshot.digest:
            raise OmopFactProjectionConflictError(
                "fact mapping vocabulary snapshot differs"
            )
        person_id = deterministic_omop_id("fact-person", fact.subject_id)
        visit_material = fact.encounter_id or item.source_key
        visit_id = deterministic_omop_id("fact-visit", fact.subject_id, visit_material)
        note_id = deterministic_omop_id("fact-note", item.source_key, fact.subject_id)
        existing_context = source_context.get(item.source_key)
        context = (person_id, visit_id, note_id)
        if existing_context is not None and existing_context != context:
            raise OmopFactProjectionConflictError(
                "one source key cannot span subjects or encounters"
            )
        source_context[item.source_key] = context
        date_value, date_losses = _fact_date(fact)
        losses.extend(
            OmopProjectionLoss(
                fact_id=fact.fact_id,
                source_key=item.source_key,
                reason_code=reason,
                fields=("effective_time",),
            )
            for reason in date_losses
        )
        if date_value is None:
            continue
        if _contains_unprojected_free_text(fact):
            losses.append(
                OmopProjectionLoss(
                    fact_id=fact.fact_id,
                    source_key=item.source_key,
                    reason_code="free_text_value_omitted",
                    fields=("value",),
                )
            )
        _upsert(
            batch_tables["person"],
            person_id,
            _person_row(person_id, fact.subject_id),
            "person",
        )
        _upsert(
            batch_tables["visit_occurrence"],
            visit_id,
            _visit_row(visit_id, person_id, fact.encounter_id, date_value),
            "visit_occurrence",
        )
        _upsert(
            batch_tables["note"],
            note_id,
            _note_row(
                note_id,
                person_id,
                visit_id,
                item.source_key,
                date_value,
            ),
            "note",
        )
        row = _domain_row(
            table,
            fact,
            mapping,
            person_id=person_id,
            visit_id=visit_id,
            date_value=date_value,
        )
        row_id = _table_row_id(table, row)
        _upsert(batch_tables[table], row_id, row, table)
        map_row = _source_map_row(mapping)
        map_id = _table_row_id("source_to_concept_map", map_row)
        _upsert(
            batch_tables["source_to_concept_map"],
            map_id,
            map_row,
            "source_to_concept_map",
        )
        provenance.append(
            OmopRowProvenance(
                provenance_id=derived_opaque_id(
                    "omopprov", table, row_id, fact.fact_id
                ),
                table=table,
                row_id=row_id,
                fact_id=fact.fact_id,
                subject_id=fact.subject_id,
                encounter_id=fact.encounter_id,
                source_key=item.source_key,
                source_revision=item.source_revision,
                evidence_ids=fact.evidence_ids,
                mapping_outcome_id=mapping.outcome_id,
                etl_run_id=run_id,
                correction_of=fact.parent_fact_ids,
                dataset_split=item.dataset_split,
            )
        )
        outcomes.append(
            OmopMappingOutcome(
                fact_id=fact.fact_id,
                source_key=item.source_key,
                table=table,
                row_id=row_id,
                mapping=mapping,
            )
        )

    if not provenance:
        return StoreResult.outcome(
            StoreState.UNSUPPORTED,
            "projection_has_no_supported_facts",
        )

    incoming_sources = {item.source_key for item in inputs}
    merged_tables, merged_provenance, merged_outcomes, merged_losses = _merge_current(
        batch_tables,
        provenance,
        outcomes,
        losses,
        previous=previous,
        incoming_sources=incoming_sources,
        mode=mode,
    )
    current_counts = {table: len(merged_tables[table]) for table in OMOP_FACT_TABLES}
    removed_fact_ids = (
        {
            item.fact_id
            for item in previous.provenance
            if item.source_key in incoming_sources
        }
        - {item.fact_id for item in merged_provenance}
        if previous is not None and mode == "replace_by_source"
        else set()
    )
    batch_output_digest = canonical_digest(
        {
            "losses": [item.to_dict() for item in losses],
            "mapping_outcomes": [item.to_dict() for item in outcomes],
            "provenance": [item.to_dict() for item in provenance],
            "tables": {
                table: [batch_tables[table][key] for key in sorted(batch_tables[table])]
                for table in OMOP_FACT_TABLES
            },
        }
    )
    run = OmopEtlRun(
        run_id=run_id,
        etl_version=etl_version,
        occurred_at=occurred_at,
        mode=mode,
        vocabulary_snapshot_digest=vocabulary_snapshot.digest,
        input_digest=input_digest,
        output_digest=batch_output_digest,
        source_keys=tuple(incoming_sources),
        input_fact_ids=tuple(item.fact.fact_id for item in inputs),
        superseded_fact_ids=tuple(removed_fact_ids),
        row_counts=current_counts,
    )
    runs_by_id = {item.run_id: item for item in (previous.etl_runs if previous else ())}
    existing_run = runs_by_id.get(run.run_id)
    if existing_run is not None and existing_run != run:
        raise OmopFactProjectionConflictError("replayed ETL run differs")
    runs_by_id[run.run_id] = run
    mapping_counts = Counter(item.mapping.state for item in merged_outcomes)
    summary = OmopProjectionSummary(
        row_counts=current_counts,
        mapping_counts={
            state: mapping_counts.get(state, 0) for state in OMOP_MAPPING_STATES
        },
        loss_counts=dict(Counter(item.reason_code for item in merged_losses)),
        current_fact_count=len(merged_provenance),
        etl_run_count=len(runs_by_id),
    )
    projection = OmopFactProjection(
        projection_id=derived_opaque_id(
            "omopproj", vocabulary_snapshot.digest, etl_version
        ),
        vocabulary_snapshot=vocabulary_snapshot,
        tables={
            table: tuple(
                merged_tables[table][key] for key in sorted(merged_tables[table])
            )
            for table in OMOP_FACT_TABLES
        },
        provenance=tuple(merged_provenance),
        mapping_outcomes=tuple(merged_outcomes),
        losses=tuple(merged_losses),
        etl_runs=tuple(runs_by_id.values()),
        summary=summary,
    )
    violations = validate_omop_fact_projection(projection)
    if violations:
        return StoreResult.outcome(
            StoreState.FAILURE,
            "projection_referential_integrity_failed",
            value=projection,
        )
    if any(item.mapping.requires_review for item in projection.mapping_outcomes):
        return StoreResult.outcome(
            StoreState.PARTIAL,
            "projection_mapping_review_required",
            value=projection,
        )
    if projection.losses:
        return StoreResult.outcome(
            StoreState.PARTIAL,
            "projection_information_loss",
            value=projection,
        )
    return StoreResult.success(projection, created=previous is None)


def _merge_current(
    batch_tables: Mapping[str, Mapping[int, dict[str, Any]]],
    batch_provenance: Sequence[OmopRowProvenance],
    batch_outcomes: Sequence[OmopMappingOutcome],
    batch_losses: Sequence[OmopProjectionLoss],
    *,
    previous: OmopFactProjection | None,
    incoming_sources: set[str],
    mode: str,
) -> tuple[
    dict[str, dict[int, dict[str, Any]]],
    tuple[OmopRowProvenance, ...],
    tuple[OmopMappingOutcome, ...],
    tuple[OmopProjectionLoss, ...],
]:
    tables: dict[str, dict[int, dict[str, Any]]] = {
        table: {} for table in OMOP_FACT_TABLES
    }
    old_provenance: tuple[OmopRowProvenance, ...] = ()
    old_outcomes: tuple[OmopMappingOutcome, ...] = ()
    old_losses: tuple[OmopProjectionLoss, ...] = ()
    removed_rows: set[tuple[str, int]] = set()
    removed_note_ids: set[int] = set()
    if previous is not None:
        old_provenance = previous.provenance
        old_outcomes = previous.mapping_outcomes
        old_losses = previous.losses
        if mode == "replace_by_source":
            removed_rows = {
                (item.table, item.row_id)
                for item in old_provenance
                if item.source_key in incoming_sources
            }
            removed_note_ids = {
                int(row["note_id"])
                for row in previous.table("note")
                if any(
                    item.source_key in incoming_sources
                    and int(row["note_id"])
                    == deterministic_omop_id(
                        "fact-note", item.source_key, item.subject_id
                    )
                    for item in old_provenance
                )
            }
        for table in OMOP_FACT_TABLES:
            for row in previous.table(table):
                row_id = _table_row_id(table, row)
                if (table, row_id) in removed_rows:
                    continue
                if table == "note" and row_id in removed_note_ids:
                    continue
                tables[table][row_id] = _plain(row)
    for table in OMOP_FACT_TABLES:
        for row_id, row in batch_tables[table].items():
            _upsert(tables[table], row_id, row, table)

    kept_provenance = tuple(
        item
        for item in old_provenance
        if mode != "replace_by_source" or item.source_key not in incoming_sources
    )
    kept_outcomes = tuple(
        item
        for item in old_outcomes
        if mode != "replace_by_source" or item.source_key not in incoming_sources
    )
    kept_losses = tuple(
        item
        for item in old_losses
        if mode != "replace_by_source" or item.source_key not in incoming_sources
    )
    merged_provenance = _deduplicate_provenance((*kept_provenance, *batch_provenance))
    merged_outcomes = _deduplicate_outcomes((*kept_outcomes, *batch_outcomes))
    merged_losses = _deduplicate_losses((*kept_losses, *batch_losses))
    _prune_supporting_rows(tables, merged_provenance, merged_outcomes)
    return tables, merged_provenance, merged_outcomes, merged_losses


def _mapping_from_fact(
    fact: ClinicalFact,
    snapshot: OmopVocabularySnapshot,
) -> OmopConceptMapping:
    value = fact.value if isinstance(fact.value, Mapping) else {}
    attributes = fact.attributes
    raw = attributes.get("mapping")
    if isinstance(raw, Mapping):
        payload = dict(raw)
        payload.setdefault("snapshot_digest", snapshot.digest)
        payload.setdefault("source_code", str(value.get("code") or fact.fact_type))
        payload.setdefault("source_system", str(value.get("system") or "local"))
        payload.setdefault("source_concept_id", 0)
        payload.setdefault("standard_concept_id", 0)
        payload.setdefault("standard_vocabulary", UNMAPPED_VOCABULARY_ID)
        payload.setdefault("standard_code", None)
        payload.setdefault("state", "unmapped")
        payload.setdefault("reason_code", "mapping_not_resolved")
        payload.setdefault("valid_start_date", "1970-01-01")
        payload.setdefault("valid_end_date", "2099-12-31")
        return OmopConceptMapping.from_dict(payload)
    return OmopConceptMapping.unmapped(
        source_system=str(value.get("system") or "local"),
        source_code=str(value.get("code") or fact.fact_type),
        snapshot_digest=snapshot.digest,
    )


def _person_row(person_id: int, subject_id: str) -> dict[str, Any]:
    return {
        "person_id": person_id,
        "gender_concept_id": 0,
        "year_of_birth": None,
        "month_of_birth": None,
        "day_of_birth": None,
        "birth_datetime": None,
        "race_concept_id": 0,
        "ethnicity_concept_id": 0,
        "location_id": None,
        "provider_id": None,
        "care_site_id": None,
        "person_source_value": subject_id,
        "gender_source_value": None,
        "gender_source_concept_id": 0,
        "race_source_value": None,
        "race_source_concept_id": 0,
        "ethnicity_source_value": None,
        "ethnicity_source_concept_id": 0,
    }


def _visit_row(
    visit_id: int,
    person_id: int,
    encounter_id: str | None,
    date_value: str,
) -> dict[str, Any]:
    return {
        "visit_occurrence_id": visit_id,
        "person_id": person_id,
        "visit_concept_id": 0,
        "visit_start_date": date_value,
        "visit_start_datetime": None,
        "visit_end_date": date_value,
        "visit_end_datetime": None,
        "visit_type_concept_id": 0,
        "provider_id": None,
        "care_site_id": None,
        "visit_source_value": encounter_id,
        "visit_source_concept_id": 0,
        "admitted_from_concept_id": 0,
        "admitted_from_source_value": None,
        "discharged_to_concept_id": 0,
        "discharged_to_source_value": None,
        "preceding_visit_occurrence_id": None,
    }


def _note_row(
    note_id: int,
    person_id: int,
    visit_id: int,
    source_key: str,
    date_value: str,
) -> dict[str, Any]:
    return {
        "note_id": note_id,
        "person_id": person_id,
        "note_date": date_value,
        "note_datetime": None,
        "note_type_concept_id": 0,
        "note_class_concept_id": 0,
        "note_title": None,
        "note_text": "",
        "encoding_concept_id": 0,
        "language_concept_id": 0,
        "provider_id": None,
        "visit_occurrence_id": visit_id,
        "visit_detail_id": None,
        "note_source_value": source_key,
        "note_event_id": None,
        "note_event_field_concept_id": 0,
    }


def _domain_row(
    table: str,
    fact: ClinicalFact,
    mapping: OmopConceptMapping,
    *,
    person_id: int,
    visit_id: int,
    date_value: str,
) -> dict[str, Any]:
    primary_key, concept_column, source_concept_column, date_column = _DOMAIN_SPEC[
        table
    ]
    row_id = deterministic_omop_id("fact-row", table, fact.fact_id)
    source_value = mapping.source_code
    base: dict[str, Any] = {
        primary_key: row_id,
        "person_id": person_id,
        concept_column: mapping.standard_concept_id,
        date_column: date_value,
        "visit_occurrence_id": visit_id,
        source_concept_column: mapping.source_concept_id,
    }
    if table == "condition_occurrence":
        base.update(
            {
                "condition_end_date": _fact_end_date(fact),
                "condition_start_datetime": None,
                "condition_end_datetime": None,
                "condition_type_concept_id": 0,
                "condition_status_concept_id": 0,
                "stop_reason": None,
                "provider_id": None,
                "visit_detail_id": None,
                "condition_source_value": source_value,
                "condition_status_source_value": fact.status,
            }
        )
    elif table == "drug_exposure":
        base.update(
            {
                "drug_exposure_end_date": _fact_end_date(fact) or date_value,
                "drug_exposure_start_datetime": None,
                "drug_exposure_end_datetime": None,
                "verbatim_end_date": _fact_end_date(fact),
                "drug_type_concept_id": 0,
                "stop_reason": None,
                "refills": None,
                "quantity": _numeric_value(fact, "quantity"),
                "days_supply": _numeric_value(fact, "days_supply"),
                "sig": None,
                "route_concept_id": 0,
                "lot_number": None,
                "provider_id": None,
                "visit_detail_id": None,
                "drug_source_value": source_value,
                "route_source_value": None,
                "dose_unit_source_value": fact.unit,
            }
        )
    elif table == "procedure_occurrence":
        base.update(
            {
                "procedure_end_date": _fact_end_date(fact),
                "procedure_datetime": None,
                "procedure_end_datetime": None,
                "procedure_type_concept_id": 0,
                "modifier_concept_id": 0,
                "quantity": _numeric_value(fact, "quantity"),
                "provider_id": None,
                "visit_detail_id": None,
                "procedure_source_value": source_value,
                "modifier_source_value": None,
            }
        )
    elif table == "measurement":
        base.update(
            {
                "measurement_datetime": None,
                "measurement_time": None,
                "measurement_type_concept_id": 0,
                "operator_concept_id": 0,
                "value_as_number": _numeric_value(fact, "numeric"),
                "value_as_concept_id": 0,
                "unit_concept_id": 0,
                "range_low": _nested_numeric(fact, "range", "low"),
                "range_high": _nested_numeric(fact, "range", "high"),
                "provider_id": None,
                "visit_detail_id": None,
                "measurement_source_value": source_value,
                "unit_source_value": fact.unit,
                "unit_source_concept_id": 0,
                "value_source_value": _safe_value_string(fact),
                "measurement_event_id": None,
                "meas_event_field_concept_id": 0,
            }
        )
    elif table == "observation":
        base.update(
            {
                "observation_datetime": None,
                "observation_type_concept_id": 0,
                "value_as_number": _numeric_value(fact, "numeric"),
                "value_as_string": _safe_value_string(fact),
                "value_as_concept_id": 0,
                "qualifier_concept_id": 0,
                "unit_concept_id": 0,
                "provider_id": None,
                "visit_detail_id": None,
                "observation_source_value": source_value,
                "qualifier_source_value": None,
                "unit_source_value": fact.unit,
                "value_source_value": _safe_value_string(fact),
                "observation_event_id": None,
                "obs_event_field_concept_id": 0,
            }
        )
    return base


def _source_map_row(mapping: OmopConceptMapping) -> dict[str, Any]:
    return {
        "source_code": mapping.source_code,
        "source_concept_id": mapping.source_concept_id,
        "source_vocabulary_id": mapping.source_system,
        "source_code_description": None,
        "target_concept_id": mapping.standard_concept_id,
        "target_vocabulary_id": mapping.standard_vocabulary,
        "valid_start_date": mapping.valid_start_date,
        "valid_end_date": mapping.valid_end_date,
        "invalid_reason": None,
    }


def _fact_date(fact: ClinicalFact) -> tuple[str | None, tuple[str, ...]]:
    effective = fact.effective_time
    candidate = effective.get("start") or effective.get("instant")
    if candidate is None:
        return None, ("effective_time_missing",)
    value = str(candidate)
    if len(value) >= 10 and _date_prefix_valid(value[:10]):
        return value[:10], ()
    return None, ("effective_time_precision_unsupported",)


def _fact_end_date(fact: ClinicalFact) -> str | None:
    value = fact.effective_time.get("end")
    if value is None:
        return None
    text = str(value)
    return text[:10] if len(text) >= 10 and _date_prefix_valid(text[:10]) else None


def _date_prefix_valid(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _date_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _date_prefix_valid(value):
        raise OmopFactProjectionError(f"{name} must be an ISO date")
    return value


def _numeric_value(fact: ClinicalFact, key: str) -> float | None:
    value = fact.value
    if isinstance(value, Mapping):
        value = value.get(key)
    elif key != "numeric":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nested_numeric(fact: ClinicalFact, parent: str, key: str) -> float | None:
    if not isinstance(fact.value, Mapping):
        return None
    nested = fact.value.get(parent)
    if not isinstance(nested, Mapping):
        return None
    value = nested.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _safe_value_string(fact: ClinicalFact) -> str | None:
    if isinstance(fact.value, Mapping):
        for key in ("category", "state", "result"):
            value = fact.value.get(key)
            if isinstance(value, str) and value:
                return value[:256]
    return None


def _contains_unprojected_free_text(fact: ClinicalFact) -> bool:
    if isinstance(fact.value, str):
        return True
    if not isinstance(fact.value, Mapping):
        return False
    return any(
        isinstance(fact.value.get(key), str) and bool(fact.value.get(key))
        for key in ("display", "narrative", "raw_text", "text")
    )


def _validate_source_subjects(inputs: Sequence[OmopFactProjectionInput]) -> None:
    by_source: dict[str, set[tuple[str, str | None]]] = defaultdict(set)
    for item in inputs:
        by_source[item.source_key].add((item.fact.subject_id, item.fact.encounter_id))
    if any(len(contexts) != 1 for contexts in by_source.values()):
        raise OmopFactProjectionConflictError(
            "one source key cannot span subjects or encounters"
        )


def _validate_split_isolation(
    inputs: Sequence[OmopFactProjectionInput],
    previous: OmopFactProjection | None,
) -> None:
    by_subject: dict[str, set[str]] = defaultdict(set)
    incoming_sources = {item.source_key for item in inputs}
    if previous is not None:
        for item in previous.provenance:
            if item.source_key in incoming_sources or item.dataset_split is None:
                continue
            by_subject[item.subject_id].add(item.dataset_split)
    for item in inputs:
        if item.dataset_split is not None:
            by_subject[item.fact.subject_id].add(item.dataset_split)
    if any(len(splits) > 1 for splits in by_subject.values()):
        raise OmopFactProjectionConflictError(
            "a subject cannot cross dataset splits in one projection"
        )


def _upsert(
    rows: dict[int, dict[str, Any]],
    row_id: int,
    row: Mapping[str, Any],
    table: str,
) -> None:
    value = _plain(row)
    existing = rows.get(row_id)
    if existing is not None and existing != value:
        if table == "visit_occurrence":
            existing_start = existing.get("visit_start_date")
            existing_end = existing.get("visit_end_date")
            incoming_start = value.get("visit_start_date")
            starts = [item for item in (existing_start, incoming_start) if item]
            ends = [item for item in (existing_end, incoming_start) if item]
            existing["visit_start_date"] = min(starts) if starts else None
            existing["visit_end_date"] = max(ends) if ends else None
            return
        if table == "note":
            existing_comparable = dict(existing)
            incoming_comparable = dict(value)
            existing_date = existing_comparable.pop("note_date")
            incoming_date = incoming_comparable.pop("note_date")
            if existing_comparable == incoming_comparable:
                dates = [item for item in (existing_date, incoming_date) if item]
                existing["note_date"] = min(dates) if dates else None
                return
        raise OmopFactProjectionConflictError(f"{table} deterministic row conflict")
    rows[row_id] = value


def _deduplicate_provenance(
    items: Sequence[OmopRowProvenance],
) -> tuple[OmopRowProvenance, ...]:
    by_fact: dict[str, OmopRowProvenance] = {}
    for item in items:
        existing = by_fact.get(item.fact_id)
        if existing is not None and existing != item:
            raise OmopFactProjectionConflictError("fact provenance conflicts")
        by_fact[item.fact_id] = item
    return tuple(sorted(by_fact.values(), key=_provenance_sort_key))


def _deduplicate_outcomes(
    items: Sequence[OmopMappingOutcome],
) -> tuple[OmopMappingOutcome, ...]:
    by_fact: dict[str, OmopMappingOutcome] = {}
    for item in items:
        existing = by_fact.get(item.fact_id)
        if existing is not None and existing != item:
            raise OmopFactProjectionConflictError("fact mapping outcomes conflict")
        by_fact[item.fact_id] = item
    return tuple(sorted(by_fact.values(), key=lambda item: (item.table, item.row_id)))


def _deduplicate_losses(
    items: Sequence[OmopProjectionLoss],
) -> tuple[OmopProjectionLoss, ...]:
    return tuple(
        sorted(
            set(items),
            key=lambda item: (item.fact_id, item.reason_code, item.fields),
        )
    )


def _prune_supporting_rows(
    tables: dict[str, dict[int, dict[str, Any]]],
    provenance: Sequence[OmopRowProvenance],
    outcomes: Sequence[OmopMappingOutcome],
) -> None:
    active_rows = {(item.table, item.row_id) for item in provenance}
    for table in OMOP_DOMAIN_TABLES:
        tables[table] = {
            row_id: row
            for row_id, row in tables[table].items()
            if (table, row_id) in active_rows
        }
    active_notes = {
        deterministic_omop_id("fact-note", item.source_key, item.subject_id)
        for item in provenance
    }
    tables["note"] = {
        row_id: row for row_id, row in tables["note"].items() if row_id in active_notes
    }
    active_visits = {int(row["visit_occurrence_id"]) for row in tables["note"].values()}
    tables["visit_occurrence"] = {
        row_id: row
        for row_id, row in tables["visit_occurrence"].items()
        if row_id in active_visits
    }
    # Visit bounds are derived from current fact dates, not historical rows.
    # Recompute after replacement so corrected or removed sources cannot leave
    # stale bounds, and every date in an incoming batch remains represented.
    visit_dates: dict[int, list[str]] = defaultdict(list)
    for table in OMOP_DOMAIN_TABLES:
        date_column = _DOMAIN_SPEC[table][3]
        for row in tables[table].values():
            visit_dates[int(row["visit_occurrence_id"])].append(row[date_column])
    for visit_id, dates in visit_dates.items():
        visit = tables["visit_occurrence"][visit_id]
        visit["visit_start_date"] = min(dates)
        visit["visit_end_date"] = max(dates)
    active_people = {int(row["person_id"]) for row in tables["note"].values()}
    tables["person"] = {
        row_id: row
        for row_id, row in tables["person"].items()
        if row_id in active_people
    }
    active_mappings = {
        (
            outcome.mapping.source_code,
            outcome.mapping.source_concept_id,
            outcome.mapping.source_system,
            outcome.mapping.standard_concept_id,
            outcome.mapping.standard_vocabulary,
            outcome.mapping.valid_start_date,
            outcome.mapping.valid_end_date,
        )
        for outcome in outcomes
    }
    tables["source_to_concept_map"] = {
        row_id: row
        for row_id, row in tables["source_to_concept_map"].items()
        if (
            row["source_code"],
            row["source_concept_id"],
            row["source_vocabulary_id"],
            row["target_concept_id"],
            row["target_vocabulary_id"],
            row["valid_start_date"],
            row["valid_end_date"],
        )
        in active_mappings
    }


def _provenance_sort_key(item: OmopRowProvenance) -> tuple[str, int, str]:
    return item.table, item.row_id, item.fact_id


def _table_row_id(table: str, row: Mapping[str, Any]) -> int:
    primary_key = _PRIMARY_KEY_BY_TABLE[table]
    if primary_key is not None:
        return _positive_integer(row[primary_key], primary_key)
    return deterministic_omop_id(
        "fact-source-map",
        row["source_vocabulary_id"],
        row["source_code"],
        row["source_concept_id"],
        row["target_concept_id"],
        row["target_vocabulary_id"],
        row["valid_start_date"],
        row["valid_end_date"],
        row["invalid_reason"],
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OmopFactProjectionError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise OmopFactProjectionError(f"{name} keys must be strings")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OmopFactProjectionError(f"{name} must be a sequence")
    return value


def _text_sequence(value: Any, name: str) -> tuple[str, ...]:
    return tuple(_text(item, name) for item in _sequence(value, name))


def _exact_keys(value: Mapping[str, Any], required: set[str], name: str) -> None:
    if set(value) != required:
        raise OmopFactProjectionError(f"{name} fields do not match contract")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise OmopFactProjectionError(f"{name} must be text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _bounded_text(value: Any, name: str, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise OmopFactProjectionError(f"{name} must be bounded text")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise OmopFactProjectionError(f"{name} must be boolean")
    return value


def _controlled(value: Any, name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise OmopFactProjectionError(f"{name} must be controlled")
    return value


def _vocabulary_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or _VOCABULARY_ID_RE.fullmatch(value) is None:
        raise OmopFactProjectionError(f"{name} must be an OMOP vocabulary identifier")
    return value


def _opaque_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise OmopFactProjectionError(f"{name} must be an opaque identifier")
    return value


def _opaque_ids(
    values: Iterable[str], name: str, *, minimum: int = 0
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise OmopFactProjectionError(f"{name} must be an identifier sequence")
    normalized = tuple(sorted(_opaque_id(item, name) for item in values))
    if len(normalized) < minimum or len(normalized) != len(set(normalized)):
        raise OmopFactProjectionError(f"{name} has invalid cardinality")
    return normalized


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise OmopFactProjectionError(f"{name} must be a sha256 digest")
    return value


def _version(value: Any, name: str) -> str:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise OmopFactProjectionError(f"{name} must be semantic")
    return value


def _timestamp(value: Any, name: str) -> str:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        raise OmopFactProjectionError(f"{name} must be timezone-aware")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise OmopFactProjectionError(f"{name} must be timezone-aware") from exc
    return value


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise OmopFactProjectionError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: Any, name: str) -> int:
    number = _integer(value, name)
    if number < 1:
        raise OmopFactProjectionError(f"{name} must be positive")
    return number


def _count_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, int]:
    data = _mapping(value, name)
    counts: dict[str, int] = {}
    for key, item in data.items():
        _controlled(key, name)
        counts[key] = _integer(item, name)
    return MappingProxyType(dict(sorted(counts.items())))


def _freeze_mapping(value: Any, name: str) -> Mapping[str, Any]:
    try:
        normalized = json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise OmopFactProjectionError(f"{name} must be JSON-safe") from exc
    if not isinstance(normalized, dict):
        raise OmopFactProjectionError(f"{name} must be an object")
    return _deep_freeze(normalized)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _contract(schema_version: str, compatibility_policy: str, cdm_version: str) -> None:
    if schema_version != OMOP_FACT_PROJECTION_SCHEMA_VERSION:
        raise OmopFactProjectionUnsupportedError(
            "projection schema version is unsupported"
        )
    if compatibility_policy != OMOP_FACT_PROJECTION_COMPATIBILITY_POLICY:
        raise OmopFactProjectionUnsupportedError(
            "projection compatibility policy is unsupported"
        )
    if cdm_version != OMOP_FACT_PROJECTION_CDM_VERSION:
        raise OmopFactProjectionUnsupportedError("OMOP CDM version is unsupported")


__all__ = [
    "OMOP_DATASET_SPLITS",
    "OMOP_DOMAIN_TABLES",
    "OMOP_FACT_PROJECTION_CDM_VERSION",
    "OMOP_FACT_PROJECTION_COMPATIBILITY_POLICY",
    "OMOP_FACT_PROJECTION_MODES",
    "OMOP_FACT_PROJECTION_SCHEMA_NAME",
    "OMOP_FACT_PROJECTION_SCHEMA_VERSION",
    "OMOP_FACT_PROJECTOR_VERSION",
    "OMOP_FACT_TABLES",
    "OMOP_MAPPING_STATES",
    "OMOP_PERMISSIVE_LICENSES",
    "OmopConceptMapping",
    "OmopEtlRun",
    "OmopFactProjection",
    "OmopFactProjectionConflictError",
    "OmopFactProjectionDeniedError",
    "OmopFactProjectionError",
    "OmopFactProjectionInput",
    "OmopFactProjectionUnsupportedError",
    "OmopFactRoundTripReport",
    "OmopMappingOutcome",
    "OmopProjectionLoss",
    "OmopProjectionSummary",
    "OmopProjectionViolation",
    "OmopRowProvenance",
    "OmopVocabularySnapshot",
    "assess_omop_fact_round_trip",
    "load_omop_fact_projection_schema",
    "project_clinical_facts_to_omop",
    "validate_omop_fact_projection",
]
