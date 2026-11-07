"""Deterministic governed datasets from saved cohorts or ingestion jobs.

Default artifacts are value-free projections: opaque source references,
labels, annotations, split custody, and digests.  Callers can explicitly opt
into de-identified values. Identified values require a snapshot-bound
authorization and a two-phase audit hook at export time. Token-vault material
is never exportable through this module.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module, resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from openmed.clinical.journey_contracts import (
    DatasetSnapshot,
    canonical_digest,
    canonical_json,
    derived_opaque_id,
    sha256_digest,
)
from openmed.structured.cohort import CohortExecution
from openmed.structured.store import StoreResult, StoreState

GOVERNED_DATASET_SCHEMA_VERSION: Final = "1.0.0"
GOVERNED_DATASET_COMPATIBILITY_POLICY: Final = "same_major"
GOVERNED_DATASET_SCHEMA_NAME: Final = "governed_dataset"
GOVERNED_DATASET_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)

_VAULT_KEYS = frozenset(
    {
        "credential",
        "credentials",
        "encryption_key",
        "password",
        "secret",
        "token",
        "token_vault",
        "vault",
        "vault_material",
    }
)
_DIRECT_IDENTIFIER_KEYS = frozenset(
    {
        "address",
        "email",
        "full_name",
        "medical_record_number",
        "mrn",
        "name",
        "phone",
        "raw",
        "raw_text",
        "source_text",
        "ssn",
    }
)


class GovernedDatasetError(ValueError):
    """Base error for malformed governed-dataset contracts."""


class GovernedDatasetConflictError(GovernedDatasetError):
    """Raised when custody, split, digest, or immutable output conflicts."""


class GovernedDatasetUnsupportedError(GovernedDatasetError):
    """Raised when a schema, format, or optional backend is unsupported."""


class DatasetSourceKind(str, Enum):
    """Supported governed selection sources."""

    COHORT = "cohort"
    INGESTION_JOB = "ingestion_job"


class DatasetPrivacyState(str, Enum):
    """Declared projection state for values held in memory."""

    DEIDENTIFIED = "deidentified"
    IDENTIFIED = "identified"


class DatasetExportFormat(str, Enum):
    """Deterministic dataset artifact formats."""

    JSONL = "jsonl"
    PARQUET = "parquet"
    ANNOTATION_JSONL = "annotation_jsonl"


class RedistributionPolicy(str, Enum):
    """Distribution posture for one input license."""

    PERMITTED = "permitted"
    RESTRICTED = "restricted"
    PROHIBITED = "prohibited"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DatasetSelection:
    """A value-free selection source bound to a named source snapshot."""

    source_kind: DatasetSourceKind
    source_id: str
    source_digest: str
    source_snapshot_id: str
    source_snapshot_digest: str
    eligible_patient_keys: tuple[str, ...] = ()
    review_excluded_count: int = 0

    def __post_init__(self) -> None:
        kind = _enum(self.source_kind, DatasetSourceKind, "source_kind")
        object.__setattr__(self, "source_kind", kind)
        _opaque_id(self.source_id, "source_id")
        _digest(self.source_digest, "source_digest")
        _opaque_id(self.source_snapshot_id, "source_snapshot_id")
        _digest(self.source_snapshot_digest, "source_snapshot_digest")
        patients = tuple(
            sorted(
                _opaque_id(item, "patient_key") for item in self.eligible_patient_keys
            )
        )
        if len(patients) != len(set(patients)):
            raise GovernedDatasetConflictError("eligible patient keys must be unique")
        if (
            type(self.review_excluded_count) is not int
            or self.review_excluded_count < 0
        ):
            raise GovernedDatasetError("review_excluded_count must be non-negative")
        if kind is DatasetSourceKind.INGESTION_JOB and patients:
            raise GovernedDatasetError(
                "ingestion job selection cannot declare cohort eligibility"
            )
        object.__setattr__(self, "eligible_patient_keys", patients)

    @classmethod
    def from_cohort_execution(cls, execution: CohortExecution) -> "DatasetSelection":
        """Select only fully resolved, eligible memberships from a saved run."""

        if not isinstance(execution, CohortExecution):
            raise TypeError("execution must be CohortExecution")
        eligible = tuple(
            item.patient_key for item in execution.memberships if item.eligible
        )
        excluded = sum(item.review_required for item in execution.memberships)
        return cls(
            source_kind=DatasetSourceKind.COHORT,
            source_id=execution.manifest.execution_id or "",
            source_digest=execution.execution_digest,
            source_snapshot_id=execution.manifest.source_snapshot.snapshot_id,
            source_snapshot_digest=execution.manifest.source_snapshot.digest,
            eligible_patient_keys=eligible,
            review_excluded_count=excluded,
        )

    @classmethod
    def from_ingestion_job(
        cls,
        *,
        job_id: str,
        job_digest: str,
        source_snapshot_id: str,
        source_snapshot_digest: str,
    ) -> "DatasetSelection":
        """Build a selection reference for one completed ingestion job."""

        return cls(
            source_kind=DatasetSourceKind.INGESTION_JOB,
            source_id=job_id,
            source_digest=job_digest,
            source_snapshot_id=source_snapshot_id,
            source_snapshot_digest=source_snapshot_digest,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical selection custody."""

        return {
            "eligible_patient_keys": list(self.eligible_patient_keys),
            "review_excluded_count": self.review_excluded_count,
            "source_digest": self.source_digest,
            "source_id": self.source_id,
            "source_kind": self.source_kind.value,
            "source_snapshot_digest": self.source_snapshot_digest,
            "source_snapshot_id": self.source_snapshot_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetSelection":
        """Parse a strict selection reference."""

        data = _mapping(value, "dataset selection")
        _exact_keys(
            data,
            {
                "eligible_patient_keys",
                "review_excluded_count",
                "source_digest",
                "source_id",
                "source_kind",
                "source_snapshot_digest",
                "source_snapshot_id",
            },
            "dataset selection",
        )
        return cls(
            source_kind=_enum(data["source_kind"], DatasetSourceKind, "source_kind"),
            source_id=_text(data["source_id"], "source_id"),
            source_digest=_text(data["source_digest"], "source_digest"),
            source_snapshot_id=_text(data["source_snapshot_id"], "source_snapshot_id"),
            source_snapshot_digest=_text(
                data["source_snapshot_digest"], "source_snapshot_digest"
            ),
            eligible_patient_keys=_text_sequence(
                data["eligible_patient_keys"], "eligible_patient_keys"
            ),
            review_excluded_count=_integer(
                data["review_excluded_count"], "review_excluded_count"
            ),
        )


@dataclass(frozen=True, slots=True)
class DatasetLicenseConstraint:
    """Digest-bound input terms and their redistribution posture."""

    source_id: str
    license_id: str
    terms_digest: str
    redistribution: RedistributionPolicy

    def __post_init__(self) -> None:
        _controlled(self.source_id, "license source_id")
        _bounded_text(self.license_id, "license_id", 128)
        _digest(self.terms_digest, "license terms_digest")
        object.__setattr__(
            self,
            "redistribution",
            _enum(
                self.redistribution,
                RedistributionPolicy,
                "redistribution policy",
            ),
        )

    @property
    def distribution_allowed(self) -> bool:
        """Return whether these exact terms permit artifact distribution."""

        return self.redistribution is RedistributionPolicy.PERMITTED

    def to_dict(self) -> dict[str, str]:
        """Return value-free license custody."""

        return {
            "license_id": self.license_id,
            "redistribution": self.redistribution.value,
            "source_id": self.source_id,
            "terms_digest": self.terms_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetLicenseConstraint":
        """Parse one strict license constraint."""

        data = _mapping(value, "license constraint")
        _exact_keys(
            data,
            {"license_id", "redistribution", "source_id", "terms_digest"},
            "license constraint",
        )
        return cls(
            source_id=_text(data["source_id"], "license source_id"),
            license_id=_text(data["license_id"], "license_id"),
            terms_digest=_text(data["terms_digest"], "license terms_digest"),
            redistribution=_enum(
                data["redistribution"],
                RedistributionPolicy,
                "redistribution policy",
            ),
        )


@dataclass(frozen=True, slots=True)
class DatasetAnnotation:
    """Value-free span label with source digest and optional evidence custody."""

    annotation_id: str
    label: str
    start: int
    end: int
    source_digest: str
    evidence_id: str | None = None

    def __post_init__(self) -> None:
        _opaque_id(self.annotation_id, "annotation_id")
        _label(self.label, "annotation label")
        if type(self.start) is not int or type(self.end) is not int:
            raise GovernedDatasetError("annotation offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise GovernedDatasetError("annotation offsets are invalid")
        _digest(self.source_digest, "annotation source_digest")
        if self.evidence_id is not None:
            _opaque_id(self.evidence_id, "annotation evidence_id")

    def to_dict(self) -> dict[str, Any]:
        """Return offsets, label, digest, and opaque evidence only."""

        return {
            "annotation_id": self.annotation_id,
            "end": self.end,
            "evidence_id": self.evidence_id,
            "label": self.label,
            "source_digest": self.source_digest,
            "start": self.start,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetAnnotation":
        """Parse a strict annotation reference."""

        data = _mapping(value, "dataset annotation")
        _exact_keys(
            data,
            {"annotation_id", "end", "evidence_id", "label", "source_digest", "start"},
            "dataset annotation",
        )
        return cls(
            annotation_id=_text(data["annotation_id"], "annotation_id"),
            label=_text(data["label"], "annotation label"),
            start=_integer(data["start"], "annotation start"),
            end=_integer(data["end"], "annotation end"),
            source_digest=_text(data["source_digest"], "annotation source_digest"),
            evidence_id=_optional_text(data["evidence_id"], "annotation evidence_id"),
        )


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """One split-bound dataset row with opaque lineage and optional values."""

    record_id: str
    patient_key: str
    split: str
    source_artifact_ids: tuple[str, ...] = ()
    source_fact_ids: tuple[str, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    annotations: tuple[DatasetAnnotation, ...] = ()
    values: Mapping[str, Any] = field(default_factory=dict, repr=False)
    privacy_state: DatasetPrivacyState = DatasetPrivacyState.DEIDENTIFIED

    def __post_init__(self) -> None:
        _opaque_id(self.record_id, "record_id")
        _opaque_id(self.patient_key, "patient_key")
        _controlled(self.split, "split")
        for name in (
            "source_artifact_ids",
            "source_fact_ids",
            "source_event_ids",
            "evidence_ids",
        ):
            references = tuple(
                sorted(_opaque_id(item, name) for item in getattr(self, name))
            )
            if len(references) != len(set(references)):
                raise GovernedDatasetConflictError(f"{name} must be unique")
            object.__setattr__(self, name, references)
        if not (
            self.source_artifact_ids
            or self.source_fact_ids
            or self.source_event_ids
            or self.evidence_ids
        ):
            raise GovernedDatasetError("dataset record requires a source reference")
        labels = tuple(sorted({_label(item, "record label") for item in self.labels}))
        if any(not isinstance(item, DatasetAnnotation) for item in self.annotations):
            raise TypeError("annotations must contain DatasetAnnotation")
        annotations = tuple(
            sorted(
                self.annotations,
                key=lambda item: (item.start, item.end, item.label, item.annotation_id),
            )
        )
        if len({item.annotation_id for item in annotations}) != len(annotations):
            raise GovernedDatasetConflictError("annotation identifiers must be unique")
        normalized_values = _plain_mapping(self.values, "record values")
        _assert_no_vault_material(normalized_values)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "annotations", annotations)
        object.__setattr__(self, "values", _freeze_json_mapping(normalized_values))
        object.__setattr__(
            self,
            "privacy_state",
            _enum(self.privacy_state, DatasetPrivacyState, "privacy_state"),
        )

    @property
    def row_digest(self) -> str:
        """Return a digest over lineage, labels, annotations, and held values."""

        return canonical_digest(self._content_dict(include_values=True))

    def export_dict(
        self,
        *,
        include_values: bool = False,
        identified: bool = False,
    ) -> dict[str, Any]:
        """Return a safe projection, requiring explicit identified context."""

        if include_values and self.privacy_state is DatasetPrivacyState.IDENTIFIED:
            if not identified:
                raise PermissionError("identified export authorization is required")
        if include_values and not identified:
            _assert_deidentified_values(self.values)
        payload = self._content_dict(include_values=include_values)
        if not include_values:
            payload["privacy_state"] = DatasetPrivacyState.DEIDENTIFIED.value
        return payload

    def _content_dict(self, *, include_values: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "annotations": [item.to_dict() for item in self.annotations],
            "evidence_ids": list(self.evidence_ids),
            "labels": list(self.labels),
            "patient_key": self.patient_key,
            "privacy_state": self.privacy_state.value,
            "record_id": self.record_id,
            "source_artifact_ids": list(self.source_artifact_ids),
            "source_event_ids": list(self.source_event_ids),
            "source_fact_ids": list(self.source_fact_ids),
            "split": self.split,
        }
        if include_values:
            payload["values"] = _plain_json_value(self.values)
        return payload


@dataclass(frozen=True, slots=True)
class DatasetBuildSpec:
    """Pinned inputs and requested safe artifacts for one dataset build."""

    dataset_id: str
    created_at: str
    selection: DatasetSelection
    query_digest: str
    policy_digest: str
    schema_digest: str
    vocabulary_digest: str
    component_versions: Mapping[str, str]
    model_versions: Mapping[str, str]
    licenses: tuple[DatasetLicenseConstraint, ...]
    formats: tuple[DatasetExportFormat, ...] = (
        DatasetExportFormat.JSONL,
        DatasetExportFormat.PARQUET,
        DatasetExportFormat.ANNOTATION_JSONL,
    )
    include_deidentified_values: bool = False
    schema_version: str = GOVERNED_DATASET_SCHEMA_VERSION
    compatibility_policy: str = GOVERNED_DATASET_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract_version(self.schema_version, self.compatibility_policy)
        _opaque_id(self.dataset_id, "dataset_id")
        _timestamp(self.created_at, "created_at")
        if not isinstance(self.selection, DatasetSelection):
            raise TypeError("selection must be DatasetSelection")
        for name in (
            "query_digest",
            "policy_digest",
            "schema_digest",
            "vocabulary_digest",
        ):
            _digest(getattr(self, name), name)
        components = _version_mapping(self.component_versions, "component_versions")
        models = _version_mapping(self.model_versions, "model_versions")
        if any(
            not isinstance(item, DatasetLicenseConstraint) for item in self.licenses
        ):
            raise TypeError("licenses must contain DatasetLicenseConstraint")
        licenses = tuple(sorted(self.licenses, key=lambda item: item.source_id))
        if not licenses or len({item.source_id for item in licenses}) != len(licenses):
            raise GovernedDatasetConflictError(
                "license constraints must be non-empty with unique sources"
            )
        formats = tuple(
            sorted(
                {
                    _enum(item, DatasetExportFormat, "dataset format")
                    for item in self.formats
                },
                key=lambda item: item.value,
            )
        )
        if not formats:
            raise GovernedDatasetError("at least one dataset format is required")
        if type(self.include_deidentified_values) is not bool:
            raise GovernedDatasetError("include_deidentified_values must be boolean")
        object.__setattr__(self, "component_versions", MappingProxyType(components))
        object.__setattr__(self, "model_versions", MappingProxyType(models))
        object.__setattr__(self, "licenses", licenses)
        object.__setattr__(self, "formats", formats)


@dataclass(frozen=True, slots=True)
class GovernedDatasetManifest:
    """Persisted manifest that extends the canonical Journey snapshot."""

    snapshot: DatasetSnapshot
    selection: DatasetSelection
    policy_digest: str
    vocabulary_digest: str
    model_versions: Mapping[str, str]
    license_constraints: tuple[DatasetLicenseConstraint, ...]
    record_digests: Mapping[str, str]
    split_records: Mapping[str, tuple[str, ...]]
    formats: tuple[DatasetExportFormat, ...]
    export_profile: str
    schema_version: str = GOVERNED_DATASET_SCHEMA_VERSION
    compatibility_policy: str = GOVERNED_DATASET_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract_version(self.schema_version, self.compatibility_policy)
        if not isinstance(self.snapshot, DatasetSnapshot):
            raise TypeError("snapshot must be DatasetSnapshot")
        if not isinstance(self.selection, DatasetSelection):
            raise TypeError("selection must be DatasetSelection")
        _digest(self.policy_digest, "policy_digest")
        _digest(self.vocabulary_digest, "vocabulary_digest")
        models = _version_mapping(self.model_versions, "model_versions")
        records = _digest_mapping(
            self.record_digests, "record_digests", opaque_keys=True
        )
        splits = _split_mapping(self.split_records)
        if {item for values in splits.values() for item in values} != set(records):
            raise GovernedDatasetConflictError(
                "split records must cover every dataset record exactly once"
            )
        if sum(len(values) for values in splits.values()) != len(records):
            raise GovernedDatasetConflictError("record appears in more than one split")
        if any(
            not isinstance(item, DatasetLicenseConstraint)
            for item in self.license_constraints
        ):
            raise TypeError("license_constraints must contain DatasetLicenseConstraint")
        licenses = tuple(
            sorted(self.license_constraints, key=lambda item: item.source_id)
        )
        formats = tuple(
            sorted(
                {
                    _enum(item, DatasetExportFormat, "dataset format")
                    for item in self.formats
                },
                key=lambda item: item.value,
            )
        )
        if self.export_profile not in {"metadata_only", "deidentified_values"}:
            raise GovernedDatasetUnsupportedError("export profile is unsupported")
        object.__setattr__(self, "model_versions", MappingProxyType(models))
        object.__setattr__(self, "record_digests", MappingProxyType(records))
        object.__setattr__(self, "split_records", MappingProxyType(splits))
        object.__setattr__(self, "license_constraints", licenses)
        object.__setattr__(self, "formats", formats)
        self._verify_snapshot()

    @property
    def distribution_allowed(self) -> bool:
        """Return whether every input permits redistribution."""

        return all(item.distribution_allowed for item in self.license_constraints)

    @property
    def identity_payload(self) -> dict[str, Any]:
        """Return all manifest fields covered by the Journey manifest hash."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "component_versions": dict(self.snapshot.component_versions),
            "created_at": self.snapshot.created_at,
            "dataset_id": self.snapshot.dataset_id,
            "export_profile": self.export_profile,
            "file_hashes": dict(self.snapshot.file_hashes),
            "formats": [item.value for item in self.formats],
            "license_constraints": [
                item.to_dict() for item in self.license_constraints
            ],
            "model_versions": dict(self.model_versions),
            "policy_digest": self.policy_digest,
            "query_digest": self.snapshot.query_hash,
            "record_digests": dict(self.record_digests),
            "schema_digest": self.snapshot.schema_hash,
            "schema_version": self.schema_version,
            "selection": self.selection.to_dict(),
            "split_records": {
                key: list(value) for key, value in self.split_records.items()
            },
            "vocabulary_digest": self.vocabulary_digest,
        }

    @property
    def manifest_digest(self) -> str:
        """Return the digest represented by ``DatasetSnapshot.manifest_hash``."""

        return canonical_digest(self.identity_payload)

    def to_dict(self) -> dict[str, Any]:
        """Return the strict persisted manifest."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "export_profile": self.export_profile,
            "formats": [item.value for item in self.formats],
            "license_constraints": [
                item.to_dict() for item in self.license_constraints
            ],
            "manifest_digest": self.manifest_digest,
            "model_versions": dict(self.model_versions),
            "policy_digest": self.policy_digest,
            "record_digests": dict(self.record_digests),
            "schema_version": self.schema_version,
            "selection": self.selection.to_dict(),
            "snapshot": self.snapshot.to_dict(),
            "split_records": {
                key: list(value) for key, value in self.split_records.items()
            },
            "vocabulary_digest": self.vocabulary_digest,
        }

    def to_json(self) -> str:
        """Return canonical manifest JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GovernedDatasetManifest":
        """Parse and verify a persisted manifest."""

        data = _mapping(value, "governed dataset manifest")
        _exact_keys(
            data,
            {
                "compatibility_policy",
                "export_profile",
                "formats",
                "license_constraints",
                "manifest_digest",
                "model_versions",
                "policy_digest",
                "record_digests",
                "schema_version",
                "selection",
                "snapshot",
                "split_records",
                "vocabulary_digest",
            },
            "governed dataset manifest",
        )
        result = cls(
            snapshot=DatasetSnapshot.from_dict(_mapping(data["snapshot"], "snapshot")),
            selection=DatasetSelection.from_dict(
                _mapping(data["selection"], "selection")
            ),
            policy_digest=_text(data["policy_digest"], "policy_digest"),
            vocabulary_digest=_text(data["vocabulary_digest"], "vocabulary_digest"),
            model_versions=_mapping(data["model_versions"], "model_versions"),
            license_constraints=tuple(
                DatasetLicenseConstraint.from_dict(_mapping(item, "license constraint"))
                for item in _sequence(
                    data["license_constraints"], "license_constraints"
                )
            ),
            record_digests=_mapping(data["record_digests"], "record_digests"),
            split_records={
                _controlled(key, "split"): _text_sequence(value, "split records")
                for key, value in _mapping(
                    data["split_records"], "split_records"
                ).items()
            },
            formats=tuple(
                _enum(item, DatasetExportFormat, "dataset format")
                for item in _sequence(data["formats"], "formats")
            ),
            export_profile=_text(data["export_profile"], "export_profile"),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        if data["manifest_digest"] != result.manifest_digest:
            raise GovernedDatasetConflictError("persisted manifest digest differs")
        return result

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "GovernedDatasetManifest":
        """Parse canonical or human-formatted JSON."""

        return cls.from_dict(_json_object(value, "governed dataset manifest"))

    def _verify_snapshot(self) -> None:
        if self.snapshot.manifest_hash != self.manifest_digest:
            raise GovernedDatasetConflictError("snapshot manifest hash differs")
        if self.snapshot.record_count != len(self.record_digests):
            raise GovernedDatasetConflictError("snapshot record count differs")
        expected_splits = {
            split: canonical_digest(
                [self.record_digests[record_id] for record_id in record_ids]
            )
            for split, record_ids in self.split_records.items()
        }
        if dict(self.snapshot.split_hashes) != expected_splits:
            raise GovernedDatasetConflictError("snapshot split hashes differ")
        if self.selection.source_snapshot_id not in self.snapshot.parent_snapshot_ids:
            raise GovernedDatasetConflictError("source snapshot custody is missing")
        attributes = dict(self.snapshot.attributes)
        expected_attributes = {
            "compatibility_policy": self.compatibility_policy,
            "export_profile": self.export_profile,
            "license_constraints": [
                item.to_dict() for item in self.license_constraints
            ],
            "model_versions": dict(self.model_versions),
            "policy_digest": self.policy_digest,
            "source_digest": self.selection.source_digest,
            "source_id": self.selection.source_id,
            "source_kind": self.selection.source_kind.value,
            "source_snapshot_digest": self.selection.source_snapshot_digest,
            "vocabulary_digest": self.vocabulary_digest,
        }
        if canonical_json(attributes) != canonical_json(expected_attributes):
            raise GovernedDatasetConflictError("snapshot governed attributes differ")
        expected_license_tags = tuple(
            sorted(
                {
                    f"license.{item.redistribution.value}"
                    for item in self.license_constraints
                }
            )
        )
        if tuple(self.snapshot.license_tags) != expected_license_tags:
            raise GovernedDatasetConflictError("snapshot license tags differ")


@dataclass(frozen=True, slots=True)
class GovernedDataset:
    """Verified in-memory records, immutable manifest, and rendered artifacts."""

    manifest: GovernedDatasetManifest
    records: tuple[DatasetRecord, ...]
    files: Mapping[str, bytes] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, GovernedDatasetManifest):
            raise TypeError("manifest must be GovernedDatasetManifest")
        if any(not isinstance(item, DatasetRecord) for item in self.records):
            raise TypeError("records must contain DatasetRecord")
        records = tuple(sorted(self.records, key=lambda item: item.record_id))
        if len({item.record_id for item in records}) != len(records):
            raise GovernedDatasetConflictError("dataset record ids must be unique")
        if {item.record_id: item.row_digest for item in records} != dict(
            self.manifest.record_digests
        ):
            raise GovernedDatasetConflictError("dataset row digests differ")
        files = {str(key): bytes(value) for key, value in self.files.items()}
        if {key: sha256_digest(value) for key, value in files.items()} != dict(
            self.manifest.snapshot.file_hashes
        ):
            raise GovernedDatasetConflictError("dataset file digests differ")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "files", MappingProxyType(files))


@dataclass(frozen=True, slots=True)
class DatasetExportAuthorization:
    """Explicit identified-export approval bound to one snapshot and policy."""

    authorization_id: str
    snapshot_id: str
    policy_digest: str
    purpose: str
    approved_by: str
    export_approved: bool
    identified_export_approved: bool

    def __post_init__(self) -> None:
        _opaque_id(self.authorization_id, "authorization_id")
        _opaque_id(self.snapshot_id, "snapshot_id")
        _digest(self.policy_digest, "authorization policy_digest")
        _controlled(self.purpose, "export purpose")
        _opaque_id(self.approved_by, "approved_by")
        if type(self.export_approved) is not bool:
            raise GovernedDatasetError("export_approved must be boolean")
        if type(self.identified_export_approved) is not bool:
            raise GovernedDatasetError("identified_export_approved must be boolean")

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free approval custody."""

        return {
            "approved_by": self.approved_by,
            "authorization_id": self.authorization_id,
            "export_approved": self.export_approved,
            "identified_export_approved": self.identified_export_approved,
            "policy_digest": self.policy_digest,
            "purpose": self.purpose,
            "snapshot_id": self.snapshot_id,
        }


@dataclass(frozen=True, slots=True)
class DatasetExportRequest:
    """Value-free request passed to an export audit authorization hook."""

    snapshot_id: str
    namespace: DatasetPrivacyState
    file_hashes: Mapping[str, str]
    distribution: bool
    purpose: str
    authorization_id: str | None = None
    authorization_digest: str | None = None

    def __post_init__(self) -> None:
        _opaque_id(self.snapshot_id, "snapshot_id")
        object.__setattr__(
            self,
            "namespace",
            _enum(self.namespace, DatasetPrivacyState, "export namespace"),
        )
        hashes = _digest_mapping(self.file_hashes, "file_hashes")
        if type(self.distribution) is not bool:
            raise GovernedDatasetError("distribution must be boolean")
        _controlled(self.purpose, "export purpose")
        if self.authorization_id is not None:
            _opaque_id(self.authorization_id, "authorization_id")
        if self.authorization_digest is not None:
            _digest(self.authorization_digest, "authorization_digest")
        if (self.authorization_id is None) is not (self.authorization_digest is None):
            raise GovernedDatasetError(
                "authorization id and digest must be supplied together"
            )
        object.__setattr__(self, "file_hashes", MappingProxyType(hashes))

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free audit request."""

        return {
            "authorization_id": self.authorization_id,
            "authorization_digest": self.authorization_digest,
            "distribution": self.distribution,
            "file_hashes": dict(self.file_hashes),
            "namespace": self.namespace.value,
            "purpose": self.purpose,
            "snapshot_id": self.snapshot_id,
        }


@dataclass(frozen=True, slots=True)
class DatasetExportReceipt:
    """Digest-only receipt recorded after an export write completes."""

    export_id: str
    request: DatasetExportRequest
    output_digest: str
    created_files: tuple[str, ...]

    def __post_init__(self) -> None:
        _opaque_id(self.export_id, "export_id")
        if not isinstance(self.request, DatasetExportRequest):
            raise TypeError("request must be DatasetExportRequest")
        _digest(self.output_digest, "output_digest")
        files = tuple(sorted(_relative_filename(item) for item in self.created_files))
        object.__setattr__(self, "created_files", files)

    def to_dict(self) -> dict[str, Any]:
        """Return the PHI-free export receipt."""

        return {
            "created_files": list(self.created_files),
            "export_id": self.export_id,
            "output_digest": self.output_digest,
            "request": self.request.to_dict(),
        }


@runtime_checkable
class DatasetExportAuditHook(Protocol):
    """Authorize before export and record a digest-only completion receipt."""

    def authorize(self, request: DatasetExportRequest) -> bool:
        """Return whether this exact export request may proceed."""

    def record(self, receipt: DatasetExportReceipt) -> None:
        """Persist or forward the completed value-free receipt."""


def build_dataset_snapshot(
    spec: DatasetBuildSpec,
    records: Sequence[DatasetRecord],
) -> StoreResult[GovernedDataset]:
    """Build a deterministic Journey snapshot and safe export artifacts."""

    try:
        if not isinstance(spec, DatasetBuildSpec):
            raise TypeError("spec must be DatasetBuildSpec")
        if any(not isinstance(item, DatasetRecord) for item in records):
            raise TypeError("records must contain DatasetRecord")
        ordered = tuple(sorted(records, key=lambda item: item.record_id))
        _validate_record_selection(spec.selection, ordered)
        _validate_split_isolation(ordered)
        if spec.include_deidentified_values and any(
            item.privacy_state is DatasetPrivacyState.IDENTIFIED for item in ordered
        ):
            return StoreResult.outcome(
                StoreState.DENIED,
                "identified_export_authorization_required",
            )
        files = _render_files(
            ordered,
            spec.formats,
            include_values=spec.include_deidentified_values,
            identified=False,
        )
        file_hashes = {name: sha256_digest(content) for name, content in files.items()}
        record_digests = {item.record_id: item.row_digest for item in ordered}
        split_records = _records_by_split(ordered)
        identity = _manifest_identity(
            spec,
            record_digests=record_digests,
            split_records=split_records,
            file_hashes=file_hashes,
        )
        manifest_digest = canonical_digest(identity)
        snapshot = DatasetSnapshot(
            snapshot_id=derived_opaque_id(
                "datasetsnapshot", spec.dataset_id, manifest_digest
            ),
            dataset_id=spec.dataset_id,
            created_at=spec.created_at,
            query_hash=spec.query_digest,
            schema_hash=spec.schema_digest,
            manifest_hash=manifest_digest,
            record_count=len(ordered),
            source_artifact_ids=tuple(
                sorted(
                    {value for item in ordered for value in item.source_artifact_ids}
                )
            ),
            source_fact_ids=tuple(
                sorted({value for item in ordered for value in item.source_fact_ids})
            ),
            parent_snapshot_ids=(spec.selection.source_snapshot_id,),
            file_hashes=file_hashes,
            split_hashes={
                split: canonical_digest(
                    [record_digests[record_id] for record_id in record_ids]
                )
                for split, record_ids in split_records.items()
            },
            component_versions=dict(spec.component_versions),
            license_tags=tuple(
                sorted(
                    {f"license.{item.redistribution.value}" for item in spec.licenses}
                )
            ),
            attributes={
                "compatibility_policy": spec.compatibility_policy,
                "export_profile": (
                    "deidentified_values"
                    if spec.include_deidentified_values
                    else "metadata_only"
                ),
                "license_constraints": [item.to_dict() for item in spec.licenses],
                "model_versions": dict(spec.model_versions),
                "policy_digest": spec.policy_digest,
                "source_digest": spec.selection.source_digest,
                "source_id": spec.selection.source_id,
                "source_kind": spec.selection.source_kind.value,
                "source_snapshot_digest": spec.selection.source_snapshot_digest,
                "vocabulary_digest": spec.vocabulary_digest,
            },
        )
        manifest = GovernedDatasetManifest(
            snapshot=snapshot,
            selection=spec.selection,
            policy_digest=spec.policy_digest,
            vocabulary_digest=spec.vocabulary_digest,
            model_versions=spec.model_versions,
            license_constraints=spec.licenses,
            record_digests=record_digests,
            split_records=split_records,
            formats=spec.formats,
            export_profile=(
                "deidentified_values"
                if spec.include_deidentified_values
                else "metadata_only"
            ),
        )
        return StoreResult.success(
            GovernedDataset(manifest=manifest, records=ordered, files=files)
        )
    except GovernedDatasetUnsupportedError:
        return StoreResult.outcome(StoreState.UNSUPPORTED, "dataset_format_unsupported")
    except GovernedDatasetConflictError:
        return StoreResult.outcome(
            StoreState.CONFLICT, "dataset_split_or_digest_conflict"
        )
    except PermissionError:
        return StoreResult.outcome(
            StoreState.DENIED, "identified_export_authorization_required"
        )
    except (GovernedDatasetError, TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "dataset_build_invalid")


class LocalDatasetExporter:
    """Write verified safe or explicitly authorized identified artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        if self.root == Path(self.root.anchor):
            raise ValueError("dataset export root must be bounded")
        self._lock = threading.RLock()

    def export(
        self,
        dataset: GovernedDataset,
        *,
        distribution: bool = False,
        audit_hook: DatasetExportAuditHook | None = None,
        purpose: str = "dataset_build",
    ) -> StoreResult[DatasetExportReceipt]:
        """Write the manifest and default safe files."""

        if not isinstance(dataset, GovernedDataset):
            return StoreResult.outcome(StoreState.FAILURE, "dataset_invalid")
        return self._export(
            dataset,
            files=dict(dataset.files),
            namespace=DatasetPrivacyState.DEIDENTIFIED,
            distribution=distribution,
            purpose=purpose,
            audit_hook=audit_hook,
            authorization_id=None,
            authorization_digest=None,
        )

    def export_identified(
        self,
        dataset: GovernedDataset,
        *,
        authorization: DatasetExportAuthorization,
        audit_hook: DatasetExportAuditHook,
        distribution: bool = False,
    ) -> StoreResult[DatasetExportReceipt]:
        """Write values only under explicit, snapshot-bound approval and audit."""

        if not isinstance(dataset, GovernedDataset):
            return StoreResult.outcome(StoreState.FAILURE, "dataset_invalid")
        if not isinstance(authorization, DatasetExportAuthorization):
            return StoreResult.outcome(StoreState.DENIED, "authorization_required")
        if authorization.snapshot_id != dataset.manifest.snapshot.snapshot_id:
            return StoreResult.outcome(
                StoreState.CONFLICT, "authorization_snapshot_conflict"
            )
        if authorization.policy_digest != dataset.manifest.policy_digest:
            return StoreResult.outcome(
                StoreState.CONFLICT, "authorization_policy_conflict"
            )
        if not authorization.export_approved:
            return StoreResult.outcome(StoreState.DENIED, "export_not_approved")
        if not authorization.identified_export_approved:
            return StoreResult.outcome(
                StoreState.DENIED, "identified_export_not_approved"
            )
        if not isinstance(audit_hook, DatasetExportAuditHook):
            return StoreResult.outcome(StoreState.DENIED, "audit_hook_required")
        try:
            files = _render_files(
                dataset.records,
                dataset.manifest.formats,
                include_values=True,
                identified=True,
            )
        except (GovernedDatasetError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "identified_export_invalid")
        return self._export(
            dataset,
            files=files,
            namespace=DatasetPrivacyState.IDENTIFIED,
            distribution=distribution,
            purpose=authorization.purpose,
            audit_hook=audit_hook,
            authorization_id=authorization.authorization_id,
            authorization_digest=canonical_digest(authorization.to_dict()),
        )

    def _export(
        self,
        dataset: GovernedDataset,
        *,
        files: Mapping[str, bytes],
        namespace: DatasetPrivacyState,
        distribution: bool,
        purpose: str,
        audit_hook: DatasetExportAuditHook | None,
        authorization_id: str | None,
        authorization_digest: str | None,
    ) -> StoreResult[DatasetExportReceipt]:
        if distribution and not dataset.manifest.distribution_allowed:
            return StoreResult.outcome(StoreState.DENIED, "license_distribution_denied")
        try:
            _controlled(purpose, "export purpose")
            rendered = dict(files)
            rendered["manifest.json"] = dataset.manifest.to_json().encode("utf-8")
            hashes = {
                name: sha256_digest(content) for name, content in rendered.items()
            }
            request = DatasetExportRequest(
                snapshot_id=dataset.manifest.snapshot.snapshot_id,
                namespace=namespace,
                file_hashes=hashes,
                distribution=distribution,
                purpose=purpose,
                authorization_id=authorization_id,
                authorization_digest=authorization_digest,
            )
        except (GovernedDatasetError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "export_request_invalid")
        if audit_hook is not None:
            try:
                if not audit_hook.authorize(request):
                    return StoreResult.outcome(
                        StoreState.DENIED, "audit_authorization_denied"
                    )
            except Exception:  # noqa: BLE001 - audit hook is caller supplied.
                return StoreResult.outcome(
                    StoreState.FAILURE, "audit_authorization_failed"
                )
        state, code, created = self._write_files(rendered)
        if state is not StoreState.SUCCESS:
            return StoreResult.outcome(state, code)
        receipt = DatasetExportReceipt(
            export_id=derived_opaque_id("datasetexport", request.to_dict()),
            request=request,
            output_digest=canonical_digest(hashes),
            created_files=tuple(created),
        )
        if audit_hook is not None:
            try:
                audit_hook.record(receipt)
            except Exception:  # noqa: BLE001 - audit hook is caller supplied.
                return StoreResult.outcome(
                    StoreState.PARTIAL,
                    "export_audit_record_failed",
                    value=receipt,
                )
        return StoreResult.success(receipt, created=bool(created))

    def _write_files(
        self, files: Mapping[str, bytes]
    ) -> tuple[StoreState, str, tuple[str, ...]]:
        with self._lock:
            created: list[str] = []
            try:
                self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
                os.chmod(self.root, 0o700)
                targets = {
                    _relative_filename(name): self.root / _relative_filename(name)
                    for name in files
                }
                for name, target in targets.items():
                    if target.is_symlink():
                        return StoreState.FAILURE, "export_path_unsafe", ()
                    if target.exists() and target.read_bytes() != files[name]:
                        return StoreState.CONFLICT, "immutable_export_conflict", ()
                for name, target in targets.items():
                    if target.exists():
                        continue
                    descriptor, temporary_name = tempfile.mkstemp(
                        prefix=".dataset-export-", dir=self.root
                    )
                    try:
                        if hasattr(os, "fchmod"):
                            os.fchmod(descriptor, 0o600)
                        with os.fdopen(descriptor, "wb") as stream:
                            stream.write(files[name])
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.link(temporary_name, target)
                        created.append(name)
                    finally:
                        try:
                            os.close(descriptor)
                        except OSError:
                            pass
                        Path(temporary_name).unlink(missing_ok=True)
            except FileExistsError:
                return StoreState.CONFLICT, "immutable_export_conflict", tuple(created)
            except OSError:
                state = StoreState.PARTIAL if created else StoreState.FAILURE
                return state, "export_write_failed", tuple(created)
            return StoreState.SUCCESS, "export_complete", tuple(created)


def load_governed_dataset_schema() -> dict[str, Any]:
    """Load the bundled JSON Schema for governed manifests."""

    text = (
        resources.files(GOVERNED_DATASET_SCHEMA_PACKAGE)
        .joinpath(f"{GOVERNED_DATASET_SCHEMA_NAME}.schema.json")
        .read_text(encoding="utf-8")
    )
    value = json.loads(text)
    if not isinstance(value, dict):  # pragma: no cover - packaged invariant
        raise RuntimeError("governed dataset schema must be an object")
    return value


def _validate_record_selection(
    selection: DatasetSelection,
    records: Sequence[DatasetRecord],
) -> None:
    if len({item.record_id for item in records}) != len(records):
        raise GovernedDatasetConflictError("dataset record ids must be unique")
    if selection.source_kind is DatasetSourceKind.COHORT:
        eligible = set(selection.eligible_patient_keys)
        if any(item.patient_key not in eligible for item in records):
            raise GovernedDatasetConflictError(
                "dataset record is outside the eligible cohort"
            )


def _validate_split_isolation(records: Sequence[DatasetRecord]) -> None:
    owners: dict[tuple[str, str], str] = {}
    for record in records:
        references = {
            ("patient", record.patient_key),
            *(("artifact", item) for item in record.source_artifact_ids),
            *(("fact", item) for item in record.source_fact_ids),
            *(("event", item) for item in record.source_event_ids),
            *(("evidence", item) for item in record.evidence_ids),
        }
        for reference in references:
            previous = owners.setdefault(reference, record.split)
            if previous != record.split:
                raise GovernedDatasetConflictError("dataset split leakage detected")


def _records_by_split(records: Sequence[DatasetRecord]) -> dict[str, tuple[str, ...]]:
    values: dict[str, list[str]] = {}
    for record in records:
        values.setdefault(record.split, []).append(record.record_id)
    return {key: tuple(sorted(items)) for key, items in sorted(values.items())}


def _render_files(
    records: Sequence[DatasetRecord],
    formats: Sequence[DatasetExportFormat],
    *,
    include_values: bool,
    identified: bool,
) -> dict[str, bytes]:
    rows = [
        item.export_dict(include_values=include_values, identified=identified)
        for item in sorted(records, key=lambda value: value.record_id)
    ]
    files: dict[str, bytes] = {}
    for format_name in formats:
        if format_name is DatasetExportFormat.JSONL:
            files["records.jsonl"] = _jsonl(rows)
        elif format_name is DatasetExportFormat.ANNOTATION_JSONL:
            annotations = []
            for record in sorted(records, key=lambda value: value.record_id):
                for annotation in record.annotations:
                    annotations.append(
                        {
                            **annotation.to_dict(),
                            "patient_key": record.patient_key,
                            "record_id": record.record_id,
                            "split": record.split,
                        }
                    )
            files["annotations.jsonl"] = _jsonl(annotations)
        elif format_name is DatasetExportFormat.PARQUET:
            files["records.parquet"] = _parquet(rows)
        else:  # pragma: no cover - enum invariant
            raise GovernedDatasetUnsupportedError("dataset format is unsupported")
    return files


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return b""
    return ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")


def _parquet(rows: Sequence[Mapping[str, Any]]) -> bytes:
    try:
        pa = import_module("pyarrow")
        parquet = import_module("pyarrow.parquet")
    except ImportError:
        raise GovernedDatasetUnsupportedError(
            "Parquet export requires the columnar extra"
        ) from None
    payloads = [canonical_json(item) for item in rows]
    record_ids = [str(item["record_id"]) for item in rows]
    splits = [str(item["split"]) for item in rows]
    table = pa.table(
        {
            "payload_json": pa.array(payloads, type=pa.string()),
            "record_id": pa.array(record_ids, type=pa.string()),
            "split": pa.array(splits, type=pa.string()),
        }
    )
    sink = pa.BufferOutputStream()
    parquet.write_table(
        table,
        sink,
        compression="zstd",
        data_page_version="1.0",
        use_dictionary=False,
        write_statistics=False,
    )
    return sink.getvalue().to_pybytes()


def _manifest_identity(
    spec: DatasetBuildSpec,
    *,
    record_digests: Mapping[str, str],
    split_records: Mapping[str, tuple[str, ...]],
    file_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "compatibility_policy": spec.compatibility_policy,
        "component_versions": dict(spec.component_versions),
        "created_at": spec.created_at,
        "dataset_id": spec.dataset_id,
        "export_profile": (
            "deidentified_values"
            if spec.include_deidentified_values
            else "metadata_only"
        ),
        "file_hashes": dict(file_hashes),
        "formats": [item.value for item in spec.formats],
        "license_constraints": [item.to_dict() for item in spec.licenses],
        "model_versions": dict(spec.model_versions),
        "policy_digest": spec.policy_digest,
        "query_digest": spec.query_digest,
        "record_digests": dict(record_digests),
        "schema_digest": spec.schema_digest,
        "schema_version": spec.schema_version,
        "selection": spec.selection.to_dict(),
        "split_records": {key: list(value) for key, value in split_records.items()},
        "vocabulary_digest": spec.vocabulary_digest,
    }


def _assert_no_vault_material(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if (
                normalized in _VAULT_KEYS
                or "vault" in normalized
                or normalized.endswith("_token")
                or normalized.startswith("token_")
            ):
                raise GovernedDatasetError("record values contain protected material")
            _assert_no_vault_material(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _assert_no_vault_material(nested)


def _assert_deidentified_values(value: Mapping[str, Any]) -> None:
    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                normalized = str(key).strip().casefold().replace("-", "_")
                if normalized in _DIRECT_IDENTIFIER_KEYS:
                    raise GovernedDatasetError(
                        "deidentified projection contains a direct identifier field"
                    )
                visit(nested)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for nested in item:
                visit(nested)

    visit(value)


def _contract_version(schema_version: str, compatibility_policy: str) -> None:
    if schema_version != GOVERNED_DATASET_SCHEMA_VERSION:
        raise GovernedDatasetUnsupportedError(
            "governed dataset schema version is unsupported"
        )
    if compatibility_policy != GOVERNED_DATASET_COMPATIBILITY_POLICY:
        raise GovernedDatasetUnsupportedError(
            "governed dataset compatibility policy is unsupported"
        )


def _version_mapping(value: Mapping[str, str], name: str) -> dict[str, str]:
    data = _mapping(value, name)
    return {
        _controlled(key, f"{name} key"): _bounded_text(item, name, 256)
        for key, item in sorted(data.items())
    }


def _digest_mapping(
    value: Mapping[str, Any],
    name: str,
    *,
    opaque_keys: bool = False,
) -> dict[str, str]:
    data = _mapping(value, name)
    return {
        (
            _opaque_id(key, f"{name} key") if opaque_keys else _relative_filename(key)
        ): _digest(item, name)
        for key, item in sorted(data.items())
    }


def _split_mapping(value: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    data = _mapping(value, "split_records")
    return {
        _controlled(key, "split"): tuple(
            sorted(_opaque_id(item, "record_id") for item in values)
        )
        for key, values in sorted(data.items())
    }


def _plain_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GovernedDatasetError(f"{name} must be an object")
    try:
        parsed = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError):
        raise GovernedDatasetError(f"{name} must contain JSON values") from None
    if not isinstance(parsed, dict):  # pragma: no cover - dict input invariant
        raise GovernedDatasetError(f"{name} must be an object")
    return parsed


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {str(key): _freeze_json_value(item) for key, item in value.items()}
    )


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json_value(item) for item in value]
    return value


def _json_object(value: str | bytes | bytearray, name: str) -> Mapping[str, Any]:
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        raise GovernedDatasetError(f"{name} is not valid JSON") from None
    return _mapping(data, name)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GovernedDatasetError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise GovernedDatasetError(f"{name} must be an array")
    return value


def _exact_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        raise GovernedDatasetError(f"{name} fields are incompatible")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GovernedDatasetError(f"{name} must be non-empty text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _text_sequence(value: Any, name: str) -> tuple[str, ...]:
    return tuple(_text(item, name) for item in _sequence(value, name))


def _bounded_text(value: Any, name: str, maximum: int) -> str:
    text = _text(value, name)
    if len(text) > maximum or any(ord(character) < 32 for character in text):
        raise GovernedDatasetError(f"{name} must be bounded text")
    return text


def _controlled(value: Any, name: str) -> str:
    text = _text(value, name)
    if _CONTROLLED_RE.fullmatch(text) is None:
        raise GovernedDatasetError(f"{name} must be controlled")
    return text


def _label(value: Any, name: str) -> str:
    text = _text(value, name)
    if _LABEL_RE.fullmatch(text) is None:
        raise GovernedDatasetError(f"{name} must be controlled")
    return text


def _opaque_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if _OPAQUE_ID_RE.fullmatch(text) is None:
        raise GovernedDatasetError(f"{name} must be opaque")
    return text


def _digest(value: Any, name: str) -> str:
    text = _text(value, name)
    if _DIGEST_RE.fullmatch(text) is None:
        raise GovernedDatasetError(f"{name} must be a digest")
    return text


def _timestamp(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TIMESTAMP_RE.fullmatch(text) is None:
        raise GovernedDatasetError(f"{name} must be an RFC 3339 timestamp")
    return text


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise GovernedDatasetError(f"{name} must be an integer")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError):
        raise GovernedDatasetUnsupportedError(f"{name} is unsupported") from None


def _relative_filename(value: Any) -> str:
    text = _text(value, "filename")
    path = Path(text)
    if path.is_absolute() or len(path.parts) != 1 or text in {".", ".."}:
        raise GovernedDatasetError("filename must be a relative basename")
    if any(character in text for character in ("/", "\\", "\x00")):
        raise GovernedDatasetError("filename must be a relative basename")
    return text


__all__ = [
    "GOVERNED_DATASET_COMPATIBILITY_POLICY",
    "GOVERNED_DATASET_SCHEMA_NAME",
    "GOVERNED_DATASET_SCHEMA_VERSION",
    "DatasetAnnotation",
    "DatasetBuildSpec",
    "DatasetExportAuditHook",
    "DatasetExportAuthorization",
    "DatasetExportFormat",
    "DatasetExportReceipt",
    "DatasetExportRequest",
    "DatasetLicenseConstraint",
    "DatasetPrivacyState",
    "DatasetRecord",
    "DatasetSelection",
    "DatasetSourceKind",
    "GovernedDataset",
    "GovernedDatasetConflictError",
    "GovernedDatasetError",
    "GovernedDatasetManifest",
    "GovernedDatasetUnsupportedError",
    "LocalDatasetExporter",
    "RedistributionPolicy",
    "build_dataset_snapshot",
    "load_governed_dataset_schema",
]
