"""Digest-bound OMOP quality and synthetic-cohort reconciliation.

The bridge in this module deliberately does not implement an OMOP data-quality
engine.  It accepts aggregate results from an explicitly selected local
subprocess or caller-provided remote runner, validates their custody, and emits
a signed report without patient-level values.  Core installations therefore
remain offline and do not acquire an external runtime or vocabulary.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import shutil
import subprocess
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib import resources
from io import BufferedReader
from types import MappingProxyType
from typing import Any, Final, cast

from openmed.clinical.journey_contracts import (
    canonical_digest,
    canonical_json,
    derived_opaque_id,
)
from openmed.structured.store import StoreResult, StoreState

from .fact_projection import (
    OMOP_DOMAIN_TABLES,
    OMOP_FACT_PROJECTION_CDM_VERSION,
    OMOP_FACT_TABLES,
    OMOP_MAPPING_STATES,
    OmopFactProjection,
    OmopFactRoundTripReport,
    validate_omop_fact_projection,
)

OMOP_QUALITY_REPORT_ARTIFACT: Final = "openmed.omop_quality_report"
OMOP_QUALITY_REQUEST_ARTIFACT: Final = "openmed.omop_quality_request"
OMOP_QUALITY_TOOL_OUTPUT_ARTIFACT: Final = "openmed.omop_quality_tool_output"
OMOP_QUALITY_REPORT_SCHEMA_VERSION: Final = "1.0.0"
OMOP_QUALITY_COMPATIBILITY_POLICY: Final = "same_major"
OMOP_QUALITY_REPORT_SCHEMA_NAME: Final = "omop_quality_report"
OMOP_QUALITY_REPORT_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"
OMOP_QUALITY_SIGNATURE_ALGORITHM: Final = "HMAC-SHA256"

OMOP_QUALITY_CATEGORIES: Final = (
    "conformance",
    "completeness",
    "plausibility",
)
OMOP_QUALITY_CHECK_STATUSES: Final = frozenset({"pass", "fail", "unknown"})
OMOP_QUALITY_SEVERITIES: Final = frozenset({"info", "warning", "error"})
OMOP_QUALITY_EXECUTION_MODES: Final = frozenset({"local", "remote"})
OMOP_QUALITY_VERDICTS: Final = frozenset({"pass", "fail", "unknown"})
OMOP_QUALITY_PERMISSIVE_LICENSES: Final = frozenset(
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

_CONTROLLED_RE: Final = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_RE: Final = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
_DATE_RE: Final = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_MAX_TOOL_OUTPUT_BYTES: Final = 1_000_000
_MIN_SIGNING_KEY_BYTES: Final = 16
_DOMAIN_DATE_COLUMNS: Final = MappingProxyType(
    {
        "condition_occurrence": "condition_start_date",
        "drug_exposure": "drug_exposure_start_date",
        "procedure_occurrence": "procedure_date",
        "measurement": "measurement_date",
        "observation": "observation_date",
    }
)


class OmopQualityError(ValueError):
    """Base error for an invalid OMOP quality contract."""


class OmopQualityConflictError(OmopQualityError):
    """Raised when a digest, split, or expectation conflicts."""


class OmopQualityDeniedError(OmopQualityError):
    """Raised when license or execution policy denies a bridge request."""


class OmopQualityUnsupportedError(OmopQualityError):
    """Raised for an unsupported schema, CDM version, or execution mode."""


class OmopQualityProtocolError(OmopQualityError):
    """Raised when an adapter violates the bounded aggregate protocol."""


@dataclass(frozen=True, slots=True)
class OmopQualitySignature:
    """Caller-owned HMAC signature metadata."""

    key_id: str
    algorithm: str
    value: str

    def __post_init__(self) -> None:
        _controlled(self.key_id, "signature key_id")
        if self.algorithm != OMOP_QUALITY_SIGNATURE_ALGORITHM:
            raise OmopQualityUnsupportedError(
                "quality signature algorithm is unsupported"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.value):
            raise OmopQualityError("quality signature value must be a SHA-256 HMAC")

    def to_dict(self) -> dict[str, str]:
        """Return stable signature metadata without key material."""

        return {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopQualitySignature":
        """Restore signature metadata."""

        data = _mapping(value, "signature")
        _exact_keys(data, {"algorithm", "key_id", "value"}, "signature")
        return cls(
            key_id=_text(data["key_id"], "signature key_id"),
            algorithm=_text(data["algorithm"], "signature algorithm"),
            value=_text(data["value"], "signature value"),
        )


@dataclass(frozen=True, slots=True)
class OmopQualityInput:
    """Value-free custody for one quality and reconciliation run."""

    cohort_id: str
    cohort_digest: str
    cohort_split: str
    projection_digest: str
    vocabulary_digest: str
    reference_snapshot_digest: str
    reference_etl: str
    reference_version: str
    reference_license: str
    cdm_version: str = OMOP_FACT_PROJECTION_CDM_VERSION
    schema_version: str = OMOP_QUALITY_REPORT_SCHEMA_VERSION
    compatibility_policy: str = OMOP_QUALITY_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _controlled(self.cohort_id, "cohort_id")
        _digest(self.cohort_digest, "cohort_digest")
        _controlled(self.cohort_split, "cohort_split")
        _digest(self.projection_digest, "projection_digest")
        _digest(self.vocabulary_digest, "vocabulary_digest")
        _digest(self.reference_snapshot_digest, "reference_snapshot_digest")
        _controlled(self.reference_etl, "reference_etl")
        _version(self.reference_version, "reference_version")
        _controlled(self.reference_license, "reference_license")
        _contract(self.schema_version, self.compatibility_policy, self.cdm_version)

    @property
    def digest(self) -> str:
        """Return the canonical digest presented to an adapter."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the complete value-free input manifest."""

        return {
            "cdm_version": self.cdm_version,
            "cohort_digest": self.cohort_digest,
            "cohort_id": self.cohort_id,
            "cohort_split": self.cohort_split,
            "compatibility_policy": self.compatibility_policy,
            "projection_digest": self.projection_digest,
            "reference_etl": self.reference_etl,
            "reference_license": self.reference_license,
            "reference_snapshot_digest": self.reference_snapshot_digest,
            "reference_version": self.reference_version,
            "schema_version": self.schema_version,
            "vocabulary_digest": self.vocabulary_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopQualityInput":
        """Restore a quality input manifest."""

        data = _mapping(value, "quality input")
        fields = {
            "cdm_version",
            "cohort_digest",
            "cohort_id",
            "cohort_split",
            "compatibility_policy",
            "projection_digest",
            "reference_etl",
            "reference_license",
            "reference_snapshot_digest",
            "reference_version",
            "schema_version",
            "vocabulary_digest",
        }
        _exact_keys(data, fields, "quality input")
        return cls(**{name: _text(data[name], name) for name in fields})


@dataclass(frozen=True, slots=True)
class OmopQualityCheck:
    """One aggregate, actionable, patient-value-free quality result."""

    check_id: str
    category: str
    status: str
    severity: str
    code: str
    remediation_code: str
    affected_rows: int
    table: str | None = None

    def __post_init__(self) -> None:
        _controlled(self.check_id, "check_id")
        if self.category not in OMOP_QUALITY_CATEGORIES:
            raise OmopQualityUnsupportedError("quality check category is unsupported")
        if self.status not in OMOP_QUALITY_CHECK_STATUSES:
            raise OmopQualityUnsupportedError("quality check status is unsupported")
        if self.severity not in OMOP_QUALITY_SEVERITIES:
            raise OmopQualityUnsupportedError("quality check severity is unsupported")
        _controlled(self.code, "quality check code")
        _controlled(self.remediation_code, "quality check remediation_code")
        _integer(self.affected_rows, "affected_rows")
        if self.table is not None and self.table not in OMOP_FACT_TABLES:
            raise OmopQualityUnsupportedError("quality check table is unsupported")
        if self.status == "pass" and self.affected_rows:
            raise OmopQualityConflictError(
                "passing quality checks cannot report affected rows"
            )

    @property
    def requires_review(self) -> bool:
        """Return whether this result needs review or remediation."""

        return self.status != "pass"

    def to_dict(self) -> dict[str, Any]:
        """Return the bounded aggregate check."""

        return {
            "affected_rows": self.affected_rows,
            "category": self.category,
            "check_id": self.check_id,
            "code": self.code,
            "remediation_code": self.remediation_code,
            "requires_review": self.requires_review,
            "severity": self.severity,
            "status": self.status,
            "table": self.table,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopQualityCheck":
        """Restore an aggregate quality check."""

        data = _mapping(value, "quality check")
        required = {
            "affected_rows",
            "category",
            "check_id",
            "code",
            "remediation_code",
            "severity",
            "status",
            "table",
        }
        allowed = required | {"requires_review"}
        if set(data) != required and set(data) != allowed:
            _exact_keys(data, required, "quality check")
        result = cls(
            check_id=_text(data["check_id"], "check_id"),
            category=_text(data["category"], "category"),
            status=_text(data["status"], "status"),
            severity=_text(data["severity"], "severity"),
            code=_text(data["code"], "code"),
            remediation_code=_text(data["remediation_code"], "remediation_code"),
            affected_rows=_integer(data["affected_rows"], "affected_rows"),
            table=_optional_text(data["table"], "table"),
        )
        if "requires_review" in data and data["requires_review"] is not (
            result.requires_review
        ):
            raise OmopQualityConflictError("quality check review flag differs")
        return result


@dataclass(frozen=True, slots=True)
class OmopQualityCategorySummary:
    """Deterministic aggregate for one standard quality category."""

    category: str
    verdict: str
    check_count: int
    failed_count: int
    unknown_count: int

    def __post_init__(self) -> None:
        if self.category not in OMOP_QUALITY_CATEGORIES:
            raise OmopQualityUnsupportedError("quality category is unsupported")
        if self.verdict not in OMOP_QUALITY_VERDICTS:
            raise OmopQualityUnsupportedError("quality category verdict is unsupported")
        for name in ("check_count", "failed_count", "unknown_count"):
            _integer(getattr(self, name), name)
        if self.failed_count + self.unknown_count > self.check_count:
            raise OmopQualityConflictError("quality category counts conflict")

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate category counts."""

        return {
            "category": self.category,
            "check_count": self.check_count,
            "failed_count": self.failed_count,
            "unknown_count": self.unknown_count,
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopQualityCategorySummary":
        """Restore an aggregate category summary."""

        data = _mapping(value, "quality category")
        _exact_keys(
            data,
            {"category", "check_count", "failed_count", "unknown_count", "verdict"},
            "quality category",
        )
        return cls(
            category=_text(data["category"], "category"),
            verdict=_text(data["verdict"], "verdict"),
            check_count=_integer(data["check_count"], "check_count"),
            failed_count=_integer(data["failed_count"], "failed_count"),
            unknown_count=_integer(data["unknown_count"], "unknown_count"),
        )


@dataclass(frozen=True, slots=True)
class OmopProjectionAggregate:
    """A count-only projection snapshot suitable for reconciliation."""

    source: str
    snapshot_digest: str
    cohort_digest: str
    cohort_split: str
    row_counts: Mapping[str, int]
    mapping_counts: Mapping[str, int]
    current_fact_count: int
    license: str
    version: str
    cdm_version: str = OMOP_FACT_PROJECTION_CDM_VERSION

    def __post_init__(self) -> None:
        _controlled(self.source, "aggregate source")
        _digest(self.snapshot_digest, "aggregate snapshot_digest")
        _digest(self.cohort_digest, "aggregate cohort_digest")
        _controlled(self.cohort_split, "aggregate cohort_split")
        rows = _count_mapping(self.row_counts, "aggregate row_counts")
        mappings = _count_mapping(self.mapping_counts, "aggregate mapping_counts")
        if set(rows) != set(OMOP_FACT_TABLES):
            raise OmopQualityError("aggregate row counts are incomplete")
        if set(mappings) != set(OMOP_MAPPING_STATES):
            raise OmopQualityError("aggregate mapping counts are incomplete")
        _integer(self.current_fact_count, "aggregate current_fact_count")
        if sum(mappings.values()) != self.current_fact_count:
            raise OmopQualityConflictError("aggregate mapping counts conflict")
        _controlled(self.license, "aggregate license")
        _version(self.version, "aggregate version")
        if self.cdm_version != OMOP_FACT_PROJECTION_CDM_VERSION:
            raise OmopQualityUnsupportedError("aggregate CDM version is unsupported")
        object.__setattr__(self, "row_counts", rows)
        object.__setattr__(self, "mapping_counts", mappings)

    @property
    def digest(self) -> str:
        """Return the canonical aggregate snapshot digest."""

        return canonical_digest(self.to_dict())

    @property
    def mapped_coverage_ppm(self) -> int:
        """Return mapped-fact coverage in integer parts per million."""

        if self.current_fact_count == 0:
            return 0
        return round(
            int(self.mapping_counts["mapped"]) * 1_000_000 / self.current_fact_count
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free aggregate snapshot."""

        return {
            "cdm_version": self.cdm_version,
            "cohort_digest": self.cohort_digest,
            "cohort_split": self.cohort_split,
            "current_fact_count": self.current_fact_count,
            "license": self.license,
            "mapped_coverage_ppm": self.mapped_coverage_ppm,
            "mapping_counts": dict(self.mapping_counts),
            "row_counts": dict(self.row_counts),
            "snapshot_digest": self.snapshot_digest,
            "source": self.source,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopProjectionAggregate":
        """Restore a count-only projection snapshot."""

        data = _mapping(value, "projection aggregate")
        required = {
            "cdm_version",
            "cohort_digest",
            "cohort_split",
            "current_fact_count",
            "license",
            "mapping_counts",
            "row_counts",
            "snapshot_digest",
            "source",
            "version",
        }
        allowed = required | {"mapped_coverage_ppm"}
        if set(data) != required and set(data) != allowed:
            _exact_keys(data, required, "projection aggregate")
        result = cls(
            source=_text(data["source"], "source"),
            snapshot_digest=_text(data["snapshot_digest"], "snapshot_digest"),
            cohort_digest=_text(data["cohort_digest"], "cohort_digest"),
            cohort_split=_text(data["cohort_split"], "cohort_split"),
            row_counts=_mapping(data["row_counts"], "row_counts"),
            mapping_counts=_mapping(data["mapping_counts"], "mapping_counts"),
            current_fact_count=_integer(
                data["current_fact_count"], "current_fact_count"
            ),
            license=_text(data["license"], "license"),
            version=_text(data["version"], "version"),
            cdm_version=_text(data["cdm_version"], "cdm_version"),
        )
        if (
            "mapped_coverage_ppm" in data
            and _integer(data["mapped_coverage_ppm"], "mapped_coverage_ppm")
            != result.mapped_coverage_ppm
        ):
            raise OmopQualityConflictError("aggregate mapping coverage differs")
        return result

    @classmethod
    def from_projection(
        cls,
        projection: OmopFactProjection,
        *,
        cohort_digest: str,
        source: str = "openmed",
        license: str = "apache-2.0",
        version: str = "3.0.0",
    ) -> "OmopProjectionAggregate":
        """Build aggregate custody from a validated OpenMed projection."""

        if not isinstance(projection, OmopFactProjection):
            raise TypeError("projection must be OmopFactProjection")
        splits = {item.dataset_split for item in projection.provenance}
        if None in splits or len(splits) != 1:
            raise OmopQualityConflictError(
                "projection must have one explicit dataset split"
            )
        return cls(
            source=source,
            snapshot_digest=projection.digest,
            cohort_digest=cohort_digest,
            cohort_split=next(iter(splits)),  # type: ignore[arg-type]
            row_counts=projection.summary.row_counts,
            mapping_counts=projection.summary.mapping_counts,
            current_fact_count=projection.summary.current_fact_count,
            license=license,
            version=version,
            cdm_version=projection.cdm_version,
        )


@dataclass(frozen=True, slots=True)
class OmopRowCountDelta:
    """One table-level row-count delta against a reference projection."""

    table: str
    openmed_count: int
    reference_count: int
    delta: int
    expected_delta: int | None
    matches_expected: bool

    def __post_init__(self) -> None:
        if self.table not in OMOP_FACT_TABLES:
            raise OmopQualityUnsupportedError("reconciliation table is unsupported")
        _integer(self.openmed_count, "openmed_count")
        _integer(self.reference_count, "reference_count")
        if type(self.delta) is not int:
            raise OmopQualityError("row delta must be an integer")
        if self.delta != self.openmed_count - self.reference_count:
            raise OmopQualityConflictError("row delta conflicts with counts")
        if self.expected_delta is not None and type(self.expected_delta) is not int:
            raise OmopQualityError("expected row delta must be an integer or null")
        expected_match = (
            True if self.expected_delta is None else self.delta == self.expected_delta
        )
        if self.matches_expected is not expected_match:
            raise OmopQualityConflictError("row delta expectation flag differs")

    def to_dict(self) -> dict[str, Any]:
        """Return one aggregate row delta."""

        return {
            "delta": self.delta,
            "expected_delta": self.expected_delta,
            "matches_expected": self.matches_expected,
            "openmed_count": self.openmed_count,
            "reference_count": self.reference_count,
            "table": self.table,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopRowCountDelta":
        """Restore one row-count delta."""

        data = _mapping(value, "row-count delta")
        _exact_keys(
            data,
            {
                "delta",
                "expected_delta",
                "matches_expected",
                "openmed_count",
                "reference_count",
                "table",
            },
            "row-count delta",
        )
        return cls(
            table=_text(data["table"], "table"),
            openmed_count=_integer(data["openmed_count"], "openmed_count"),
            reference_count=_integer(data["reference_count"], "reference_count"),
            delta=_signed_integer(data["delta"], "delta"),
            expected_delta=_optional_signed_integer(
                data["expected_delta"], "expected_delta"
            ),
            matches_expected=_boolean(data["matches_expected"], "matches_expected"),
        )


@dataclass(frozen=True, slots=True)
class OmopReconciliation:
    """Expected and observed aggregate differences for a frozen cohort."""

    cohort_digest: str
    cohort_split: str
    openmed_snapshot_digest: str
    reference_snapshot_digest: str
    openmed_aggregate_digest: str
    reference_aggregate_digest: str
    row_deltas: tuple[OmopRowCountDelta, ...]
    mapped_coverage_delta_ppm: int
    expected_mapped_coverage_delta_ppm: int | None
    semantic_difference_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _digest(self.cohort_digest, "reconciliation cohort_digest")
        _controlled(self.cohort_split, "reconciliation cohort_split")
        _digest(self.openmed_snapshot_digest, "openmed_snapshot_digest")
        _digest(self.reference_snapshot_digest, "reference_snapshot_digest")
        _digest(self.openmed_aggregate_digest, "openmed_aggregate_digest")
        _digest(self.reference_aggregate_digest, "reference_aggregate_digest")
        deltas = tuple(sorted(self.row_deltas, key=lambda item: item.table))
        if any(not isinstance(item, OmopRowCountDelta) for item in deltas):
            raise TypeError("row_deltas must contain OmopRowCountDelta values")
        if {item.table for item in deltas} != set(OMOP_FACT_TABLES):
            raise OmopQualityError("reconciliation row deltas are incomplete")
        _signed_integer(self.mapped_coverage_delta_ppm, "mapped_coverage_delta_ppm")
        if self.expected_mapped_coverage_delta_ppm is not None:
            _signed_integer(
                self.expected_mapped_coverage_delta_ppm,
                "expected_mapped_coverage_delta_ppm",
            )
        differences = tuple(
            sorted(
                {
                    _controlled(item, "semantic difference code")
                    for item in self.semantic_difference_codes
                }
            )
        )
        object.__setattr__(self, "row_deltas", deltas)
        object.__setattr__(self, "semantic_difference_codes", differences)

    @property
    def expectations_met(self) -> bool:
        """Return whether all explicitly pinned deltas match."""

        coverage_matches = (
            self.expected_mapped_coverage_delta_ppm is None
            or self.mapped_coverage_delta_ppm == self.expected_mapped_coverage_delta_ppm
        )
        return coverage_matches and all(
            item.matches_expected for item in self.row_deltas
        )

    @property
    def digest(self) -> str:
        """Return the canonical reconciliation digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, aggregate-only reconciliation evidence."""

        return {
            "cohort_digest": self.cohort_digest,
            "cohort_split": self.cohort_split,
            "expectations_met": self.expectations_met,
            "expected_mapped_coverage_delta_ppm": (
                self.expected_mapped_coverage_delta_ppm
            ),
            "mapped_coverage_delta_ppm": self.mapped_coverage_delta_ppm,
            "openmed_aggregate_digest": self.openmed_aggregate_digest,
            "openmed_snapshot_digest": self.openmed_snapshot_digest,
            "reference_aggregate_digest": self.reference_aggregate_digest,
            "reference_snapshot_digest": self.reference_snapshot_digest,
            "row_deltas": [item.to_dict() for item in self.row_deltas],
            "semantic_difference_codes": list(self.semantic_difference_codes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopReconciliation":
        """Restore reconciliation evidence and verify derived fields."""

        data = _mapping(value, "reconciliation")
        required = {
            "cohort_digest",
            "cohort_split",
            "expected_mapped_coverage_delta_ppm",
            "mapped_coverage_delta_ppm",
            "openmed_aggregate_digest",
            "openmed_snapshot_digest",
            "reference_aggregate_digest",
            "reference_snapshot_digest",
            "row_deltas",
            "semantic_difference_codes",
        }
        allowed = required | {"expectations_met"}
        if set(data) != required and set(data) != allowed:
            _exact_keys(data, required, "reconciliation")
        result = cls(
            cohort_digest=_text(data["cohort_digest"], "cohort_digest"),
            cohort_split=_text(data["cohort_split"], "cohort_split"),
            openmed_snapshot_digest=_text(
                data["openmed_snapshot_digest"], "openmed_snapshot_digest"
            ),
            reference_snapshot_digest=_text(
                data["reference_snapshot_digest"], "reference_snapshot_digest"
            ),
            openmed_aggregate_digest=_text(
                data["openmed_aggregate_digest"], "openmed_aggregate_digest"
            ),
            reference_aggregate_digest=_text(
                data["reference_aggregate_digest"], "reference_aggregate_digest"
            ),
            row_deltas=tuple(
                OmopRowCountDelta.from_dict(_mapping(item, "row-count delta"))
                for item in _sequence(data["row_deltas"], "row_deltas")
            ),
            mapped_coverage_delta_ppm=_signed_integer(
                data["mapped_coverage_delta_ppm"], "mapped_coverage_delta_ppm"
            ),
            expected_mapped_coverage_delta_ppm=_optional_signed_integer(
                data["expected_mapped_coverage_delta_ppm"],
                "expected_mapped_coverage_delta_ppm",
            ),
            semantic_difference_codes=_text_sequence(
                data["semantic_difference_codes"], "semantic_difference_codes"
            ),
        )
        if "expectations_met" in data and data["expectations_met"] is not (
            result.expectations_met
        ):
            raise OmopQualityConflictError("reconciliation expectation flag differs")
        return result


@dataclass(frozen=True, slots=True)
class OmopQualityReport:
    """Signed normalized quality evidence with no patient-level values."""

    report_id: str
    quality_input: OmopQualityInput
    input_digest: str
    source_output_digest: str
    tool_name: str
    tool_version: str
    execution_mode: str
    checks: tuple[OmopQualityCheck, ...]
    categories: tuple[OmopQualityCategorySummary, ...]
    reconciliation: OmopReconciliation
    quality_verdict: str
    requires_review: bool
    report_digest: str = ""
    signature: OmopQualitySignature | None = field(default=None, repr=False)
    artifact_type: str = OMOP_QUALITY_REPORT_ARTIFACT
    cdm_version: str = OMOP_FACT_PROJECTION_CDM_VERSION
    schema_version: str = OMOP_QUALITY_REPORT_SCHEMA_VERSION
    compatibility_policy: str = OMOP_QUALITY_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if self.artifact_type != OMOP_QUALITY_REPORT_ARTIFACT:
            raise OmopQualityUnsupportedError("quality report artifact is unsupported")
        _contract(self.schema_version, self.compatibility_policy, self.cdm_version)
        _controlled(self.report_id, "report_id")
        if not isinstance(self.quality_input, OmopQualityInput):
            raise TypeError("quality_input must be OmopQualityInput")
        _digest(self.input_digest, "input_digest")
        _digest(self.source_output_digest, "source_output_digest")
        if self.input_digest != self.quality_input.digest:
            raise OmopQualityConflictError("quality report input digest differs")
        _controlled(self.tool_name, "tool_name")
        _version(self.tool_version, "tool_version")
        if self.execution_mode not in OMOP_QUALITY_EXECUTION_MODES:
            raise OmopQualityUnsupportedError("quality execution mode is unsupported")
        checks = tuple(
            sorted(
                self.checks,
                key=lambda item: (item.category, item.check_id, item.table or ""),
            )
        )
        if any(not isinstance(item, OmopQualityCheck) for item in checks):
            raise TypeError("checks must contain OmopQualityCheck values")
        if len({item.check_id for item in checks}) != len(checks):
            raise OmopQualityConflictError("quality check identifiers conflict")
        categories = tuple(sorted(self.categories, key=lambda item: item.category))
        if any(not isinstance(item, OmopQualityCategorySummary) for item in categories):
            raise TypeError("categories contain invalid values")
        if categories != _summarize_categories(checks):
            raise OmopQualityConflictError("quality category summaries differ")
        if not isinstance(self.reconciliation, OmopReconciliation):
            raise TypeError("reconciliation must be OmopReconciliation")
        if self.reconciliation.cohort_digest != self.quality_input.cohort_digest:
            raise OmopQualityConflictError("report cohort digest differs")
        if self.reconciliation.cohort_split != self.quality_input.cohort_split:
            raise OmopQualityConflictError("report cohort split differs")
        if (
            self.reconciliation.openmed_snapshot_digest
            != self.quality_input.projection_digest
        ):
            raise OmopQualityConflictError("report projection digest differs")
        if (
            self.reconciliation.reference_snapshot_digest
            != self.quality_input.reference_snapshot_digest
        ):
            raise OmopQualityConflictError("report reference snapshot digest differs")
        expected_verdict = _quality_verdict(categories, self.reconciliation)
        if self.quality_verdict != expected_verdict:
            raise OmopQualityConflictError("quality report verdict differs")
        expected_review = expected_verdict != "pass"
        if self.requires_review is not expected_review:
            raise OmopQualityConflictError("quality report review flag differs")
        expected_id = derived_opaque_id(
            "omopquality",
            self.input_digest,
            self.source_output_digest,
            self.reconciliation.digest,
        )
        if self.report_id != expected_id:
            raise OmopQualityConflictError("quality report identifier differs")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "categories", categories)
        expected_digest = canonical_digest(self._payload(include_digest=False))
        if self.report_digest:
            _digest(self.report_digest, "report_digest")
            if self.report_digest != expected_digest:
                raise OmopQualityConflictError("quality report digest differs")
        else:
            object.__setattr__(self, "report_digest", expected_digest)
        if self.signature is not None and not isinstance(
            self.signature, OmopQualitySignature
        ):
            raise TypeError("signature must be OmopQualitySignature or None")

    def _payload(
        self,
        *,
        include_digest: bool,
        include_signature: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_type": self.artifact_type,
            "categories": [item.to_dict() for item in self.categories],
            "cdm_version": self.cdm_version,
            "checks": [item.to_dict() for item in self.checks],
            "compatibility_policy": self.compatibility_policy,
            "execution_mode": self.execution_mode,
            "input": self.quality_input.to_dict(),
            "input_digest": self.input_digest,
            "quality_verdict": self.quality_verdict,
            "reconciliation": self.reconciliation.to_dict(),
            "report_id": self.report_id,
            "requires_review": self.requires_review,
            "schema_version": self.schema_version,
            "source_output_digest": self.source_output_digest,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
        }
        if include_digest:
            payload["report_digest"] = self.report_digest
        if include_signature:
            payload["signature"] = (
                self.signature.to_dict() if self.signature is not None else None
            )
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Return the persisted report, including digest and signature."""

        return self._payload(include_digest=True, include_signature=True)

    def to_json(self) -> str:
        """Return deterministic compact JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OmopQualityReport":
        """Restore a report and verify all derived, non-secret fields."""

        data = _mapping(value, "quality report")
        _exact_keys(
            data,
            {
                "artifact_type",
                "categories",
                "cdm_version",
                "checks",
                "compatibility_policy",
                "execution_mode",
                "input",
                "input_digest",
                "quality_verdict",
                "reconciliation",
                "report_digest",
                "report_id",
                "requires_review",
                "schema_version",
                "signature",
                "source_output_digest",
                "tool_name",
                "tool_version",
            },
            "quality report",
        )
        signature = data["signature"]
        if signature is None:
            raise OmopQualityError("persisted quality report requires a signature")
        return cls(
            report_id=_text(data["report_id"], "report_id"),
            quality_input=OmopQualityInput.from_dict(
                _mapping(data["input"], "quality input")
            ),
            input_digest=_text(data["input_digest"], "input_digest"),
            source_output_digest=_text(
                data["source_output_digest"], "source_output_digest"
            ),
            tool_name=_text(data["tool_name"], "tool_name"),
            tool_version=_text(data["tool_version"], "tool_version"),
            execution_mode=_text(data["execution_mode"], "execution_mode"),
            checks=tuple(
                OmopQualityCheck.from_dict(_mapping(item, "quality check"))
                for item in _sequence(data["checks"], "checks")
            ),
            categories=tuple(
                OmopQualityCategorySummary.from_dict(_mapping(item, "quality category"))
                for item in _sequence(data["categories"], "categories")
            ),
            reconciliation=OmopReconciliation.from_dict(
                _mapping(data["reconciliation"], "reconciliation")
            ),
            quality_verdict=_text(data["quality_verdict"], "quality_verdict"),
            requires_review=_boolean(data["requires_review"], "requires_review"),
            report_digest=_text(data["report_digest"], "report_digest"),
            signature=OmopQualitySignature.from_dict(_mapping(signature, "signature")),
            artifact_type=_text(data["artifact_type"], "artifact_type"),
            cdm_version=_text(data["cdm_version"], "cdm_version"),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> "OmopQualityReport":
        """Restore a report from deterministic JSON."""

        if not isinstance(value, str):
            raise OmopQualityError("quality report JSON must be text")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise OmopQualityError("quality report JSON is invalid") from exc
        return cls.from_dict(_mapping(decoded, "quality report"))


OmopQualityRemoteRunner = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def projection_quality_checks(
    projection: OmopFactProjection,
    *,
    round_trip: OmopFactRoundTripReport | None = None,
) -> tuple[OmopQualityCheck, ...]:
    """Create safe built-in checks for a fact projection.

    These checks complement, but do not impersonate, a standard external OMOP
    quality engine.  They cover OpenMed referential integrity, mapping
    completeness, round-trip loss, and primary event-date presence.
    """

    if not isinstance(projection, OmopFactProjection):
        raise TypeError("projection must be OmopFactProjection")
    violations = validate_omop_fact_projection(projection)
    unmapped = sum(
        count
        for state, count in projection.summary.mapping_counts.items()
        if state != "mapped"
    )
    missing_dates = 0
    for table in OMOP_DOMAIN_TABLES:
        date_column = _DOMAIN_DATE_COLUMNS[table]
        missing_dates += sum(
            1
            for row in projection.table(table)
            if not isinstance(row.get(date_column), str)
            or _DATE_RE.fullmatch(str(row.get(date_column))) is None
        )
    checks = [
        OmopQualityCheck(
            check_id="openmed.projection.referential_integrity",
            category="conformance",
            status="fail" if violations else "pass",
            severity="error",
            code=(
                "projection_referential_integrity_failed"
                if violations
                else "projection_referential_integrity_passed"
            ),
            remediation_code="repair_projection_references",
            affected_rows=len(violations),
        ),
        OmopQualityCheck(
            check_id="openmed.projection.mapping_coverage",
            category="completeness",
            status="fail" if unmapped else "pass",
            severity="warning",
            code="mapping_incomplete" if unmapped else "mapping_complete",
            remediation_code="review_unmapped_concepts",
            affected_rows=unmapped,
        ),
        OmopQualityCheck(
            check_id="openmed.projection.event_dates",
            category="plausibility",
            status="fail" if missing_dates else "pass",
            severity="error",
            code="event_date_missing" if missing_dates else "event_dates_present",
            remediation_code="repair_event_dates",
            affected_rows=missing_dates,
        ),
    ]
    if round_trip is not None:
        if not isinstance(round_trip, OmopFactRoundTripReport):
            raise TypeError("round_trip must be OmopFactRoundTripReport or None")
        losses = (
            len(round_trip.loss_fact_ids)
            + len(round_trip.missing_fact_ids)
            + len(round_trip.unexpected_fact_ids)
        )
        checks.append(
            OmopQualityCheck(
                check_id="openmed.projection.round_trip",
                category="completeness",
                status="fail" if losses else "pass",
                severity="warning",
                code="round_trip_loss" if losses else "round_trip_lossless",
                remediation_code="review_projection_loss",
                affected_rows=losses,
            )
        )
    return tuple(checks)


def reconcile_omop_aggregates(
    openmed: OmopProjectionAggregate,
    reference: OmopProjectionAggregate,
    *,
    expected_table_deltas: Mapping[str, int] | None = None,
    expected_mapped_coverage_delta_ppm: int | None = None,
    semantic_difference_codes: Sequence[str] = (),
) -> OmopReconciliation:
    """Compare two count-only projections of the same frozen cohort."""

    if not isinstance(openmed, OmopProjectionAggregate) or not isinstance(
        reference, OmopProjectionAggregate
    ):
        raise TypeError("reconciliation requires two projection aggregates")
    if reference.license not in OMOP_QUALITY_PERMISSIVE_LICENSES:
        raise OmopQualityDeniedError("reference projection license is not permitted")
    if openmed.cohort_digest != reference.cohort_digest:
        raise OmopQualityConflictError("projection cohort digests differ")
    if openmed.cohort_split != reference.cohort_split:
        raise OmopQualityConflictError("projection dataset splits differ")
    if openmed.cdm_version != reference.cdm_version:
        raise OmopQualityUnsupportedError("projection CDM versions differ")
    expected = _expected_deltas(expected_table_deltas)
    rows = tuple(
        OmopRowCountDelta(
            table=table,
            openmed_count=int(openmed.row_counts[table]),
            reference_count=int(reference.row_counts[table]),
            delta=int(openmed.row_counts[table]) - int(reference.row_counts[table]),
            expected_delta=expected.get(table),
            matches_expected=(
                table not in expected
                or int(openmed.row_counts[table]) - int(reference.row_counts[table])
                == expected[table]
            ),
        )
        for table in OMOP_FACT_TABLES
    )
    coverage_delta = openmed.mapped_coverage_ppm - reference.mapped_coverage_ppm
    return OmopReconciliation(
        cohort_digest=openmed.cohort_digest,
        cohort_split=openmed.cohort_split,
        openmed_snapshot_digest=openmed.snapshot_digest,
        reference_snapshot_digest=reference.snapshot_digest,
        openmed_aggregate_digest=openmed.digest,
        reference_aggregate_digest=reference.digest,
        row_deltas=rows,
        mapped_coverage_delta_ppm=coverage_delta,
        expected_mapped_coverage_delta_ppm=expected_mapped_coverage_delta_ppm,
        semantic_difference_codes=tuple(semantic_difference_codes),
    )


def build_omop_quality_tool_output(
    quality_input: OmopQualityInput,
    checks: Sequence[OmopQualityCheck],
    *,
    tool_name: str,
    tool_version: str,
    execution_mode: str,
) -> dict[str, Any]:
    """Build the exact aggregate JSON object an adapter must return."""

    if not isinstance(quality_input, OmopQualityInput):
        raise TypeError("quality_input must be OmopQualityInput")
    _controlled(tool_name, "tool_name")
    _version(tool_version, "tool_version")
    if execution_mode not in OMOP_QUALITY_EXECUTION_MODES:
        raise OmopQualityUnsupportedError("quality execution mode is unsupported")
    normalized = tuple(checks)
    if any(not isinstance(item, OmopQualityCheck) for item in normalized):
        raise TypeError("checks must contain OmopQualityCheck values")
    payload: dict[str, Any] = {
        "artifact_type": OMOP_QUALITY_TOOL_OUTPUT_ARTIFACT,
        "checks": [item.to_dict() for item in normalized],
        "compatibility_policy": OMOP_QUALITY_COMPATIBILITY_POLICY,
        "execution_mode": execution_mode,
        "input_digest": quality_input.digest,
        "schema_version": OMOP_QUALITY_REPORT_SCHEMA_VERSION,
        "tool_name": tool_name,
        "tool_version": tool_version,
    }
    payload["output_digest"] = canonical_digest(payload)
    return payload


def normalize_omop_quality_output(
    output: Mapping[str, Any],
    *,
    quality_input: OmopQualityInput,
    reconciliation: OmopReconciliation,
    signing_key: bytes | str,
    key_id: str = "omop-quality",
    expected_execution_mode: str | None = None,
) -> OmopQualityReport:
    """Validate aggregate adapter output and return a signed report."""

    data = _mapping(output, "quality tool output")
    _exact_keys(
        data,
        {
            "artifact_type",
            "checks",
            "compatibility_policy",
            "execution_mode",
            "input_digest",
            "output_digest",
            "schema_version",
            "tool_name",
            "tool_version",
        },
        "quality tool output",
    )
    if data["artifact_type"] != OMOP_QUALITY_TOOL_OUTPUT_ARTIFACT:
        raise OmopQualityUnsupportedError("quality tool artifact is unsupported")
    _contract(
        _text(data["schema_version"], "schema_version"),
        _text(data["compatibility_policy"], "compatibility_policy"),
        quality_input.cdm_version,
    )
    mode = _text(data["execution_mode"], "execution_mode")
    if mode not in OMOP_QUALITY_EXECUTION_MODES:
        raise OmopQualityUnsupportedError("quality execution mode is unsupported")
    if expected_execution_mode is not None and mode != expected_execution_mode:
        raise OmopQualityConflictError("quality execution mode differs")
    if data["input_digest"] != quality_input.digest:
        raise OmopQualityConflictError("quality tool input digest differs")
    supplied_output_digest = _text(data["output_digest"], "output_digest")
    _digest(supplied_output_digest, "output_digest")
    unsigned_output = dict(data)
    unsigned_output.pop("output_digest")
    if canonical_digest(unsigned_output) != supplied_output_digest:
        raise OmopQualityConflictError("quality tool output digest differs")
    checks = tuple(
        OmopQualityCheck.from_dict(_mapping(item, "quality check"))
        for item in _sequence(data["checks"], "checks")
    )
    categories = _summarize_categories(checks)
    verdict = _quality_verdict(categories, reconciliation)
    report = OmopQualityReport(
        report_id=derived_opaque_id(
            "omopquality",
            quality_input.digest,
            supplied_output_digest,
            reconciliation.digest,
        ),
        quality_input=quality_input,
        input_digest=quality_input.digest,
        source_output_digest=supplied_output_digest,
        tool_name=_text(data["tool_name"], "tool_name"),
        tool_version=_text(data["tool_version"], "tool_version"),
        execution_mode=mode,
        checks=checks,
        categories=categories,
        reconciliation=reconciliation,
        quality_verdict=verdict,
        requires_review=verdict != "pass",
    )
    return sign_omop_quality_report(report, signing_key, key_id=key_id)


def sign_omop_quality_report(
    report: OmopQualityReport,
    key: bytes | str,
    *,
    key_id: str = "omop-quality",
) -> OmopQualityReport:
    """Return a copy signed with caller-owned HMAC key material."""

    if not isinstance(report, OmopQualityReport):
        raise TypeError("report must be OmopQualityReport")
    _controlled(key_id, "signature key_id")
    key_bytes = _key_bytes(key)
    message = canonical_json(report._payload(include_digest=True)).encode("utf-8")
    signature = OmopQualitySignature(
        key_id=key_id,
        algorithm=OMOP_QUALITY_SIGNATURE_ALGORITHM,
        value=hmac.new(key_bytes, message, hashlib.sha256).hexdigest(),
    )
    return replace(report, signature=signature)


def verify_omop_quality_report(
    report: OmopQualityReport,
    key: bytes | str,
) -> bool:
    """Verify report derivations and its caller-owned HMAC signature."""

    if not isinstance(report, OmopQualityReport) or report.signature is None:
        return False
    try:
        key_bytes = _key_bytes(key)
        restored = OmopQualityReport.from_dict(report.to_dict())
    except (OmopQualityError, TypeError, ValueError):
        return False
    if restored.signature is None:
        return False
    message = canonical_json(restored._payload(include_digest=True)).encode("utf-8")
    expected = hmac.new(key_bytes, message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, restored.signature.value)


def run_omop_quality_subprocess(
    quality_input: OmopQualityInput,
    *,
    command: Sequence[str | os.PathLike[str]],
    reconciliation: OmopReconciliation,
    signing_key: bytes | str,
    key_id: str = "omop-quality",
    timeout: float = 300.0,
    environment: Mapping[str, str] | None = None,
) -> StoreResult[OmopQualityReport]:
    """Run an explicitly selected local aggregate adapter without a shell."""

    preflight = _quality_preflight(quality_input, reconciliation)
    if preflight is not None:
        return preflight
    try:
        argv = _resolve_command(command)
        timeout_value = _positive_finite(timeout, "timeout")
        payload = canonical_json(_quality_request_payload(quality_input)).encode(
            "utf-8"
        )
        raw_output = _run_bounded_adapter(
            argv,
            payload,
            timeout=timeout_value,
            environment=_safe_environment(environment),
        )
    except FileNotFoundError:
        return StoreResult.outcome(
            StoreState.UNSUPPORTED, "quality_adapter_unavailable"
        )
    except subprocess.TimeoutExpired:
        return StoreResult.outcome(StoreState.FAILURE, "quality_adapter_timeout")
    except OmopQualityProtocolError:
        return StoreResult.outcome(StoreState.FAILURE, "quality_output_invalid")
    except (OSError, OmopQualityError, TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "quality_adapter_failed")
    try:
        output = _decode_tool_output(raw_output)
        report = normalize_omop_quality_output(
            output,
            quality_input=quality_input,
            reconciliation=reconciliation,
            signing_key=signing_key,
            key_id=key_id,
            expected_execution_mode="local",
        )
    except OmopQualityDeniedError:
        return StoreResult.outcome(StoreState.DENIED, "quality_policy_denied")
    except OmopQualityConflictError:
        return StoreResult.outcome(StoreState.CONFLICT, "quality_digest_conflict")
    except OmopQualityUnsupportedError:
        return StoreResult.outcome(StoreState.UNSUPPORTED, "quality_output_unsupported")
    except (OmopQualityError, TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "quality_output_invalid")
    return _report_result(report)


def run_omop_quality_remote(
    quality_input: OmopQualityInput,
    *,
    runner: OmopQualityRemoteRunner,
    reconciliation: OmopReconciliation,
    signing_key: bytes | str,
    key_id: str = "omop-quality",
) -> StoreResult[OmopQualityReport]:
    """Run an explicitly injected remote job without a bundled network client."""

    preflight = _quality_preflight(quality_input, reconciliation)
    if preflight is not None:
        return preflight
    if not callable(runner):
        return StoreResult.outcome(StoreState.UNSUPPORTED, "quality_runner_unsupported")
    try:
        output = runner(_quality_request_payload(quality_input))
        report = normalize_omop_quality_output(
            output,
            quality_input=quality_input,
            reconciliation=reconciliation,
            signing_key=signing_key,
            key_id=key_id,
            expected_execution_mode="remote",
        )
    except OmopQualityDeniedError:
        return StoreResult.outcome(StoreState.DENIED, "quality_policy_denied")
    except OmopQualityConflictError:
        return StoreResult.outcome(StoreState.CONFLICT, "quality_digest_conflict")
    except OmopQualityUnsupportedError:
        return StoreResult.outcome(StoreState.UNSUPPORTED, "quality_output_unsupported")
    except Exception:  # noqa: BLE001 - sanitize the caller-owned transport boundary
        return StoreResult.outcome(StoreState.FAILURE, "quality_remote_failed")
    return _report_result(report)


def load_omop_quality_report_schema() -> dict[str, Any]:
    """Load the bundled OMOP quality-report JSON Schema."""

    resource = resources.files(OMOP_QUALITY_REPORT_SCHEMA_PACKAGE).joinpath(
        f"{OMOP_QUALITY_REPORT_SCHEMA_NAME}.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _quality_request_payload(quality_input: OmopQualityInput) -> dict[str, Any]:
    return {
        "artifact_type": OMOP_QUALITY_REQUEST_ARTIFACT,
        "compatibility_policy": OMOP_QUALITY_COMPATIBILITY_POLICY,
        "input": quality_input.to_dict(),
        "input_digest": quality_input.digest,
        "schema_version": OMOP_QUALITY_REPORT_SCHEMA_VERSION,
    }


def _quality_preflight(
    quality_input: OmopQualityInput,
    reconciliation: OmopReconciliation,
) -> StoreResult[OmopQualityReport] | None:
    if not isinstance(quality_input, OmopQualityInput) or not isinstance(
        reconciliation, OmopReconciliation
    ):
        return StoreResult.outcome(StoreState.FAILURE, "quality_input_invalid")
    if quality_input.reference_license not in OMOP_QUALITY_PERMISSIVE_LICENSES:
        return StoreResult.outcome(
            StoreState.DENIED, "quality_reference_license_denied"
        )
    if reconciliation.cohort_digest != quality_input.cohort_digest:
        return StoreResult.outcome(StoreState.CONFLICT, "quality_cohort_conflict")
    if reconciliation.cohort_split != quality_input.cohort_split:
        return StoreResult.outcome(StoreState.CONFLICT, "quality_split_conflict")
    if reconciliation.openmed_snapshot_digest != quality_input.projection_digest:
        return StoreResult.outcome(StoreState.CONFLICT, "quality_projection_conflict")
    if (
        reconciliation.reference_snapshot_digest
        != quality_input.reference_snapshot_digest
    ):
        return StoreResult.outcome(StoreState.CONFLICT, "quality_reference_conflict")
    return None


def _report_result(report: OmopQualityReport) -> StoreResult[OmopQualityReport]:
    if report.quality_verdict == "fail":
        return StoreResult.outcome(
            StoreState.FAILURE,
            "quality_checks_failed",
            value=report,
        )
    if report.quality_verdict == "unknown":
        category_counts = {
            item.category: item.check_count for item in report.categories
        }
        if not any(category_counts.values()):
            return StoreResult.outcome(
                StoreState.UNKNOWN,
                "quality_checks_unknown",
                value=report,
            )
        return StoreResult.outcome(
            StoreState.PARTIAL,
            "quality_report_partial",
            value=report,
        )
    return StoreResult.success(report, created=True)


def _summarize_categories(
    checks: Sequence[OmopQualityCheck],
) -> tuple[OmopQualityCategorySummary, ...]:
    grouped: dict[str, list[OmopQualityCheck]] = {
        category: [] for category in OMOP_QUALITY_CATEGORIES
    }
    for check in checks:
        grouped[check.category].append(check)
    summaries = []
    for category in OMOP_QUALITY_CATEGORIES:
        members = grouped[category]
        counts = Counter(item.status for item in members)
        if counts["fail"]:
            verdict = "fail"
        elif counts["unknown"] or not members:
            verdict = "unknown"
        else:
            verdict = "pass"
        summaries.append(
            OmopQualityCategorySummary(
                category=category,
                verdict=verdict,
                check_count=len(members),
                failed_count=counts["fail"],
                unknown_count=counts["unknown"],
            )
        )
    return tuple(sorted(summaries, key=lambda item: item.category))


def _quality_verdict(
    categories: Sequence[OmopQualityCategorySummary],
    reconciliation: OmopReconciliation,
) -> str:
    if not reconciliation.expectations_met or any(
        item.verdict == "fail" for item in categories
    ):
        return "fail"
    if any(item.verdict == "unknown" for item in categories):
        return "unknown"
    return "pass"


def _run_bounded_adapter(
    argv: tuple[str, ...],
    payload: bytes,
    *,
    timeout: float,
    environment: Mapping[str, str],
) -> bytes:
    """Bound captured output while feeding input and reap the owned process."""

    process = subprocess.Popen(  # noqa: S603 - explicit caller-selected argv
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=environment,
    )
    stdin = process.stdin
    stdout = cast(BufferedReader, process.stdout)
    assert stdin is not None and stdout is not None
    timed_out = threading.Event()

    def expire() -> None:
        timed_out.set()
        if process.poll() is None:
            process.kill()

    def feed() -> None:
        try:
            stdin.write(payload)
            stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                stdin.close()
            except OSError:
                pass

    writer = threading.Thread(target=feed, name="openmed-quality-input")
    watchdog = threading.Timer(timeout, expire)
    output = bytearray()
    try:
        writer.start()
        watchdog.start()
        while chunk := stdout.read1(
            min(65536, _MAX_TOOL_OUTPUT_BYTES + 1 - len(output))
        ):
            output.extend(chunk)
            if len(output) > _MAX_TOOL_OUTPUT_BYTES:
                raise OmopQualityProtocolError("quality adapter output size is invalid")
        process.wait()
        if timed_out.is_set():
            raise subprocess.TimeoutExpired(argv, timeout)
        if process.returncode != 0:
            raise OmopQualityError("quality adapter failed")
        return bytes(output)
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
        process.wait()
        if writer.ident is not None:
            writer.join()
        else:
            stdin.close()
        if watchdog.ident is not None:
            watchdog.join()
        stdout.close()


def _decode_tool_output(value: bytes) -> Mapping[str, Any]:
    if not value or len(value) > _MAX_TOOL_OUTPUT_BYTES:
        raise OmopQualityProtocolError("quality adapter output size is invalid")
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OmopQualityProtocolError("quality adapter output is invalid") from exc
    return _mapping(decoded, "quality adapter output")


def _resolve_command(
    command: Sequence[str | os.PathLike[str]],
) -> tuple[str, ...]:
    if isinstance(command, (str, bytes, os.PathLike)) or not isinstance(
        command, Sequence
    ):
        raise TypeError("command must be a non-empty argv sequence")
    argv = tuple(os.fspath(item) for item in command)
    if not argv or any(not item or "\x00" in item for item in argv):
        raise OmopQualityError("command must contain bounded argv values")
    executable = shutil.which(argv[0])
    if executable is None:
        raise FileNotFoundError(argv[0])
    return (os.path.abspath(executable), *argv[1:])


def _safe_environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    result = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", os.defpath),
    }
    if environment is None:
        return result
    if not isinstance(environment, Mapping):
        raise TypeError("environment must be a string mapping or None")
    for key, value in environment.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or "\x00" in key
            or "=" in key
            or "\x00" in value
        ):
            raise OmopQualityError("environment contains an invalid entry")
        result[key] = value
    return result


def _expected_deltas(value: Mapping[str, int] | None) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("expected_table_deltas must be a mapping or None")
    result: dict[str, int] = {}
    for table, delta in value.items():
        if table not in OMOP_FACT_TABLES:
            raise OmopQualityUnsupportedError("expected row delta table is unsupported")
        result[table] = _signed_integer(delta, f"expected delta for {table}")
    return result


def _key_bytes(key: bytes | str) -> bytes:
    if isinstance(key, str):
        value = key.encode("utf-8")
    elif isinstance(key, bytes):
        value = key
    else:
        raise TypeError("quality signing key must be bytes or text")
    if len(value) < _MIN_SIGNING_KEY_BYTES:
        raise OmopQualityError("quality signing key must contain at least 16 bytes")
    return value


def _contract(schema_version: str, compatibility_policy: str, cdm_version: str) -> None:
    if schema_version != OMOP_QUALITY_REPORT_SCHEMA_VERSION:
        raise OmopQualityUnsupportedError("quality schema version is unsupported")
    if compatibility_policy != OMOP_QUALITY_COMPATIBILITY_POLICY:
        raise OmopQualityUnsupportedError("quality compatibility policy is unsupported")
    if cdm_version != OMOP_FACT_PROJECTION_CDM_VERSION:
        raise OmopQualityUnsupportedError("quality CDM version is unsupported")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OmopQualityError(f"{field_name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise OmopQualityError(f"{field_name} keys must be strings")
    return value


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OmopQualityError(f"{field_name} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    field_name: str,
) -> None:
    if set(value) != expected:
        raise OmopQualityError(f"{field_name} fields are invalid")


def _text(value: Any, field_name: str) -> str:
    if type(value) is not str or not value or len(value) > 512:
        raise OmopQualityError(f"{field_name} must be bounded non-empty text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise OmopQualityError(f"{field_name} contains a control character")
    return value


def _optional_text(value: Any, field_name: str) -> str | None:
    return None if value is None else _text(value, field_name)


def _text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    return tuple(_text(item, field_name) for item in _sequence(value, field_name))


def _controlled(value: Any, field_name: str) -> str:
    if type(value) is not str or _CONTROLLED_RE.fullmatch(value) is None:
        raise OmopQualityError(f"{field_name} must be a controlled identifier")
    return value


def _digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise OmopQualityError(f"{field_name} must be a SHA-256 digest")
    return value


def _version(value: Any, field_name: str) -> str:
    if type(value) is not str or _VERSION_RE.fullmatch(value) is None:
        raise OmopQualityError(f"{field_name} must be a semantic version")
    return value


def _integer(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise OmopQualityError(f"{field_name} must be a non-negative integer")
    return value


def _signed_integer(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise OmopQualityError(f"{field_name} must be an integer")
    return value


def _optional_signed_integer(value: Any, field_name: str) -> int | None:
    return None if value is None else _signed_integer(value, field_name)


def _boolean(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise OmopQualityError(f"{field_name} must be a boolean")
    return value


def _positive_finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OmopQualityError(f"{field_name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise OmopQualityError(f"{field_name} must be a positive finite number")
    return result


def _count_mapping(value: Any, field_name: str) -> Mapping[str, int]:
    data = _mapping(value, field_name)
    return MappingProxyType(
        {
            _text(key, f"{field_name} key"): _integer(item, f"{field_name} value")
            for key, item in sorted(data.items())
        }
    )


__all__ = [
    "OMOP_QUALITY_CATEGORIES",
    "OMOP_QUALITY_CHECK_STATUSES",
    "OMOP_QUALITY_COMPATIBILITY_POLICY",
    "OMOP_QUALITY_EXECUTION_MODES",
    "OMOP_QUALITY_PERMISSIVE_LICENSES",
    "OMOP_QUALITY_REPORT_ARTIFACT",
    "OMOP_QUALITY_REPORT_SCHEMA_VERSION",
    "OMOP_QUALITY_SEVERITIES",
    "OMOP_QUALITY_SIGNATURE_ALGORITHM",
    "OMOP_QUALITY_VERDICTS",
    "OmopProjectionAggregate",
    "OmopQualityCategorySummary",
    "OmopQualityCheck",
    "OmopQualityConflictError",
    "OmopQualityDeniedError",
    "OmopQualityError",
    "OmopQualityInput",
    "OmopQualityProtocolError",
    "OmopQualityReport",
    "OmopQualitySignature",
    "OmopQualityUnsupportedError",
    "OmopReconciliation",
    "OmopRowCountDelta",
    "build_omop_quality_tool_output",
    "load_omop_quality_report_schema",
    "normalize_omop_quality_output",
    "projection_quality_checks",
    "reconcile_omop_aggregates",
    "run_omop_quality_remote",
    "run_omop_quality_subprocess",
    "sign_omop_quality_report",
    "verify_omop_quality_report",
]
