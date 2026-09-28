"""Value-safe contracts for the ingestion-to-fact pipeline lineage."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any, ClassVar, TypeVar

from openmed.clinical.journey_contracts import (
    canonical_digest,
    canonical_json,
    derived_opaque_id,
)

PIPELINE_LINEAGE_SCHEMA_VERSION = "1.0.0"
PIPELINE_LINEAGE_COMPATIBILITY_POLICY = "same_major"
PIPELINE_STAGE_SCHEMA_NAMES = ("pipeline_stage", "stage_invalidation")
PIPELINE_STAGES = (
    "source_adaptation",
    "privacy_policy",
    "document_routing",
    "extraction",
    "assertion",
    "temporality",
    "relation_extraction",
    "grounding",
    "fact_normalization",
    "validation",
    "durable_writes",
)
PIPELINE_STAGE_STATES = frozenset(
    {
        "success",
        "partial",
        "unknown",
        "conflict",
        "unsupported",
        "denied",
        "failure",
    }
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)

R = TypeVar("R", bound="PipelineLineageRecord")


class PipelineLineageError(ValueError):
    """Value-safe validation error for pipeline lineage records."""


class PipelineLineageRecord:
    """Strict deterministic JSON behavior for pipeline lineage records."""

    _fields: ClassVar[frozenset[str]]

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible metadata."""

        raise NotImplementedError

    def to_json(self) -> str:
        """Return canonical JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls: type[R], payload: Mapping[str, Any]) -> R:
        """Parse one strict mapping."""

        raise NotImplementedError

    @classmethod
    def from_json(cls: type[R], payload: str) -> R:
        """Parse strict JSON without accepting non-finite values."""

        try:
            value = json.loads(payload, parse_constant=_reject_constant)
        except (TypeError, json.JSONDecodeError):
            raise PipelineLineageError("pipeline lineage must be valid JSON") from None
        if not isinstance(value, dict):
            raise PipelineLineageError("pipeline lineage must be an object")
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class PipelineStageManifest(PipelineLineageRecord):
    """One PHI-free, append-only stage execution manifest."""

    stage_manifest_id: str
    job_id: str
    stage: str
    sequence: int
    state: str
    input_digests: tuple[str, ...]
    output_digests: tuple[str, ...]
    input_record_ids: tuple[str, ...]
    output_record_ids: tuple[str, ...]
    component: str
    component_version: str
    policy_digest: str
    parent_stage_manifest_ids: tuple[str, ...]
    recorded_at: str
    reason_code: str | None = None
    schema_version: str = PIPELINE_LINEAGE_SCHEMA_VERSION
    compatibility_policy: str = PIPELINE_LINEAGE_COMPATIBILITY_POLICY

    _fields = frozenset(
        {
            "compatibility_policy",
            "component",
            "component_version",
            "input_digests",
            "input_record_ids",
            "job_id",
            "output_digests",
            "output_record_ids",
            "parent_stage_manifest_ids",
            "policy_digest",
            "reason_code",
            "recorded_at",
            "schema_version",
            "sequence",
            "stage",
            "stage_manifest_id",
            "state",
        }
    )

    def __post_init__(self) -> None:
        _opaque(self.stage_manifest_id, "stage_manifest_id")
        _opaque(self.job_id, "job_id")
        if self.stage not in PIPELINE_STAGES:
            raise PipelineLineageError("pipeline stage is unsupported")
        expected_sequence = PIPELINE_STAGES.index(self.stage) + 1
        if type(self.sequence) is not int or self.sequence != expected_sequence:
            raise PipelineLineageError("pipeline stage sequence is invalid")
        if self.state not in PIPELINE_STAGE_STATES:
            raise PipelineLineageError("pipeline stage state is unsupported")
        inputs = _digests(self.input_digests, "input_digests", minimum=1)
        outputs = _digests(self.output_digests, "output_digests")
        input_ids = _opaque_ids(self.input_record_ids, "input_record_ids")
        output_ids = _opaque_ids(self.output_record_ids, "output_record_ids")
        parents = _opaque_ids(
            self.parent_stage_manifest_ids,
            "parent_stage_manifest_ids",
        )
        _controlled(self.component, "component")
        _version(self.component_version, "component_version")
        _digest(self.policy_digest, "policy_digest")
        _timestamp(self.recorded_at, "recorded_at")
        if self.state == "success" and self.reason_code is not None:
            raise PipelineLineageError("successful stage cannot have a reason code")
        if self.state != "success":
            _controlled(self.reason_code, "reason_code")
        _schema(self.schema_version, self.compatibility_policy)
        object.__setattr__(self, "input_digests", inputs)
        object.__setattr__(self, "output_digests", outputs)
        object.__setattr__(self, "input_record_ids", input_ids)
        object.__setattr__(self, "output_record_ids", output_ids)
        object.__setattr__(self, "parent_stage_manifest_ids", parents)

    @property
    def manifest_digest(self) -> str:
        """Return the digest for the complete safe manifest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return value-free stage lineage metadata."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "component": self.component,
            "component_version": self.component_version,
            "input_digests": list(self.input_digests),
            "input_record_ids": list(self.input_record_ids),
            "job_id": self.job_id,
            "output_digests": list(self.output_digests),
            "output_record_ids": list(self.output_record_ids),
            "parent_stage_manifest_ids": list(self.parent_stage_manifest_ids),
            "policy_digest": self.policy_digest,
            "reason_code": self.reason_code,
            "recorded_at": self.recorded_at,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "stage": self.stage,
            "stage_manifest_id": self.stage_manifest_id,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineStageManifest":
        """Parse one strict stage manifest."""

        data = _strict(payload, cls._fields, "pipeline stage manifest")
        try:
            return cls(
                stage_manifest_id=data["stage_manifest_id"],
                job_id=data["job_id"],
                stage=data["stage"],
                sequence=data["sequence"],
                state=data["state"],
                input_digests=_tuple(data["input_digests"], "input_digests"),
                output_digests=_tuple(data["output_digests"], "output_digests"),
                input_record_ids=_tuple(data["input_record_ids"], "input_record_ids"),
                output_record_ids=_tuple(
                    data["output_record_ids"], "output_record_ids"
                ),
                component=data["component"],
                component_version=data["component_version"],
                policy_digest=data["policy_digest"],
                parent_stage_manifest_ids=_tuple(
                    data["parent_stage_manifest_ids"],
                    "parent_stage_manifest_ids",
                ),
                recorded_at=data["recorded_at"],
                reason_code=data["reason_code"],
                schema_version=data["schema_version"],
                compatibility_policy=data["compatibility_policy"],
            )
        except KeyError:
            raise PipelineLineageError(
                "pipeline stage manifest is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class PipelineStageInvalidation(PipelineLineageRecord):
    """Append-only record invalidating one replaced stage descendant."""

    invalidation_id: str
    job_id: str
    stage_manifest_id: str
    replacement_job_id: str
    reason_code: str
    recorded_at: str
    schema_version: str = PIPELINE_LINEAGE_SCHEMA_VERSION
    compatibility_policy: str = PIPELINE_LINEAGE_COMPATIBILITY_POLICY

    _fields = frozenset(
        {
            "compatibility_policy",
            "invalidation_id",
            "job_id",
            "reason_code",
            "recorded_at",
            "replacement_job_id",
            "schema_version",
            "stage_manifest_id",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "invalidation_id",
            "job_id",
            "stage_manifest_id",
            "replacement_job_id",
        ):
            _opaque(getattr(self, name), name)
        if self.job_id == self.replacement_job_id:
            raise PipelineLineageError("replacement job must be a new derivation")
        _controlled(self.reason_code, "reason_code")
        _timestamp(self.recorded_at, "recorded_at")
        _schema(self.schema_version, self.compatibility_policy)

    def to_dict(self) -> dict[str, Any]:
        """Return value-free invalidation metadata."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "invalidation_id": self.invalidation_id,
            "job_id": self.job_id,
            "reason_code": self.reason_code,
            "recorded_at": self.recorded_at,
            "replacement_job_id": self.replacement_job_id,
            "schema_version": self.schema_version,
            "stage_manifest_id": self.stage_manifest_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineStageInvalidation":
        """Parse one strict stage invalidation."""

        data = _strict(payload, cls._fields, "pipeline stage invalidation")
        try:
            return cls(**data)
        except KeyError:
            raise PipelineLineageError(
                "pipeline stage invalidation is missing a required field"
            ) from None


def build_stage_manifest(
    *,
    job_id: str,
    stage: str,
    state: str,
    input_digests: Sequence[str],
    output_digests: Sequence[str],
    input_record_ids: Sequence[str],
    output_record_ids: Sequence[str],
    component: str,
    component_version: str,
    policy_digest: str,
    parent_stage_manifest_ids: Sequence[str],
    recorded_at: str,
    reason_code: str | None = None,
) -> PipelineStageManifest:
    """Build a deterministic stage manifest from value-free materials."""

    sequence = PIPELINE_STAGES.index(stage) + 1
    identity = {
        "component": component,
        "component_version": component_version,
        "input_digests": sorted(input_digests),
        "input_record_ids": sorted(input_record_ids),
        "job_id": job_id,
        "output_digests": sorted(output_digests),
        "output_record_ids": sorted(output_record_ids),
        "parent_stage_manifest_ids": sorted(parent_stage_manifest_ids),
        "policy_digest": policy_digest,
        "reason_code": reason_code,
        "schema_version": PIPELINE_LINEAGE_SCHEMA_VERSION,
        "sequence": sequence,
        "stage": stage,
        "state": state,
    }
    return PipelineStageManifest(
        stage_manifest_id=derived_opaque_id("stage", identity),
        job_id=job_id,
        stage=stage,
        sequence=sequence,
        state=state,
        input_digests=tuple(input_digests),
        output_digests=tuple(output_digests),
        input_record_ids=tuple(input_record_ids),
        output_record_ids=tuple(output_record_ids),
        component=component,
        component_version=component_version,
        policy_digest=policy_digest,
        parent_stage_manifest_ids=tuple(parent_stage_manifest_ids),
        recorded_at=recorded_at,
        reason_code=reason_code,
    )


def build_stage_invalidation(
    manifest: PipelineStageManifest,
    *,
    replacement_job_id: str,
    reason_code: str,
    recorded_at: str,
) -> PipelineStageInvalidation:
    """Build a deterministic invalidation for a replaced stage manifest."""

    return PipelineStageInvalidation(
        invalidation_id=derived_opaque_id(
            "invalidation",
            manifest.stage_manifest_id,
            replacement_job_id,
            reason_code,
        ),
        job_id=manifest.job_id,
        stage_manifest_id=manifest.stage_manifest_id,
        replacement_job_id=replacement_job_id,
        reason_code=reason_code,
        recorded_at=recorded_at,
    )


def load_pipeline_lineage_schema(name: str) -> dict[str, Any]:
    """Load one bundled pipeline-lineage JSON Schema."""

    normalized = name.removeprefix("ingestion_").removesuffix(".schema.json")
    if normalized not in PIPELINE_STAGE_SCHEMA_NAMES:
        raise KeyError("unknown pipeline lineage schema")
    resource = resources.files("openmed.core.schemas.json").joinpath(
        f"ingestion_{normalized}.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _strict(
    payload: Mapping[str, Any],
    fields: frozenset[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise PipelineLineageError(f"{name} fields are invalid")
    return dict(payload)


def _tuple(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise PipelineLineageError(f"{name} must be an array")
    return tuple(value)


def _digests(
    values: Sequence[str],
    name: str,
    *,
    minimum: int = 0,
) -> tuple[str, ...]:
    if not isinstance(values, tuple) or len(values) < minimum:
        raise PipelineLineageError(f"{name} has invalid cardinality")
    for value in values:
        _digest(value, name)
    if len(set(values)) != len(values):
        raise PipelineLineageError(f"{name} must be unique")
    return tuple(sorted(values))


def _opaque_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise PipelineLineageError(f"{name} must be a tuple")
    for value in values:
        _opaque(value, name)
    if len(set(values)) != len(values):
        raise PipelineLineageError(f"{name} must be unique")
    return tuple(sorted(values))


def _digest(value: Any, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise PipelineLineageError(f"{name} must contain SHA-256 digests")


def _opaque(value: Any, name: str) -> None:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise PipelineLineageError(f"{name} must be an opaque identifier")


def _controlled(value: Any, name: str) -> None:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise PipelineLineageError(f"{name} must be a controlled identifier")


def _version(value: Any, name: str) -> None:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise PipelineLineageError(f"{name} must be a bounded version")


def _timestamp(value: Any, name: str) -> None:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        raise PipelineLineageError(f"{name} must be a timezone-aware timestamp")


def _schema(version: str, compatibility_policy: str) -> None:
    if version != PIPELINE_LINEAGE_SCHEMA_VERSION:
        raise PipelineLineageError("pipeline lineage schema is unsupported")
    if compatibility_policy != PIPELINE_LINEAGE_COMPATIBILITY_POLICY:
        raise PipelineLineageError("pipeline lineage compatibility is unsupported")


def _reject_constant(value: str) -> None:
    raise PipelineLineageError("pipeline lineage contains a non-finite number")


__all__ = [
    "PIPELINE_LINEAGE_COMPATIBILITY_POLICY",
    "PIPELINE_LINEAGE_SCHEMA_VERSION",
    "PIPELINE_STAGE_SCHEMA_NAMES",
    "PIPELINE_STAGE_STATES",
    "PIPELINE_STAGES",
    "PipelineLineageError",
    "PipelineStageInvalidation",
    "PipelineStageManifest",
    "build_stage_invalidation",
    "build_stage_manifest",
    "load_pipeline_lineage_schema",
]
