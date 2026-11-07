"""Versioned, value-safe results for structured evidence adapters."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Any, ClassVar, TypeVar

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    EvidenceLocator,
    canonical_json,
)

EVIDENCE_ADAPTER_SCHEMA_VERSION = "1.0.0"
EVIDENCE_ADAPTER_COMPATIBILITY_POLICY = "same_major"
EVIDENCE_ADAPTER_SCHEMA_PACKAGE = "openmed.core.schemas.json"
EVIDENCE_ADAPTER_SCHEMA_NAMES = ("result", "quarantine")
EVIDENCE_SOURCE_FORMATS = frozenset(
    {
        "text",
        "document",
        "fhir_r4",
        "hl7v2",
        "cda",
        "csv",
        "xlsx",
        "pdf",
        "ocr",
        "image",
        "dicom_sr",
    }
)
EVIDENCE_COORDINATE_CONVENTIONS = frozenset(
    {
        "unicode_text_v1",
        "existing_document_v1",
        "fhir_json_pointer_v1",
        "hl7v2_field_component_v1",
        "cda_document_path_v1",
        "table_cell_v1",
        "pdf_top_left_v1",
        "ocr_top_left_v1",
        "image_top_left_v1",
        "dicom_element_v1",
    }
)
EVIDENCE_QUARANTINE_CLASSIFICATIONS = frozenset(
    {"ambiguous", "malformed", "partial", "unsafe", "unsupported"}
)

_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SEMVER_RE = re.compile(r"^[1-9][0-9]*(?:\.[0-9]+){0,2}$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
_SENSITIVE_KEYS = frozenset(
    {
        "credential",
        "name",
        "note",
        "patient",
        "payload",
        "phi",
        "raw",
        "secret",
        "source_text",
        "text",
        "token",
        "value",
        "vault",
    }
)

R = TypeVar("R", bound="EvidenceAdapterRecord")


class EvidenceAdapterContractError(ValueError):
    """Value-safe validation error for evidence-adapter records."""


class EvidenceAdapterRecord:
    """Strict deterministic JSON behavior for adapter records."""

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
        """Parse strict JSON with duplicate and non-finite values rejected."""

        return cls.from_dict(_strict_json(payload))


@dataclass(frozen=True, slots=True)
class StructuredEvidenceResult(EvidenceAdapterRecord):
    """Immutable artifact and exact evidence coordinates for one source."""

    result_id: str
    artifact: ClinicalArtifact
    locators: tuple[EvidenceLocator, ...]
    source_digest: str
    source_byte_size: int
    source_format: str
    source_version: str
    parser_id: str
    parser_version: str
    coordinate_convention: str
    normalization_transform: Mapping[str, Any]
    compatibility_policy: str = EVIDENCE_ADAPTER_COMPATIBILITY_POLICY
    schema_version: str = EVIDENCE_ADAPTER_SCHEMA_VERSION

    _fields = frozenset(
        {
            "artifact",
            "compatibility_policy",
            "coordinate_convention",
            "locators",
            "normalization_transform",
            "parser_id",
            "parser_version",
            "result_id",
            "schema_version",
            "source_byte_size",
            "source_digest",
            "source_format",
            "source_version",
        }
    )

    def __post_init__(self) -> None:
        _opaque(self.result_id, "result_id")
        if not isinstance(self.artifact, ClinicalArtifact):
            raise EvidenceAdapterContractError("artifact is invalid")
        if not isinstance(self.locators, tuple):
            raise EvidenceAdapterContractError("locators must be a tuple")
        if not all(isinstance(item, EvidenceLocator) for item in self.locators):
            raise EvidenceAdapterContractError("locators contain an invalid record")
        if any(item.artifact_id != self.artifact.artifact_id for item in self.locators):
            raise EvidenceAdapterContractError("locator artifact differs")
        locator_ids = tuple(item.locator_id for item in self.locators)
        if len(set(locator_ids)) != len(locator_ids):
            raise EvidenceAdapterContractError("locator identifiers must be unique")
        object.__setattr__(
            self,
            "locators",
            tuple(sorted(self.locators, key=lambda item: item.locator_id)),
        )
        _digest(self.source_digest, "source_digest")
        _non_negative(self.source_byte_size, "source_byte_size")
        if self.source_format not in EVIDENCE_SOURCE_FORMATS:
            raise EvidenceAdapterContractError("source_format is unsupported")
        _version(self.source_version, "source_version")
        _controlled(self.parser_id, "parser_id")
        _semver(self.parser_version, "parser_version")
        if self.coordinate_convention not in EVIDENCE_COORDINATE_CONVENTIONS:
            raise EvidenceAdapterContractError("coordinate_convention is unsupported")
        object.__setattr__(
            self,
            "normalization_transform",
            _safe_mapping(self.normalization_transform),
        )
        _compatibility(self.compatibility_policy)
        _schema(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return value-free adapter metadata and Journey records."""

        return {
            "artifact": self.artifact.to_dict(),
            "compatibility_policy": self.compatibility_policy,
            "coordinate_convention": self.coordinate_convention,
            "locators": [item.to_dict() for item in self.locators],
            "normalization_transform": _plain(self.normalization_transform),
            "parser_id": self.parser_id,
            "parser_version": self.parser_version,
            "result_id": self.result_id,
            "schema_version": self.schema_version,
            "source_byte_size": self.source_byte_size,
            "source_digest": self.source_digest,
            "source_format": self.source_format,
            "source_version": self.source_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StructuredEvidenceResult":
        """Parse one strict structured-evidence result."""

        data = _strict(payload, cls._fields, "structured evidence result")
        try:
            locators = data["locators"]
            if not isinstance(locators, list):
                raise EvidenceAdapterContractError("locators must be an array")
            return cls(
                result_id=data["result_id"],
                artifact=ClinicalArtifact.from_dict(data["artifact"]),
                locators=tuple(EvidenceLocator.from_dict(item) for item in locators),
                source_digest=data["source_digest"],
                source_byte_size=data["source_byte_size"],
                source_format=data["source_format"],
                source_version=data["source_version"],
                parser_id=data["parser_id"],
                parser_version=data["parser_version"],
                coordinate_convention=data["coordinate_convention"],
                normalization_transform=data["normalization_transform"],
                compatibility_policy=data["compatibility_policy"],
                schema_version=data["schema_version"],
            )
        except KeyError:
            raise EvidenceAdapterContractError(
                "structured evidence result is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class StructuredEvidenceQuarantine(EvidenceAdapterRecord):
    """Value-free isolation record for an unsafe or unsupported source."""

    quarantine_id: str
    source_digest: str
    source_byte_size: int
    source_format: str
    source_version: str
    parser_id: str
    parser_version: str
    classification: str
    reason_code: str
    candidate_count: int
    failure_count: int
    created_at: str
    compatibility_policy: str = EVIDENCE_ADAPTER_COMPATIBILITY_POLICY
    schema_version: str = EVIDENCE_ADAPTER_SCHEMA_VERSION

    _fields = frozenset(
        {
            "candidate_count",
            "classification",
            "compatibility_policy",
            "created_at",
            "failure_count",
            "parser_id",
            "parser_version",
            "quarantine_id",
            "reason_code",
            "schema_version",
            "source_byte_size",
            "source_digest",
            "source_format",
            "source_version",
        }
    )

    def __post_init__(self) -> None:
        _opaque(self.quarantine_id, "quarantine_id")
        _digest(self.source_digest, "source_digest")
        _non_negative(self.source_byte_size, "source_byte_size")
        if self.source_format not in EVIDENCE_SOURCE_FORMATS:
            raise EvidenceAdapterContractError("source_format is unsupported")
        _version(self.source_version, "source_version")
        _controlled(self.parser_id, "parser_id")
        _semver(self.parser_version, "parser_version")
        if self.classification not in EVIDENCE_QUARANTINE_CLASSIFICATIONS:
            raise EvidenceAdapterContractError(
                "quarantine classification is unsupported"
            )
        _controlled(self.reason_code, "reason_code")
        _non_negative(self.candidate_count, "candidate_count")
        _positive(self.failure_count, "failure_count")
        _timestamp(self.created_at, "created_at")
        _compatibility(self.compatibility_policy)
        _schema(self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free quarantine mapping."""

        return {
            "candidate_count": self.candidate_count,
            "classification": self.classification,
            "compatibility_policy": self.compatibility_policy,
            "created_at": self.created_at,
            "failure_count": self.failure_count,
            "parser_id": self.parser_id,
            "parser_version": self.parser_version,
            "quarantine_id": self.quarantine_id,
            "reason_code": self.reason_code,
            "schema_version": self.schema_version,
            "source_byte_size": self.source_byte_size,
            "source_digest": self.source_digest,
            "source_format": self.source_format,
            "source_version": self.source_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StructuredEvidenceQuarantine":
        """Parse one strict evidence quarantine."""

        data = _strict(payload, cls._fields, "structured evidence quarantine")
        try:
            return cls(**data)
        except KeyError:
            raise EvidenceAdapterContractError(
                "structured evidence quarantine is missing a required field"
            ) from None


def load_evidence_adapter_schema(name: str) -> dict[str, Any]:
    """Load one bundled structured-evidence JSON Schema."""

    normalized = name.removeprefix("evidence_adapter_").removesuffix(".schema.json")
    if normalized not in EVIDENCE_ADAPTER_SCHEMA_NAMES:
        raise KeyError("unknown evidence adapter schema")
    resource = resources.files(EVIDENCE_ADAPTER_SCHEMA_PACKAGE).joinpath(
        f"evidence_adapter_{normalized}.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_all_evidence_adapter_schemas() -> dict[str, dict[str, Any]]:
    """Load all bundled structured-evidence JSON Schemas."""

    return {
        name: load_evidence_adapter_schema(name)
        for name in EVIDENCE_ADAPTER_SCHEMA_NAMES
    }


def _strict(
    payload: Mapping[str, Any], fields: frozenset[str], record_name: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise EvidenceAdapterContractError(f"{record_name} must be an object")
    data = dict(payload)
    if set(data) != fields:
        raise EvidenceAdapterContractError(
            f"{record_name} has missing or unknown fields"
        )
    return data


def _strict_json(payload: str) -> Mapping[str, Any]:
    if not isinstance(payload, str):
        raise EvidenceAdapterContractError("record JSON must be text")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise EvidenceAdapterContractError("record JSON is invalid") from None
    if not isinstance(value, Mapping):
        raise EvidenceAdapterContractError("record JSON must contain an object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite number")


def _safe_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise EvidenceAdapterContractError(
            "normalization_transform must be a non-empty object"
        )
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or _CONTROLLED_RE.fullmatch(key) is None
            or key.lower() in _SENSITIVE_KEYS
        ):
            raise EvidenceAdapterContractError(
                "normalization_transform contains an unsafe key"
            )
        normalized[key] = _safe_value(item)
    return MappingProxyType(dict(sorted(normalized.items())))


def _safe_value(value: Any) -> Any:
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise EvidenceAdapterContractError("transform number must be finite")
        return value
    if isinstance(value, str):
        if _VERSION_RE.fullmatch(value) is None and _DIGEST_RE.fullmatch(value) is None:
            raise EvidenceAdapterContractError(
                "transform text must be a controlled token"
            )
        return value
    if isinstance(value, Mapping):
        return _safe_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_safe_value(item) for item in value)
    raise EvidenceAdapterContractError("transform value is unsupported")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _opaque(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise EvidenceAdapterContractError(f"{field_name} must be an opaque identifier")
    return value


def _digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise EvidenceAdapterContractError(f"{field_name} must be a SHA-256 digest")
    return value


def _controlled(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise EvidenceAdapterContractError(
            f"{field_name} must be a controlled identifier"
        )
    return value


def _version(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise EvidenceAdapterContractError(f"{field_name} must be a safe version")
    return value


def _semver(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SEMVER_RE.fullmatch(value) is None:
        raise EvidenceAdapterContractError(f"{field_name} must be a version")
    return value


def _non_negative(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise EvidenceAdapterContractError(f"{field_name} must be non-negative")
    return value


def _positive(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise EvidenceAdapterContractError(f"{field_name} must be positive")
    return value


def _timestamp(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        raise EvidenceAdapterContractError(
            f"{field_name} must be a timezone-aware ISO timestamp"
        )
    return value


def _compatibility(value: Any) -> str:
    if value != EVIDENCE_ADAPTER_COMPATIBILITY_POLICY:
        raise EvidenceAdapterContractError("compatibility policy is unsupported")
    return str(value)


def _schema(value: Any) -> str:
    if value != EVIDENCE_ADAPTER_SCHEMA_VERSION:
        raise EvidenceAdapterContractError("evidence adapter schema is unsupported")
    return str(value)


__all__ = [
    "EVIDENCE_ADAPTER_COMPATIBILITY_POLICY",
    "EVIDENCE_ADAPTER_SCHEMA_NAMES",
    "EVIDENCE_ADAPTER_SCHEMA_PACKAGE",
    "EVIDENCE_ADAPTER_SCHEMA_VERSION",
    "EVIDENCE_COORDINATE_CONVENTIONS",
    "EVIDENCE_QUARANTINE_CLASSIFICATIONS",
    "EVIDENCE_SOURCE_FORMATS",
    "EvidenceAdapterContractError",
    "StructuredEvidenceQuarantine",
    "StructuredEvidenceResult",
    "load_all_evidence_adapter_schemas",
    "load_evidence_adapter_schema",
]
