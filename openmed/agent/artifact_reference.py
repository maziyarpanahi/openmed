"""Content-free references to artifacts produced by agent workflows.

References contain only bounded categorical metadata. They do not contain a
path, URL, filename, inline payload, or patient-derived identifier, and this
module never opens, fetches, or verifies the referenced artifact.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

ARTIFACT_REFERENCE_VERSION: Final = 1
MAX_ARTIFACT_BYTE_SIZE: Final = (1 << 63) - 1

_ARTIFACT_ID_RE = re.compile(r"art_[0-9a-f]{32}")
_SCHEMA_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+\.v[1-9][0-9]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ALLOWED_FIELDS = frozenset(
    {"version", "artifact_id", "kind", "schema_id", "sha256", "byte_size"}
)
_REQUIRED_FIELDS = frozenset(
    {"artifact_id", "kind", "schema_id", "sha256", "byte_size"}
)
_ORDERED_FIELDS = (
    "version",
    "artifact_id",
    "kind",
    "schema_id",
    "sha256",
    "byte_size",
)


class ArtifactKind(str, Enum):
    """Closed vocabulary for agent-produced artifact categories."""

    EVIDENCE = "evidence"
    PREVIEW = "preview"
    FHIR = "fhir"
    OMOP = "omop"
    EVALUATION = "evaluation"


class ArtifactReferenceError(ValueError):
    """Raised when an artifact reference fails closed validation.

    Args:
        code: Stable machine-readable validation code.
        field_name: Optional public field associated with the failure.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Immutable metadata-only reference to an agent artifact.

    Args:
        artifact_id: Opaque ``art_`` identifier containing 128 random bits.
        kind: Closed artifact category.
        schema_id: Canonical, versioned schema identifier.
        sha256: Lowercase hexadecimal SHA-256 digest supplied by the caller.
        byte_size: Positive artifact size supplied by the caller.
        version: Artifact-reference envelope version.
    """

    artifact_id: str
    kind: ArtifactKind
    schema_id: str
    sha256: str
    byte_size: int
    version: int = ARTIFACT_REFERENCE_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != ARTIFACT_REFERENCE_VERSION:
            raise ArtifactReferenceError("invalid_version", "version")
        _validate_string(
            self.artifact_id,
            _ARTIFACT_ID_RE,
            "invalid_artifact_id",
            "artifact_id",
        )
        if not isinstance(self.kind, ArtifactKind):
            raise ArtifactReferenceError("unknown_kind", "kind")
        _validate_string(
            self.schema_id, _SCHEMA_ID_RE, "invalid_schema_id", "schema_id"
        )
        _validate_string(self.sha256, _SHA256_RE, "invalid_sha256", "sha256")
        if (
            type(self.byte_size) is not int
            or not 0 < self.byte_size <= MAX_ARTIFACT_BYTE_SIZE
        ):
            raise ArtifactReferenceError("invalid_byte_size", "byte_size")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactReference":
        """Build a reference from a strict metadata-only mapping."""

        if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
            raise ArtifactReferenceError("not_a_mapping")
        try:
            fields = set(data)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ArtifactReferenceError("not_a_mapping") from None
        if fields - _ALLOWED_FIELDS:
            raise ArtifactReferenceError("unknown_field")
        if _REQUIRED_FIELDS - fields:
            raise ArtifactReferenceError("missing_field")

        try:
            values = {field_name: data[field_name] for field_name in fields}
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ArtifactReferenceError("unreadable_mapping") from None
        return cls(
            version=values.get("version", ARTIFACT_REFERENCE_VERSION),
            artifact_id=values["artifact_id"],
            kind=_parse_kind(values["kind"]),
            schema_id=values["schema_id"],
            sha256=values["sha256"],
            byte_size=values["byte_size"],
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "ArtifactReference":
        """Build a reference from a strict JSON object."""

        try:
            data = json.loads(payload, object_pairs_hook=_strict_json_object)
        except (
            json.JSONDecodeError,
            ArtifactReferenceError,
            TypeError,
            UnicodeDecodeError,
        ):
            pass
        else:
            return cls.from_dict(data)
        raise ArtifactReferenceError("malformed_json")

    def to_dict(self) -> dict[str, str | int]:
        """Return deterministic metadata-only fields."""

        values: dict[str, str | int] = {
            "version": self.version,
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "schema_id": self.schema_id,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
        }
        return {field_name: values[field_name] for field_name in _ORDERED_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with deterministic key ordering."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def validate_artifact_references(
    references: Iterable[ArtifactReference],
) -> tuple[ArtifactReference, ...]:
    """Return an immutable reference sequence after rejecting duplicate IDs."""

    if isinstance(references, (str, bytes, bytearray, Mapping)):
        raise ArtifactReferenceError("invalid_reference_collection")
    try:
        values = tuple(references)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ArtifactReferenceError("invalid_reference_collection") from None

    artifact_ids: set[str] = set()
    for reference in values:
        if type(reference) is not ArtifactReference:
            raise ArtifactReferenceError("invalid_reference")
        if reference.artifact_id in artifact_ids:
            raise ArtifactReferenceError("duplicate_artifact_id", "artifact_id")
        artifact_ids.add(reference.artifact_id)
    return values


def _parse_kind(value: Any) -> ArtifactKind:
    if isinstance(value, ArtifactKind):
        return value
    if type(value) is str:
        try:
            return ArtifactKind(value)
        except ValueError:
            pass
    raise ArtifactReferenceError("unknown_kind", "kind")


def _validate_string(
    value: Any,
    pattern: re.Pattern[str],
    code: str,
    field_name: str,
) -> None:
    if type(value) is not str or len(value) > 128 or pattern.fullmatch(value) is None:
        raise ArtifactReferenceError(code, field_name)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactReferenceError("duplicate_field")
        result[key] = value
    return result


__all__ = [
    "ARTIFACT_REFERENCE_VERSION",
    "MAX_ARTIFACT_BYTE_SIZE",
    "ArtifactKind",
    "ArtifactReference",
    "ArtifactReferenceError",
    "validate_artifact_references",
]
