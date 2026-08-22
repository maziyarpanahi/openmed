"""Deterministic, local-only verification for release provenance evidence.

The verifier compares content-addressed release claims without retaining the
build inputs that produced them.  It is intentionally a small boundary:
callers provide a source revision, normalized build-input mapping, dependency
lock digest, artifact digests, and provenance schema identifier.  Reports
contain only validated identifiers and digests, never paths, credentials, or
source payloads.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

RELEASE_PROVENANCE_SCHEMA_VERSION = "openmed.reproducible_release.v1"
SCHEMA_VERSION = RELEASE_PROVENANCE_SCHEMA_VERSION
PROVENANCE_SCHEMA = "https://slsa.dev/provenance/v1"

MISMATCH_SCHEMA = "schema"
MISMATCH_SOURCE_REVISION = "source_revision"
MISMATCH_BUILD_INPUTS = "build_inputs"
MISMATCH_DEPENDENCY_LOCK = "dependency_lock"
MISMATCH_ARTIFACT_HASH = "artifact_hash"
MISMATCH_PROVENANCE_SCHEMA = "provenance_schema"
MISMATCH_CATEGORIES: Final = (
    MISMATCH_SCHEMA,
    MISMATCH_SOURCE_REVISION,
    MISMATCH_BUILD_INPUTS,
    MISMATCH_DEPENDENCY_LOCK,
    MISMATCH_ARTIFACT_HASH,
    MISMATCH_PROVENANCE_SCHEMA,
)

_DIGEST_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_SOURCE_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_SCHEMA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")
_FIELD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_BUILD_INPUT_KEY_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+@-]{0,127}$")


class ReleaseProvenanceError(ValueError):
    """Raised when release provenance input cannot be safely normalized."""


class ReleaseProvenanceVerificationError(ReleaseProvenanceError):
    """Raised when a verification report is explicitly required to pass."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _hash_json(value: Any) -> str:
    return _hash_bytes(_canonical_json(value).encode("utf-8"))


def _normalise_digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise ReleaseProvenanceError(f"{field_name} must be a sha256 digest")
    digest = value.lower()
    return digest if digest.startswith("sha256:") else f"sha256:{digest}"


def _normalise_source_revision(value: Any) -> str:
    if not isinstance(value, str) or not _SOURCE_REVISION_RE.fullmatch(value):
        raise ReleaseProvenanceError(
            "source_revision must be a 40- or 64-character hexadecimal revision"
        )
    return value.lower()


def _normalise_schema_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SCHEMA_ID_RE.fullmatch(value):
        raise ReleaseProvenanceError(f"{field_name} must be a safe schema identifier")
    return value


def _normalise_artifact_id(value: Any) -> str:
    if not isinstance(value, str) or not _ARTIFACT_ID_RE.fullmatch(value):
        raise ReleaseProvenanceError(
            "artifact identifiers must be non-empty path-free identifiers"
        )
    return value


def _normalise_artifact_hashes(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ReleaseProvenanceError("artifact_hashes must be an object")

    normalized: dict[str, str] = {}
    for artifact_id, digest in value.items():
        normalized_id = _normalise_artifact_id(artifact_id)
        normalized[normalized_id] = _normalise_digest(
            digest,
            "artifact hash",
        )
    return dict(sorted(normalized.items()))


def _normalise_build_key(value: Any) -> str:
    if not isinstance(value, str) or not _BUILD_INPUT_KEY_RE.fullmatch(value):
        raise ReleaseProvenanceError("build input keys must be visible strings")
    return value


def _normalise_build_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _normalise_build_key(key)
            normalized[normalized_key] = _normalise_build_value(item)
        return dict(sorted(normalized.items()))

    if isinstance(value, (list, tuple)):
        return [_normalise_build_value(item) for item in value]

    if isinstance(value, (set, frozenset)):
        normalized_items = [_normalise_build_value(item) for item in value]
        return sorted(normalized_items, key=_canonical_json)

    if isinstance(value, (bytes, bytearray, memoryview)):
        content = bytes(value)
        return {
            "kind": "bytes",
            "length": len(content),
            "sha256": _hash_bytes(content),
        }

    if isinstance(value, PathLike):
        return {"kind": "path"}

    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReleaseProvenanceError("build inputs must contain finite numbers")
        return value

    raise ReleaseProvenanceError("build inputs contain an unsupported value type")


def compute_build_inputs_digest(build_inputs: Mapping[str, Any]) -> str:
    """Return a deterministic digest for normalized build inputs.

    Args:
        build_inputs: JSON-like toolchain and build configuration values. Mapping
            keys are sorted recursively, sequences preserve order, sets are
            sorted by canonical JSON, bytes are content-hashed, and path-like
            values are represented only by a path marker.

    Returns:
        A lowercase ``sha256:`` digest. The normalized values are not retained
        by the release provenance record.
    """

    if not isinstance(build_inputs, Mapping):
        raise ReleaseProvenanceError("build_inputs must be an object")
    normalized = _normalise_build_value(build_inputs)
    return _hash_json(
        {
            "kind": "openmed-release-build-inputs",
            "schema_version": SCHEMA_VERSION,
            "inputs": normalized,
        }
    )


def _read_content(value: Any, field_name: str) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, (str, PathLike)):
        try:
            return Path(value).read_bytes()
        except (OSError, TypeError, ValueError):
            raise ReleaseProvenanceError(f"cannot read {field_name} input") from None
    raise ReleaseProvenanceError(f"{field_name} must be bytes or a local path")


def compute_dependency_lock_digest(lock: bytes | str | PathLike[str]) -> str:
    """Hash a local dependency-lock payload without retaining its contents.

    Args:
        lock: Lock-file bytes or a local path to read.

    Returns:
        A lowercase ``sha256:`` digest.
    """

    return _hash_bytes(_read_content(lock, "dependency lock"))


def compute_artifact_digest(artifact: bytes | str | PathLike[str]) -> str:
    """Hash one local release artifact without retaining its contents.

    Args:
        artifact: Artifact bytes or a local path to read.

    Returns:
        A lowercase ``sha256:`` digest.
    """

    return _hash_bytes(_read_content(artifact, "artifact"))


def compute_artifact_hashes(
    artifacts: Mapping[str, bytes | str | PathLike[str]],
) -> dict[str, str]:
    """Hash named local artifacts into a deterministic safe mapping.

    Args:
        artifacts: Path-free artifact identifiers mapped to bytes or local paths.

    Returns:
        Artifact identifiers mapped to lowercase ``sha256:`` digests.
    """

    if not isinstance(artifacts, Mapping):
        raise ReleaseProvenanceError("artifacts must be an object")
    return {
        artifact_id: compute_artifact_digest(artifacts[artifact_id])
        for artifact_id in sorted(_normalise_artifact_id(key) for key in artifacts)
    }


@dataclass(frozen=True)
class ReleaseProvenance:
    """Safe, content-addressed claims for one release build."""

    schema_version: str
    source_revision: str
    build_inputs_digest: str
    dependency_lock_digest: str
    artifact_hashes: Mapping[str, str]
    provenance_schema: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _normalise_schema_id(self.schema_version, "schema_version"),
        )
        object.__setattr__(
            self,
            "source_revision",
            _normalise_source_revision(self.source_revision),
        )
        object.__setattr__(
            self,
            "build_inputs_digest",
            _normalise_digest(self.build_inputs_digest, "build_inputs_digest"),
        )
        object.__setattr__(
            self,
            "dependency_lock_digest",
            _normalise_digest(
                self.dependency_lock_digest,
                "dependency_lock_digest",
            ),
        )
        object.__setattr__(
            self,
            "artifact_hashes",
            MappingProxyType(_normalise_artifact_hashes(self.artifact_hashes)),
        )
        object.__setattr__(
            self,
            "provenance_schema",
            _normalise_schema_id(self.provenance_schema, "provenance_schema"),
        )

    @classmethod
    def from_inputs(
        cls,
        *,
        source_revision: str,
        build_inputs: Mapping[str, Any],
        dependency_lock_digest: str | None = None,
        artifact_hashes: Mapping[str, str] | None = None,
        dependency_lock: bytes | str | PathLike[str] | None = None,
        artifacts: Mapping[str, bytes | str | PathLike[str]] | None = None,
        provenance_schema: str = PROVENANCE_SCHEMA,
        schema_version: str = SCHEMA_VERSION,
    ) -> "ReleaseProvenance":
        """Build a safe record from local release inputs.

        Either a dependency-lock digest or local lock content must be supplied.
        Artifact digests may be supplied directly, or local artifact content can
        be hashed with ``artifacts``. Raw values are used only to derive digests.
        """

        if (dependency_lock_digest is None) == (dependency_lock is None):
            raise ReleaseProvenanceError(
                "provide exactly one dependency lock digest or local lock input"
            )
        if (artifact_hashes is None) == (artifacts is None):
            raise ReleaseProvenanceError(
                "provide exactly one artifact hash mapping or local artifacts"
            )

        resolved_lock_digest = (
            _normalise_digest(dependency_lock_digest, "dependency_lock_digest")
            if dependency_lock_digest is not None
            else compute_dependency_lock_digest(dependency_lock)
        )
        resolved_artifact_hashes = (
            _normalise_artifact_hashes(artifact_hashes)
            if artifact_hashes is not None
            else compute_artifact_hashes(artifacts)
        )
        return cls(
            schema_version=schema_version,
            source_revision=source_revision,
            build_inputs_digest=compute_build_inputs_digest(build_inputs),
            dependency_lock_digest=resolved_lock_digest,
            artifact_hashes=resolved_artifact_hashes,
            provenance_schema=provenance_schema,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ReleaseProvenance":
        """Load a strict safe record or derive its build-input digest once."""

        if not isinstance(payload, Mapping):
            raise ReleaseProvenanceError("release provenance must be an object")

        required = {
            "schema_version",
            "source_revision",
            "dependency_lock_digest",
            "provenance_schema",
        }
        if not required.issubset(payload):
            raise ReleaseProvenanceError(
                "release provenance payload is missing required fields"
            )

        has_digest = "build_inputs_digest" in payload
        has_inputs = "build_inputs" in payload
        if has_digest == has_inputs:
            raise ReleaseProvenanceError(
                "provide exactly one build-input digest or build-input object"
            )

        has_hashes = "artifact_hashes" in payload
        has_artifacts = "artifacts" in payload
        if has_hashes == has_artifacts:
            raise ReleaseProvenanceError(
                "provide exactly one artifact hash mapping or artifact object"
            )

        allowed = required | {
            "build_inputs_digest" if has_digest else "build_inputs",
            "artifact_hashes" if has_hashes else "artifacts",
        }
        if set(payload) != allowed:
            raise ReleaseProvenanceError(
                "release provenance payload contains unsupported fields"
            )

        build_inputs_digest = (
            payload["build_inputs_digest"]
            if has_digest
            else compute_build_inputs_digest(payload["build_inputs"])
        )
        artifact_hashes = (
            payload["artifact_hashes"]
            if has_hashes
            else compute_artifact_hashes(payload["artifacts"])
        )
        return cls(
            schema_version=payload["schema_version"],
            source_revision=payload["source_revision"],
            build_inputs_digest=build_inputs_digest,
            dependency_lock_digest=payload["dependency_lock_digest"],
            artifact_hashes=artifact_hashes,
            provenance_schema=payload["provenance_schema"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic evidence without raw input values."""

        return {
            "artifact_hashes": dict(self.artifact_hashes),
            "build_inputs_digest": self.build_inputs_digest,
            "dependency_lock_digest": self.dependency_lock_digest,
            "provenance_schema": self.provenance_schema,
            "schema_version": self.schema_version,
            "source_revision": self.source_revision,
        }

    def to_evidence(self) -> dict[str, Any]:
        """Return the same safe representation under the evidence name."""

        return self.to_dict()


@dataclass(frozen=True)
class ProvenanceMismatch:
    """One stable, raw-value-free mismatch diagnostic."""

    category: str
    field: str
    expected: str | None
    actual: str | None

    def __post_init__(self) -> None:
        if self.category not in MISMATCH_CATEGORIES:
            raise ReleaseProvenanceError("unsupported provenance mismatch category")
        if not isinstance(self.field, str) or not _FIELD_RE.fullmatch(self.field):
            raise ReleaseProvenanceError("provenance mismatch field is not safe")
        for value in (self.expected, self.actual):
            if value is not None and (
                not isinstance(value, str) or not _SCHEMA_ID_RE.fullmatch(value)
            ):
                raise ReleaseProvenanceError(
                    "provenance mismatch values must be safe identifiers"
                )

    def to_dict(self) -> dict[str, str | None]:
        """Return a stable machine-readable mismatch."""

        return {
            "actual": self.actual,
            "category": self.category,
            "expected": self.expected,
            "field": self.field,
        }


@dataclass(frozen=True)
class ReleaseVerificationReport:
    """Deterministic verification result containing only safe diagnostics."""

    valid: bool
    mismatches: tuple[ProvenanceMismatch, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise ReleaseProvenanceError("verification validity must be boolean")
        object.__setattr__(
            self,
            "schema_version",
            _normalise_schema_id(self.schema_version, "schema_version"),
        )
        normalized = tuple(
            mismatch
            if isinstance(mismatch, ProvenanceMismatch)
            else ProvenanceMismatch(*mismatch)
            for mismatch in self.mismatches
        )
        order = {category: index for index, category in enumerate(MISMATCH_CATEGORIES)}
        ordered = tuple(
            sorted(
                normalized,
                key=lambda item: (order[item.category], item.field),
            )
        )
        object.__setattr__(self, "mismatches", ordered)
        object.__setattr__(self, "valid", not ordered)

    @property
    def mismatch_categories(self) -> tuple[str, ...]:
        """Return distinct mismatch categories in stable schema order."""

        present = {mismatch.category for mismatch in self.mismatches}
        return tuple(
            category for category in MISMATCH_CATEGORIES if category in present
        )

    @property
    def categories(self) -> tuple[str, ...]:
        """Return :attr:`mismatch_categories` as a concise alias."""

        return self.mismatch_categories

    def verify(self) -> bool:
        """Return whether all release provenance claims matched."""

        return self.valid

    def raise_if_invalid(self) -> None:
        """Raise a safe exception when one or more claims do not match."""

        if not self.valid:
            categories = ", ".join(self.mismatch_categories)
            raise ReleaseProvenanceVerificationError(
                f"release provenance verification failed: {categories}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic evidence suitable for logs or storage."""

        return {
            "mismatch_categories": list(self.mismatch_categories),
            "mismatches": [mismatch.to_dict() for mismatch in self.mismatches],
            "schema_version": self.schema_version,
            "valid": self.valid,
        }

    def __bool__(self) -> bool:
        return self.valid


def build_release_provenance(
    *,
    source_revision: str,
    build_inputs: Mapping[str, Any],
    dependency_lock_digest: str | None = None,
    artifact_hashes: Mapping[str, str] | None = None,
    dependency_lock: bytes | str | PathLike[str] | None = None,
    artifacts: Mapping[str, bytes | str | PathLike[str]] | None = None,
    provenance_schema: str = PROVENANCE_SCHEMA,
    schema_version: str = SCHEMA_VERSION,
) -> ReleaseProvenance:
    """Build a safe release provenance record from local-only inputs.

    Args:
        source_revision: Immutable 40- or 64-character source revision.
        build_inputs: JSON-like normalized toolchain and build configuration.
        dependency_lock_digest: Content digest for the declared dependency lock.
        artifact_hashes: Path-free artifact identifiers and content digests.
        dependency_lock: Optional local lock bytes or path when a digest is not
            already available.
        artifacts: Optional path-free artifact identifiers mapped to bytes or
            local paths when digests are not already available.
        provenance_schema: Schema identifier of the attached provenance claim.
        schema_version: Version of this verifier's record schema.

    Returns:
        A record whose evidence representation contains no raw inputs.
    """

    return ReleaseProvenance.from_inputs(
        source_revision=source_revision,
        build_inputs=build_inputs,
        dependency_lock_digest=dependency_lock_digest,
        artifact_hashes=artifact_hashes,
        dependency_lock=dependency_lock,
        artifacts=artifacts,
        provenance_schema=provenance_schema,
        schema_version=schema_version,
    )


def verify_release_provenance(
    expected: ReleaseProvenance | Mapping[str, Any],
    actual: ReleaseProvenance | Mapping[str, Any],
) -> ReleaseVerificationReport:
    """Compare two release provenance records without network access.

    Args:
        expected: Claims derived from the declared source and toolchain.
        actual: Claims derived from the release build and artifacts.

    Returns:
        A deterministic report. Mismatch categories are stable and the report
        includes only safe identifiers and digests.
    """

    expected_record = _coerce_record(expected)
    actual_record = _coerce_record(actual)
    mismatches: list[ProvenanceMismatch] = []

    _compare_field(
        mismatches,
        MISMATCH_SCHEMA,
        "schema_version",
        expected_record.schema_version,
        actual_record.schema_version,
    )
    _compare_field(
        mismatches,
        MISMATCH_SOURCE_REVISION,
        "source_revision",
        expected_record.source_revision,
        actual_record.source_revision,
    )
    _compare_field(
        mismatches,
        MISMATCH_BUILD_INPUTS,
        "build_inputs_digest",
        expected_record.build_inputs_digest,
        actual_record.build_inputs_digest,
    )
    _compare_field(
        mismatches,
        MISMATCH_DEPENDENCY_LOCK,
        "dependency_lock_digest",
        expected_record.dependency_lock_digest,
        actual_record.dependency_lock_digest,
    )
    _compare_field(
        mismatches,
        MISMATCH_PROVENANCE_SCHEMA,
        "provenance_schema",
        expected_record.provenance_schema,
        actual_record.provenance_schema,
    )

    artifact_ids = sorted(
        set(expected_record.artifact_hashes) | set(actual_record.artifact_hashes)
    )
    for artifact_id in artifact_ids:
        _compare_field(
            mismatches,
            MISMATCH_ARTIFACT_HASH,
            f"artifact_hashes.{artifact_id}",
            expected_record.artifact_hashes.get(artifact_id),
            actual_record.artifact_hashes.get(artifact_id),
        )

    return ReleaseVerificationReport(valid=not mismatches, mismatches=tuple(mismatches))


def check_release_provenance(
    expected: ReleaseProvenance | Mapping[str, Any],
    actual: ReleaseProvenance | Mapping[str, Any],
) -> ReleaseVerificationReport:
    """Return :func:`verify_release_provenance` under a check-style name."""

    return verify_release_provenance(expected, actual)


def _coerce_record(value: ReleaseProvenance | Mapping[str, Any]) -> ReleaseProvenance:
    if isinstance(value, ReleaseProvenance):
        return value
    if isinstance(value, Mapping):
        return ReleaseProvenance.from_mapping(value)
    raise ReleaseProvenanceError("expected a release provenance record")


def _compare_field(
    mismatches: list[ProvenanceMismatch],
    category: str,
    field: str,
    expected: str | None,
    actual: str | None,
) -> None:
    if expected != actual:
        mismatches.append(
            ProvenanceMismatch(
                category=category,
                field=field,
                expected=expected,
                actual=actual,
            )
        )


ReleaseProvenanceRecord = ReleaseProvenance
ProvenanceVerificationReport = ReleaseVerificationReport

__all__ = [
    "MISMATCH_ARTIFACT_HASH",
    "MISMATCH_BUILD_INPUTS",
    "MISMATCH_CATEGORIES",
    "MISMATCH_DEPENDENCY_LOCK",
    "MISMATCH_PROVENANCE_SCHEMA",
    "MISMATCH_SCHEMA",
    "MISMATCH_SOURCE_REVISION",
    "PROVENANCE_SCHEMA",
    "RELEASE_PROVENANCE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ProvenanceMismatch",
    "ProvenanceVerificationReport",
    "ReleaseProvenance",
    "ReleaseProvenanceError",
    "ReleaseProvenanceRecord",
    "ReleaseProvenanceVerificationError",
    "ReleaseVerificationReport",
    "build_release_provenance",
    "check_release_provenance",
    "compute_artifact_digest",
    "compute_artifact_hashes",
    "compute_build_inputs_digest",
    "compute_dependency_lock_digest",
    "verify_release_provenance",
]
