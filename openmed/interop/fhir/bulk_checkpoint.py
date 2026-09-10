"""Deterministic, PHI-safe checkpoints for FHIR Bulk Data pagination.

The FHIR Bulk Data protocol exposes opaque page tokens. A resumable export
needs to remember which token, policy, and endpoint scope it used without
persisting the token or any resource payload. This module stores only stable
SHA-256 digests and aggregate progress counters. It performs no network or
filesystem work unless a caller explicitly asks to read or write a manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

CHECKPOINT_MANIFEST_VERSION: Final[int] = 1
BULK_CHECKPOINT_MANIFEST_VERSION: Final[int] = CHECKPOINT_MANIFEST_VERSION
CHECKPOINT_SCHEMA_VERSION: Final[int] = CHECKPOINT_MANIFEST_VERSION

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RESOURCE_TYPE_RE = re.compile(r"^[A-Z][A-Za-z0-9]{0,63}$")
_SERIALIZED_FIELDS = frozenset(
    {
        "manifest_version",
        "resource_type",
        "page_token_digest",
        "policy_fingerprint",
        "endpoint_scope",
        "pages_processed",
        "resources_processed",
    }
)


class BulkCheckpointError(ValueError):
    """Base error for malformed or unsafe Bulk Data checkpoint manifests."""


class BulkCheckpointSchemaError(BulkCheckpointError):
    """Raised when a checkpoint does not match the supported JSON shape."""


class BulkCheckpointCompatibilityError(BulkCheckpointError):
    """Raised when a checkpoint cannot safely resume the requested export."""


def _canonical_identity(value: Any, field_name: str) -> bytes:
    """Encode one caller-supplied identity without exposing it in errors."""

    if isinstance(value, str):
        if not value and field_name != "page_token":
            raise BulkCheckpointError(f"{field_name} must be non-empty")
        return value.encode("utf-8")
    if isinstance(value, bytes):
        if not value and field_name != "page_token":
            raise BulkCheckpointError(f"{field_name} must be non-empty")
        return value
    if value is None and field_name == "page_token":
        return b"null"

    if isinstance(value, Mapping) and any(type(key) is not str for key in value):
        raise BulkCheckpointError(f"{field_name} keys must be strings")

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise BulkCheckpointError(
            f"{field_name} must be a JSON-compatible value"
        ) from None
    if not encoded and field_name != "page_token":
        raise BulkCheckpointError(f"{field_name} must be non-empty")
    return encoded


def _digest(value: Any, field_name: str) -> str:
    """Return the canonical SHA-256 representation for an identity value."""

    material = _canonical_identity(value, field_name)
    return f"sha256:{hashlib.sha256(material).hexdigest()}"


def digest_page_token(page_token: str | bytes | None) -> str:
    """Return a stable digest for an opaque FHIR page token.

    ``None`` is encoded as a distinct end-of-pagination marker. The token is
    never returned, logged, or included in the checkpoint representation.
    """

    if page_token is not None and not isinstance(page_token, (str, bytes)):
        raise BulkCheckpointError("page_token must be text, bytes, or None")
    return _digest(page_token, "page_token")


def fingerprint_policy(policy: Any) -> str:
    """Return a deterministic digest for a policy name or JSON configuration."""

    return _digest(policy, "policy")


def fingerprint_endpoint_scope(endpoint_scope: Any) -> str:
    """Return a PHI-safe digest for the export endpoint scope."""

    return _digest(endpoint_scope, "endpoint_scope")


def _require_digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise BulkCheckpointSchemaError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise BulkCheckpointSchemaError(f"{field_name} must be a non-negative integer")
    return value


def _require_manifest_version(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise BulkCheckpointSchemaError("manifest_version must be a positive integer")
    return value


def _require_resource_type(value: Any) -> str:
    if type(value) is not str or _RESOURCE_TYPE_RE.fullmatch(value) is None:
        raise BulkCheckpointSchemaError("resource_type must be a valid FHIR type")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


@dataclass(frozen=True)
class BulkCheckpointManifest:
    """A counts-only checkpoint for one FHIR Bulk Data resource stream.

    ``endpoint_scope`` is the digest returned by
    :func:`fingerprint_endpoint_scope`, not the caller's raw endpoint value.
    The same is true of ``page_token_digest`` and ``policy_fingerprint``.
    """

    resource_type: str
    page_token_digest: str
    policy_fingerprint: str
    endpoint_scope: str
    pages_processed: int = 0
    resources_processed: int = 0
    manifest_version: int = CHECKPOINT_MANIFEST_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "manifest_version", _require_manifest_version(self.manifest_version)
        )
        object.__setattr__(
            self, "resource_type", _require_resource_type(self.resource_type)
        )
        for field_name in (
            "page_token_digest",
            "policy_fingerprint",
            "endpoint_scope",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_digest(getattr(self, field_name), field_name),
            )
        for field_name in ("pages_processed", "resources_processed"):
            object.__setattr__(
                self,
                field_name,
                _require_non_negative_int(getattr(self, field_name), field_name),
            )

    @property
    def schema_version(self) -> int:
        """Return the manifest version under the repository's schema alias."""

        return self.manifest_version

    @property
    def endpoint_scope_digest(self) -> str:
        """Return the stored endpoint-scope digest."""

        return self.endpoint_scope

    @property
    def next_page_token_digest(self) -> str:
        """Return the page-token digest under a protocol-oriented alias."""

        return self.page_token_digest

    @property
    def progress(self) -> dict[str, int]:
        """Return aggregate progress without resource identifiers or payloads."""

        return {
            "pages_processed": self.pages_processed,
            "resources_processed": self.resources_processed,
        }

    @property
    def counts(self) -> dict[str, int]:
        """Return the counts-only progress mapping."""

        return self.progress

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, payload-free manifest representation."""

        return {
            "manifest_version": self.manifest_version,
            "resource_type": self.resource_type,
            "page_token_digest": self.page_token_digest,
            "policy_fingerprint": self.policy_fingerprint,
            "endpoint_scope": self.endpoint_scope,
            "pages_processed": self.pages_processed,
            "resources_processed": self.resources_processed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BulkCheckpointManifest":
        """Build a manifest from a strict JSON-compatible mapping."""

        if not isinstance(payload, Mapping):
            raise BulkCheckpointSchemaError("checkpoint must be a JSON object")

        data = dict(payload)
        if "manifest_version" not in data and "schema_version" in data:
            data["manifest_version"] = data.pop("schema_version")
        if set(data) != _SERIALIZED_FIELDS:
            raise BulkCheckpointSchemaError("checkpoint has missing or unknown fields")
        if any(type(key) is not str for key in data):
            raise BulkCheckpointSchemaError("checkpoint field names must be strings")

        return cls(
            manifest_version=data["manifest_version"],
            resource_type=data["resource_type"],
            page_token_digest=data["page_token_digest"],
            policy_fingerprint=data["policy_fingerprint"],
            endpoint_scope=data["endpoint_scope"],
            pages_processed=data["pages_processed"],
            resources_processed=data["resources_processed"],
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BulkCheckpointManifest":
        """Alias for :meth:`from_dict` used by other OpenMed manifests."""

        return cls.from_dict(payload)

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the manifest deterministically without raw identity data."""

        if indent is not None and (type(indent) is not int or indent < 0):
            raise ValueError("indent must be a non-negative integer or None")
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "BulkCheckpointManifest":
        """Parse a manifest JSON document and validate its schema."""

        if isinstance(payload, (bytes, bytearray)):
            try:
                payload = bytes(payload).decode("utf-8")
            except UnicodeDecodeError:
                raise BulkCheckpointSchemaError(
                    "checkpoint JSON is not UTF-8"
                ) from None
        if type(payload) is not str:
            raise BulkCheckpointSchemaError("checkpoint JSON must be text")
        try:
            decoded = json.loads(
                payload,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError):
            raise BulkCheckpointSchemaError("checkpoint JSON is invalid") from None
        if not isinstance(decoded, Mapping):
            raise BulkCheckpointSchemaError("checkpoint JSON must be an object")
        return cls.from_dict(decoded)

    def write(self, path: str | Path) -> Path:
        """Persist this manifest through an atomic local-file replacement."""

        return write_checkpoint(self, path)

    @classmethod
    def read(cls, path: str | Path) -> "BulkCheckpointManifest":
        """Load and validate a manifest from a local file."""

        loaded = load_checkpoint(path)
        if not isinstance(loaded, cls):
            raise BulkCheckpointSchemaError("checkpoint file has an invalid type")
        return loaded

    def with_progress(
        self,
        *,
        page_token: str | bytes | None,
        pages_processed: int | None = None,
        resources_processed: int | None = None,
    ) -> "BulkCheckpointManifest":
        """Return a copy with a new token digest and aggregate progress."""

        return replace(
            self,
            page_token_digest=digest_page_token(page_token),
            pages_processed=(
                self.pages_processed if pages_processed is None else pages_processed
            ),
            resources_processed=(
                self.resources_processed
                if resources_processed is None
                else resources_processed
            ),
        )

    def validate_resume(
        self,
        *,
        resource_type: str,
        page_token: str | bytes | None,
        policy: Any,
        endpoint_scope: Any,
        manifest_version: int = CHECKPOINT_MANIFEST_VERSION,
    ) -> None:
        """Raise if the supplied resume context does not match this manifest."""

        validate_resume(
            self,
            resource_type=resource_type,
            page_token=page_token,
            policy=policy,
            endpoint_scope=endpoint_scope,
            manifest_version=manifest_version,
        )

    def is_compatible(
        self,
        *,
        resource_type: str,
        page_token: str | bytes | None,
        policy: Any,
        endpoint_scope: Any,
        manifest_version: int = CHECKPOINT_MANIFEST_VERSION,
    ) -> bool:
        """Return whether the supplied context can safely resume this manifest."""

        return is_resume_compatible(
            self,
            resource_type=resource_type,
            page_token=page_token,
            policy=policy,
            endpoint_scope=endpoint_scope,
            manifest_version=manifest_version,
        )


def create_checkpoint(
    resource_type: str,
    page_token: str | bytes | None,
    policy: Any,
    endpoint_scope: Any,
    *,
    pages_processed: int = 0,
    resources_processed: int = 0,
    manifest_version: int = CHECKPOINT_MANIFEST_VERSION,
) -> BulkCheckpointManifest:
    """Create a local-only checkpoint from the current export context.

    The page token, policy input, and endpoint scope are immediately reduced
    to digests. Only the resource type and aggregate counters remain visible
    in the returned manifest.
    """

    return BulkCheckpointManifest(
        resource_type=resource_type,
        page_token_digest=digest_page_token(page_token),
        policy_fingerprint=fingerprint_policy(policy),
        endpoint_scope=fingerprint_endpoint_scope(endpoint_scope),
        pages_processed=pages_processed,
        resources_processed=resources_processed,
        manifest_version=manifest_version,
    )


build_checkpoint = create_checkpoint


def validate_resume(
    checkpoint: BulkCheckpointManifest,
    *,
    resource_type: str,
    page_token: str | bytes | None,
    policy: Any,
    endpoint_scope: Any,
    manifest_version: int = CHECKPOINT_MANIFEST_VERSION,
) -> None:
    """Fail closed unless all resume identity fields match.

    Error messages identify only the incompatible field names. They never
    include the caller's page token, policy configuration, or endpoint scope.
    """

    if not isinstance(checkpoint, BulkCheckpointManifest):
        raise BulkCheckpointCompatibilityError("checkpoint is not a valid manifest")

    try:
        checkpoint.__post_init__()
        expected_version = _require_manifest_version(manifest_version)
        expected_resource_type = _require_resource_type(resource_type)
        expected_page_token_digest = digest_page_token(page_token)
        expected_policy_fingerprint = fingerprint_policy(policy)
        expected_endpoint_scope = fingerprint_endpoint_scope(endpoint_scope)
    except BulkCheckpointError:
        raise BulkCheckpointCompatibilityError("resume context is invalid") from None

    mismatches: list[str] = []
    if checkpoint.manifest_version != expected_version:
        mismatches.append("manifest version")
    if checkpoint.resource_type != expected_resource_type:
        mismatches.append("resource type")
    if checkpoint.page_token_digest != expected_page_token_digest:
        mismatches.append("page token")
    if checkpoint.policy_fingerprint != expected_policy_fingerprint:
        mismatches.append("policy")
    if checkpoint.endpoint_scope != expected_endpoint_scope:
        mismatches.append("endpoint scope")

    if mismatches:
        raise BulkCheckpointCompatibilityError(
            "checkpoint is incompatible with the requested resume context: "
            + ", ".join(mismatches)
        )


validate_resume_compatibility = validate_resume
assert_resume_compatible = validate_resume


def is_resume_compatible(
    checkpoint: BulkCheckpointManifest,
    *,
    resource_type: str,
    page_token: str | bytes | None,
    policy: Any,
    endpoint_scope: Any,
    manifest_version: int = CHECKPOINT_MANIFEST_VERSION,
) -> bool:
    """Return ``False`` for any malformed or incompatible resume context."""

    try:
        validate_resume(
            checkpoint,
            resource_type=resource_type,
            page_token=page_token,
            policy=policy,
            endpoint_scope=endpoint_scope,
            manifest_version=manifest_version,
        )
    except BulkCheckpointError:
        return False
    return True


def write_checkpoint(
    checkpoint: BulkCheckpointManifest,
    path: str | Path,
) -> Path:
    """Atomically write a validated checkpoint to a local JSON file."""

    if not isinstance(checkpoint, BulkCheckpointManifest):
        raise BulkCheckpointSchemaError("checkpoint must be a manifest")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = (checkpoint.to_json() + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


save_checkpoint = write_checkpoint


def load_checkpoint(path: str | Path) -> BulkCheckpointManifest:
    """Read and validate a checkpoint from a local JSON file."""

    try:
        payload = Path(path).read_bytes()
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError):
        raise BulkCheckpointSchemaError("checkpoint file could not be read") from None
    return BulkCheckpointManifest.from_json(payload)


BulkCheckpoint = BulkCheckpointManifest
FHIRBulkCheckpoint = BulkCheckpointManifest
FHIRBulkCheckpointManifest = BulkCheckpointManifest


__all__ = [
    "BULK_CHECKPOINT_MANIFEST_VERSION",
    "CHECKPOINT_MANIFEST_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "BulkCheckpoint",
    "BulkCheckpointCompatibilityError",
    "BulkCheckpointError",
    "BulkCheckpointManifest",
    "BulkCheckpointSchemaError",
    "FHIRBulkCheckpoint",
    "FHIRBulkCheckpointManifest",
    "assert_resume_compatible",
    "build_checkpoint",
    "create_checkpoint",
    "digest_page_token",
    "fingerprint_endpoint_scope",
    "fingerprint_policy",
    "is_resume_compatible",
    "load_checkpoint",
    "save_checkpoint",
    "validate_resume",
    "validate_resume_compatibility",
    "write_checkpoint",
]
