"""Deterministic, privacy-safe names for local export artifacts.

An export path is an operational identifier, not a place to persist source
metadata.  This module therefore accepts only a small typed metadata contract:
an artifact type, format, schema version, and an already-derived fingerprint.
The resulting name contains no source identifiers, paths, timestamps, or other
clock-derived values unless a caller supplies an explicit timestamp.

The implementation is intentionally local-only.  Fingerprints use canonical
JSON and SHA-256; no model registry, clock, filesystem, or network service is
consulted while a name is built.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Final, NoReturn

EXPORT_FILENAME_SCHEMA_VERSION: Final[str] = "openmed.export_filename.v1"
SCHEMA_VERSION: Final[int] = 1
FINGERPRINT_ALGORITHM: Final[str] = "sha256"
DEFAULT_FINGERPRINT_LENGTH: Final[int] = 12
MIN_FINGERPRINT_LENGTH: Final[int] = 6
MAX_FINGERPRINT_LENGTH: Final[int] = 64
MAX_COMPONENT_LENGTH: Final[int] = 64

_COMPONENT_RE = re.compile(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?")
_HEX_FINGERPRINT_RE = re.compile(r"[0-9a-f]{6,64}")
_UUID_RE = re.compile(
    r"(?<![a-z0-9])[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}(?![a-z0-9])"
)
_NUMERIC_IDENTIFIER_RE = re.compile(r"(?<![a-z0-9])\d{6,}(?![a-z0-9])")
_NAMED_IDENTIFIER_RE = re.compile(
    r"(?:^|[-_.])(?:accession|account|case|encounter|member|mrn|patient|"
    r"record|subject)[-_.]?(?:\d{2,}|[a-f0-9]{8,})(?:$|[-_.])"
)
_EXPLICIT_IDENTIFIER_RE = re.compile(
    r"(?:^|[-_.])(?:identifier|raw|sensitive|id)(?:$|[-_.])"
)


class ExportNamingError(ValueError):
    """Raised when export metadata cannot be represented safely in a name."""


ExportFilenameError = ExportNamingError


def _fail(field: str, reason: str) -> NoReturn:
    """Raise a value-only error that never includes the rejected value."""

    raise ExportNamingError(f"{field} {reason}")


def _reject_path_syntax(value: str, field: str) -> None:
    """Reject path and control syntax before any value is normalized."""

    if any(character in value for character in ("/", "\\", "\x00")):
        _fail(field, "contains a path separator or null byte")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        _fail(field, "contains a control character")


def _looks_like_raw_identifier(value: str) -> bool:
    """Return whether a filename token looks like a direct identifier."""

    return bool(
        _UUID_RE.search(value)
        or _NUMERIC_IDENTIFIER_RE.search(value)
        or _NAMED_IDENTIFIER_RE.search(value)
        or _EXPLICIT_IDENTIFIER_RE.search(value)
    )


def _normalise_component(value: object, field: str) -> str:
    """Validate and normalize one non-sensitive filename component."""

    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    _reject_path_syntax(value, field)
    candidate = value.strip().lower()
    if not candidate:
        _fail(field, "must not be empty")
    if len(candidate) > MAX_COMPONENT_LENGTH:
        _fail(field, "is too long")
    if candidate in {".", ".."} or not _COMPONENT_RE.fullmatch(candidate):
        _fail(field, "contains unsupported filename characters")
    if _looks_like_raw_identifier(candidate):
        _fail(field, "must not contain a raw identifier")
    return candidate


def _normalise_schema_version(value: object) -> str:
    """Normalize a string or integer schema version without exposing it."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError("schema_version must be a string or integer")
    return _normalise_component(str(value), "schema_version")


def _normalise_extension(value: object, format_name: str) -> str:
    """Normalize an optional extension, defaulting it to the export format."""

    if value is None:
        return format_name
    if not isinstance(value, str):
        raise TypeError("extension must be a string")
    candidate = value[1:] if value.startswith(".") else value
    if not candidate:
        _fail("extension", "must not be empty")
    return _normalise_component(candidate, "extension")


def _normalise_fingerprint(value: object) -> str:
    """Return a canonical ``sha256:`` fingerprint without echoing input."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            _fail("fingerprint", "must not be empty")
        return f"{FINGERPRINT_ALGORITHM}:{hashlib.sha256(raw).hexdigest()}"
    if not isinstance(value, str):
        raise TypeError("fingerprint must be hexadecimal text or bytes")

    _reject_path_syntax(value, "fingerprint")
    candidate = value.strip().lower()
    if candidate.startswith(f"{FINGERPRINT_ALGORITHM}:"):
        candidate = candidate.split(":", 1)[1]
    if not _HEX_FINGERPRINT_RE.fullmatch(candidate):
        _fail("fingerprint", "must be a short hexadecimal digest")
    return f"{FINGERPRINT_ALGORITHM}:{candidate}"


def _normalise_fingerprint_length(value: object) -> int:
    """Validate the requested visible fingerprint prefix length."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("fingerprint_length must be an integer")
    if not MIN_FINGERPRINT_LENGTH <= value <= MAX_FINGERPRINT_LENGTH:
        _fail(
            "fingerprint_length",
            f"must be between {MIN_FINGERPRINT_LENGTH} and {MAX_FINGERPRINT_LENGTH}",
        )
    return value


def _format_datetime(value: datetime) -> str:
    """Render an explicitly supplied datetime without punctuation or paths."""

    resolved = value
    suffix = ""
    if value.tzinfo is not None:
        resolved = value.astimezone(timezone.utc)
        suffix = "z"
    fraction = f"{resolved.microsecond:06d}" if resolved.microsecond else ""
    return resolved.strftime("%Y%m%dt%H%M%S") + fraction + suffix


def _normalise_explicit_timestamp(value: object) -> str | None:
    """Normalize a caller-supplied ISO date/time into a safe filename token."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return _format_datetime(value)
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    if not isinstance(value, str):
        raise TypeError("explicit_timestamp must be an ISO date or datetime")

    _reject_path_syntax(value, "explicit_timestamp")
    candidate = value.strip()
    if not candidate:
        _fail("explicit_timestamp", "must not be empty")
    try:
        if "t" in candidate.lower():
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            return _format_datetime(parsed)
        parsed_date = date.fromisoformat(candidate)
    except (TypeError, ValueError):
        _fail("explicit_timestamp", "must be an ISO date or datetime")
    return parsed_date.strftime("%Y%m%d")


def _canonical_json(value: Any) -> bytes:
    """Serialize a fingerprint source without exposing serialization errors."""

    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExportNamingError(
            "fingerprint source must be finite and JSON-serializable"
        ) from exc
    return serialized.encode("utf-8")


def fingerprint_for(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for local metadata.

    The input is hashed in memory and is never included in the return value,
    exception text, or a report.  Mappings are canonicalized by key order.
    """

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    else:
        raw = _canonical_json(value)
    return f"{FINGERPRINT_ALGORITHM}:{hashlib.sha256(raw).hexdigest()}"


def short_fingerprint(value: Any, *, length: int = DEFAULT_FINGERPRINT_LENGTH) -> str:
    """Return the visible hexadecimal prefix for a local fingerprint source."""

    resolved_length = _normalise_fingerprint_length(length)
    return _normalise_fingerprint(fingerprint_for(value)).split(":", 1)[1][
        :resolved_length
    ]


@dataclass(frozen=True, slots=True)
class ExportArtifactMetadata:
    """Typed, non-source metadata used to name one export artifact.

    ``fingerprint`` must already be a hexadecimal digest (or bytes to hash in
    memory).  Call :func:`fingerprint_for` when the provenance input is not a
    digest.  Keeping provenance out of this object prevents accidental reprs,
    logs, and reports from retaining raw values.
    """

    artifact_type: str
    format: str
    schema_version: str | int
    fingerprint: str | bytes
    extension: str | None = None
    explicit_timestamp: str | date | datetime | None = None

    def __post_init__(self) -> None:
        artifact_type = _normalise_component(self.artifact_type, "artifact_type")
        format_name = _normalise_component(self.format, "format")
        schema_version = _normalise_schema_version(self.schema_version)
        fingerprint = _normalise_fingerprint(self.fingerprint)
        extension = _normalise_extension(self.extension, format_name)
        explicit_timestamp = _normalise_explicit_timestamp(self.explicit_timestamp)
        object.__setattr__(self, "artifact_type", artifact_type)
        object.__setattr__(self, "format", format_name)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "extension", extension)
        object.__setattr__(self, "explicit_timestamp", explicit_timestamp)

    @property
    def format_name(self) -> str:
        """Return ``format`` under the descriptive alias used by callers."""

        return self.format

    @property
    def provenance_fingerprint(self) -> str:
        """Return the canonical provenance digest."""

        return self.fingerprint

    def to_dict(self) -> dict[str, Any]:
        """Return stable metadata without any source or identifier values."""

        result: dict[str, Any] = {
            "artifact_type": self.artifact_type,
            "format": self.format,
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "extension": self.extension,
        }
        if self.explicit_timestamp is not None:
            result["explicit_timestamp"] = self.explicit_timestamp
        return result


ArtifactMetadata = ExportArtifactMetadata
ExportMetadata = ExportArtifactMetadata


def _metadata_from_mapping(value: Mapping[str, Any]) -> ExportArtifactMetadata:
    """Build typed metadata from an allowlisted mapping."""

    allowed = {
        "artifact_type",
        "format",
        "format_name",
        "schema_version",
        "fingerprint",
        "provenance_fingerprint",
        "extension",
        "explicit_timestamp",
        "timestamp",
    }
    if any(key not in allowed for key in value):
        _fail("metadata", "contains unsupported fields")

    format_value = value.get("format")
    format_name_value = value.get("format_name")
    if format_value is not None and format_name_value is not None:
        _fail("metadata", "contains conflicting format fields")
    fingerprint_value = value.get("fingerprint")
    provenance_value = value.get("provenance_fingerprint")
    if fingerprint_value is not None and provenance_value is not None:
        _fail("metadata", "contains conflicting fingerprint fields")
    timestamp_value = value.get("explicit_timestamp")
    timestamp_alias = value.get("timestamp")
    if timestamp_value is not None and timestamp_alias is not None:
        _fail("metadata", "contains conflicting timestamp fields")

    return ExportArtifactMetadata(
        artifact_type=value.get("artifact_type"),
        format=format_value if format_value is not None else format_name_value,
        schema_version=value.get("schema_version"),
        fingerprint=(
            fingerprint_value if fingerprint_value is not None else provenance_value
        ),
        extension=value.get("extension"),
        explicit_timestamp=(
            timestamp_value if timestamp_value is not None else timestamp_alias
        ),
    )


def build_export_filename(
    metadata: ExportArtifactMetadata | Mapping[str, Any] | None = None,
    *,
    artifact_type: str | None = None,
    format: str | None = None,
    format_name: str | None = None,
    schema_version: str | int | None = None,
    fingerprint: str | bytes | None = None,
    provenance_fingerprint: str | bytes | None = None,
    extension: str | None = None,
    explicit_timestamp: str | date | datetime | None = None,
    timestamp: str | date | datetime | None = None,
    fingerprint_length: int = DEFAULT_FINGERPRINT_LENGTH,
) -> str:
    """Build one deterministic relative filename from typed metadata.

    A metadata object or allowlisted mapping can be passed as the first
    argument.  Alternatively, the typed fields can be supplied as keyword
    arguments.  The returned value is always a single relative filename with
    no path separator.  No timestamp is generated; ``explicit_timestamp`` is
    included only when the caller supplies it.

    Raises:
        ExportNamingError: If a component is unsafe, ambiguous, or resembles
            a raw identifier.
        TypeError: If a field has the wrong type.
    """

    resolved_length = _normalise_fingerprint_length(fingerprint_length)
    supplied_fields = (
        artifact_type,
        format,
        format_name,
        schema_version,
        fingerprint,
        provenance_fingerprint,
        extension,
        explicit_timestamp,
        timestamp,
    )
    if metadata is not None and any(field is not None for field in supplied_fields):
        _fail("metadata", "cannot be combined with individual fields")

    if metadata is None:
        if format is not None and format_name is not None:
            _fail("format", "has conflicting aliases")
        if fingerprint is not None and provenance_fingerprint is not None:
            _fail("fingerprint", "has conflicting aliases")
        if explicit_timestamp is not None and timestamp is not None:
            _fail("explicit_timestamp", "has conflicting aliases")
        metadata = ExportArtifactMetadata(
            artifact_type=artifact_type,
            format=format if format is not None else format_name,
            schema_version=schema_version,
            fingerprint=(
                fingerprint if fingerprint is not None else provenance_fingerprint
            ),
            extension=extension,
            explicit_timestamp=(
                explicit_timestamp if explicit_timestamp is not None else timestamp
            ),
        )
    elif isinstance(metadata, Mapping):
        metadata = _metadata_from_mapping(metadata)
    elif not isinstance(metadata, ExportArtifactMetadata):
        raise TypeError("metadata must be export metadata or a mapping")

    parts = [
        metadata.artifact_type,
        metadata.format,
        "schema",
        metadata.schema_version,
        _normalise_fingerprint(metadata.fingerprint).split(":", 1)[1][:resolved_length],
    ]
    if metadata.explicit_timestamp is not None:
        parts.append(metadata.explicit_timestamp)
    return "-".join(parts) + f".{metadata.extension}"


def make_export_filename(
    *,
    artifact_type: str,
    format: str,
    schema_version: str | int,
    fingerprint: str | bytes,
    extension: str | None = None,
    explicit_timestamp: str | date | datetime | None = None,
    fingerprint_length: int = DEFAULT_FINGERPRINT_LENGTH,
) -> str:
    """Keyword-only convenience wrapper for :func:`build_export_filename`."""

    return build_export_filename(
        artifact_type=artifact_type,
        format=format,
        schema_version=schema_version,
        fingerprint=fingerprint,
        extension=extension,
        explicit_timestamp=explicit_timestamp,
        fingerprint_length=fingerprint_length,
    )


def export_naming_policy() -> dict[str, Any]:
    """Return a stable, value-free description of the filename policy."""

    return {
        "schema_version": EXPORT_FILENAME_SCHEMA_VERSION,
        "hash_algorithm": FINGERPRINT_ALGORITHM,
        "default_fingerprint_length": DEFAULT_FINGERPRINT_LENGTH,
        "fingerprint_length": {
            "minimum": MIN_FINGERPRINT_LENGTH,
            "maximum": MAX_FINGERPRINT_LENGTH,
        },
        "timestamp_policy": "omitted_unless_explicitly_supplied",
        "raw_identifier_policy": "reject",
        "path_policy": "relative_single_filename",
    }


get_export_naming_policy = export_naming_policy


__all__ = [
    "ArtifactMetadata",
    "DEFAULT_FINGERPRINT_LENGTH",
    "EXPORT_FILENAME_SCHEMA_VERSION",
    "ExportArtifactMetadata",
    "ExportFilenameError",
    "ExportMetadata",
    "ExportNamingError",
    "FINGERPRINT_ALGORITHM",
    "MAX_COMPONENT_LENGTH",
    "MAX_FINGERPRINT_LENGTH",
    "MIN_FINGERPRINT_LENGTH",
    "SCHEMA_VERSION",
    "build_export_filename",
    "export_naming_policy",
    "fingerprint_for",
    "get_export_naming_policy",
    "make_export_filename",
    "short_fingerprint",
]
