"""Deterministic, privacy-safe result envelopes for scriptable CLI work.

The envelope deliberately has no free-text payload.  A caller can report the
outcome, a bounded category, numeric counters, fingerprints for named local
artifacts, and a small set of remediation codes.  This keeps machine output
stable across terminal widths and locales and prevents rejected input from
being copied into logs or exceptions.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import IO, Any, TypeVar

SCHEMA_VERSION = 1
MAX_REMEDIATION_CODES = 3

_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9_.-]{0,63})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HASH_CHUNK_SIZE = 1024 * 1024

_ENVELOPE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "category",
        "counters",
        "artifacts",
        "remediation_codes",
    }
)
_ARTIFACT_KEYS = frozenset({"name", "sha256", "size_bytes"})


class ResultEnvelopeError(ValueError):
    """Raised when an envelope or one of its typed fields is invalid.

    Error messages are intentionally constant and never include rejected
    values.  A caller can safely send the exception to a log sink.
    """


class ResultStatus(str, Enum):
    """The finite set of outcomes represented by an envelope."""

    SUCCESS = "success"
    FAILURE = "failure"


class ResultCategory(str, Enum):
    """Bounded categories for successful and failed CLI outcomes."""

    SUCCESS = "success"
    INPUT = "input"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    RUNTIME = "runtime"
    INTEGRITY = "integrity"


class RemediationCode(str, Enum):
    """Finite remediation hints that do not carry user-provided text."""

    CHECK_INPUT = "check_input"
    CHECK_CONFIGURATION = "check_configuration"
    VERIFY_ARTIFACT = "verify_artifact"
    RETRY_COMMAND = "retry_command"
    CONTACT_OPERATOR = "contact_operator"


def _invalid(message: str) -> ResultEnvelopeError:
    return ResultEnvelopeError(message)


def _is_identifier(value: Any) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _coerce_enum(value: Any, enum_type: type[Enum], message: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            pass
    raise _invalid(message)


def _normalize_counters(
    counters: Mapping[str, int] | Iterable[tuple[str, int]] | None,
) -> tuple[tuple[str, int], ...]:
    if counters is None:
        return ()
    if isinstance(counters, Mapping):
        entries = list(counters.items())
    else:
        try:
            entries = list(counters)
        except (TypeError, ValueError):
            raise _invalid("counters must be a mapping of non-negative integers")

    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for entry in entries:
        if (
            not isinstance(entry, (tuple, list))
            or len(entry) != 2
            or not _is_identifier(entry[0])
            or entry[0] in seen
            or isinstance(entry[1], bool)
            or not isinstance(entry[1], int)
            or entry[1] < 0
        ):
            raise _invalid("counters must be a mapping of non-negative integers")
        key, value = entry
        seen.add(key)
        normalized.append((key, value))
    return tuple(sorted(normalized))


def _normalize_artifacts(
    artifacts: Iterable["ArtifactFingerprint | Mapping[str, Any]"] | None,
) -> tuple["ArtifactFingerprint", ...]:
    if artifacts is None:
        return ()
    try:
        entries = list(artifacts)
    except (TypeError, ValueError):
        raise _invalid("artifacts must be a sequence of fingerprints")

    normalized: list[ArtifactFingerprint] = []
    seen: set[str] = set()
    for entry in entries:
        if isinstance(entry, ArtifactFingerprint):
            artifact = entry
        elif isinstance(entry, Mapping):
            artifact = ArtifactFingerprint.from_dict(entry)
        else:
            raise _invalid("artifacts must be a sequence of fingerprints")
        if artifact.name in seen:
            raise _invalid("artifact names must be unique")
        seen.add(artifact.name)
        normalized.append(artifact)
    return tuple(sorted(normalized, key=lambda artifact: artifact.name))


def _normalize_remediation_codes(
    codes: Iterable[RemediationCode | str] | None,
) -> tuple[RemediationCode, ...]:
    if codes is None:
        return ()
    if isinstance(codes, (str, bytes)):
        raise _invalid("remediation_codes must be a sequence of bounded codes")
    try:
        entries = list(codes)
    except (TypeError, ValueError):
        raise _invalid("remediation_codes must be a sequence of bounded codes")

    normalized = {
        _coerce_enum(
            entry,
            RemediationCode,
            "remediation_codes contain an unsupported code",
        )
        for entry in entries
    }
    if len(normalized) > MAX_REMEDIATION_CODES:
        raise _invalid("remediation_codes exceed the bounded maximum")
    return tuple(sorted(normalized, key=lambda code: code.value))


@dataclass(frozen=True)
class ArtifactFingerprint:
    """A hash-only description of one named local artifact.

    ``name`` is a logical identifier, never a source path.  The class stores
    no artifact bytes and therefore cannot accidentally serialize their
    contents.
    """

    name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not _is_identifier(self.name):
            raise _invalid("artifact names must be lowercase logical identifiers")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise _invalid("artifact sha256 must be lowercase hexadecimal")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise _invalid("artifact size_bytes must be a non-negative integer")

    @classmethod
    def from_bytes(
        cls,
        name: str,
        content: bytes | bytearray | memoryview,
    ) -> "ArtifactFingerprint":
        """Return a fingerprint for in-memory artifact bytes."""

        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise _invalid("artifact content must be bytes")
        try:
            payload = bytes(content)
        except (TypeError, ValueError):
            raise _invalid("artifact content must be bytes")
        return cls(
            name=name,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    @classmethod
    def from_file(cls, name: str, path: str | Path) -> "ArtifactFingerprint":
        """Return a fingerprint for a local file without serializing its path."""

        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
                    digest.update(chunk)
                    size_bytes += len(chunk)
        except (OSError, TypeError, ValueError):
            raise _invalid("artifact file could not be fingerprinted")
        return cls(name=name, sha256=digest.hexdigest(), size_bytes=size_bytes)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactFingerprint":
        """Parse the exact wire representation of a fingerprint."""

        if not isinstance(payload, Mapping) or set(payload) != _ARTIFACT_KEYS:
            raise _invalid("artifact fingerprint has an unsupported shape")
        return cls(
            name=payload["name"],
            sha256=payload["sha256"],
            size_bytes=payload["size_bytes"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the hash-only JSON representation."""

        return {
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ResultEnvelope:
    """Versioned JSON result with only bounded, audit-safe fields."""

    status: ResultStatus | str
    category: ResultCategory | str
    counters: Mapping[str, int] | Iterable[tuple[str, int]] = ()
    artifacts: Iterable[ArtifactFingerprint | Mapping[str, Any]] = ()
    remediation_codes: Iterable[RemediationCode | str] = ()

    def __post_init__(self) -> None:
        status = _coerce_enum(
            self.status,
            ResultStatus,
            "status must be success or failure",
        )
        category = _coerce_enum(
            self.category,
            ResultCategory,
            "category is not supported",
        )
        counters = _normalize_counters(self.counters)
        artifacts = _normalize_artifacts(self.artifacts)
        remediation_codes = _normalize_remediation_codes(self.remediation_codes)

        if status is ResultStatus.SUCCESS and category is not ResultCategory.SUCCESS:
            raise _invalid("successful envelopes must use the success category")
        if status is ResultStatus.FAILURE and category is ResultCategory.SUCCESS:
            raise _invalid("failed envelopes must use a failure category")
        if status is ResultStatus.SUCCESS and remediation_codes:
            raise _invalid("successful envelopes cannot include remediation codes")

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "counters", counters)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "remediation_codes", remediation_codes)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete stable wire representation."""

        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status.value,
            "category": self.category.value,
            "counters": dict(self.counters),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "remediation_codes": [code.value for code in self.remediation_codes],
        }

    def to_json(self) -> str:
        """Return canonical compact JSON independent of locale or terminal size."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def write_json(self, stream: IO[str]) -> None:
        """Write one newline-terminated canonical JSON document."""

        stream.write(self.to_json())
        stream.write("\n")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResultEnvelope":
        """Parse and validate the exact envelope wire representation."""

        if not isinstance(payload, Mapping) or set(payload) != _ENVELOPE_KEYS:
            raise _invalid("result envelope has an unsupported shape")
        if (
            isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], int)
            or payload["schema_version"] != SCHEMA_VERSION
        ):
            raise _invalid("result envelope schema version is unsupported")
        if not isinstance(payload["counters"], Mapping):
            raise _invalid("result envelope counters must be an object")
        if not isinstance(payload["artifacts"], list):
            raise _invalid("result envelope artifacts must be an array")
        if not isinstance(payload["remediation_codes"], list):
            raise _invalid("result envelope remediation_codes must be an array")
        return cls(
            status=payload["status"],
            category=payload["category"],
            counters=payload["counters"],
            artifacts=payload["artifacts"],
            remediation_codes=payload["remediation_codes"],
        )

    @classmethod
    def from_json(cls, document: str) -> "ResultEnvelope":
        """Parse one JSON document without echoing malformed input."""

        if not isinstance(document, str):
            raise _invalid("result envelope JSON must be text")
        try:
            payload = json.loads(document)
        except (TypeError, ValueError):
            raise _invalid("result envelope JSON is invalid")
        return cls.from_dict(payload)


# The CLI-specific name is useful to callers while ResultEnvelope remains the
# concise canonical type name for imports and documentation.
CliResultEnvelope = ResultEnvelope


def create_success_envelope(
    *,
    counters: Mapping[str, int] | Iterable[tuple[str, int]] | None = None,
    artifacts: Iterable[ArtifactFingerprint | Mapping[str, Any]] = (),
) -> ResultEnvelope:
    """Create a successful envelope with deterministic typed fields."""

    return ResultEnvelope(
        status=ResultStatus.SUCCESS,
        category=ResultCategory.SUCCESS,
        counters=counters,
        artifacts=artifacts,
    )


def create_failure_envelope(
    category: ResultCategory | str,
    *,
    counters: Mapping[str, int] | Iterable[tuple[str, int]] | None = None,
    artifacts: Iterable[ArtifactFingerprint | Mapping[str, Any]] = (),
    remediation_codes: Iterable[RemediationCode | str] = (),
) -> ResultEnvelope:
    """Create a failed envelope without accepting a free-text error message."""

    return ResultEnvelope(
        status=ResultStatus.FAILURE,
        category=category,
        counters=counters,
        artifacts=artifacts,
        remediation_codes=remediation_codes,
    )


_EnvelopeT = TypeVar("_EnvelopeT", bound=ResultEnvelope)


def serialize_envelope(envelope: _EnvelopeT) -> str:
    """Serialize an envelope after checking its concrete type."""

    if not isinstance(envelope, ResultEnvelope):
        raise _invalid("value must be a result envelope")
    return envelope.to_json()


__all__ = [
    "ArtifactFingerprint",
    "CliResultEnvelope",
    "MAX_REMEDIATION_CODES",
    "RemediationCode",
    "ResultCategory",
    "ResultEnvelope",
    "ResultEnvelopeError",
    "ResultStatus",
    "SCHEMA_VERSION",
    "create_failure_envelope",
    "create_success_envelope",
    "serialize_envelope",
]
