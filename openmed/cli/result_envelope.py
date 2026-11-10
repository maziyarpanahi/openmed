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
import os
import re
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import IO, Any, TypeVar

SCHEMA_VERSION = 1
MAX_REMEDIATION_CODES = 3
MAX_COUNTERS = 128
MAX_ARTIFACTS = 64
MAX_JSON_CHARS = 1_048_576
MAX_SAFE_INTEGER = (1 << 53) - 1

_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9_.-]{0,63})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HASH_CHUNK_SIZE = 1024 * 1024
_MAX_ENUM_VALUE_LENGTH = 64
_MAX_WIRE_KEY_LENGTH = 64

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
    return (
        type(value) is str
        and 0 < len(value) <= _MAX_WIRE_KEY_LENGTH
        and _IDENTIFIER.fullmatch(value) is not None
    )


_ValueT = TypeVar("_ValueT")
_EnumT = TypeVar("_EnumT", bound=Enum)


def _coerce_enum(value: Any, enum_type: type[_EnumT], message: str) -> _EnumT:
    if type(value) is enum_type:
        return value
    if type(value) is str and len(value) <= _MAX_ENUM_VALUE_LENGTH:
        try:
            return enum_type(value)
        except ValueError:
            pass
    raise _invalid(message)


def _bounded_values(
    values: Iterable[_ValueT],
    *,
    maximum: int,
    message: str,
) -> tuple[_ValueT, ...]:
    """Copy at most ``maximum`` values without relaying iterator failures."""

    try:
        iterator = iter(values)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise _invalid(message) from None

    copied: list[_ValueT] = []
    for index in range(maximum + 1):
        try:
            value = next(iterator)
        except StopIteration:
            return tuple(copied)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise _invalid(message) from None
        if index == maximum:
            raise _invalid(message)
        copied.append(value)
    raise AssertionError("bounded iteration did not terminate")


def _copy_wire_mapping(
    payload: Mapping[str, Any],
    expected_keys: frozenset[str],
    message: str,
) -> dict[str, Any]:
    """Copy an exact wire object while containing hostile mapping hooks."""

    if not isinstance(payload, Mapping):
        raise _invalid(message)
    keys = _bounded_values(payload, maximum=len(expected_keys), message=message)
    if (
        any(type(key) is not str or len(key) > _MAX_WIRE_KEY_LENGTH for key in keys)
        or len(set(keys)) != len(keys)
        or frozenset(keys) != expected_keys
    ):
        raise _invalid(message)
    try:
        return {key: payload[key] for key in expected_keys}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise _invalid(message) from None


def _normalize_counters(
    counters: Mapping[str, int] | Iterable[tuple[str, int]] | None,
) -> tuple[tuple[str, int], ...]:
    if counters is None:
        return ()
    source: Iterable[tuple[str, int]]
    if isinstance(counters, Mapping):
        try:
            source = counters.items()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise _invalid("counters must be a bounded mapping of integers") from None
    else:
        source = counters
    entries = _bounded_values(
        source,
        maximum=MAX_COUNTERS,
        message="counters must be a bounded mapping of integers",
    )

    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for entry in entries:
        if (
            type(entry) not in {tuple, list}
            or len(entry) != 2
            or not _is_identifier(entry[0])
            or entry[0] in seen
            or type(entry[1]) is not int
            or entry[1] < 0
            or entry[1] > MAX_SAFE_INTEGER
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
    entries = _bounded_values(
        artifacts,
        maximum=MAX_ARTIFACTS,
        message="artifacts must be a bounded sequence of fingerprints",
    )

    normalized: list[ArtifactFingerprint] = []
    seen: set[str] = set()
    for entry in entries:
        if type(entry) is ArtifactFingerprint:
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
    entries = _bounded_values(
        codes,
        maximum=MAX_REMEDIATION_CODES,
        message="remediation_codes must be a sequence of bounded codes",
    )

    normalized = {
        _coerce_enum(
            entry,
            RemediationCode,
            "remediation_codes contain an unsupported code",
        )
        for entry in entries
    }
    return tuple(sorted(normalized, key=lambda code: code.value))


@dataclass(frozen=True, slots=True)
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
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or _SHA256.fullmatch(self.sha256) is None
        ):
            raise _invalid("artifact sha256 must be lowercase hexadecimal")
        if (
            type(self.size_bytes) is not int
            or self.size_bytes < 0
            or self.size_bytes > MAX_SAFE_INTEGER
        ):
            raise _invalid("artifact size_bytes must be a bounded non-negative integer")

    @classmethod
    def from_bytes(
        cls,
        name: str,
        content: bytes | bytearray | memoryview,
    ) -> "ArtifactFingerprint":
        """Return a fingerprint for in-memory artifact bytes."""

        if not _is_identifier(name):
            raise _invalid("artifact names must be lowercase logical identifiers")
        if type(content) not in {bytes, bytearray, memoryview}:
            raise _invalid("artifact content must be bytes")
        try:
            payload = bytes(content)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise _invalid("artifact content must be bytes") from None
        return cls(
            name=name,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    @classmethod
    def from_file(cls, name: str, path: str | Path) -> "ArtifactFingerprint":
        """Return a fingerprint for a local file without serializing its path."""

        if not _is_identifier(name):
            raise _invalid("artifact names must be lowercase logical identifiers")
        descriptor: int | None = None
        verification_descriptor: int | None = None
        digest = hashlib.sha256()
        size_bytes = 0
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            artifact_path = Path(path)
            initial_path_stat = os.stat(artifact_path, follow_symlinks=False)
            if (
                not stat.S_ISREG(initial_path_stat.st_mode)
                or initial_path_stat.st_size < 0
                or initial_path_stat.st_size > MAX_SAFE_INTEGER
            ):
                raise OSError
            descriptor = os.open(os.fspath(artifact_path), flags)
            opened_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened_stat.st_mode)
                or opened_stat.st_size < 0
                or opened_stat.st_size > MAX_SAFE_INTEGER
                or not os.path.samestat(initial_path_stat, opened_stat)
                or _portable_stat_state(initial_path_stat)
                != _portable_stat_state(opened_stat)
            ):
                raise OSError

            with os.fdopen(descriptor, "rb") as handle:
                descriptor = None
                while size_bytes <= opened_stat.st_size:
                    remaining = opened_stat.st_size - size_bytes + 1
                    chunk = handle.read(min(_HASH_CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    digest.update(chunk)
                    size_bytes += len(chunk)
                final_descriptor_stat = os.fstat(handle.fileno())

            verification_descriptor = os.open(os.fspath(artifact_path), flags)
            verification_stat = os.fstat(verification_descriptor)
            final_path_stat = os.stat(artifact_path, follow_symlinks=False)
            if (
                not stat.S_ISREG(verification_stat.st_mode)
                or not stat.S_ISREG(final_path_stat.st_mode)
                or not os.path.samestat(opened_stat, verification_stat)
                or not os.path.samestat(verification_stat, final_path_stat)
                or _stable_stat_state(final_descriptor_stat)
                != _stable_stat_state(opened_stat)
                or _stable_stat_state(verification_stat)
                != _stable_stat_state(opened_stat)
                or _portable_stat_state(final_path_stat)
                != _portable_stat_state(opened_stat)
                or size_bytes != opened_stat.st_size
            ):
                raise OSError
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise _invalid("artifact file could not be fingerprinted") from None
        finally:
            for open_descriptor in (descriptor, verification_descriptor):
                if open_descriptor is not None:
                    try:
                        os.close(open_descriptor)
                    except OSError:
                        pass
        return cls(name=name, sha256=digest.hexdigest(), size_bytes=size_bytes)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactFingerprint":
        """Parse the exact wire representation of a fingerprint."""

        values = _copy_wire_mapping(
            payload,
            _ARTIFACT_KEYS,
            "artifact fingerprint has an unsupported shape",
        )
        return cls(
            name=values["name"],
            sha256=values["sha256"],
            size_bytes=values["size_bytes"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the hash-only JSON representation."""

        return {
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True, init=False)
class ResultEnvelope:
    """Versioned JSON result with only bounded, audit-safe fields."""

    status: ResultStatus
    category: ResultCategory
    counters: tuple[tuple[str, int], ...]
    artifacts: tuple[ArtifactFingerprint, ...]
    remediation_codes: tuple[RemediationCode, ...]

    def __init__(
        self,
        status: ResultStatus | str,
        category: ResultCategory | str,
        counters: Mapping[str, int] | Iterable[tuple[str, int]] | None = None,
        artifacts: (Iterable[ArtifactFingerprint | Mapping[str, Any]] | None) = None,
        remediation_codes: Iterable[RemediationCode | str] | None = None,
    ) -> None:
        status = _coerce_enum(
            status,
            ResultStatus,
            "status must be success or failure",
        )
        category = _coerce_enum(
            category,
            ResultCategory,
            "category is not supported",
        )
        normalized_counters = _normalize_counters(counters)
        normalized_artifacts = _normalize_artifacts(artifacts)
        normalized_codes = _normalize_remediation_codes(remediation_codes)

        if status is ResultStatus.SUCCESS and category is not ResultCategory.SUCCESS:
            raise _invalid("successful envelopes must use the success category")
        if status is ResultStatus.FAILURE and category is ResultCategory.SUCCESS:
            raise _invalid("failed envelopes must use a failure category")
        if status is ResultStatus.SUCCESS and normalized_codes:
            raise _invalid("successful envelopes cannot include remediation codes")

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "counters", normalized_counters)
        object.__setattr__(self, "artifacts", normalized_artifacts)
        object.__setattr__(self, "remediation_codes", normalized_codes)

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

        values = _copy_wire_mapping(
            payload,
            _ENVELOPE_KEYS,
            "result envelope has an unsupported shape",
        )
        if (
            type(values["schema_version"]) is not int
            or values["schema_version"] != SCHEMA_VERSION
        ):
            raise _invalid("result envelope schema version is unsupported")
        if not isinstance(values["counters"], Mapping):
            raise _invalid("result envelope counters must be an object")
        if type(values["artifacts"]) is not list:
            raise _invalid("result envelope artifacts must be an array")
        if type(values["remediation_codes"]) is not list:
            raise _invalid("result envelope remediation_codes must be an array")
        return cls(
            status=values["status"],
            category=values["category"],
            counters=values["counters"],
            artifacts=values["artifacts"],
            remediation_codes=values["remediation_codes"],
        )

    @classmethod
    def from_json(cls, document: str) -> "ResultEnvelope":
        """Parse one JSON document without echoing malformed input."""

        if type(document) is not str or len(document) > MAX_JSON_CHARS:
            raise _invalid("result envelope JSON must be text")
        try:
            payload = json.loads(
                document,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise _invalid("result envelope JSON is invalid") from None
        return cls.from_dict(payload)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys instead of silently taking the last."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    """Reject non-standard NaN and infinity constants."""

    del value
    raise ValueError("non-standard JSON constant")


def _stable_stat_state(source_stat: os.stat_result) -> tuple[int, ...]:
    """Return descriptor fields that must remain stable while hashing."""

    return (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
        source_stat.st_mode,
    )


def _portable_stat_state(source_stat: os.stat_result) -> tuple[int, int, int]:
    """Return fields consistent between path and descriptor stats."""

    return (
        stat.S_IFMT(source_stat.st_mode),
        source_stat.st_size,
        source_stat.st_mtime_ns,
    )


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


def serialize_envelope(envelope: ResultEnvelope) -> str:
    """Serialize an envelope after checking its concrete type."""

    if type(envelope) is not ResultEnvelope:
        raise _invalid("value must be a result envelope")
    return envelope.to_json()


__all__ = [
    "ArtifactFingerprint",
    "CliResultEnvelope",
    "MAX_ARTIFACTS",
    "MAX_COUNTERS",
    "MAX_JSON_CHARS",
    "MAX_REMEDIATION_CODES",
    "MAX_SAFE_INTEGER",
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
