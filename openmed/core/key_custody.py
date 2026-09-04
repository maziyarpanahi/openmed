"""Validate key custody metadata without handling key material.

The validator accepts only descriptive lifecycle metadata.  It never accepts,
copies, serializes, or logs key bytes.  Validation is deliberately local and
deterministic so signing and surrogate workflows can run it before selecting a
key from their own custody system.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

KEY_CUSTODY_SCHEMA_VERSION = 1

KEY_STATES = ("active", "rotated", "retired", "destroyed")
KEY_PURPOSES = (
    "attestation",
    "audit",
    "authentication",
    "backup",
    "encryption",
    "integrity",
    "key_agreement",
    "key_wrapping",
    "signing",
    "surrogate",
)
KEY_ALGORITHMS = (
    "aes-128-gcm",
    "aes-256-gcm",
    "chacha20-poly1305",
    "ecdsa-p256-sha256",
    "ecdsa-p384-sha384",
    "ed25519",
    "hmac-sha256",
    "hmac-sha512",
    "hkdf-sha256",
    "rsa-pkcs1-sha256",
    "rsa-pss-sha256",
    "rsa-pss-sha384",
    "rsa-pss-sha512",
    "x25519",
)

_ALGORITHM_SET = frozenset(KEY_ALGORITHMS)
_PURPOSE_SET = frozenset(KEY_PURPOSES)
_STATE_SET = frozenset(KEY_STATES)
_KEY_ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9._:/-]{2,127}\Z")
_FIELD_SEPARATOR_PATTERN = re.compile(r"[^a-z0-9]+")
_SECRET_FIELD_PARTS = frozenset(
    {
        "bytes",
        "ciphertext",
        "credential",
        "credentials",
        "der",
        "iv",
        "key",
        "material",
        "nonce",
        "password",
        "pem",
        "plaintext",
        "private",
        "secret",
        "seed",
        "token",
        "value",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "algorithm",
        "created_at",
        "destroyed_at",
        "key_id",
        "purpose",
        "purposes",
        "retired_at",
        "rotated_at",
        "rotation_at",
        "state",
        "status",
        "transitions",
    }
)
_TRANSITION_FIELDS = frozenset({"at", "state", "status", "timestamp"})
_PURPOSE_ALIASES = {
    "attestation": "attestation",
    "audit": "audit",
    "authentication": "authentication",
    "backup": "backup",
    "encryption": "encryption",
    "integrity": "integrity",
    "key-agreement": "key_agreement",
    "key-wrapping": "key_wrapping",
    "release-signing": "signing",
    "signing": "signing",
    "surrogate": "surrogate",
    "surrogate-vault": "surrogate",
}
_ALGORITHM_ALIASES = {
    "aes-128-gcm": "aes-128-gcm",
    "aes-256-gcm": "aes-256-gcm",
    "chacha20-poly1305": "chacha20-poly1305",
    "ecdsa-p256": "ecdsa-p256-sha256",
    "ecdsa-p256-sha256": "ecdsa-p256-sha256",
    "ecdsa-p384": "ecdsa-p384-sha384",
    "ecdsa-p384-sha384": "ecdsa-p384-sha384",
    "ed25519": "ed25519",
    "hmac-sha256": "hmac-sha256",
    "hmac-sha512": "hmac-sha512",
    "hkdf-sha256": "hkdf-sha256",
    "rsa-pkcs1-sha256": "rsa-pkcs1-sha256",
    "rsa-pss-sha256": "rsa-pss-sha256",
    "rsa-pss-sha384": "rsa-pss-sha384",
    "rsa-pss-sha512": "rsa-pss-sha512",
    "x25519": "x25519",
}
_PURPOSE_ALGORITHMS = {
    "attestation": frozenset(
        {
            "ecdsa-p256-sha256",
            "ecdsa-p384-sha384",
            "ed25519",
            "hmac-sha256",
            "hmac-sha512",
            "rsa-pkcs1-sha256",
            "rsa-pss-sha256",
            "rsa-pss-sha384",
            "rsa-pss-sha512",
        }
    ),
    "audit": frozenset(
        {
            "ecdsa-p256-sha256",
            "ecdsa-p384-sha384",
            "ed25519",
            "hmac-sha256",
            "hmac-sha512",
            "rsa-pkcs1-sha256",
            "rsa-pss-sha256",
            "rsa-pss-sha384",
            "rsa-pss-sha512",
        }
    ),
    "authentication": frozenset(
        {
            "ecdsa-p256-sha256",
            "ecdsa-p384-sha384",
            "ed25519",
            "hmac-sha256",
            "hmac-sha512",
            "rsa-pkcs1-sha256",
            "rsa-pss-sha256",
            "rsa-pss-sha384",
            "rsa-pss-sha512",
        }
    ),
    "backup": frozenset({"aes-128-gcm", "aes-256-gcm", "chacha20-poly1305"}),
    "encryption": frozenset({"aes-128-gcm", "aes-256-gcm", "chacha20-poly1305"}),
    "integrity": frozenset({"hmac-sha256", "hmac-sha512"}),
    "key_agreement": frozenset({"x25519", "ecdsa-p256-sha256", "ecdsa-p384-sha384"}),
    "key_wrapping": frozenset(
        {"aes-128-gcm", "aes-256-gcm", "chacha20-poly1305", "hkdf-sha256"}
    ),
    "signing": frozenset(
        {
            "ecdsa-p256-sha256",
            "ecdsa-p384-sha384",
            "ed25519",
            "hmac-sha256",
            "hmac-sha512",
            "rsa-pkcs1-sha256",
            "rsa-pss-sha256",
            "rsa-pss-sha384",
            "rsa-pss-sha512",
        }
    ),
    "surrogate": frozenset(
        {
            "aes-128-gcm",
            "aes-256-gcm",
            "chacha20-poly1305",
            "hkdf-sha256",
            "hmac-sha256",
            "hmac-sha512",
        }
    ),
}
_NEXT_STATES = {
    "active": frozenset({"rotated", "retired"}),
    "rotated": frozenset({"retired"}),
    "retired": frozenset({"destroyed"}),
    "destroyed": frozenset(),
}


@dataclass(frozen=True)
class KeyCustodyMetadata:
    """Descriptive metadata for one key, never the key itself.

    ``created_at`` and lifecycle timestamps must be timezone-aware ISO-8601
    strings or timezone-aware :class:`datetime` values.  ``state`` defaults to
    ``"active"`` for metadata produced by older callers that did not record a
    state explicitly.
    """

    key_id: str
    purpose: str
    algorithm: str
    created_at: datetime | str
    state: str = "active"
    rotated_at: datetime | str | None = None
    retired_at: datetime | str | None = None
    destroyed_at: datetime | str | None = None
    transitions: tuple[Mapping[str, Any], ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        """Return metadata fields for local validation."""

        return {
            "key_id": self.key_id,
            "purpose": self.purpose,
            "algorithm": self.algorithm,
            "created_at": self.created_at,
            "state": self.state,
            "rotated_at": self.rotated_at,
            "retired_at": self.retired_at,
            "destroyed_at": self.destroyed_at,
            "transitions": self.transitions,
        }


@dataclass(frozen=True)
class KeyCustodyViolation:
    """A privacy-safe validation failure identified by code and location."""

    code: str
    record_index: int | None = None
    field: str | None = None

    def __str__(self) -> str:
        """Render a message without including any input value."""

        location = "metadata"
        if self.record_index is not None:
            location = f"record {self.record_index}"
        if self.field:
            location = f"{location}.{self.field}"
        return f"{location}: {self.code}"

    @property
    def message(self) -> str:
        """Return the same safe text as :func:`str`, without input values."""

        return str(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe violation without raw metadata values."""

        payload: dict[str, Any] = {"code": self.code}
        if self.record_index is not None:
            payload["record_index"] = self.record_index
        if self.field is not None:
            payload["field"] = self.field
        return payload


@dataclass(frozen=True)
class KeyCustodyValidationResult:
    """Deterministic, privacy-safe outcome of custody metadata validation."""

    records_checked: int
    violations: tuple[KeyCustodyViolation, ...] = ()
    key_id_digests: tuple[str, ...] = ()
    active_purposes: tuple[str, ...] = ()
    state_counts: tuple[tuple[str, int], ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether every checked record passed validation."""

        return not self.violations

    @property
    def ok(self) -> bool:
        """Alias for :attr:`valid` used by other OpenMed validators."""

        return self.valid

    @property
    def is_valid(self) -> bool:
        """Alias for :attr:`valid` for callers using predicate terminology."""

        return self.valid

    @property
    def passed(self) -> bool:
        """Alias for :attr:`valid` for report-oriented callers."""

        return self.valid

    @property
    def errors(self) -> tuple[KeyCustodyViolation, ...]:
        """Return validation violations without exposing input values."""

        return self.violations

    def to_dict(self) -> dict[str, Any]:
        """Serialize a safe summary suitable for logs or audit artifacts."""

        return {
            "schema_version": KEY_CUSTODY_SCHEMA_VERSION,
            "valid": self.valid,
            "records_checked": self.records_checked,
            "key_id_digests": list(self.key_id_digests),
            "active_purposes": list(self.active_purposes),
            "state_counts": dict(self.state_counts),
            "violations": [violation.to_dict() for violation in self.violations],
        }

    def to_json(self) -> str:
        """Serialize the safe summary deterministically."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def raise_for_errors(self) -> None:
        """Raise a safe exception if any metadata failed validation."""

        if not self.valid:
            raise KeyCustodyValidationError(self)


class KeyCustodyValidationError(ValueError):
    """Raised when metadata is required to be valid but is not."""

    def __init__(self, result: KeyCustodyValidationResult) -> None:
        if result.valid:
            raise ValueError("a valid custody result cannot raise a validation error")
        self.result = result
        super().__init__(
            "key custody metadata validation failed with "
            f"{len(result.violations)} violation(s)"
        )


@dataclass(frozen=True)
class _ParsedMetadata:
    record_index: int
    key_id: str
    purpose: str
    algorithm: str
    state: str


class KeyCustodyValidator:
    """Stateless local validator for key custody metadata."""

    def validate(
        self,
        records: Mapping[str, Any]
        | KeyCustodyMetadata
        | Iterable[Mapping[str, Any] | KeyCustodyMetadata],
    ) -> KeyCustodyValidationResult:
        """Validate one metadata mapping or an iterable of mappings."""

        return validate_key_custody_metadata(records)

    def require_valid(
        self,
        records: Mapping[str, Any]
        | KeyCustodyMetadata
        | Iterable[Mapping[str, Any] | KeyCustodyMetadata],
    ) -> KeyCustodyValidationResult:
        """Validate metadata and raise a safe error when it fails."""

        result = self.validate(records)
        result.raise_for_errors()
        return result


def validate_key_custody_metadata(
    records: Mapping[str, Any]
    | KeyCustodyMetadata
    | Iterable[Mapping[str, Any] | KeyCustodyMetadata],
) -> KeyCustodyValidationResult:
    """Validate descriptive key lifecycle metadata.

    The input may be one mapping or an iterable of mappings.  The function
    accepts no key bytes, byte buffers, private-key fields, or secret-like
    fields.  It returns a safe result instead of echoing invalid values.

    Validation is local and deterministic: no clock, filesystem, environment,
    or network state is consulted.
    """

    if _is_bytes_like(records):
        return KeyCustodyValidationResult(
            records_checked=0,
            violations=(
                KeyCustodyViolation("key_material_rejected", field="metadata"),
            ),
        )

    if isinstance(records, KeyCustodyMetadata) or isinstance(records, Mapping):
        raw_records: Sequence[Any] = (records,)
    elif isinstance(records, Iterable) and not isinstance(records, str):
        raw_records = tuple(records)
    else:
        raise TypeError("key custody metadata must be a mapping or iterable")

    violations: list[KeyCustodyViolation] = []
    parsed_records: list[_ParsedMetadata] = []
    key_id_digests: list[str] = []

    for record_index, raw_record in enumerate(raw_records):
        parsed = _validate_record(raw_record, record_index, violations)
        if parsed is not None:
            parsed_records.append(parsed)
            key_id_digests.append(_digest_key_id(parsed.key_id))

    _validate_cross_record_invariants(parsed_records, violations)

    state_counts = tuple(
        (state, sum(record.state == state for record in parsed_records))
        for state in KEY_STATES
    )
    active_purposes = tuple(
        sorted(
            {record.purpose for record in parsed_records if record.state == "active"}
        )
    )

    return KeyCustodyValidationResult(
        records_checked=len(raw_records),
        violations=tuple(violations),
        key_id_digests=tuple(key_id_digests),
        active_purposes=active_purposes,
        state_counts=state_counts,
    )


def require_valid_key_custody_metadata(
    records: Mapping[str, Any]
    | KeyCustodyMetadata
    | Iterable[Mapping[str, Any] | KeyCustodyMetadata],
) -> KeyCustodyValidationResult:
    """Validate metadata and raise without revealing invalid input values."""

    result = validate_key_custody_metadata(records)
    result.raise_for_errors()
    return result


def _validate_record(
    raw_record: Any,
    record_index: int,
    violations: list[KeyCustodyViolation],
) -> _ParsedMetadata | None:
    if isinstance(raw_record, KeyCustodyMetadata):
        record: Mapping[Any, Any] = raw_record.to_mapping()
    elif isinstance(raw_record, Mapping):
        record = raw_record
    elif _is_bytes_like(raw_record):
        violations.append(
            KeyCustodyViolation("key_material_rejected", record_index, "metadata")
        )
        return None
    else:
        violations.append(KeyCustodyViolation("invalid_record", record_index))
        return None

    _validate_field_names(record, record_index, violations)
    byte_path = _find_bytes_path(record)
    if byte_path is not None:
        violations.append(
            KeyCustodyViolation("key_material_rejected", record_index, byte_path)
        )

    key_id = _read_key_id(record, record_index, violations)
    purpose = _read_purpose(record, record_index, violations)
    algorithm = _read_algorithm(record, record_index, violations)
    created_at = _read_timestamp(
        record, "created_at", record_index, violations, required=True
    )
    rotated_at = _read_alias_timestamp(
        record,
        "rotated_at",
        "rotation_at",
        record_index,
        violations,
    )
    retired_at = _read_timestamp(
        record, "retired_at", record_index, violations, required=False
    )
    destroyed_at = _read_timestamp(
        record, "destroyed_at", record_index, violations, required=False
    )
    state = _read_state(record, record_index, violations)
    transition_events = _read_transitions(record, record_index, violations)

    if transition_events:
        _validate_transition_sequence(
            transition_events,
            state,
            created_at,
            record_index,
            violations,
        )
        event_times = {
            event_state: timestamp for event_state, timestamp in transition_events
        }
        for field, event_state in (
            ("rotated_at", "rotated"),
            ("retired_at", "retired"),
            ("destroyed_at", "destroyed"),
        ):
            explicit_time = {
                "rotated_at": rotated_at,
                "retired_at": retired_at,
                "destroyed_at": destroyed_at,
            }[field]
            event_time = event_times.get(event_state)
            if explicit_time is not None and event_time is not None:
                if explicit_time != event_time:
                    violations.append(
                        KeyCustodyViolation(
                            "transition_timestamp_mismatch",
                            record_index,
                            field,
                        )
                    )
        rotated_at = rotated_at or event_times.get("rotated")
        retired_at = retired_at or event_times.get("retired")
        destroyed_at = destroyed_at or event_times.get("destroyed")

    _validate_lifecycle(
        state=state,
        created_at=created_at,
        rotated_at=rotated_at,
        retired_at=retired_at,
        destroyed_at=destroyed_at,
        record_index=record_index,
        violations=violations,
    )

    if (
        purpose is not None
        and algorithm is not None
        and algorithm not in _PURPOSE_ALGORITHMS[purpose]
    ):
        violations.append(
            KeyCustodyViolation(
                "incompatible_algorithm",
                record_index,
                "algorithm",
            )
        )

    if key_id is None or purpose is None or algorithm is None or state is None:
        return None
    return _ParsedMetadata(record_index, key_id, purpose, algorithm, state)


def _validate_field_names(
    record: Mapping[Any, Any],
    record_index: int,
    violations: list[KeyCustodyViolation],
) -> None:
    for field in record:
        if not isinstance(field, str):
            code = (
                "key_material_rejected"
                if _is_bytes_like(field)
                else "invalid_field_name"
            )
            violations.append(KeyCustodyViolation(code, record_index, "field"))
            continue
        if field in _RECORD_FIELDS:
            continue
        code = (
            "sensitive_field_rejected"
            if _is_secret_field(field)
            else "unsupported_field"
        )
        violations.append(KeyCustodyViolation(code, record_index, field))


def _read_key_id(
    record: Mapping[Any, Any],
    record_index: int,
    violations: list[KeyCustodyViolation],
) -> str | None:
    value = record.get("key_id")
    if "key_id" not in record:
        violations.append(KeyCustodyViolation("missing_field", record_index, "key_id"))
        return None
    if not isinstance(value, str):
        violations.append(KeyCustodyViolation("invalid_type", record_index, "key_id"))
        return None
    if not _KEY_ID_PATTERN.fullmatch(value):
        violations.append(
            KeyCustodyViolation("invalid_identifier", record_index, "key_id")
        )
        return None
    if _looks_like_pem(value) or _looks_like_encoded_material(value):
        violations.append(
            KeyCustodyViolation("key_material_rejected", record_index, "key_id")
        )
        return None
    return value


def _read_purpose(
    record: Mapping[Any, Any],
    record_index: int,
    violations: list[KeyCustodyViolation],
) -> str | None:
    present = [name for name in ("purpose", "purposes") if name in record]
    if not present:
        violations.append(KeyCustodyViolation("missing_field", record_index, "purpose"))
        return None
    if len(present) > 1:
        violations.append(
            KeyCustodyViolation("duplicate_field", record_index, "purpose")
        )
        return None

    field = present[0]
    value = record[field]
    if field == "purposes":
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            violations.append(KeyCustodyViolation("invalid_type", record_index, field))
            return None
        if len(value) != 1:
            violations.append(
                KeyCustodyViolation("overlapping_purpose", record_index, field)
            )
            return None
        value = value[0]
    if not isinstance(value, str):
        violations.append(KeyCustodyViolation("invalid_type", record_index, field))
        return None
    normalized = _normalize_token(value)
    purpose = _PURPOSE_ALIASES.get(normalized)
    if purpose not in _PURPOSE_SET:
        violations.append(KeyCustodyViolation("invalid_purpose", record_index, field))
        return None
    return purpose


def _read_algorithm(
    record: Mapping[Any, Any],
    record_index: int,
    violations: list[KeyCustodyViolation],
) -> str | None:
    if "algorithm" not in record:
        violations.append(
            KeyCustodyViolation("missing_field", record_index, "algorithm")
        )
        return None
    value = record["algorithm"]
    if not isinstance(value, str):
        violations.append(
            KeyCustodyViolation("invalid_type", record_index, "algorithm")
        )
        return None
    algorithm = _ALGORITHM_ALIASES.get(_normalize_token(value))
    if algorithm not in _ALGORITHM_SET:
        violations.append(
            KeyCustodyViolation("invalid_algorithm", record_index, "algorithm")
        )
        return None
    return algorithm


def _read_state(
    record: Mapping[Any, Any],
    record_index: int,
    violations: list[KeyCustodyViolation],
) -> str | None:
    present = [name for name in ("state", "status") if name in record]
    if len(present) > 1:
        violations.append(KeyCustodyViolation("duplicate_field", record_index, "state"))
        return None
    if not present:
        return "active"
    field = present[0]
    value = record[field]
    if not isinstance(value, str):
        violations.append(KeyCustodyViolation("invalid_type", record_index, field))
        return None
    state = _normalize_token(value)
    if state not in _STATE_SET:
        violations.append(KeyCustodyViolation("invalid_state", record_index, field))
        return None
    return state


def _read_alias_timestamp(
    record: Mapping[Any, Any],
    primary: str,
    alias: str,
    record_index: int,
    violations: list[KeyCustodyViolation],
) -> datetime | None:
    if primary in record and alias in record:
        violations.append(KeyCustodyViolation("duplicate_field", record_index, primary))
        return None
    field = primary if primary in record else alias
    if field not in record:
        return None
    if record[field] is None:
        return None
    return _parse_timestamp(record[field], record_index, field, violations)


def _read_timestamp(
    record: Mapping[Any, Any],
    field: str,
    record_index: int,
    violations: list[KeyCustodyViolation],
    *,
    required: bool,
) -> datetime | None:
    if field not in record:
        if required:
            violations.append(KeyCustodyViolation("missing_field", record_index, field))
        return None
    if record[field] is None and not required:
        return None
    return _parse_timestamp(record[field], record_index, field, violations)


def _parse_timestamp(
    value: Any,
    record_index: int,
    field: str,
    violations: list[KeyCustodyViolation],
) -> datetime | None:
    if not isinstance(value, (str, datetime)):
        violations.append(KeyCustodyViolation("invalid_type", record_index, field))
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            violations.append(
                KeyCustodyViolation("invalid_timestamp", record_index, field)
            )
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        violations.append(
            KeyCustodyViolation("timestamp_requires_timezone", record_index, field)
        )
        return None
    return parsed.astimezone(timezone.utc)


def _read_transitions(
    record: Mapping[Any, Any],
    record_index: int,
    violations: list[KeyCustodyViolation],
) -> tuple[tuple[str, datetime], ...]:
    if "transitions" not in record:
        return ()
    value = record["transitions"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        violations.append(
            KeyCustodyViolation("invalid_type", record_index, "transitions")
        )
        return ()

    events: list[tuple[str, datetime]] = []
    for transition_index, raw_transition in enumerate(value):
        field_prefix = f"transitions[{transition_index}]"
        if _is_bytes_like(raw_transition):
            violations.append(
                KeyCustodyViolation("key_material_rejected", record_index, field_prefix)
            )
            continue
        if not isinstance(raw_transition, Mapping):
            violations.append(
                KeyCustodyViolation("invalid_transition", record_index, field_prefix)
            )
            continue
        for field in raw_transition:
            if not isinstance(field, str):
                violations.append(
                    KeyCustodyViolation(
                        "invalid_field_name", record_index, field_prefix
                    )
                )
            elif field not in _TRANSITION_FIELDS:
                code = (
                    "sensitive_field_rejected"
                    if _is_secret_field(field)
                    else "unsupported_field"
                )
                violations.append(
                    KeyCustodyViolation(
                        code,
                        record_index,
                        f"{field_prefix}.{field}",
                    )
                )
        state_fields = [
            field for field in ("state", "status") if field in raw_transition
        ]
        time_fields = [
            field for field in ("at", "timestamp") if field in raw_transition
        ]
        if len(state_fields) != 1 or len(time_fields) != 1:
            violations.append(
                KeyCustodyViolation("invalid_transition", record_index, field_prefix)
            )
            continue
        state_value = raw_transition[state_fields[0]]
        if not isinstance(state_value, str):
            violations.append(
                KeyCustodyViolation(
                    "invalid_type", record_index, f"{field_prefix}.state"
                )
            )
            continue
        state = _normalize_token(state_value)
        if state not in _STATE_SET:
            violations.append(
                KeyCustodyViolation(
                    "invalid_state", record_index, f"{field_prefix}.state"
                )
            )
            continue
        timestamp = _parse_timestamp(
            raw_transition[time_fields[0]],
            record_index,
            f"{field_prefix}.at",
            violations,
        )
        if timestamp is not None:
            events.append((state, timestamp))
    return tuple(events)


def _validate_transition_sequence(
    events: tuple[tuple[str, datetime], ...],
    declared_state: str | None,
    created_at: datetime | None,
    record_index: int,
    violations: list[KeyCustodyViolation],
) -> None:
    previous_state: str | None = None
    previous_time = created_at
    for state, timestamp in events:
        if previous_state is None:
            if state != "active":
                violations.append(
                    KeyCustodyViolation(
                        "invalid_transition", record_index, "transitions"
                    )
                )
        elif state not in _NEXT_STATES[previous_state]:
            violations.append(
                KeyCustodyViolation("invalid_transition", record_index, "transitions")
            )
        if previous_time is not None and timestamp < previous_time:
            violations.append(
                KeyCustodyViolation("invalid_transition", record_index, "transitions")
            )
        previous_state = state
        previous_time = timestamp
    if declared_state is not None and previous_state != declared_state:
        violations.append(
            KeyCustodyViolation("state_transition_mismatch", record_index, "state")
        )


def _validate_lifecycle(
    *,
    state: str | None,
    created_at: datetime | None,
    rotated_at: datetime | None,
    retired_at: datetime | None,
    destroyed_at: datetime | None,
    record_index: int,
    violations: list[KeyCustodyViolation],
) -> None:
    if state is None:
        return

    lifecycle = (
        ("rotated_at", rotated_at),
        ("retired_at", retired_at),
        ("destroyed_at", destroyed_at),
    )
    previous = created_at
    for field, timestamp in lifecycle:
        if timestamp is None:
            continue
        if previous is not None and timestamp <= previous:
            violations.append(
                KeyCustodyViolation("invalid_transition", record_index, field)
            )
        previous = timestamp

    if state == "active":
        if any(timestamp is not None for _, timestamp in lifecycle):
            violations.append(
                KeyCustodyViolation("invalid_transition", record_index, "state")
            )
    elif state == "rotated":
        if rotated_at is None:
            violations.append(
                KeyCustodyViolation("missing_transition", record_index, "rotated_at")
            )
        if retired_at is not None or destroyed_at is not None:
            violations.append(
                KeyCustodyViolation("invalid_transition", record_index, "state")
            )
    elif state == "retired":
        if retired_at is None:
            violations.append(
                KeyCustodyViolation("missing_transition", record_index, "retired_at")
            )
        if destroyed_at is not None:
            violations.append(
                KeyCustodyViolation("invalid_transition", record_index, "state")
            )
    elif state == "destroyed":
        if retired_at is None:
            violations.append(
                KeyCustodyViolation("invalid_transition", record_index, "retired_at")
            )
        if destroyed_at is None:
            violations.append(
                KeyCustodyViolation("missing_transition", record_index, "destroyed_at")
            )


def _validate_cross_record_invariants(
    records: Sequence[_ParsedMetadata],
    violations: list[KeyCustodyViolation],
) -> None:
    seen_ids: dict[str, int] = {}
    active_purposes: dict[str, int] = {}
    for record in records:
        record_index = record.record_index
        if record.key_id in seen_ids:
            violations.append(
                KeyCustodyViolation("duplicate_key_id", record_index, "key_id")
            )
        else:
            seen_ids[record.key_id] = record_index
        if record.state == "active":
            if record.purpose in active_purposes:
                violations.append(
                    KeyCustodyViolation(
                        "overlapping_purpose",
                        record_index,
                        "purpose",
                    )
                )
            else:
                active_purposes[record.purpose] = record_index


def _normalize_token(value: str) -> str:
    return value.strip().casefold().replace("_", "-").replace(" ", "-")


def _is_secret_field(field: str) -> bool:
    parts = [part for part in _FIELD_SEPARATOR_PATTERN.split(field.casefold()) if part]
    return bool(set(parts) & _SECRET_FIELD_PARTS) or any(
        marker in "".join(parts)
        for marker in ("privatekey", "secretkey", "keymaterial", "rawkey")
    )


def _is_bytes_like(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview))


def _find_bytes_path(value: Any, path: str = "metadata") -> str | None:
    if _is_bytes_like(value):
        return path
    if isinstance(value, Mapping):
        for field, nested in value.items():
            field_name = field if isinstance(field, str) else "field"
            nested_path = _find_bytes_path(nested, f"{path}.{field_name}")
            if nested_path is not None:
                return nested_path.removeprefix("metadata.")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            nested_path = _find_bytes_path(nested, f"{path}[{index}]")
            if nested_path is not None:
                return nested_path.removeprefix("metadata.")
    return None


def _looks_like_pem(value: str) -> bool:
    return "-----begin " in value.casefold() or "-----end " in value.casefold()


def _looks_like_encoded_material(value: str) -> bool:
    if len(value) < 96:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9+/=_-]+", value))


def _digest_key_id(key_id: str) -> str:
    digest = hashlib.sha256(
        ("openmed-key-custody-id-v1\x00" + key_id).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


KeyCustodyRecord = KeyCustodyMetadata
KeyCustodyReport = KeyCustodyValidationResult
validate_key_custody = validate_key_custody_metadata
require_valid_key_custody = require_valid_key_custody_metadata


__all__ = [
    "KEY_ALGORITHMS",
    "KEY_CUSTODY_SCHEMA_VERSION",
    "KEY_PURPOSES",
    "KEY_STATES",
    "KeyCustodyMetadata",
    "KeyCustodyRecord",
    "KeyCustodyReport",
    "KeyCustodyValidationError",
    "KeyCustodyValidationResult",
    "KeyCustodyValidator",
    "KeyCustodyViolation",
    "require_valid_key_custody",
    "require_valid_key_custody_metadata",
    "validate_key_custody",
    "validate_key_custody_metadata",
]
