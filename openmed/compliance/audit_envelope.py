"""Parse signed audit-envelope metadata without retaining payload content.

Audit envelopes are transport records, not compliance certifications.  This
module accepts a small, closed JSON shape and keeps only the header, signature
metadata, payload size/type, and a canonical SHA-256 fingerprint.  Payload
values are inspected transiently to validate their fingerprint and bounds;
they are never copied into the parsed object, reports, or exceptions.

The parser performs structural signature validation only.  A caller that owns
the relevant key can verify the signature over the canonical envelope bytes in
its own trust boundary.  No key lookup, filesystem access, clock read, or
network call is performed here.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final

AUDIT_ENVELOPE_SCHEMA_VERSION: Final = 1
"""The supported audit-envelope schema version."""

AUDIT_ENVELOPE_SCHEMA: Final = "openmed.compliance.audit_envelope.v1"
"""Stable schema identifier for audit-envelope metadata."""

AUDIT_ENVELOPE_REPORT_TYPE: Final = "audit_envelope_metadata"
"""Report type emitted by :meth:`AuditEnvelope.to_report`."""

AUDIT_ENVELOPE_MAX_BYTES: Final = 64 * 1024
"""Maximum encoded envelope size accepted by the parser."""

AUDIT_ENVELOPE_MAX_PAYLOAD_BYTES: Final = 8 * 1024 * 1024
"""Maximum canonical payload size inspected transiently by the parser."""

AUDIT_ENVELOPE_MAX_HEADER_FIELDS: Final = 12
AUDIT_ENVELOPE_MAX_HEADER_VALUE_LENGTH: Final = 128
AUDIT_ENVELOPE_MAX_SIGNATURE_LENGTH: Final = 4096
AUDIT_ENVELOPE_MAX_JSON_DEPTH: Final = 32
AUDIT_ENVELOPE_MAX_JSON_ITEMS: Final = 4096

# Discoverable aliases used by callers that prefer shorter bound names.
MAX_ENVELOPE_BYTES: Final = AUDIT_ENVELOPE_MAX_BYTES
MAX_PAYLOAD_BYTES: Final = AUDIT_ENVELOPE_MAX_PAYLOAD_BYTES
MAX_HEADER_FIELDS: Final = AUDIT_ENVELOPE_MAX_HEADER_FIELDS
MAX_HEADER_VALUE_LENGTH: Final = AUDIT_ENVELOPE_MAX_HEADER_VALUE_LENGTH
MAX_SIGNATURE_LENGTH: Final = AUDIT_ENVELOPE_MAX_SIGNATURE_LENGTH

_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")
_SIGNATURE_VALUE_RE: Final = re.compile(r"^[A-Za-z0-9._~:/+=-]{1,4096}$")
_CONTENT_TYPE_RE: Final = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/"
    r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$"
)
_UTC: Final = timezone.utc
_MISSING: Final = object()
_NO_DEFAULT: Final = object()

_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "version",
        "report_type",
        "type",
        "header",
        "headers",
        "metadata",
        "signature",
        "signature_algorithm",
        "signature_key_id",
        "signature_value",
        "payload",
        "payload_fingerprint",
        "payload_hash",
        "payload_digest",
        "payload_size",
        "payload_type",
        # These fields are accepted at the top level as a convenience for
        # line-oriented envelope producers that flatten their header.
        "envelope_id",
        "id",
        "producer",
        "source",
        "created_at",
        "issued_at",
        "timestamp",
        "content_type",
        "format",
    }
)
_HEADER_FIELDS: Final = frozenset(
    {
        "schema_version",
        "version",
        "envelope_id",
        "id",
        "producer",
        "source",
        "created_at",
        "issued_at",
        "timestamp",
        "content_type",
        "format",
        "signature",
        "signature_algorithm",
        "signature_key_id",
        "signature_value",
        "payload_fingerprint",
        "payload_hash",
        "payload_digest",
        "payload_size",
        "payload_type",
    }
)
_SIGNATURE_FIELDS: Final = frozenset(
    {"algorithm", "key_id", "value", "signature", "signature_value"}
)
_PAYLOAD_TYPES: Final = frozenset(
    {"object", "array", "string", "number", "boolean", "null", "omitted"}
)


class _DuplicateJsonKey(ValueError):
    """Internal marker for a duplicate JSON object key."""


class _InvalidJsonConstant(ValueError):
    """Internal marker for NaN and Infinity in JSON input."""


class AuditEnvelopeError(ValueError):
    """Base class for safe, value-free audit-envelope failures."""

    error_code: Final = "invalid_envelope"

    def __init__(
        self,
        message: str = "audit envelope is invalid",
        *,
        code: str | None = None,
        field_name: str | None = None,
    ) -> None:
        # Messages are authored by this module and never interpolate caller
        # values.  Keeping the guard here protects future call sites too.
        safe_message = (
            message if isinstance(message, str) else "audit envelope is invalid"
        )
        super().__init__(safe_message)
        self.code = code or self.error_code
        self.field_name = field_name

    def to_dict(self) -> dict[str, Any]:
        """Return a redacted diagnostic containing no rejected values."""

        result: dict[str, Any] = {
            "error": self.error_code,
            "code": self.code,
            "redacted": True,
        }
        if self.field_name is not None:
            result["field"] = self.field_name
        return result


class AuditEnvelopeParseError(AuditEnvelopeError):
    """Raised when an envelope cannot be decoded or is not an object."""

    error_code: Final = "malformed_envelope"


class AuditEnvelopeValidationError(AuditEnvelopeError):
    """Raised when decoded values violate the closed envelope schema."""

    error_code: Final = "invalid_envelope"


class AuditEnvelopeBoundsError(AuditEnvelopeValidationError):
    """Raised when an envelope, header, payload, or signature exceeds bounds."""

    error_code: Final = "envelope_bounds_exceeded"


class AuditEnvelopeSignatureError(AuditEnvelopeValidationError):
    """Raised when signature metadata is absent or malformed."""

    error_code: Final = "invalid_signature"


class AuditEnvelopeUnsignedError(AuditEnvelopeSignatureError):
    """Raised when an envelope does not carry a non-empty signature."""

    error_code: Final = "unsigned_envelope"


# Short compatibility aliases keep the public error family easy to discover.
AuditEnvelopeMalformedError = AuditEnvelopeParseError
AuditEnvelopeBoundError = AuditEnvelopeBoundsError


def _raise(
    error_type: type[AuditEnvelopeError],
    message: str,
    *,
    field_name: str | None = None,
) -> None:
    raise error_type(message, field_name=field_name) from None


def _require_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope field must be an object",
            field_name=field_name,
        )
    result = dict(value)
    if not all(isinstance(key, str) for key in result):
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope object keys must be text",
            field_name=field_name,
        )
    return result


def _require_allowed_keys(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    field_name: str,
    maximum: int | None = None,
) -> None:
    if maximum is not None and len(value) > maximum:
        _raise(
            AuditEnvelopeBoundsError,
            "audit envelope field has too many entries",
            field_name=field_name,
        )
    if not set(value).issubset(allowed):
        # Do not include an untrusted key in the exception; it may itself be
        # a sensitive identifier.
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope contains an unsupported field",
            field_name=field_name,
        )


def _require_text(
    value: Any,
    *,
    field_name: str,
    maximum_length: int,
    pattern: re.Pattern[str] | None = None,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope metadata must be text",
            field_name=field_name,
        )
    if len(value) > maximum_length:
        _raise(
            AuditEnvelopeBoundsError,
            "audit envelope metadata exceeds its bound",
            field_name=field_name,
        )
    if not allow_empty and (not value or value != value.strip()):
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope metadata is not canonical",
            field_name=field_name,
        )
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for character in value
    ):
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope metadata contains unsupported characters",
            field_name=field_name,
        )
    if pattern is not None and pattern.fullmatch(value) is None:
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope metadata is not a safe identifier",
            field_name=field_name,
        )
    return value


def _require_identifier(value: Any, *, field_name: str) -> str:
    return _require_text(
        value,
        field_name=field_name,
        maximum_length=AUDIT_ENVELOPE_MAX_HEADER_VALUE_LENGTH,
        pattern=_IDENTIFIER_RE,
    )


def _require_digest(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope fingerprint must be a SHA-256 digest",
            field_name=field_name,
        )
    return value


def _require_non_negative_int(
    value: Any,
    *,
    field_name: str,
    maximum: int,
) -> int:
    if type(value) is not int or value < 0:
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope count must be a non-negative integer",
            field_name=field_name,
        )
    if value > maximum:
        _raise(
            AuditEnvelopeBoundsError,
            "audit envelope count exceeds its bound",
            field_name=field_name,
        )
    return value


def _require_schema_version(value: Any, *, field_name: str) -> int:
    # Numeric JSON 1.0 is deliberately rejected.  String aliases are accepted
    # only for interoperability with line-oriented producers and normalize to
    # the one supported integer version.
    if type(value) is int and value == AUDIT_ENVELOPE_SCHEMA_VERSION:
        return value
    if isinstance(value, str) and value in {
        "1",
        "audit-envelope.v1",
        AUDIT_ENVELOPE_SCHEMA,
    }:
        return AUDIT_ENVELOPE_SCHEMA_VERSION
    _raise(
        AuditEnvelopeValidationError,
        "audit envelope schema version is unsupported",
        field_name=field_name,
    )


def _canonical_json_bytes(value: Any, *, field_name: str) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, UnicodeError, ValueError, OverflowError):
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope payload is not canonical JSON",
            field_name=field_name,
        )
    return encoded


def fingerprint_payload(
    payload: Any,
    *,
    max_bytes: int = AUDIT_ENVELOPE_MAX_PAYLOAD_BYTES,
) -> str:
    """Return the canonical SHA-256 fingerprint of a JSON payload.

    The payload is serialized only for hashing.  The returned digest is safe
    to place in logs and audit records; the payload itself is never returned.
    """

    _validate_bound(
        max_bytes, maximum=AUDIT_ENVELOPE_MAX_PAYLOAD_BYTES, field_name="max_bytes"
    )
    encoded = _canonical_json_bytes(payload, field_name="payload")
    if len(encoded) > max_bytes:
        _raise(
            AuditEnvelopeBoundsError,
            "audit envelope payload exceeds its bound",
            field_name="payload",
        )
    return _sha256_bytes(encoded)


# Common naming aliases for callers that use hash/digest terminology.
payload_fingerprint = fingerprint_payload
compute_payload_fingerprint = fingerprint_payload


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _validate_bound(value: Any, *, maximum: int, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        _raise(
            AuditEnvelopeValidationError,
            "parser bound must be a positive integer",
            field_name=field_name,
        )
    if value > maximum:
        _raise(
            AuditEnvelopeBoundsError,
            "parser bound exceeds the supported maximum",
            field_name=field_name,
        )
    return value


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise _InvalidJsonConstant


def _check_json_shape(
    value: Any, *, depth: int = 0, items: list[int] | None = None
) -> None:
    if items is None:
        items = [0]
    if depth > AUDIT_ENVELOPE_MAX_JSON_DEPTH:
        _raise(
            AuditEnvelopeBoundsError,
            "audit envelope JSON is too deeply nested",
            field_name="envelope",
        )
    if isinstance(value, Mapping):
        items[0] += len(value)
        if items[0] > AUDIT_ENVELOPE_MAX_JSON_ITEMS:
            _raise(
                AuditEnvelopeBoundsError,
                "audit envelope JSON has too many entries",
                field_name="envelope",
            )
        for nested in value.values():
            _check_json_shape(nested, depth=depth + 1, items=items)
    elif isinstance(value, list):
        items[0] += len(value)
        if items[0] > AUDIT_ENVELOPE_MAX_JSON_ITEMS:
            _raise(
                AuditEnvelopeBoundsError,
                "audit envelope JSON has too many entries",
                field_name="envelope",
            )
        for nested in value:
            _check_json_shape(nested, depth=depth + 1, items=items)


def _load_document(
    value: Any,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        document = dict(value)
        if not all(isinstance(key, str) for key in document):
            _raise(
                AuditEnvelopeParseError,
                "audit envelope object keys must be text",
                field_name="envelope",
            )
        encoded = _canonical_json_bytes(document, field_name="envelope")
    elif isinstance(value, str):
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError:
            _raise(
                AuditEnvelopeParseError,
                "audit envelope text is not valid UTF-8",
                field_name="envelope",
            )
        if len(encoded) > max_bytes:
            _raise(
                AuditEnvelopeBoundsError,
                "audit envelope exceeds its byte bound",
                field_name="envelope",
            )
        try:
            document = json.loads(
                value,
                object_pairs_hook=_json_object_without_duplicates,
                parse_constant=_reject_json_constant,
            )
        except (
            json.JSONDecodeError,
            UnicodeError,
            _DuplicateJsonKey,
            _InvalidJsonConstant,
        ):
            _raise(
                AuditEnvelopeParseError,
                "audit envelope JSON is malformed",
                field_name="envelope",
            )
    elif isinstance(value, bytes):
        encoded = value
        if len(encoded) > max_bytes:
            _raise(
                AuditEnvelopeBoundsError,
                "audit envelope exceeds its byte bound",
                field_name="envelope",
            )
        try:
            document = json.loads(
                value,
                object_pairs_hook=_json_object_without_duplicates,
                parse_constant=_reject_json_constant,
            )
        except (
            json.JSONDecodeError,
            UnicodeError,
            _DuplicateJsonKey,
            _InvalidJsonConstant,
        ):
            _raise(
                AuditEnvelopeParseError,
                "audit envelope JSON is malformed",
                field_name="envelope",
            )
    else:
        _raise(
            AuditEnvelopeParseError,
            "audit envelope must be JSON text or an object",
            field_name="envelope",
        )

    if len(encoded) > max_bytes:
        _raise(
            AuditEnvelopeBoundsError,
            "audit envelope exceeds its byte bound",
            field_name="envelope",
        )
    if not isinstance(document, Mapping):
        _raise(
            AuditEnvelopeParseError,
            "audit envelope JSON must contain an object",
            field_name="envelope",
        )
    _check_json_shape(document)
    return dict(document)


def _values_equal(left: Any, right: Any) -> bool:
    try:
        return bool(left == right)
    except Exception:
        return False


def _coalesce(
    source: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    field_name: str,
    default: Any = _NO_DEFAULT,
) -> Any:
    values = [source[name] for name in names if name in source]
    if not values:
        if default is _NO_DEFAULT:
            _raise(
                AuditEnvelopeValidationError,
                "audit envelope field is missing",
                field_name=field_name,
            )
        return default
    first = values[0]
    if any(not _values_equal(first, value) for value in values[1:]):
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope field aliases conflict",
            field_name=field_name,
        )
    return first


def _field_from_header_or_top(
    document: Mapping[str, Any],
    header: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    field_name: str,
    default: Any = _NO_DEFAULT,
) -> Any:
    top_value = _coalesce(document, names, field_name=field_name, default=_MISSING)
    header_value = _coalesce(header, names, field_name=field_name, default=_MISSING)
    if top_value is not _MISSING and header_value is not _MISSING:
        if not _values_equal(top_value, header_value):
            _raise(
                AuditEnvelopeValidationError,
                "audit envelope header aliases conflict",
                field_name=field_name,
            )
        return top_value
    if top_value is not _MISSING:
        return top_value
    if header_value is not _MISSING:
        return header_value
    if default is not _NO_DEFAULT:
        return default
    _raise(
        AuditEnvelopeValidationError,
        "audit envelope field is missing",
        field_name=field_name,
    )


def _parse_header(document: Mapping[str, Any]) -> "AuditEnvelopeHeader":
    header_values: dict[str, Any] = {}
    for name in ("header", "headers", "metadata"):
        if name not in document:
            continue
        candidate = _require_mapping(document[name], field_name="header")
        if header_values and not _values_equal(header_values, candidate):
            _raise(
                AuditEnvelopeValidationError,
                "audit envelope header aliases conflict",
                field_name="header",
            )
        header_values = candidate

    _require_allowed_keys(
        header_values,
        _HEADER_FIELDS,
        field_name="header",
        maximum=AUDIT_ENVELOPE_MAX_HEADER_FIELDS,
    )

    for name in _HEADER_FIELDS:
        if name in document and name not in header_values:
            header_values[name] = document[name]

    version = _field_from_header_or_top(
        document,
        header_values,
        ("schema_version", "version"),
        field_name="schema_version",
    )
    parsed_version = _require_schema_version(version, field_name="schema_version")

    envelope_id = _field_from_header_or_top(
        document,
        header_values,
        ("envelope_id", "id"),
        field_name="envelope_id",
        default="unspecified",
    )
    producer = _field_from_header_or_top(
        document,
        header_values,
        ("producer", "source"),
        field_name="producer",
        default="unspecified",
    )
    created_at = _field_from_header_or_top(
        document,
        header_values,
        ("created_at", "issued_at", "timestamp"),
        field_name="created_at",
        default=None,
    )
    content_type = _field_from_header_or_top(
        document,
        header_values,
        ("content_type", "format"),
        field_name="content_type",
        default="application/json",
    )

    return AuditEnvelopeHeader(
        schema_version=parsed_version,
        envelope_id=_require_identifier(envelope_id, field_name="envelope_id"),
        producer=_require_identifier(producer, field_name="producer"),
        created_at=_parse_timestamp(created_at, field_name="created_at"),
        content_type=_require_content_type(content_type),
    )


def _require_content_type(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) > AUDIT_ENVELOPE_MAX_HEADER_VALUE_LENGTH
    ):
        _raise(
            AuditEnvelopeBoundsError,
            "audit envelope content type exceeds its bound",
            field_name="content_type",
        )
    if _CONTENT_TYPE_RE.fullmatch(value) is None:
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope content type is invalid",
            field_name="content_type",
        )
    return value


def _parse_timestamp(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 40 or not value.endswith("Z"):
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope timestamp is invalid",
            field_name=field_name,
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope timestamp is invalid",
            field_name=field_name,
        )
    if parsed.tzinfo is None or parsed.utcoffset() != _UTC.utcoffset(None):
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope timestamp is invalid",
            field_name=field_name,
        )
    canonical = parsed.astimezone(_UTC)
    timespec = "microseconds" if canonical.microsecond else "seconds"
    expected = canonical.isoformat(timespec=timespec).replace("+00:00", "Z")
    if expected != value:
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope timestamp is not canonical",
            field_name=field_name,
        )
    return value


def _parse_signature(
    document: Mapping[str, Any],
    header: Mapping[str, Any],
) -> "AuditEnvelopeSignature":
    source = _field_from_header_or_top(
        document,
        header,
        ("signature",),
        field_name="signature",
        default=_MISSING,
    )
    separate: dict[str, Any] = {}
    for target, names in (
        ("algorithm", ("signature_algorithm",)),
        ("key_id", ("signature_key_id",)),
        ("value", ("signature_value",)),
    ):
        value = _field_from_header_or_top(
            document,
            header,
            names,
            field_name=f"signature.{target}",
            default=_MISSING,
        )
        if value is not _MISSING:
            separate[target] = value

    if source is _MISSING and not separate:
        raise AuditEnvelopeUnsignedError(
            "audit envelope signature is required",
            field_name="signature",
        ) from None
    if source is _MISSING:
        source = separate
    elif separate:
        if isinstance(source, Mapping):
            source_mapping = dict(source)
            if any(
                key in source_mapping and not _values_equal(source_mapping[key], value)
                for key, value in separate.items()
            ):
                _raise(
                    AuditEnvelopeSignatureError,
                    "audit envelope signature aliases conflict",
                    field_name="signature",
                )
            source = {**source_mapping, **separate}
        else:
            _raise(
                AuditEnvelopeSignatureError,
                "audit envelope signature aliases conflict",
                field_name="signature",
            )

    if isinstance(source, str):
        algorithm = "opaque"
        key_id = "unspecified"
        signature_value = source
    else:
        signature_mapping = _require_mapping(source, field_name="signature")
        _require_allowed_keys(
            signature_mapping, _SIGNATURE_FIELDS, field_name="signature"
        )
        algorithm = _coalesce(
            signature_mapping,
            ("algorithm",),
            field_name="signature.algorithm",
            default="opaque",
        )
        key_id = _coalesce(
            signature_mapping,
            ("key_id",),
            field_name="signature.key_id",
            default="unspecified",
        )
        signature_value = _coalesce(
            signature_mapping,
            ("value", "signature", "signature_value"),
            field_name="signature.value",
            default=_MISSING,
        )
        if signature_value is _MISSING:
            raise AuditEnvelopeUnsignedError(
                "audit envelope signature is required",
                field_name="signature.value",
            ) from None

    if (
        signature_value is None
        or signature_value == ""
        or (
            isinstance(signature_value, str)
            and signature_value.lower() in {"none", "null", "unsigned"}
        )
        or (
            isinstance(algorithm, str)
            and algorithm.lower() in {"none", "null", "unsigned"}
        )
    ):
        raise AuditEnvelopeUnsignedError(
            "audit envelope signature is required",
            field_name="signature.value",
        ) from None
    if (
        not isinstance(signature_value, str)
        or _SIGNATURE_VALUE_RE.fullmatch(signature_value) is None
    ):
        _raise(
            AuditEnvelopeSignatureError,
            "audit envelope signature value is invalid",
            field_name="signature.value",
        )
    return AuditEnvelopeSignature(
        algorithm=_require_identifier(algorithm, field_name="signature.algorithm"),
        key_id=_require_identifier(key_id, field_name="signature.key_id"),
        value=_require_text(
            signature_value,
            field_name="signature.value",
            maximum_length=AUDIT_ENVELOPE_MAX_SIGNATURE_LENGTH,
            pattern=_SIGNATURE_VALUE_RE,
        ),
    )


def _payload_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    _raise(
        AuditEnvelopeValidationError,
        "audit envelope payload type is unsupported",
        field_name="payload",
    )
    return "omitted"


def _parse_payload(
    document: Mapping[str, Any],
    header: Mapping[str, Any],
    *,
    max_payload_bytes: int,
) -> tuple[str, int | None, str, bool]:
    payload_present = "payload" in document
    payload = document.get("payload")
    payload_size_value = _field_from_header_or_top(
        document,
        header,
        ("payload_size",),
        field_name="payload_size",
        default=_MISSING,
    )
    payload_type_value = _field_from_header_or_top(
        document,
        header,
        ("payload_type",),
        field_name="payload_type",
        default=_MISSING,
    )
    fingerprint_value = _field_from_header_or_top(
        document,
        header,
        ("payload_fingerprint", "payload_hash", "payload_digest"),
        field_name="payload_fingerprint",
        default=_MISSING,
    )

    if payload_present:
        encoded = _canonical_json_bytes(payload, field_name="payload")
        actual_size = len(encoded)
        if actual_size > max_payload_bytes:
            _raise(
                AuditEnvelopeBoundsError,
                "audit envelope payload exceeds its bound",
                field_name="payload",
            )
        actual_type = _payload_type(payload)
        if payload_size_value is _MISSING:
            payload_size = actual_size
        else:
            payload_size = _require_non_negative_int(
                payload_size_value,
                field_name="payload_size",
                maximum=max_payload_bytes,
            )
            if payload_size != actual_size:
                _raise(
                    AuditEnvelopeValidationError,
                    "audit envelope payload size does not match",
                    field_name="payload_size",
                )
        if payload_type_value is _MISSING:
            payload_type = actual_type
        else:
            if payload_type_value not in _PAYLOAD_TYPES - {"omitted"}:
                _raise(
                    AuditEnvelopeValidationError,
                    "audit envelope payload type is invalid",
                    field_name="payload_type",
                )
            if payload_type_value != actual_type:
                _raise(
                    AuditEnvelopeValidationError,
                    "audit envelope payload type does not match",
                    field_name="payload_type",
                )
            payload_type = payload_type_value
        computed_fingerprint = _sha256_bytes(encoded)
        if fingerprint_value is _MISSING:
            fingerprint = computed_fingerprint
        else:
            fingerprint = _require_digest(
                fingerprint_value, field_name="payload_fingerprint"
            )
            if fingerprint != computed_fingerprint:
                _raise(
                    AuditEnvelopeValidationError,
                    "audit envelope payload fingerprint does not match",
                    field_name="payload_fingerprint",
                )
        return fingerprint, payload_size, payload_type, True

    if fingerprint_value is _MISSING:
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope payload fingerprint is required",
            field_name="payload_fingerprint",
        )
    fingerprint = _require_digest(fingerprint_value, field_name="payload_fingerprint")
    if payload_size_value is _MISSING or payload_size_value is None:
        payload_size = None
    else:
        payload_size = _require_non_negative_int(
            payload_size_value,
            field_name="payload_size",
            maximum=max_payload_bytes,
        )
    if payload_type_value is _MISSING:
        payload_type = "omitted"
    else:
        if payload_type_value not in _PAYLOAD_TYPES:
            _raise(
                AuditEnvelopeValidationError,
                "audit envelope payload type is invalid",
                field_name="payload_type",
            )
        payload_type = payload_type_value
    return fingerprint, payload_size, payload_type, False


@dataclass(frozen=True)
class AuditEnvelopeHeader:
    """Bounded, non-payload metadata from an audit envelope."""

    schema_version: int
    envelope_id: str = "unspecified"
    producer: str = "unspecified"
    created_at: str | None = None
    content_type: str = "application/json"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _require_schema_version(self.schema_version, field_name="schema_version"),
        )
        object.__setattr__(
            self,
            "envelope_id",
            _require_identifier(self.envelope_id, field_name="envelope_id"),
        )
        object.__setattr__(
            self,
            "producer",
            _require_identifier(self.producer, field_name="producer"),
        )
        object.__setattr__(
            self,
            "created_at",
            _parse_timestamp(self.created_at, field_name="created_at"),
        )
        object.__setattr__(
            self, "content_type", _require_content_type(self.content_type)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return only validated header metadata."""

        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "envelope_id": self.envelope_id,
            "producer": self.producer,
            "content_type": self.content_type,
        }
        result["created_at"] = self.created_at
        return result


@dataclass(frozen=True)
class AuditEnvelopeSignature:
    """Structurally validated signature metadata.

    The signature value is retained only for a caller-side trust-boundary
    verification step and is hidden from ``repr`` and all redacted reports.
    """

    algorithm: str
    key_id: str
    value: str = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "algorithm",
            _require_identifier(self.algorithm, field_name="signature.algorithm"),
        )
        object.__setattr__(
            self,
            "key_id",
            _require_identifier(self.key_id, field_name="signature.key_id"),
        )
        object.__setattr__(
            self,
            "value",
            _require_text(
                self.value,
                field_name="signature.value",
                maximum_length=AUDIT_ENVELOPE_MAX_SIGNATURE_LENGTH,
                pattern=_SIGNATURE_VALUE_RE,
            ),
        )

    @property
    def fingerprint(self) -> str:
        """Return a safe digest of the signature value."""

        return _sha256_bytes(self.value.encode("utf-8"))

    @property
    def present(self) -> bool:
        """Return ``True`` for this required, non-empty signature."""

        return True

    def to_dict(self, *, include_value: bool = False) -> dict[str, Any]:
        """Return signature metadata, omitting the value by default."""

        result: dict[str, Any] = {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "fingerprint": self.fingerprint,
            "present": True,
        }
        if include_value:
            result["value"] = self.value
        return result


@dataclass(frozen=True)
class AuditEnvelopeReport:
    """PHI-safe metadata report for one parsed audit envelope."""

    schema_version: int
    envelope_id: str
    producer: str
    created_at: str | None
    content_type: str
    payload_fingerprint: str
    payload_size: int | None
    payload_type: str
    payload_present: bool
    signature_algorithm: str
    signature_key_id: str
    signature_fingerprint: str
    signed: bool = True
    report_type: str = AUDIT_ENVELOPE_REPORT_TYPE

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata with no payload or signature value."""

        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "envelope_id": self.envelope_id,
            "producer": self.producer,
            "content_type": self.content_type,
            "payload_fingerprint": self.payload_fingerprint,
            "payload_size": self.payload_size,
            "payload_type": self.payload_type,
            "payload_present": self.payload_present,
            "signature": {
                "algorithm": self.signature_algorithm,
                "key_id": self.signature_key_id,
                "fingerprint": self.signature_fingerprint,
                "present": self.signed,
            },
            "signed": self.signed,
        }
        result["created_at"] = self.created_at
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the redacted report deterministically."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )


@dataclass(frozen=True)
class AuditEnvelope:
    """A parsed audit envelope that contains metadata but no raw payload."""

    header: AuditEnvelopeHeader
    signature: AuditEnvelopeSignature
    payload_fingerprint: str
    payload_size: int | None
    payload_type: str
    payload_present: bool

    def __post_init__(self) -> None:
        if not isinstance(self.header, AuditEnvelopeHeader):
            raise TypeError("audit envelope header must be AuditEnvelopeHeader")
        if not isinstance(self.signature, AuditEnvelopeSignature):
            raise TypeError("audit envelope signature must be AuditEnvelopeSignature")
        object.__setattr__(
            self,
            "payload_fingerprint",
            _require_digest(self.payload_fingerprint, field_name="payload_fingerprint"),
        )
        if self.payload_size is not None:
            object.__setattr__(
                self,
                "payload_size",
                _require_non_negative_int(
                    self.payload_size,
                    field_name="payload_size",
                    maximum=AUDIT_ENVELOPE_MAX_PAYLOAD_BYTES,
                ),
            )
        if self.payload_type not in _PAYLOAD_TYPES:
            _raise(
                AuditEnvelopeValidationError,
                "audit envelope payload type is invalid",
                field_name="payload_type",
            )
        if type(self.payload_present) is not bool:
            raise TypeError("audit envelope payload_present must be boolean")

    @property
    def schema_version(self) -> int:
        """Return the envelope schema version."""

        return self.header.schema_version

    @property
    def envelope_id(self) -> str:
        """Return the validated envelope identifier."""

        return self.header.envelope_id

    @property
    def producer(self) -> str:
        """Return the validated producer identifier."""

        return self.header.producer

    @property
    def signature_value(self) -> str:
        """Return the signature for caller-side verification."""

        return self.signature.value

    @property
    def signed(self) -> bool:
        """Return whether the required signature metadata is present."""

        return True

    @property
    def report(self) -> AuditEnvelopeReport:
        """Return a PHI-safe report object."""

        return self.to_report()

    def to_report(self) -> AuditEnvelopeReport:
        """Return metadata only; neither payload nor signature value is copied."""

        return AuditEnvelopeReport(
            schema_version=self.schema_version,
            envelope_id=self.envelope_id,
            producer=self.producer,
            created_at=self.header.created_at,
            content_type=self.header.content_type,
            payload_fingerprint=self.payload_fingerprint,
            payload_size=self.payload_size,
            payload_type=self.payload_type,
            payload_present=self.payload_present,
            signature_algorithm=self.signature.algorithm,
            signature_key_id=self.signature.key_id,
            signature_fingerprint=self.signature.fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the redacted metadata report as a dictionary."""

        return self.to_report().to_dict()

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize redacted metadata without retaining the payload."""

        return self.to_report().to_json(indent=indent)

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        max_payload_bytes: int = AUDIT_ENVELOPE_MAX_PAYLOAD_BYTES,
    ) -> "AuditEnvelope":
        """Parse a mapping using the same bounded path as JSON input."""

        return parse_audit_envelope(value, max_payload_bytes=max_payload_bytes)

    @classmethod
    def from_json(
        cls,
        value: str | bytes,
        *,
        max_bytes: int = AUDIT_ENVELOPE_MAX_BYTES,
        max_payload_bytes: int = AUDIT_ENVELOPE_MAX_PAYLOAD_BYTES,
    ) -> "AuditEnvelope":
        """Parse strict JSON text or UTF-8 bytes."""

        return parse_audit_envelope(
            value,
            max_bytes=max_bytes,
            max_payload_bytes=max_payload_bytes,
        )


class AuditEnvelopeParser:
    """Reusable bounded parser configuration."""

    def __init__(
        self,
        *,
        max_bytes: int = AUDIT_ENVELOPE_MAX_BYTES,
        max_payload_bytes: int = AUDIT_ENVELOPE_MAX_PAYLOAD_BYTES,
    ) -> None:
        self.max_bytes = _validate_bound(
            max_bytes,
            maximum=AUDIT_ENVELOPE_MAX_BYTES,
            field_name="max_bytes",
        )
        self.max_payload_bytes = _validate_bound(
            max_payload_bytes,
            maximum=AUDIT_ENVELOPE_MAX_PAYLOAD_BYTES,
            field_name="max_payload_bytes",
        )

    def parse(self, value: Mapping[str, Any] | str | bytes) -> AuditEnvelope:
        """Parse one envelope with this parser's fixed bounds."""

        return parse_audit_envelope(
            value,
            max_bytes=self.max_bytes,
            max_payload_bytes=self.max_payload_bytes,
        )


def parse_audit_envelope(
    value: Mapping[str, Any] | str | bytes,
    *,
    max_bytes: int = AUDIT_ENVELOPE_MAX_BYTES,
    max_payload_bytes: int = AUDIT_ENVELOPE_MAX_PAYLOAD_BYTES,
    max_envelope_bytes: int | None = None,
) -> AuditEnvelope:
    """Parse an audit envelope and discard payload content before returning.

    ``value`` may be a mapping or strict JSON text/bytes.  The optional
    ``max_envelope_bytes`` name is an alias for ``max_bytes``; conflicting
    bounds fail closed.  All failures use :class:`AuditEnvelopeError` and do
    not echo caller-provided values.
    """

    if max_envelope_bytes is not None:
        if max_bytes != AUDIT_ENVELOPE_MAX_BYTES and max_bytes != max_envelope_bytes:
            _raise(
                AuditEnvelopeValidationError,
                "parser bounds conflict",
                field_name="max_bytes",
            )
        max_bytes = max_envelope_bytes
    checked_max_bytes = _validate_bound(
        max_bytes,
        maximum=AUDIT_ENVELOPE_MAX_BYTES,
        field_name="max_bytes",
    )
    checked_max_payload = _validate_bound(
        max_payload_bytes,
        maximum=AUDIT_ENVELOPE_MAX_PAYLOAD_BYTES,
        field_name="max_payload_bytes",
    )
    document = _load_document(value, max_bytes=checked_max_bytes)
    _require_allowed_keys(document, _TOP_LEVEL_FIELDS, field_name="envelope")
    header = _parse_header(document)
    header_values = _header_values_for_parse(document)
    signature = _parse_signature(document, header_values)
    fingerprint, payload_size, payload_type, payload_present = _parse_payload(
        document,
        header_values,
        max_payload_bytes=checked_max_payload,
    )
    report_type = _coalesce(
        document,
        ("report_type", "type"),
        field_name="report_type",
        default=_MISSING,
    )
    if report_type is not _MISSING and report_type not in {
        AUDIT_ENVELOPE_REPORT_TYPE,
        "audit_envelope",
        "openmed.audit_envelope",
    }:
        _raise(
            AuditEnvelopeValidationError,
            "audit envelope report type is unsupported",
            field_name="report_type",
        )
    return AuditEnvelope(
        header=header,
        signature=signature,
        payload_fingerprint=fingerprint,
        payload_size=payload_size,
        payload_type=payload_type,
        payload_present=payload_present,
    )


def _header_values_for_parse(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the validated header mapping used by the parser internals."""

    header_values: dict[str, Any] = {}
    for name in ("header", "headers", "metadata"):
        if name not in document:
            continue
        candidate = _require_mapping(document[name], field_name="header")
        if header_values and not _values_equal(header_values, candidate):
            _raise(
                AuditEnvelopeValidationError,
                "audit envelope header aliases conflict",
                field_name="header",
            )
        header_values = candidate
    _require_allowed_keys(
        header_values,
        _HEADER_FIELDS,
        field_name="header",
        maximum=AUDIT_ENVELOPE_MAX_HEADER_FIELDS,
    )
    for name in _HEADER_FIELDS:
        if name in document and name not in header_values:
            header_values[name] = document[name]
    return header_values


def redacted_audit_envelope_report(
    value: AuditEnvelope | Mapping[str, Any] | str | bytes,
) -> dict[str, Any]:
    """Parse if needed and return only redacted envelope metadata."""

    envelope = (
        value if isinstance(value, AuditEnvelope) else parse_audit_envelope(value)
    )
    return envelope.to_dict()


def render_audit_envelope_report(
    value: AuditEnvelope | Mapping[str, Any] | str | bytes,
    *,
    indent: int | None = 2,
) -> str:
    """Render a deterministic JSON report without payload text."""

    envelope = (
        value if isinstance(value, AuditEnvelope) else parse_audit_envelope(value)
    )
    return envelope.to_json(indent=indent)


def create_audit_envelope(
    payload: Any,
    *,
    signature: str | Mapping[str, Any],
    envelope_id: str = "synthetic-envelope",
    producer: str = "synthetic",
    created_at: str | None = None,
    content_type: str = "application/json",
    schema_version: int = AUDIT_ENVELOPE_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build a transient synthetic envelope with a canonical payload digest.

    The returned mapping intentionally contains the payload because it is a
    producer-side construction helper.  Parsed :class:`AuditEnvelope` objects
    never retain it.  Callers must keep real sensitive payloads outside logs,
    reports, and committed fixtures.
    """

    encoded = _canonical_json_bytes(payload, field_name="payload")
    if len(encoded) > AUDIT_ENVELOPE_MAX_PAYLOAD_BYTES:
        _raise(
            AuditEnvelopeBoundsError,
            "audit envelope payload exceeds its bound",
            field_name="payload",
        )
    header = AuditEnvelopeHeader(
        schema_version=schema_version,
        envelope_id=envelope_id,
        producer=producer,
        created_at=created_at,
        content_type=content_type,
    )
    # Validate the signature now so the constructor cannot create an unsigned
    # synthetic record that the parser would later reject.
    signature_metadata = _parse_signature(
        {"signature": signature},
        {},
    )
    return {
        "schema_version": header.schema_version,
        "header": header.to_dict(),
        "signature": {
            "algorithm": signature_metadata.algorithm,
            "key_id": signature_metadata.key_id,
            "value": signature_metadata.value,
        },
        "payload_fingerprint": _sha256_bytes(encoded),
        "payload_size": len(encoded),
        "payload_type": _payload_type(payload),
        "payload": payload,
    }


build_audit_envelope = create_audit_envelope
parse_envelope = parse_audit_envelope
redact_audit_envelope = redacted_audit_envelope_report


__all__ = [
    "AUDIT_ENVELOPE_MAX_BYTES",
    "AUDIT_ENVELOPE_MAX_HEADER_FIELDS",
    "AUDIT_ENVELOPE_MAX_HEADER_VALUE_LENGTH",
    "AUDIT_ENVELOPE_MAX_JSON_DEPTH",
    "AUDIT_ENVELOPE_MAX_JSON_ITEMS",
    "AUDIT_ENVELOPE_MAX_PAYLOAD_BYTES",
    "AUDIT_ENVELOPE_MAX_SIGNATURE_LENGTH",
    "AUDIT_ENVELOPE_REPORT_TYPE",
    "AUDIT_ENVELOPE_SCHEMA",
    "AUDIT_ENVELOPE_SCHEMA_VERSION",
    "AuditEnvelope",
    "AuditEnvelopeBoundError",
    "AuditEnvelopeBoundsError",
    "AuditEnvelopeError",
    "AuditEnvelopeHeader",
    "AuditEnvelopeMalformedError",
    "AuditEnvelopeParseError",
    "AuditEnvelopeParser",
    "AuditEnvelopeReport",
    "AuditEnvelopeSignature",
    "AuditEnvelopeSignatureError",
    "AuditEnvelopeUnsignedError",
    "AuditEnvelopeValidationError",
    "MAX_ENVELOPE_BYTES",
    "MAX_HEADER_FIELDS",
    "MAX_HEADER_VALUE_LENGTH",
    "MAX_PAYLOAD_BYTES",
    "MAX_SIGNATURE_LENGTH",
    "build_audit_envelope",
    "compute_payload_fingerprint",
    "create_audit_envelope",
    "fingerprint_payload",
    "parse_audit_envelope",
    "parse_envelope",
    "payload_fingerprint",
    "redact_audit_envelope",
    "redacted_audit_envelope_report",
    "render_audit_envelope_report",
]
