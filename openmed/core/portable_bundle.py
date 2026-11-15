"""Signed portable model-bundle manifests shared by every OpenMed kit.

A portable bundle is a directory that holds model files plus two documents:

* ``openmed-bundle.json``: the manifest. Its bytes must be the RFC 8785 (JCS)
  canonical encoding of the manifest value, restricted to the I-JSON subset
  without non-integer numbers. The content identity of a bundle is
  ``sha256:`` plus the hex SHA-256 of exactly these bytes.
* ``openmed-bundle.sig.json``: the signature envelope, also canonical JSON. It
  repeats the content identity and carries one or more Ed25519 signatures over
  the DSSE pre-authentication encoding of the manifest bytes.

This module is the Python reference reader, verifier, and signer. Swift,
Kotlin, and TypeScript kits implement the same checks in the same order and
must produce the same reason code for the golden vectors in
``tests/fixtures/portable_bundle/``.

Verification is offline and value-free: it never opens a network connection,
never executes bundled code, never follows references outside the bundle root,
and reports only controlled reason codes, digests, counts, and indexes. Trust
roots and signing keys are always supplied by the caller and never bundled.
Verification is separate from cache activation; a later activation step must
refuse any bundle whose result is not ``ok``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any

PORTABLE_BUNDLE_SCHEMA = "openmed.portable_bundle"
PORTABLE_BUNDLE_SCHEMA_VERSION = 1
ENVELOPE_SCHEMA = "openmed.portable_bundle.signatures"
ENVELOPE_SCHEMA_VERSION = 1
TRUST_POLICY_SCHEMA = "openmed.portable_bundle.trust_policy"
TRUST_POLICY_SCHEMA_VERSION = 1
VERIFICATION_SCHEMA = "openmed.portable_bundle.verification"
VERIFICATION_SCHEMA_VERSION = 1

MANIFEST_FILENAME = "openmed-bundle.json"
ENVELOPE_FILENAME = "openmed-bundle.sig.json"
PAYLOAD_TYPE = "application/vnd.openmed.portable-bundle.v1+json"
SIGNATURE_ALGORITHM = "Ed25519"

DEFAULT_MAX_MANIFEST_BYTES = 1 << 20
DEFAULT_MAX_ENVELOPE_BYTES = 64 << 10
DEFAULT_MAX_TOTAL_BYTES = 64 << 30
DEFAULT_MAX_FILES = 4096
MAX_SIGNATURES = 16
MAX_SAFE_INTEGER = (1 << 53) - 1

SCHEMA_PACKAGE = "openmed.core.schemas.json"
SCHEMA_NAMES = (
    "portable_bundle",
    "portable_bundle_envelope",
    "portable_bundle_trust_policy",
    "portable_bundle_verification",
)

FORMAT_PLATFORMS: Mapping[str, frozenset[str]] = {
    "coreml": frozenset({"ios", "macos", "watchos", "visionos"}),
    "mlx": frozenset({"ios", "macos"}),
    "onnx": frozenset({"android", "ios", "macos", "linux", "windows", "web"}),
    "ort": frozenset({"android", "ios"}),
    "transformersjs": frozenset({"web"}),
    "pytorch": frozenset({"linux", "macos", "windows"}),
}
"""Platforms each runtime format may declare; other pairs fail closed."""

FORMAT_PRECISIONS: Mapping[str, frozenset[str]] = {
    "coreml": frozenset({"fp32", "fp16", "int8", "int4"}),
    "mlx": frozenset({"fp32", "fp16", "bf16", "int8", "int4", "int2"}),
    "onnx": frozenset({"fp32", "fp16", "int8", "int4"}),
    "ort": frozenset({"fp32", "fp16", "int8"}),
    "transformersjs": frozenset({"fp32", "fp16", "int8", "int4"}),
    "pytorch": frozenset({"fp32", "fp16", "bf16"}),
}
"""Precisions each runtime format may declare; other pairs fail closed."""

PLATFORMS = frozenset(
    {"ios", "macos", "watchos", "visionos", "android", "web", "linux", "windows"}
)
PRECISIONS = frozenset({"fp32", "fp16", "bf16", "int8", "int4", "int2"})
FILE_ROLES = frozenset(
    {
        "calibration",
        "config",
        "decoder",
        "evidence",
        "graph",
        "labels",
        "license",
        "metadata",
        "policy",
        "provenance",
        "segmenter",
        "tokenizer",
        "weights",
    }
)
OFFSET_UNITS = frozenset({"unicode_scalar", "utf16_code_unit", "utf8_byte"})


class PortableBundleReason(str, Enum):
    """Stable, value-free reason codes shared by every portable-bundle reader.

    Codes are listed in the order the verifier evaluates them. Readers in every
    language must report the first failing check in this order.
    """

    MANIFEST_TOO_LARGE = "manifest_too_large"
    ENVELOPE_TOO_LARGE = "envelope_too_large"
    ENVELOPE_MALFORMED = "envelope_malformed"
    ENVELOPE_VERSION_UNSUPPORTED = "envelope_version_unsupported"
    CONTENT_ID_MISMATCH = "content_id_mismatch"
    BUNDLE_REVOKED = "bundle_revoked"
    SIGNATURE_MISSING = "signature_missing"
    SIGNATURE_ALGORITHM_UNSUPPORTED = "signature_algorithm_unsupported"
    UNTRUSTED_KEY = "untrusted_key"
    KEY_REVOKED = "key_revoked"
    KEY_NOT_YET_VALID = "key_not_yet_valid"
    KEY_EXPIRED = "key_expired"
    SIGNATURE_INVALID = "signature_invalid"
    SIGNATURE_VERIFIER_UNAVAILABLE = "signature_verifier_unavailable"
    SIGNATURE_THRESHOLD_NOT_MET = "signature_threshold_not_met"
    MANIFEST_NOT_CANONICAL = "manifest_not_canonical"
    SCHEMA_UNSUPPORTED = "schema_unsupported"
    SCHEMA_VERSION_UNSUPPORTED = "schema_version_unsupported"
    UNKNOWN_CRITICAL_FIELD = "unknown_critical_field"
    MANIFEST_MALFORMED = "manifest_malformed"
    PATH_UNSAFE = "path_unsafe"
    EXECUTABLE_FILE_FORBIDDEN = "executable_file_forbidden"
    RUNTIME_COMBINATION_UNSUPPORTED = "runtime_combination_unsupported"
    DUPLICATE_PATH = "duplicate_path"
    PATH_CONFLICT = "path_conflict"
    FILE_ORDER_INVALID = "file_order_invalid"
    FILE_REFERENCE_UNDECLARED = "file_reference_undeclared"
    FALLBACK_INVALID = "fallback_invalid"
    UNKNOWN_CRITICAL_EXTENSION = "unknown_critical_extension"
    SIZE_LIMIT_EXCEEDED = "size_limit_exceeded"
    BUNDLE_NOT_YET_VALID = "bundle_not_yet_valid"
    BUNDLE_EXPIRED = "bundle_expired"
    ROLLBACK_REJECTED = "rollback_rejected"
    SDK_INCOMPATIBLE = "sdk_incompatible"
    POLICY_INCOMPATIBLE = "policy_incompatible"
    PLATFORM_INCOMPATIBLE = "platform_incompatible"
    FILE_MISSING = "file_missing"
    FILE_SIZE_MISMATCH = "file_size_mismatch"
    FILE_DIGEST_MISMATCH = "file_digest_mismatch"
    FILE_UNDECLARED = "file_undeclared"
    TRUST_POLICY_INVALID = "trust_policy_invalid"
    CONTEXT_INVALID = "context_invalid"
    SIGNING_KEY_INVALID = "signing_key_invalid"


REASON_CODES: tuple[str, ...] = tuple(reason.value for reason in PortableBundleReason)
"""Every stable reason code, in evaluation order."""


class PortableBundleError(ValueError):
    """Raised for invalid inputs; the message is the reason code only.

    Attributes:
        reason: The controlled :class:`PortableBundleReason`.
    """

    def __init__(self, reason: PortableBundleReason) -> None:
        """Store the controlled reason and use its value as the message."""

        self.reason = reason
        super().__init__(reason.value)


class _Refusal(Exception):
    def __init__(self, reason: PortableBundleReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


def _fail(reason: PortableBundleReason) -> _Refusal:
    return _Refusal(reason)


_R = PortableBundleReason
_MALFORMED = _R.MANIFEST_MALFORMED

_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_BUNDLE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,4}")
_LICENSE = re.compile(r"[a-z0-9][a-z0-9.+-]{0,63}")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_MEDIA_TYPE = re.compile(r"[a-z]+/[a-z0-9][a-z0-9.+-]{0,126}")
_EXTENSION = re.compile(r"[a-z0-9][a-z0-9-]{0,31}(?:\.[a-z0-9][a-z0-9_-]{0,31}){1,7}")
_SEMVER = re.compile(r"(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8})){2}")
_VERSION = re.compile(r"(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8})){0,3}")
_TIMESTAMP = re.compile(
    r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z"
)
_KID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_PATH_SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._+-]{0,127}")
_B64URL = re.compile(r"[A-Za-z0-9_-]+")
_MAX_PATH_CHARS = 512
_MAX_PATH_SEGMENTS = 16
_MAX_DEPTH = 32
_RESERVED_PATHS = frozenset({MANIFEST_FILENAME, ENVELOPE_FILENAME})
_HASH_CHUNK = 1 << 20

_TOP_KEYS = (
    "schema",
    "schema_version",
    "bundle",
    "model",
    "tokenizer",
    "labels",
    "offsets",
    "decoder",
    "segmenter",
    "policy",
    "calibration",
    "compatibility",
    "runtimes",
    "licenses",
    "provenance",
    "evidence",
    "files",
    "extensions",
    "critical_extensions",
)
_BUNDLE_KEYS = ("id", "version", "sequence", "issued_at", "expires_at")
_MODEL_KEYS = ("id", "revision", "family", "architecture", "task", "languages")
_TOKENIZER_KEYS = ("kind", "contract_version", "files")
_LABELS_KEYS = ("path", "count", "scheme")
_OFFSETS_KEYS = ("unit", "contract_version")
_DECODER_KEYS = ("kind", "contract_version")
_SEGMENTER_KEYS = ("id", "files")
_POLICY_KEYS = ("min_version", "max_version_exclusive", "path")
_CALIBRATION_KEYS = ("profile_version", "path")
_COMPATIBILITY_KEYS = ("sdks",)
_SDK_KEYS = ("name", "min", "max_exclusive")
_RUNTIME_KEYS = (
    "id",
    "format",
    "precision",
    "entrypoint",
    "files",
    "platforms",
    "min_memory_bytes",
    "min_storage_bytes",
    "fallback",
)
_PLATFORM_KEYS = ("platform", "min_os", "max_os_exclusive")
_LICENSE_KEYS = ("id", "files")
_PROVENANCE_KEYS = ("source_digest", "export_record")
_EVIDENCE_KEYS = ("kind", "path")
_FILE_KEYS = ("path", "media_type", "size_bytes", "sha256", "role", "executable")
_ENVELOPE_KEYS = ("schema", "schema_version", "content_id", "signatures")
_SIGNATURE_KEYS = ("kid", "alg", "sig")


# --------------------------------------------------------------------------
# Canonical JSON (RFC 8785 subset)
# --------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    """Return the RFC 8785 (JCS) canonical UTF-8 encoding of ``value``.

    The portable-bundle profile accepts ``None``, booleans, integers in the
    IEEE 754 safe range, strings without unpaired surrogates, lists, and
    mappings with string keys. Object members are sorted by UTF-16 code units,
    strings use the ECMAScript ``JSON.stringify`` escapes, and no insignificant
    whitespace is emitted. Non-integer numbers are rejected so every reader
    produces identical bytes without a floating-point formatter.

    Args:
        value: JSON-compatible value to encode.

    Returns:
        Canonical UTF-8 bytes.

    Raises:
        PortableBundleError: ``manifest_not_canonical`` when the value falls
            outside the supported subset.
    """

    parts: list[str] = []
    try:
        _encode_canonical(value, parts, 0)
        return "".join(parts).encode("utf-8")
    except (_Refusal, UnicodeError, RecursionError):
        raise PortableBundleError(_R.MANIFEST_NOT_CANONICAL) from None


def _encode_canonical(value: Any, out: list[str], depth: int) -> None:
    if depth > _MAX_DEPTH:
        raise _fail(_R.MANIFEST_NOT_CANONICAL)
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif type(value) is int:
        if not -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
            raise _fail(_R.MANIFEST_NOT_CANONICAL)
        out.append(str(value))
    elif isinstance(value, str):
        out.append(_canonical_string(value))
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _encode_canonical(item, out, depth + 1)
        out.append("]")
    elif isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise _fail(_R.MANIFEST_NOT_CANONICAL)
        out.append("{")
        for index, key in enumerate(sorted(value, key=_utf16_key)):
            if index:
                out.append(",")
            out.append(_canonical_string(key))
            out.append(":")
            _encode_canonical(value[key], out, depth + 1)
        out.append("}")
    else:
        raise _fail(_R.MANIFEST_NOT_CANONICAL)


def _utf16_key(key: str) -> bytes:
    return key.encode("utf-16-be", "surrogatepass")


def _canonical_string(value: str) -> str:
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise _fail(_R.MANIFEST_NOT_CANONICAL)
    # ``json.dumps`` with ``ensure_ascii=False`` emits exactly the escapes of
    # ECMAScript JSON.stringify for well-formed strings.
    return json.dumps(value, ensure_ascii=False)


def _reject_number(_text: str) -> Any:
    raise ValueError("non-integer numbers are not canonical")


def _parse_canonical(data: bytes) -> Any:
    try:
        text = data.decode("utf-8")
        value = json.loads(
            text, parse_float=_reject_number, parse_constant=_reject_number
        )
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise _fail(_R.MANIFEST_NOT_CANONICAL) from None
    try:
        encoded = canonical_json_bytes(value)
    except PortableBundleError:
        raise _fail(_R.MANIFEST_NOT_CANONICAL) from None
    if encoded != data:
        raise _fail(_R.MANIFEST_NOT_CANONICAL)
    return value


def content_id(manifest_bytes: bytes) -> str:
    """Return the ``sha256:<hex>`` content identity of canonical manifest bytes.

    Args:
        manifest_bytes: Exact bytes of ``openmed-bundle.json``.

    Returns:
        The bundle content identity.
    """

    return "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()


def signing_input(manifest_bytes: bytes) -> bytes:
    """Return the DSSE pre-authentication encoding that signatures cover.

    ``PAE = "DSSEv1" SP len(type) SP type SP len(body) SP body`` with lengths as
    ASCII decimal byte counts, ``type`` equal to :data:`PAYLOAD_TYPE`, and
    ``body`` equal to the canonical manifest bytes. The payload type binds the
    signature to this manifest format so it cannot be replayed elsewhere.

    Args:
        manifest_bytes: Exact bytes of ``openmed-bundle.json``.

    Returns:
        Bytes passed to Ed25519 sign and verify.
    """

    payload_type = PAYLOAD_TYPE.encode("ascii")
    header = f"DSSEv1 {len(payload_type)} {PAYLOAD_TYPE} {len(manifest_bytes)} "
    return header.encode("ascii") + manifest_bytes


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(text: object, length: int) -> bytes | None:
    if not isinstance(text, str) or _B64URL.fullmatch(text) is None:
        return None
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (ValueError, TypeError):
        return None
    if len(raw) != length or _b64url_encode(raw) != text:
        return None
    return raw


# --------------------------------------------------------------------------
# Trust policy and verification context
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustedKey:
    """One caller-controlled Ed25519 trust anchor.

    Args:
        kid: Key identifier that signatures reference.
        public_key: Raw 32-byte Ed25519 public key.
        not_before: First instant (``YYYY-MM-DDTHH:MM:SSZ``) the key is valid.
        not_after: First instant the key is no longer valid.
        revoked: Whether the key has been revoked.
    """

    kid: str
    public_key: bytes = field(repr=False)
    not_before: str
    not_after: str
    revoked: bool = False

    def __post_init__(self) -> None:
        """Validate the key without echoing its value."""

        if (
            not isinstance(self.kid, str)
            or _KID.fullmatch(self.kid) is None
            or not isinstance(self.public_key, bytes)
            or len(self.public_key) != 32
            or not _is_timestamp(self.not_before)
            or not _is_timestamp(self.not_after)
            or self.not_before >= self.not_after
            or not isinstance(self.revoked, bool)
        ):
            raise PortableBundleError(_R.TRUST_POLICY_INVALID)

    def to_dict(self) -> dict[str, Any]:
        """Return the JWK-style trust-policy entry for this key."""

        return {
            "crv": "Ed25519",
            "kid": self.kid,
            "kty": "OKP",
            "not_after": self.not_after,
            "not_before": self.not_before,
            "status": "revoked" if self.revoked else "active",
            "x": _b64url_encode(self.public_key),
        }


@dataclass(frozen=True)
class TrustPolicy:
    """Offline trust roots, rotation windows, and revocations.

    Args:
        keys: Trusted Ed25519 keys; overlapping windows support rotation.
        threshold: Number of distinct valid trusted signatures required.
        revoked_content_ids: Bundle content identities that must be refused.
    """

    keys: tuple[TrustedKey, ...]
    threshold: int = 1
    revoked_content_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Validate the policy and normalise collections."""

        keys = tuple(self.keys)
        revoked = frozenset(self.revoked_content_ids)
        if (
            not keys
            or len(keys) > 64
            or not all(isinstance(key, TrustedKey) for key in keys)
            or len({key.kid for key in keys}) != len(keys)
            or type(self.threshold) is not int
            or not 1 <= self.threshold <= MAX_SIGNATURES
            or not all(
                isinstance(item, str) and _SHA256.fullmatch(item) for item in revoked
            )
        ):
            raise PortableBundleError(_R.TRUST_POLICY_INVALID)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "revoked_content_ids", revoked)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TrustPolicy:
        """Parse a JSON trust-policy document.

        Args:
            data: Mapping that follows ``portable_bundle_trust_policy.schema.json``.

        Returns:
            A validated :class:`TrustPolicy`.

        Raises:
            PortableBundleError: ``trust_policy_invalid`` for any malformed input.
        """

        invalid = PortableBundleError(_R.TRUST_POLICY_INVALID)
        expected = {
            "schema",
            "schema_version",
            "threshold",
            "keys",
            "revoked_content_ids",
        }
        if not isinstance(data, Mapping) or set(data) != expected:
            raise invalid
        if (
            data["schema"] != TRUST_POLICY_SCHEMA
            or type(data["schema_version"]) is not int
            or data["schema_version"] != TRUST_POLICY_SCHEMA_VERSION
            or not isinstance(data["keys"], list)
            or not isinstance(data["revoked_content_ids"], list)
        ):
            raise invalid
        keys = []
        key_fields = {"kid", "kty", "crv", "x", "not_before", "not_after", "status"}
        for entry in data["keys"]:
            if not isinstance(entry, Mapping) or set(entry) != key_fields:
                raise invalid
            public_key = _b64url_decode(entry["x"], 32)
            if (
                entry["kty"] != "OKP"
                or entry["crv"] != "Ed25519"
                or public_key is None
                or entry["status"] not in {"active", "revoked"}
            ):
                raise invalid
            keys.append(
                TrustedKey(
                    kid=entry["kid"],
                    public_key=public_key,
                    not_before=entry["not_before"],
                    not_after=entry["not_after"],
                    revoked=entry["status"] == "revoked",
                )
            )
        return cls(
            keys=tuple(keys),
            threshold=data["threshold"],
            revoked_content_ids=frozenset(data["revoked_content_ids"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON trust-policy document for this policy."""

        return {
            "keys": [key.to_dict() for key in self.keys],
            "revoked_content_ids": sorted(self.revoked_content_ids),
            "schema": TRUST_POLICY_SCHEMA,
            "schema_version": TRUST_POLICY_SCHEMA_VERSION,
            "threshold": self.threshold,
        }

    def key(self, kid: str) -> TrustedKey | None:
        """Return the trusted key with identifier ``kid``, if any."""

        for candidate in self.keys:
            if candidate.kid == kid:
                return candidate
        return None


@dataclass(frozen=True)
class VerificationTarget:
    """Runtime environment a bundle must be compatible with.

    Args:
        platform: One of :data:`PLATFORMS`.
        sdk_name: Reading SDK, for example ``openmed-python`` or
            ``openmedkit-swift``.
        sdk_version: Dotted numeric SDK version.
        runtimes: Runtime formats the reader can load.
        os_version: Dotted numeric OS version, or ``None`` when not applicable.
        policy_version: Privacy-policy schema version the reader enforces.
        memory_bytes: Memory available to the runtime, if known.
        storage_bytes: Storage available to the cache, if known.
    """

    platform: str
    sdk_name: str
    sdk_version: str
    runtimes: tuple[str, ...]
    os_version: str | None = None
    policy_version: int | None = None
    memory_bytes: int | None = None
    storage_bytes: int | None = None

    def __post_init__(self) -> None:
        """Validate controlled target fields."""

        runtimes = tuple(self.runtimes)
        if (
            self.platform not in PLATFORMS
            or not isinstance(self.sdk_name, str)
            or _TOKEN.fullmatch(self.sdk_name) is None
            or not _is_version(self.sdk_version)
            or not all(item in FORMAT_PLATFORMS for item in runtimes)
            or (self.os_version is not None and not _is_version(self.os_version))
            or not _optional_uint(self.policy_version)
            or not _optional_uint(self.memory_bytes)
            or not _optional_uint(self.storage_bytes)
        ):
            raise PortableBundleError(_R.CONTEXT_INVALID)
        object.__setattr__(self, "runtimes", runtimes)


@dataclass(frozen=True)
class VerificationContext:
    """Caller-controlled verification inputs.

    Args:
        now: Verification instant as ``YYYY-MM-DDTHH:MM:SSZ`` or an aware
            :class:`datetime`; defaults to the current UTC time.
        target: Runtime target; ``None`` skips compatibility selection.
        sequence_floors: Highest accepted sequence per bundle id
            (anti-rollback state owned by the caller).
        understood_extensions: Extension names this reader implements.
        max_manifest_bytes: Manifest size limit.
        max_envelope_bytes: Signature-envelope size limit.
        max_total_bytes: Limit on the sum of declared file sizes.
        max_files: Limit on the number of declared files.
    """

    now: str | datetime | None = None
    target: VerificationTarget | None = None
    sequence_floors: Mapping[str, int] = field(default_factory=dict)
    understood_extensions: frozenset[str] = frozenset()
    max_manifest_bytes: int = DEFAULT_MAX_MANIFEST_BYTES
    max_envelope_bytes: int = DEFAULT_MAX_ENVELOPE_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    max_files: int = DEFAULT_MAX_FILES

    def __post_init__(self) -> None:
        """Normalise ``now`` and validate limits."""

        now = self.now
        if now is None:
            now = datetime.now(timezone.utc)
        if isinstance(now, datetime):
            if now.tzinfo is None or now.utcoffset() is None:
                raise PortableBundleError(_R.CONTEXT_INVALID)
            now = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        floors = dict(self.sequence_floors)
        if (
            not _is_timestamp(now)
            or (
                self.target is not None
                and not isinstance(self.target, VerificationTarget)
            )
            or not all(
                isinstance(key, str) and _optional_uint(value) and value is not None
                for key, value in floors.items()
            )
            or not all(
                isinstance(item, str) and _EXTENSION.fullmatch(item)
                for item in self.understood_extensions
            )
            or not all(
                type(limit) is int and limit > 0
                for limit in (
                    self.max_manifest_bytes,
                    self.max_envelope_bytes,
                    self.max_total_bytes,
                    self.max_files,
                )
            )
        ):
            raise PortableBundleError(_R.CONTEXT_INVALID)
        object.__setattr__(self, "now", now)
        object.__setattr__(self, "sequence_floors", floors)
        object.__setattr__(
            self, "understood_extensions", frozenset(self.understood_extensions)
        )


@dataclass(frozen=True)
class PortableBundleVerification:
    """Bounded, value-free verification result.

    Attributes:
        ok: Whether every evaluated check passed.
        reason: First failing reason code, or ``None`` when ``ok``.
        content_id: Manifest content identity, when it could be computed.
        valid_signatures: Distinct trusted keys with a valid signature.
        eligible_runtimes: Indexes of runtime variants usable on the target.
        files_checked: Declared files whose size and digest matched.
        files_verified: Whether the file tree was checked and matched.
    """

    ok: bool
    reason: str | None
    content_id: str | None
    valid_signatures: int = 0
    eligible_runtimes: tuple[int, ...] = ()
    files_checked: int = 0
    files_verified: bool = False

    @property
    def eligible_for_activation(self) -> bool:
        """Return whether a separate activation step may consider the bundle."""

        return self.ok and self.files_verified

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON result document (``portable_bundle_verification``)."""

        return {
            "content_id": self.content_id,
            "eligible_runtimes": list(self.eligible_runtimes),
            "files_checked": self.files_checked,
            "files_verified": self.files_verified,
            "ok": self.ok,
            "reason": self.reason,
            "schema": VERIFICATION_SCHEMA,
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "valid_signatures": self.valid_signatures,
        }


def _optional_uint(value: object) -> bool:
    return value is None or (type(value) is int and 0 <= value <= MAX_SAFE_INTEGER)


def _is_timestamp(value: object) -> bool:
    return isinstance(value, str) and _TIMESTAMP.fullmatch(value) is not None


def _is_version(value: object) -> bool:
    return isinstance(value, str) and _VERSION.fullmatch(value) is not None


def _version_key(value: str) -> tuple[int, ...]:
    parts = [int(part) for part in value.split(".")]
    return tuple(parts + [0] * (4 - len(parts)))


# --------------------------------------------------------------------------
# Manifest structure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PortableBundleManifest:
    """A canonical, structurally valid portable-bundle manifest.

    Unknown non-critical extensions are kept verbatim, so
    ``PortableBundleManifest.from_bytes(data).canonical_bytes == data``.

    Attributes:
        canonical_bytes: Exact canonical manifest bytes.
        content_id: ``sha256:`` identity of :attr:`canonical_bytes`.
    """

    canonical_bytes: bytes = field(repr=False)
    content_id: str
    _value: Mapping[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        understood_extensions: Iterable[str] = (),
    ) -> PortableBundleManifest:
        """Parse and validate canonical manifest bytes.

        Args:
            data: Exact bytes of ``openmed-bundle.json``.
            understood_extensions: Extension names the caller implements.

        Returns:
            The validated manifest.

        Raises:
            PortableBundleError: With the first failing structural reason.
        """

        try:
            value = _parse_canonical(bytes(data))
            _validate_manifest(value, frozenset(understood_extensions))
        except _Refusal as refusal:
            raise PortableBundleError(refusal.reason) from None
        return cls(bytes(data), content_id(bytes(data)), value)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        understood_extensions: Iterable[str] = (),
    ) -> PortableBundleManifest:
        """Canonicalize and validate a manifest mapping.

        Args:
            value: Manifest mapping following ``portable_bundle.schema.json``.
            understood_extensions: Extension names the caller implements.

        Returns:
            The validated manifest.

        Raises:
            PortableBundleError: With the first failing structural reason.
        """

        return cls.from_bytes(
            canonical_json_bytes(value), understood_extensions=understood_extensions
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached copy of the manifest value."""

        return json.loads(self.canonical_bytes.decode("utf-8"))

    @property
    def bundle_id(self) -> str:
        """Return the declared bundle identifier."""

        return str(self._value["bundle"]["id"])

    @property
    def sequence(self) -> int:
        """Return the anti-rollback sequence number."""

        return int(self._value["bundle"]["sequence"])

    @property
    def runtime_ids(self) -> tuple[str, ...]:
        """Return runtime variant identifiers in manifest order."""

        return tuple(str(item["id"]) for item in self._value["runtimes"])

    def fallback_chain(self, runtime_id: str) -> tuple[str, ...]:
        """Return ``runtime_id`` followed by its declared fallbacks.

        Args:
            runtime_id: Identifier of a runtime variant in this manifest.

        Returns:
            Runtime identifiers to try in order.

        Raises:
            KeyError: If ``runtime_id`` is not declared.
        """

        by_id = {item["id"]: item for item in self._value["runtimes"]}
        if runtime_id not in by_id:
            raise KeyError("runtime is not declared")
        chain = [runtime_id]
        while (fallback := by_id[chain[-1]]["fallback"]) is not None:
            chain.append(fallback)
        return tuple(chain)


@dataclass(frozen=True)
class _Validated:
    bundle_id: str
    sequence: int
    issued_at: str
    expires_at: str
    sdks: tuple[tuple[str, str, str | None], ...]
    policy_range: tuple[int, int]
    runtimes: tuple[Mapping[str, Any], ...]
    files: tuple[Mapping[str, Any], ...]


def _object(value: object, keys: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _fail(_MALFORMED)
    if not set(value) <= set(keys):
        raise _fail(_R.UNKNOWN_CRITICAL_FIELD)
    if len(value) != len(keys):
        raise _fail(_MALFORMED)
    return value


def _string(value: object, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise _fail(_MALFORMED)
    return value


def _choice(value: object, choices: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise _fail(_MALFORMED)
    return value


def _uint(value: object, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= MAX_SAFE_INTEGER:
        raise _fail(_MALFORMED)
    return value


def _array(value: object, minimum: int, maximum: int) -> list[Any]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise _fail(_MALFORMED)
    return value


def _unique(items: Sequence[Any]) -> None:
    if len(set(items)) != len(items):
        raise _fail(_MALFORMED)


def _path(value: object) -> str:
    if not isinstance(value, str):
        raise _fail(_MALFORMED)
    segments = value.split("/")
    if (
        not 0 < len(value) <= _MAX_PATH_CHARS
        or len(segments) > _MAX_PATH_SEGMENTS
        or not all(_PATH_SEGMENT.fullmatch(segment) for segment in segments)
        or value.lower() in _RESERVED_PATHS
    ):
        raise _fail(_R.PATH_UNSAFE)
    return value


def _path_list(value: object, minimum: int, maximum: int) -> list[str]:
    items = [_path(item) for item in _array(value, minimum, maximum)]
    _unique(items)
    return items


def _nullable(value: object, parse: Any) -> Any:
    return None if value is None else parse(value)


def _version(value: object) -> str:
    if not _is_version(value):
        raise _fail(_MALFORMED)
    return str(value)


def _validate_manifest(value: Any, understood: frozenset[str]) -> _Validated:
    if not isinstance(value, dict):
        raise _fail(_MALFORMED)
    if value.get("schema") != PORTABLE_BUNDLE_SCHEMA:
        raise _fail(_R.SCHEMA_UNSUPPORTED)
    version = value.get("schema_version")
    if type(version) is not int:
        raise _fail(_MALFORMED)
    if version != PORTABLE_BUNDLE_SCHEMA_VERSION:
        raise _fail(_R.SCHEMA_VERSION_UNSUPPORTED)
    top = _object(value, _TOP_KEYS)
    references: list[str] = []

    bundle = _object(top["bundle"], _BUNDLE_KEYS)
    bundle_id = _string(bundle["id"], _BUNDLE_ID)
    _string(bundle["version"], _SEMVER)
    sequence = _uint(bundle["sequence"])
    issued_at = _string(bundle["issued_at"], _TIMESTAMP)
    expires_at = _string(bundle["expires_at"], _TIMESTAMP)
    if issued_at >= expires_at:
        raise _fail(_MALFORMED)

    model = _object(top["model"], _MODEL_KEYS)
    _string(model["id"], _MODEL_ID)
    _nullable(model["revision"], lambda item: _string(item, _REVISION))
    _string(model["family"], _TOKEN)
    _string(model["architecture"], _TOKEN)
    _string(model["task"], _TOKEN)
    languages = [_string(item, _LANGUAGE) for item in _array(model["languages"], 1, 32)]
    _unique(languages)

    if top["tokenizer"] is not None:
        tokenizer = _object(top["tokenizer"], _TOKENIZER_KEYS)
        _string(tokenizer["kind"], _TOKEN)
        _uint(tokenizer["contract_version"], 1)
        references += _path_list(tokenizer["files"], 1, 64)

    if top["labels"] is not None:
        labels = _object(top["labels"], _LABELS_KEYS)
        references.append(_path(labels["path"]))
        _uint(labels["count"], 1)
        _string(labels["scheme"], _TOKEN)

    offsets = _object(top["offsets"], _OFFSETS_KEYS)
    _choice(offsets["unit"], OFFSET_UNITS)
    _uint(offsets["contract_version"], 1)

    decoder = _object(top["decoder"], _DECODER_KEYS)
    _string(decoder["kind"], _TOKEN)
    _uint(decoder["contract_version"], 1)

    if top["segmenter"] is not None:
        segmenter = _object(top["segmenter"], _SEGMENTER_KEYS)
        _string(segmenter["id"], _TOKEN)
        references += _path_list(segmenter["files"], 1, 64)

    policy = _object(top["policy"], _POLICY_KEYS)
    policy_min = _uint(policy["min_version"], 1)
    policy_max = _uint(policy["max_version_exclusive"], 1)
    if policy_min >= policy_max:
        raise _fail(_MALFORMED)
    if policy["path"] is not None:
        references.append(_path(policy["path"]))

    if top["calibration"] is not None:
        calibration = _object(top["calibration"], _CALIBRATION_KEYS)
        _uint(calibration["profile_version"], 1)
        references.append(_path(calibration["path"]))

    compatibility = _object(top["compatibility"], _COMPATIBILITY_KEYS)
    sdks = []
    for entry in _array(compatibility["sdks"], 1, 16):
        sdk = _object(entry, _SDK_KEYS)
        name = _string(sdk["name"], _TOKEN)
        minimum = _version(sdk["min"])
        maximum = _nullable(sdk["max_exclusive"], _version)
        if maximum is not None and _version_key(minimum) >= _version_key(maximum):
            raise _fail(_MALFORMED)
        sdks.append((name, minimum, maximum))
    _unique([item[0] for item in sdks])

    runtimes = []
    for entry in _array(top["runtimes"], 1, 32):
        runtime = _object(entry, _RUNTIME_KEYS)
        _string(runtime["id"], _TOKEN)
        runtime_format = _choice(runtime["format"], frozenset(FORMAT_PLATFORMS))
        precision = _choice(runtime["precision"], PRECISIONS)
        entrypoint = _path(runtime["entrypoint"])
        runtime_files = _path_list(runtime["files"], 1, 1024)
        if entrypoint not in runtime_files:
            raise _fail(_MALFORMED)
        references += runtime_files
        platforms = []
        for item in _array(runtime["platforms"], 1, 16):
            platform = _object(item, _PLATFORM_KEYS)
            name = _choice(platform["platform"], PLATFORMS)
            min_os = _nullable(platform["min_os"], _version)
            max_os = _nullable(platform["max_os_exclusive"], _version)
            if (
                min_os is not None
                and max_os is not None
                and _version_key(min_os) >= _version_key(max_os)
            ):
                raise _fail(_MALFORMED)
            platforms.append(name)
        _unique(platforms)
        _uint(runtime["min_memory_bytes"])
        _uint(runtime["min_storage_bytes"])
        _nullable(runtime["fallback"], lambda item: _string(item, _TOKEN))
        if (
            precision not in FORMAT_PRECISIONS[runtime_format]
            or not set(platforms) <= FORMAT_PLATFORMS[runtime_format]
        ):
            raise _fail(_R.RUNTIME_COMBINATION_UNSUPPORTED)
        runtimes.append(runtime)

    for entry in _array(top["licenses"], 1, 16):
        license_entry = _object(entry, _LICENSE_KEYS)
        _string(license_entry["id"], _LICENSE)
        references += _path_list(license_entry["files"], 0, 64)

    provenance = _object(top["provenance"], _PROVENANCE_KEYS)
    _nullable(provenance["source_digest"], lambda item: _string(item, _SHA256))
    if provenance["export_record"] is not None:
        references.append(_path(provenance["export_record"]))

    for entry in _array(top["evidence"], 0, 64):
        evidence = _object(entry, _EVIDENCE_KEYS)
        _string(evidence["kind"], _TOKEN)
        references.append(_path(evidence["path"]))

    files = []
    for entry in _array(top["files"], 1, 65536):
        item = _object(entry, _FILE_KEYS)
        _path(item["path"])
        _string(item["media_type"], _MEDIA_TYPE)
        _uint(item["size_bytes"])
        _string(item["sha256"], _SHA256)
        _choice(item["role"], FILE_ROLES)
        if not isinstance(item["executable"], bool):
            raise _fail(_MALFORMED)
        if item["executable"]:
            raise _fail(_R.EXECUTABLE_FILE_FORBIDDEN)
        files.append(item)

    extensions = top["extensions"]
    if not isinstance(extensions, dict) or len(extensions) > 64:
        raise _fail(_MALFORMED)
    for name in extensions:
        _string(name, _EXTENSION)
    critical = [
        _string(item, _EXTENSION) for item in _array(top["critical_extensions"], 0, 64)
    ]
    _unique(critical)
    if not set(critical) <= set(extensions):
        raise _fail(_MALFORMED)

    # Cross-field checks, in the normative order.
    paths = [str(item["path"]) for item in files]
    folded = [path.lower() for path in paths]
    if len(set(folded)) != len(folded):
        raise _fail(_R.DUPLICATE_PATH)
    folded_set = set(folded)
    for path in folded:
        segments = path.split("/")
        for index in range(1, len(segments)):
            if "/".join(segments[:index]) in folded_set:
                raise _fail(_R.PATH_CONFLICT)
    if any(
        paths[index].encode("utf-8") >= paths[index + 1].encode("utf-8")
        for index in range(len(paths) - 1)
    ):
        raise _fail(_R.FILE_ORDER_INVALID)
    declared = set(paths)
    if any(reference not in declared for reference in references):
        raise _fail(_R.FILE_REFERENCE_UNDECLARED)

    runtime_ids = [str(item["id"]) for item in runtimes]
    _unique(runtime_ids)
    by_id = {str(item["id"]): item for item in runtimes}
    for runtime in runtimes:
        fallback = runtime["fallback"]
        if fallback is None:
            continue
        if fallback not in by_id or fallback == runtime["id"]:
            raise _fail(_R.FALLBACK_INVALID)
        seen = {runtime["id"]}
        current = fallback
        while current is not None:
            if current in seen:
                raise _fail(_R.FALLBACK_INVALID)
            seen.add(current)
            current = by_id[current]["fallback"]
            if current is not None and current not in by_id:
                raise _fail(_R.FALLBACK_INVALID)

    if not set(critical) <= understood:
        raise _fail(_R.UNKNOWN_CRITICAL_EXTENSION)

    return _Validated(
        bundle_id=bundle_id,
        sequence=sequence,
        issued_at=issued_at,
        expires_at=expires_at,
        sdks=tuple(sdks),
        policy_range=(policy_min, policy_max),
        runtimes=tuple(runtimes),
        files=tuple(files),
    )


# --------------------------------------------------------------------------
# Signature envelope
# --------------------------------------------------------------------------


def _parse_envelope(data: bytes) -> tuple[str, list[Mapping[str, Any]]]:
    malformed = _R.ENVELOPE_MALFORMED
    try:
        value = _parse_canonical(data)
    except _Refusal:
        raise _fail(malformed) from None
    if not isinstance(value, dict) or value.get("schema") != ENVELOPE_SCHEMA:
        raise _fail(malformed)
    version = value.get("schema_version")
    if type(version) is not int:
        raise _fail(malformed)
    if version != ENVELOPE_SCHEMA_VERSION:
        raise _fail(_R.ENVELOPE_VERSION_UNSUPPORTED)
    if set(value) != set(_ENVELOPE_KEYS):
        raise _fail(malformed)
    identity = value["content_id"]
    signatures = value["signatures"]
    if (
        not isinstance(identity, str)
        or _SHA256.fullmatch(identity) is None
        or not isinstance(signatures, list)
        or len(signatures) > MAX_SIGNATURES
    ):
        raise _fail(malformed)
    for entry in signatures:
        if (
            not isinstance(entry, dict)
            or set(entry) != set(_SIGNATURE_KEYS)
            or not isinstance(entry["kid"], str)
            or _KID.fullmatch(entry["kid"]) is None
            or not isinstance(entry["alg"], str)
            or not isinstance(entry["sig"], str)
        ):
            raise _fail(malformed)
    return identity, signatures


def _ed25519_verify(public_key: bytes, signature: bytes, message: bytes) -> bool:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError:
        raise _fail(_R.SIGNATURE_VERIFIER_UNAVAILABLE) from None
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except (InvalidSignature, ValueError):
        return False
    return True


def _check_signatures(
    signatures: Sequence[Mapping[str, Any]],
    manifest_bytes: bytes,
    policy: TrustPolicy,
    now: str,
) -> int:
    if not signatures:
        raise _fail(_R.SIGNATURE_MISSING)
    message = signing_input(manifest_bytes)
    valid: set[str] = set()
    first_failure: PortableBundleReason | None = None
    for entry in signatures:
        kid = str(entry["kid"])
        if kid in valid:
            continue
        reason = _signature_failure(entry, kid, message, policy, now)
        if reason is None:
            valid.add(kid)
        elif first_failure is None:
            first_failure = reason
    if len(valid) >= policy.threshold:
        return len(valid)
    raise _fail(first_failure or _R.SIGNATURE_THRESHOLD_NOT_MET)


def _signature_failure(
    entry: Mapping[str, Any],
    kid: str,
    message: bytes,
    policy: TrustPolicy,
    now: str,
) -> PortableBundleReason | None:
    if entry["alg"] != SIGNATURE_ALGORITHM:
        return _R.SIGNATURE_ALGORITHM_UNSUPPORTED
    key = policy.key(kid)
    if key is None:
        return _R.UNTRUSTED_KEY
    if key.revoked:
        return _R.KEY_REVOKED
    if now < key.not_before:
        return _R.KEY_NOT_YET_VALID
    if now >= key.not_after:
        return _R.KEY_EXPIRED
    signature = _b64url_decode(entry["sig"], 64)
    if signature is None or not _ed25519_verify(key.public_key, signature, message):
        return _R.SIGNATURE_INVALID
    return None


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


class _DirectoryTree:
    """Read-only view of a bundle directory that never follows symlinks."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._paths: dict[str, Path] = {}
        if root.is_symlink() or not root.is_dir():
            raise _fail(_R.FILE_MISSING)

        def walk_error(_error: OSError) -> None:
            raise _fail(_R.FILE_MISSING)

        for directory, directories, names in os.walk(
            root, followlinks=False, onerror=walk_error
        ):
            parent = Path(directory)
            for name in [*directories, *names]:
                candidate = parent / name
                mode = candidate.lstat().st_mode
                if stat.S_ISLNK(mode):
                    raise _fail(_R.PATH_UNSAFE)
                if stat.S_ISDIR(mode):
                    continue
                if not stat.S_ISREG(mode):
                    raise _fail(_R.PATH_UNSAFE)
                self._paths[candidate.relative_to(root).as_posix()] = candidate

    def paths(self) -> list[str]:
        return sorted(self._paths)

    def size(self, path: str) -> int | None:
        target = self._paths.get(path)
        return None if target is None else target.lstat().st_size

    def digest(self, path: str) -> str:
        hasher = hashlib.sha256()
        with self._paths[path].open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK):
                hasher.update(chunk)
        return "sha256:" + hasher.hexdigest()


class _MemoryTree:
    def __init__(self, files: Mapping[str, bytes]) -> None:
        if not all(
            isinstance(key, str) and isinstance(value, (bytes, bytearray))
            for key, value in files.items()
        ):
            raise PortableBundleError(_R.CONTEXT_INVALID)
        self._files = {key: bytes(value) for key, value in files.items()}

    def paths(self) -> list[str]:
        return sorted(self._files)

    def size(self, path: str) -> int | None:
        data = self._files.get(path)
        return None if data is None else len(data)

    def digest(self, path: str) -> str:
        return "sha256:" + hashlib.sha256(self._files[path]).hexdigest()


def verify_portable_bundle(
    manifest_bytes: bytes,
    envelope_bytes: bytes,
    *,
    trust_policy: TrustPolicy,
    context: VerificationContext | None = None,
    files: Mapping[str, bytes] | str | os.PathLike[str] | None = None,
) -> PortableBundleVerification:
    """Verify a portable bundle offline and return a value-free result.

    Checks run in the normative order documented in
    ``docs/models/portable-bundle.md``; the first failure is reported. No check
    opens a network connection or executes bundled content.

    Args:
        manifest_bytes: Exact bytes of ``openmed-bundle.json``.
        envelope_bytes: Exact bytes of ``openmed-bundle.sig.json``.
        trust_policy: Caller-controlled trust roots.
        context: Verification instant, target, rollback floors, and limits.
        files: Bundle file tree as a ``path -> bytes`` mapping or a bundle
            directory. ``None`` verifies the manifest and signatures only.

    Returns:
        The verification result. ``files_verified`` is ``True`` only when a
        tree was supplied and matched the manifest exactly.

    Raises:
        PortableBundleError: ``context_invalid`` or ``trust_policy_invalid``
            when caller inputs are malformed.
    """

    if not isinstance(trust_policy, TrustPolicy):
        raise PortableBundleError(_R.TRUST_POLICY_INVALID)
    context = context if context is not None else VerificationContext()
    if not isinstance(context, VerificationContext):
        raise PortableBundleError(_R.CONTEXT_INVALID)
    manifest_bytes = bytes(manifest_bytes)
    envelope_bytes = bytes(envelope_bytes)
    identity: str | None = None
    valid_signatures = 0
    eligible: tuple[int, ...] = ()
    checked = 0
    now = str(context.now)

    def rejected(reason: PortableBundleReason) -> PortableBundleVerification:
        return PortableBundleVerification(
            ok=False,
            reason=reason.value,
            content_id=identity,
            valid_signatures=valid_signatures,
            eligible_runtimes=eligible,
            files_checked=checked,
        )

    try:
        if len(manifest_bytes) > context.max_manifest_bytes:
            raise _fail(_R.MANIFEST_TOO_LARGE)
        identity = content_id(manifest_bytes)
        if len(envelope_bytes) > context.max_envelope_bytes:
            raise _fail(_R.ENVELOPE_TOO_LARGE)
        declared_identity, signatures = _parse_envelope(envelope_bytes)
        if declared_identity != identity:
            raise _fail(_R.CONTENT_ID_MISMATCH)
        if identity in trust_policy.revoked_content_ids:
            raise _fail(_R.BUNDLE_REVOKED)
        valid_signatures = _check_signatures(
            signatures, manifest_bytes, trust_policy, now
        )
        value = _parse_canonical(manifest_bytes)
        manifest = _validate_manifest(value, context.understood_extensions)
        if len(manifest.files) > context.max_files or (
            sum(int(item["size_bytes"]) for item in manifest.files)
            > context.max_total_bytes
        ):
            raise _fail(_R.SIZE_LIMIT_EXCEEDED)
        if now < manifest.issued_at:
            raise _fail(_R.BUNDLE_NOT_YET_VALID)
        if now >= manifest.expires_at:
            raise _fail(_R.BUNDLE_EXPIRED)
        floor = context.sequence_floors.get(manifest.bundle_id)
        if floor is not None and manifest.sequence < floor:
            raise _fail(_R.ROLLBACK_REJECTED)
        eligible = _eligible_runtimes(manifest, context.target)
        if files is not None:
            tree: _DirectoryTree | _MemoryTree
            if isinstance(files, Mapping):
                tree = _MemoryTree(files)
            else:
                tree = _DirectoryTree(Path(files))
            for item in manifest.files:
                path = str(item["path"])
                size = tree.size(path)
                if size is None:
                    raise _fail(_R.FILE_MISSING)
                if size != item["size_bytes"]:
                    raise _fail(_R.FILE_SIZE_MISMATCH)
                if tree.digest(path) != item["sha256"]:
                    raise _fail(_R.FILE_DIGEST_MISMATCH)
                checked += 1
            declared = {str(item["path"]) for item in manifest.files}
            if any(
                path not in declared and path not in _RESERVED_PATHS
                for path in tree.paths()
            ):
                raise _fail(_R.FILE_UNDECLARED)
    except _Refusal as refusal:
        return rejected(refusal.reason)
    except OSError:
        return rejected(_R.FILE_MISSING)
    return PortableBundleVerification(
        ok=True,
        reason=None,
        content_id=identity,
        valid_signatures=valid_signatures,
        eligible_runtimes=eligible,
        files_checked=checked,
        files_verified=files is not None,
    )


def verify_portable_bundle_dir(
    root: str | os.PathLike[str],
    *,
    trust_policy: TrustPolicy,
    context: VerificationContext | None = None,
) -> PortableBundleVerification:
    """Verify a bundle directory that contains its manifest and envelope.

    Args:
        root: Bundle directory.
        trust_policy: Caller-controlled trust roots.
        context: Verification instant, target, rollback floors, and limits.

    Returns:
        The verification result for the directory and its file tree.
    """

    base = Path(root)
    context = context if context is not None else VerificationContext()
    try:
        manifest_bytes = _read_bounded(
            base / MANIFEST_FILENAME, context.max_manifest_bytes
        )
        envelope_bytes = _read_bounded(
            base / ENVELOPE_FILENAME, context.max_envelope_bytes
        )
    except _Refusal as refusal:
        return PortableBundleVerification(
            ok=False, reason=refusal.reason.value, content_id=None
        )
    return verify_portable_bundle(
        manifest_bytes,
        envelope_bytes,
        trust_policy=trust_policy,
        context=context,
        files=base,
    )


def _read_bounded(path: Path, limit: int) -> bytes:
    too_large = (
        _R.MANIFEST_TOO_LARGE
        if path.name == MANIFEST_FILENAME
        else _R.ENVELOPE_TOO_LARGE
    )
    try:
        if path.is_symlink() or not path.is_file():
            raise _fail(_R.FILE_MISSING)
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
    except OSError:
        raise _fail(_R.FILE_MISSING) from None
    if len(data) > limit:
        raise _fail(too_large)
    return data


def _eligible_runtimes(
    manifest: _Validated, target: VerificationTarget | None
) -> tuple[int, ...]:
    if target is None:
        return tuple(range(len(manifest.runtimes)))
    sdk = next((item for item in manifest.sdks if item[0] == target.sdk_name), None)
    version = _version_key(target.sdk_version)
    if (
        sdk is None
        or version < _version_key(sdk[1])
        or (sdk[2] is not None and version >= _version_key(sdk[2]))
    ):
        raise _fail(_R.SDK_INCOMPATIBLE)
    low, high = manifest.policy_range
    if target.policy_version is not None and not low <= target.policy_version < high:
        raise _fail(_R.POLICY_INCOMPATIBLE)
    eligible = tuple(
        index
        for index, runtime in enumerate(manifest.runtimes)
        if _runtime_fits(runtime, target)
    )
    if not eligible:
        raise _fail(_R.PLATFORM_INCOMPATIBLE)
    return eligible


def _runtime_fits(runtime: Mapping[str, Any], target: VerificationTarget) -> bool:
    if runtime["format"] not in target.runtimes:
        return False
    if (
        target.memory_bytes is not None
        and runtime["min_memory_bytes"] > target.memory_bytes
    ):
        return False
    if (
        target.storage_bytes is not None
        and runtime["min_storage_bytes"] > target.storage_bytes
    ):
        return False
    for platform in runtime["platforms"]:
        if platform["platform"] != target.platform:
            continue
        low, high = platform["min_os"], platform["max_os_exclusive"]
        if low is None and high is None:
            return True
        if target.os_version is None:
            return False
        version = _version_key(target.os_version)
        return (low is None or version >= _version_key(low)) and (
            high is None or version < _version_key(high)
        )
    return False


# --------------------------------------------------------------------------
# Producing bundles
# --------------------------------------------------------------------------


def encode_portable_manifest(
    manifest: Mapping[str, Any],
    *,
    understood_extensions: Iterable[str] = (),
) -> bytes:
    """Return validated canonical bytes for a manifest mapping.

    Args:
        manifest: Manifest mapping following ``portable_bundle.schema.json``.
        understood_extensions: Extension names the producer implements.

    Returns:
        Canonical manifest bytes ready to write and sign.

    Raises:
        PortableBundleError: With the first failing structural reason.
    """

    return PortableBundleManifest.from_mapping(
        manifest, understood_extensions=understood_extensions
    ).canonical_bytes


def sign_portable_manifest(
    manifest_bytes: bytes,
    signers: Sequence[tuple[str, bytes]],
) -> bytes:
    """Return a canonical signature envelope for canonical manifest bytes.

    Signing keys are caller-controlled raw 32-byte Ed25519 seeds. They are used
    only in memory and are never written, logged, or embedded. Multiple signers
    produce a rotation-ready envelope with one signature per key.

    Args:
        manifest_bytes: Canonical manifest bytes from
            :func:`encode_portable_manifest`.
        signers: ``(kid, seed)`` pairs.

    Returns:
        Canonical bytes for ``openmed-bundle.sig.json``.

    Raises:
        PortableBundleError: ``manifest_not_canonical`` for non-canonical
            input, ``signing_key_invalid`` for malformed keys, or
            ``signature_verifier_unavailable`` without ``cryptography``.
    """

    manifest_bytes = bytes(manifest_bytes)
    try:
        _parse_canonical(manifest_bytes)
    except _Refusal as refusal:
        raise PortableBundleError(refusal.reason) from None
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError:
        raise PortableBundleError(_R.SIGNATURE_VERIFIER_UNAVAILABLE) from None
    if not signers or len(signers) > MAX_SIGNATURES:
        raise PortableBundleError(_R.SIGNING_KEY_INVALID)
    message = signing_input(manifest_bytes)
    entries = []
    for kid, seed in signers:
        if (
            not isinstance(kid, str)
            or _KID.fullmatch(kid) is None
            or not isinstance(seed, (bytes, bytearray))
            or len(seed) != 32
        ):
            raise PortableBundleError(_R.SIGNING_KEY_INVALID)
        signature = Ed25519PrivateKey.from_private_bytes(bytes(seed)).sign(message)
        entries.append(
            {"alg": SIGNATURE_ALGORITHM, "kid": kid, "sig": _b64url_encode(signature)}
        )
    return canonical_json_bytes(
        {
            "content_id": content_id(manifest_bytes),
            "schema": ENVELOPE_SCHEMA,
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "signatures": entries,
        }
    )


def ed25519_public_key(seed: bytes) -> bytes:
    """Return the raw public key for a caller-controlled Ed25519 seed.

    Args:
        seed: Raw 32-byte Ed25519 private seed.

    Returns:
        Raw 32-byte public key suitable for :class:`TrustedKey`.

    Raises:
        PortableBundleError: ``signing_key_invalid`` or
            ``signature_verifier_unavailable``.
    """

    if not isinstance(seed, (bytes, bytearray)) or len(seed) != 32:
        raise PortableBundleError(_R.SIGNING_KEY_INVALID)
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError:
        raise PortableBundleError(_R.SIGNATURE_VERIFIER_UNAVAILABLE) from None
    return (
        Ed25519PrivateKey.from_private_bytes(bytes(seed))
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )


def inventory_files(
    root: str | os.PathLike[str],
    *,
    roles: Mapping[str, str] | None = None,
    media_types: Mapping[str, str] | None = None,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return sorted ``files`` entries for every regular file under ``root``.

    The manifest and signature envelope are excluded. Paths must satisfy the
    portable path grammar; symlinks and special files are refused. Every file
    is measured here; a precomputed inventory with ``path``, ``size_bytes``,
    and ``sha256`` rows (for example an export-provenance file list) may be
    supplied as ``records`` and is cross-checked, never trusted on its own.

    Args:
        root: Bundle directory.
        roles: Optional ``path -> role`` overrides; defaults are inferred.
        media_types: Optional ``path -> media type`` overrides.
        records: Optional precomputed ``path``/``size_bytes``/``sha256`` rows.

    Returns:
        Manifest ``files`` entries sorted by path.

    Raises:
        PortableBundleError: ``path_unsafe`` for unsupported paths or links,
            ``file_missing`` when a record names an absent file, or
            ``file_size_mismatch``/``file_digest_mismatch`` when a record
            disagrees with the tree.
    """

    base = Path(root)
    try:
        tree = _DirectoryTree(base)
    except _Refusal as refusal:
        raise PortableBundleError(refusal.reason) from None
    paths = [path for path in tree.paths() if path not in _RESERVED_PATHS]
    measured = {}
    for path in paths:
        try:
            _path(path)
        except _Refusal:
            raise PortableBundleError(_R.PATH_UNSAFE) from None
        measured[path] = (tree.size(path), tree.digest(path))
    for record in records or ():
        record_path = record.get("path") if isinstance(record, Mapping) else None
        if not isinstance(record_path, str) or record_path not in measured:
            raise PortableBundleError(_R.FILE_MISSING)
        size, digest = measured[record_path]
        if record.get("size_bytes") != size:
            raise PortableBundleError(_R.FILE_SIZE_MISMATCH)
        if record.get("sha256") != digest:
            raise PortableBundleError(_R.FILE_DIGEST_MISMATCH)
    roles = dict(roles or {})
    media_types = dict(media_types or {})
    entries = []
    for path in sorted(measured, key=lambda item: item.encode("utf-8")):
        size, digest = measured[path]
        entries.append(
            {
                "executable": False,
                "media_type": media_types.get(path) or _infer_media_type(path),
                "path": path,
                "role": roles.get(path) or _infer_role(path),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return entries


_MEDIA_TYPES = {
    ".json": "application/json",
    ".jsonl": "application/jsonl",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".onnx": "application/onnx",
    ".ort": "application/vnd.onnxruntime.ort",
    ".safetensors": "application/vnd.safetensors",
    ".npz": "application/x-npz",
    ".mlmodel": "application/vnd.apple.coreml",
    ".bin": "application/octet-stream",
}

_ROLE_NAMES = {
    "added_tokens.json": "tokenizer",
    "config.json": "config",
    "id2label.json": "labels",
    "label_map.json": "labels",
    "merges.txt": "tokenizer",
    "sentencepiece.bpe.model": "tokenizer",
    "special_tokens_map.json": "tokenizer",
    "spiece.model": "tokenizer",
    "tokenizer.json": "tokenizer",
    "tokenizer.model": "tokenizer",
    "tokenizer_config.json": "tokenizer",
    "vocab.json": "tokenizer",
    "vocab.txt": "tokenizer",
}


def _infer_media_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return _MEDIA_TYPES.get(suffix, "application/octet-stream")


def _infer_role(path: str) -> str:
    name = Path(path).name
    lowered = name.lower()
    if name in _ROLE_NAMES:
        return _ROLE_NAMES[name]
    if lowered.startswith(("license", "copying", "notice")):
        return "license"
    suffix = Path(path).suffix.lower()
    if suffix in {".onnx", ".ort", ".mlmodel"}:
        return "graph"
    if suffix in {".safetensors", ".npz", ".bin", ".weights"} or "weights" in lowered:
        return "weights"
    return "metadata"


def write_portable_bundle(
    root: str | os.PathLike[str],
    manifest_bytes: bytes,
    envelope_bytes: bytes,
) -> tuple[Path, Path]:
    """Write canonical manifest and envelope bytes into a bundle directory.

    Existing manifest or envelope files are refused rather than overwritten.

    Args:
        root: Bundle directory.
        manifest_bytes: Canonical manifest bytes.
        envelope_bytes: Canonical envelope bytes for the same manifest.

    Returns:
        Paths of the written manifest and envelope.

    Raises:
        PortableBundleError: ``content_id_mismatch`` when the envelope does not
            name the manifest, or ``path_conflict`` when either file exists.
    """

    base = Path(root)
    try:
        identity, _signatures = _parse_envelope(bytes(envelope_bytes))
        _parse_canonical(bytes(manifest_bytes))
    except _Refusal as refusal:
        raise PortableBundleError(refusal.reason) from None
    if identity != content_id(bytes(manifest_bytes)):
        raise PortableBundleError(_R.CONTENT_ID_MISMATCH)
    manifest_path = base / MANIFEST_FILENAME
    envelope_path = base / ENVELOPE_FILENAME
    try:
        with manifest_path.open("xb") as handle:
            handle.write(bytes(manifest_bytes))
        with envelope_path.open("xb") as handle:
            handle.write(bytes(envelope_bytes))
    except FileExistsError:
        raise PortableBundleError(_R.PATH_CONFLICT) from None
    return manifest_path, envelope_path


def load_portable_bundle_schema(name: str = "portable_bundle") -> dict[str, Any]:
    """Load one bundled portable-bundle JSON Schema.

    Args:
        name: One of :data:`SCHEMA_NAMES`.

    Returns:
        The parsed JSON Schema.

    Raises:
        KeyError: If ``name`` is not a portable-bundle schema.
    """

    if name not in SCHEMA_NAMES:
        raise KeyError("unknown portable bundle schema")
    resource = resources.files(SCHEMA_PACKAGE).joinpath(f"{name}.schema.json")
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "DEFAULT_MAX_ENVELOPE_BYTES",
    "DEFAULT_MAX_FILES",
    "DEFAULT_MAX_MANIFEST_BYTES",
    "DEFAULT_MAX_TOTAL_BYTES",
    "ENVELOPE_FILENAME",
    "ENVELOPE_SCHEMA",
    "ENVELOPE_SCHEMA_VERSION",
    "FILE_ROLES",
    "FORMAT_PLATFORMS",
    "FORMAT_PRECISIONS",
    "MANIFEST_FILENAME",
    "MAX_SAFE_INTEGER",
    "MAX_SIGNATURES",
    "OFFSET_UNITS",
    "PAYLOAD_TYPE",
    "PLATFORMS",
    "PORTABLE_BUNDLE_SCHEMA",
    "PORTABLE_BUNDLE_SCHEMA_VERSION",
    "PRECISIONS",
    "REASON_CODES",
    "SCHEMA_NAMES",
    "SIGNATURE_ALGORITHM",
    "TRUST_POLICY_SCHEMA",
    "TRUST_POLICY_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA",
    "VERIFICATION_SCHEMA_VERSION",
    "PortableBundleError",
    "PortableBundleManifest",
    "PortableBundleReason",
    "PortableBundleVerification",
    "TrustPolicy",
    "TrustedKey",
    "VerificationContext",
    "VerificationTarget",
    "canonical_json_bytes",
    "content_id",
    "ed25519_public_key",
    "encode_portable_manifest",
    "inventory_files",
    "load_portable_bundle_schema",
    "sign_portable_manifest",
    "signing_input",
    "verify_portable_bundle",
    "verify_portable_bundle_dir",
    "write_portable_bundle",
]
