"""Caller-owned signing-key rotation for deterministic audit reports.

``AuditReport`` already stores a non-secret key identifier alongside its
HMAC-SHA256 signature.  This module keeps key custody outside OpenMed while
providing the small adapter needed to sign with a current key and verify
reports signed by either a current or retained former key.

Key providers are deliberately callbacks or mappings supplied by the caller.
The provider and the resolved key material are never copied into an audit
report or included in an exception raised by this module.

The same caller-owned keys can also sign value-free ``sha256:`` digests for a
named purpose, such as the audit continuity manifest. Digest signatures use a
separate per-purpose key derivation, so a report signature can never be
replayed as a digest signature or the other way round.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol, TypeAlias

from .audit import AuditReport

AUDIT_SIGNATURE_ALGORITHM: Final = "HMAC-SHA256"
MIN_AUDIT_HMAC_KEY_BYTES: Final = 32
MAX_AUDIT_HMAC_KEY_BYTES: Final = 4_096
AUDIT_DIGEST_PURPOSES: Final = frozenset({"audit-continuity"})
AUDIT_DIGEST_ROLES: Final = frozenset({"signer", "outgoing", "incoming"})
_KEY_ID_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_SIGNATURE_VALUE_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_KEY_DERIVATION_CONTEXT: Final = b"openmed.audit-key-rotation.v1\x00"
_DIGEST_DERIVATION_CONTEXT: Final = b"openmed.audit-digest-signature.v1\x00"

DigestSignatureStatus: TypeAlias = Literal["valid", "invalid", "unavailable"]
_AUDIT_REPORT_FIELDS: Final = frozenset(
    {
        "policy",
        "resolved_profile",
        "detectors",
        "safety_sweep",
        "spans",
        "thresholds",
        "residual_risk",
        "openmed_version",
        "manifest_hash",
        "document_length",
        "input_hash",
        "deidentified_text_hash",
        "grounding",
        "repro_hash",
        "signature",
    }
)

KeyMaterial: TypeAlias = bytes | str


class AuditKeyProvider(Protocol):
    """Resolve caller-owned HMAC key material by its non-secret key ID."""

    def __call__(self, key_id: str) -> KeyMaterial:
        """Return the key for ``key_id`` without exposing it to OpenMed."""


KeyProvider: TypeAlias = AuditKeyProvider | Mapping[str, KeyMaterial]


class AuditKeyRotationError(ValueError):
    """Raised when a rotation provider or report cannot satisfy the contract."""


class AuditKeyUnavailableError(AuditKeyRotationError):
    """Raised when a provider cannot resolve key material for a key ID."""


def _require_key_id(value: Any) -> str:
    if type(value) is not str or not _KEY_ID_RE.fullmatch(value):
        raise AuditKeyRotationError(
            "key_id must be a safe, non-secret metadata identifier"
        )
    return value


def _require_signed_digest(value: Any) -> str:
    if type(value) is not str or not _DIGEST_RE.fullmatch(value):
        raise AuditKeyRotationError("signed digest must be a sha256:<hex> digest")
    return value


def _require_purpose(value: Any) -> str:
    if type(value) is not str or value not in AUDIT_DIGEST_PURPOSES:
        raise AuditKeyRotationError("digest signature purpose is not supported")
    return value


def _require_role(value: Any) -> str:
    if type(value) is not str or value not in AUDIT_DIGEST_ROLES:
        raise AuditKeyRotationError("digest signature role is not supported")
    return value


def _validate_provider(provider: Any) -> None:
    if not callable(provider) and not isinstance(provider, Mapping):
        raise TypeError("key_provider must be callable or a key-id mapping")


def _resolve_key(provider: KeyProvider, key_id: str) -> bytes:
    try:
        value = provider[key_id] if isinstance(provider, Mapping) else provider(key_id)
    except KeyError:
        raise AuditKeyUnavailableError(
            "key provider has no key for the requested key_id"
        ) from None
    except Exception:
        # Provider exceptions are caller-owned and may contain sensitive
        # details.  Do not preserve or interpolate those messages.
        raise AuditKeyUnavailableError(
            "key provider failed to resolve the requested key_id"
        ) from None

    try:
        if type(value) is bytes:
            key = value
        elif type(value) is str:
            key = value.encode("utf-8")
        else:
            raise TypeError
    except Exception:
        raise AuditKeyRotationError(
            "key provider returned empty or unsupported key material"
        ) from None
    if not MIN_AUDIT_HMAC_KEY_BYTES <= len(key) <= MAX_AUDIT_HMAC_KEY_BYTES:
        raise AuditKeyRotationError(
            "key provider returned key material outside the supported size range"
        )
    return key


def _report_key(key: bytes, key_id: str) -> bytes:
    """Derive a per-ID HMAC key so new signatures bind their key metadata."""

    return hmac.new(
        key,
        _KEY_DERIVATION_CONTEXT + key_id.encode("ascii"),
        hashlib.sha256,
    ).digest()


def _digest_signature_value(
    key: bytes,
    *,
    digest: str,
    key_id: str,
    purpose: str,
    role: str,
) -> str:
    """Return the hex HMAC binding a digest to its key ID, purpose, and role."""

    derived = hmac.new(
        key,
        _DIGEST_DERIVATION_CONTEXT
        + purpose.encode("ascii")
        + b"\x00"
        + key_id.encode("ascii"),
        hashlib.sha256,
    ).digest()
    message = json.dumps(
        {"digest": digest, "key_id": key_id, "purpose": purpose, "role": role},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hmac.new(derived, message, hashlib.sha256).hexdigest()


def _sign_digest(
    provider: KeyProvider,
    digest: str,
    *,
    key_id: str,
    purpose: str,
    role: str,
) -> str:
    _require_signed_digest(digest)
    _require_key_id(key_id)
    _require_purpose(purpose)
    _require_role(role)
    key = _resolve_key(provider, key_id)
    return _digest_signature_value(
        key, digest=digest, key_id=key_id, purpose=purpose, role=role
    )


def _digest_signature_status(
    provider: KeyProvider,
    digest: Any,
    value: Any,
    *,
    key_id: Any,
    purpose: str,
    role: Any,
) -> DigestSignatureStatus:
    """Classify a digest signature without exposing provider or key details.

    Returns ``"unavailable"`` when the provider cannot resolve usable key
    material for ``key_id``, ``"invalid"`` for malformed input or a
    signature mismatch, and ``"valid"`` only for an exact match.
    """

    try:
        _require_signed_digest(digest)
        _require_key_id(key_id)
        _require_purpose(purpose)
        _require_role(role)
    except AuditKeyRotationError:
        return "invalid"
    if type(value) is not str or not _SIGNATURE_VALUE_RE.fullmatch(value):
        return "invalid"
    try:
        key = _resolve_key(provider, key_id)
    except AuditKeyRotationError:
        return "unavailable"
    expected = _digest_signature_value(
        key, digest=digest, key_id=key_id, purpose=purpose, role=role
    )
    return "valid" if hmac.compare_digest(expected, value) else "invalid"


def _snapshot_report_mapping(report: Mapping[str, Any]) -> dict[str, Any]:
    try:
        items = list(itertools.islice(report.items(), len(_AUDIT_REPORT_FIELDS) + 1))
    except Exception:
        raise AuditKeyRotationError(
            "audit report mapping could not be parsed"
        ) from None
    if len(items) > len(_AUDIT_REPORT_FIELDS):
        raise AuditKeyRotationError("audit report mapping could not be parsed")

    snapshot: dict[str, Any] = {}
    for item in items:
        if type(item) not in {list, tuple} or len(item) != 2:
            raise AuditKeyRotationError("audit report mapping could not be parsed")
        key, value = item
        if type(key) is not str or key in snapshot:
            raise AuditKeyRotationError("audit report mapping could not be parsed")
        snapshot[key] = value
    if set(snapshot) - _AUDIT_REPORT_FIELDS:
        raise AuditKeyRotationError("audit report mapping could not be parsed")
    return snapshot


def _coerce_report(report: AuditReport | Mapping[str, Any]) -> AuditReport:
    if type(report) is AuditReport:
        return report
    if not isinstance(report, Mapping):
        raise TypeError("audit_report must be an AuditReport or mapping")
    try:
        snapshot = _snapshot_report_mapping(report)
        parsed = AuditReport.from_dict(snapshot)
        if parsed.to_dict() != snapshot:
            raise ValueError
        return parsed
    except Exception:
        # Parsed report failures must not echo untrusted report fields.
        raise AuditKeyRotationError(
            "audit report mapping could not be parsed"
        ) from None


@dataclass(frozen=True, slots=True)
class AuditKeyRotationSigner:
    """Sign reports with one caller-selected active key ID.

    The provider is retained only as a caller-owned reference.  Construct a
    new signer, or pass ``key_id`` to :meth:`sign`, when the active signing key
    rotates.  Retired keys remain available to the verifier through its own
    provider until the caller's retention policy permits removal.
    """

    key_id: str
    key_provider: KeyProvider = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_key_id(self.key_id)
        _validate_provider(self.key_provider)

    def sign(self, report: AuditReport, *, key_id: str | None = None) -> AuditReport:
        """Sign ``report`` in place and return it.

        Args:
            report: The deterministic report to sign.
            key_id: Optional active-key override for this signing operation.

        Raises:
            AuditKeyRotationError: If the provider cannot resolve valid key
                material or the report cannot be signed.
            TypeError: If ``report`` is not an :class:`AuditReport`.
        """

        if type(report) is not AuditReport:
            raise TypeError("audit_report must be an AuditReport")
        effective_key_id = self.key_id if key_id is None else _require_key_id(key_id)
        key = _resolve_key(self.key_provider, effective_key_id)
        try:
            return report.sign(
                _report_key(key, effective_key_id),
                key_id=effective_key_id,
            )
        except Exception:
            raise AuditKeyRotationError("audit report could not be signed") from None

    def sign_digest(
        self,
        digest: str,
        *,
        purpose: str,
        role: str = "signer",
        key_id: str | None = None,
    ) -> str:
        """Return a hex HMAC-SHA256 signature over a value-free digest.

        Args:
            digest: A ``sha256:<hex>`` digest to sign.
            purpose: Signature purpose from :data:`AUDIT_DIGEST_PURPOSES`.
            role: Signer role from :data:`AUDIT_DIGEST_ROLES`, for example
                ``"outgoing"`` or ``"incoming"`` during a key rollover.
            key_id: Optional key-ID override for this signing operation.

        Returns:
            The 64-character hex signature value. It never contains key
            material.

        Raises:
            AuditKeyRotationError: If an argument is malformed or the provider
                cannot resolve valid key material.
        """

        effective_key_id = self.key_id if key_id is None else _require_key_id(key_id)
        return _sign_digest(
            self.key_provider,
            digest,
            key_id=effective_key_id,
            purpose=purpose,
            role=role,
        )


@dataclass(frozen=True, slots=True)
class AuditKeyRotationVerifier:
    """Verify reports using the key ID stored in each report signature.

    A verification provider should retain both the active and any still-valid
    retired keys.  Verification fails closed when the report is unsigned, the
    key ID is unknown, the provider fails, or the report content is changed.
    """

    key_provider: KeyProvider = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_provider(self.key_provider)

    def verify(
        self,
        report: AuditReport | Mapping[str, Any],
        *,
        original_text: str | None = None,
        deidentified_text: str | None = None,
    ) -> bool:
        """Return whether a report verifies against its caller-owned key.

        ``original_text`` and ``deidentified_text`` are optional in-memory
        bindings.  They are hashed by ``AuditReport`` and are never serialized
        by this module.
        """

        try:
            audit_report = _coerce_report(report)
        except (AuditKeyRotationError, TypeError):
            return False
        signature = audit_report.signature
        if signature is None or signature.algorithm != AUDIT_SIGNATURE_ALGORITHM:
            return False
        try:
            key_id = _require_key_id(signature.key_id)
            key = _resolve_key(self.key_provider, key_id)
            bindings = {
                "original_text": original_text,
                "deidentified_text": deidentified_text,
            }
            if audit_report.verify(_report_key(key, key_id), **bindings):
                return True
            # Reports signed directly by AuditReport before this adapter existed
            # used the provider key without the per-ID derivation.
            return audit_report.verify(key, **bindings)
        except Exception:
            # Verification is intentionally fail-closed.  In particular, do
            # not surface caller-provider errors that could contain key data.
            return False

    def verify_digest(
        self,
        digest: str,
        signature: str,
        *,
        key_id: str,
        purpose: str,
        role: str = "signer",
    ) -> bool:
        """Return whether ``signature`` signs ``digest`` for the given key ID.

        Verification fails closed for malformed input, an unknown or
        unavailable key ID, provider failures, and signature mismatches.

        Args:
            digest: The ``sha256:<hex>`` digest that was signed.
            signature: The hex signature value to check.
            key_id: Non-secret key identifier the signature claims.
            purpose: Signature purpose from :data:`AUDIT_DIGEST_PURPOSES`.
            role: Signer role from :data:`AUDIT_DIGEST_ROLES`.
        """

        try:
            status = _digest_signature_status(
                self.key_provider,
                digest,
                signature,
                key_id=key_id,
                purpose=purpose,
                role=role,
            )
        except Exception:
            return False
        return status == "valid"


AuditReportSigner = AuditKeyRotationSigner
AuditReportVerifier = AuditKeyRotationVerifier


def sign_audit_report(
    report: AuditReport,
    *,
    key_id: str,
    key_provider: KeyProvider,
) -> AuditReport:
    """Sign an audit report using a caller-owned key provider."""

    return AuditKeyRotationSigner(key_id=key_id, key_provider=key_provider).sign(report)


def verify_audit_report(
    report: AuditReport | Mapping[str, Any],
    *,
    key_provider: KeyProvider,
    original_text: str | None = None,
    deidentified_text: str | None = None,
) -> bool:
    """Verify an audit report using its embedded key ID and a key provider."""

    return AuditKeyRotationVerifier(key_provider=key_provider).verify(
        report,
        original_text=original_text,
        deidentified_text=deidentified_text,
    )


__all__ = [
    "AUDIT_DIGEST_PURPOSES",
    "AUDIT_DIGEST_ROLES",
    "AUDIT_SIGNATURE_ALGORITHM",
    "MAX_AUDIT_HMAC_KEY_BYTES",
    "MIN_AUDIT_HMAC_KEY_BYTES",
    "AuditKeyProvider",
    "AuditKeyRotationError",
    "AuditKeyUnavailableError",
    "AuditKeyRotationSigner",
    "AuditKeyRotationVerifier",
    "AuditReportSigner",
    "AuditReportVerifier",
    "KeyMaterial",
    "KeyProvider",
    "sign_audit_report",
    "verify_audit_report",
]
