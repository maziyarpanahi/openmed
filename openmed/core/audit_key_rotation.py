"""Caller-owned signing-key rotation for deterministic audit reports.

``AuditReport`` already stores a non-secret key identifier alongside its
HMAC-SHA256 signature.  This module keeps key custody outside OpenMed while
providing the small adapter needed to sign with a current key and verify
reports signed by either a current or retained former key.

Key providers are deliberately callbacks or mappings supplied by the caller.
The provider and the resolved key material are never copied into an audit
report or included in an exception raised by this module.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, TypeAlias

from .audit import AuditReport

AUDIT_SIGNATURE_ALGORITHM: Final = "HMAC-SHA256"
_KEY_ID_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")

KeyMaterial: TypeAlias = bytes | str


class AuditKeyProvider(Protocol):
    """Resolve caller-owned HMAC key material by its non-secret key ID."""

    def __call__(self, key_id: str) -> KeyMaterial:
        """Return the key for ``key_id`` without exposing it to OpenMed."""


KeyProvider: TypeAlias = AuditKeyProvider | Mapping[str, KeyMaterial]


class AuditKeyRotationError(ValueError):
    """Raised when a rotation provider or report cannot satisfy the contract."""


def _require_key_id(value: Any) -> str:
    if not isinstance(value, str) or not _KEY_ID_RE.fullmatch(value):
        raise AuditKeyRotationError(
            "key_id must be a safe, non-secret metadata identifier"
        )
    return value


def _validate_provider(provider: Any) -> None:
    if not callable(provider) and not isinstance(provider, Mapping):
        raise TypeError("key_provider must be callable or a key-id mapping")


def _resolve_key(provider: KeyProvider, key_id: str) -> KeyMaterial:
    try:
        value = provider[key_id] if isinstance(provider, Mapping) else provider(key_id)
    except KeyError:
        raise AuditKeyRotationError(
            "key provider has no key for the requested key_id"
        ) from None
    except Exception:
        # Provider exceptions are caller-owned and may contain sensitive
        # details.  Do not preserve or interpolate those messages.
        raise AuditKeyRotationError(
            "key provider failed to resolve the requested key_id"
        ) from None

    if not isinstance(value, (bytes, str)) or not value:
        raise AuditKeyRotationError(
            "key provider returned empty or unsupported key material"
        )
    return value


def _coerce_report(report: AuditReport | Mapping[str, Any]) -> AuditReport:
    if isinstance(report, AuditReport):
        return report
    if not isinstance(report, Mapping):
        raise TypeError("audit_report must be an AuditReport or mapping")
    try:
        return AuditReport.from_dict(report)
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

        if not isinstance(report, AuditReport):
            raise TypeError("audit_report must be an AuditReport")
        effective_key_id = self.key_id if key_id is None else _require_key_id(key_id)
        key = _resolve_key(self.key_provider, effective_key_id)
        try:
            return report.sign(key, key_id=effective_key_id)
        except (TypeError, ValueError):
            raise AuditKeyRotationError("audit report could not be signed") from None


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
        except AuditKeyRotationError:
            return False
        signature = audit_report.signature
        if signature is None or signature.algorithm != AUDIT_SIGNATURE_ALGORITHM:
            return False
        try:
            key_id = _require_key_id(signature.key_id)
            key = _resolve_key(self.key_provider, key_id)
            return audit_report.verify(
                key,
                original_text=original_text,
                deidentified_text=deidentified_text,
            )
        except Exception:
            # Verification is intentionally fail-closed.  In particular, do
            # not surface caller-provider errors that could contain key data.
            return False


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
    "AUDIT_SIGNATURE_ALGORITHM",
    "AuditKeyProvider",
    "AuditKeyRotationError",
    "AuditKeyRotationSigner",
    "AuditKeyRotationVerifier",
    "AuditReportSigner",
    "AuditReportVerifier",
    "KeyMaterial",
    "KeyProvider",
    "sign_audit_report",
    "verify_audit_report",
]
