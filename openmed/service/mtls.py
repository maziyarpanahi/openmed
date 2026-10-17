"""Mutual-TLS client certificate authentication for the REST service."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import math
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote_to_bytes

from cryptography import x509
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.x509.verification import (
    PolicyBuilder,
    Store,
    VerificationError,
)
from fastapi import Request
from starlette.responses import Response

from openmed.core.config import PROFILE_ENV_VAR

from .auth import AuthPrincipal, parse_bool, parse_non_negative_float
from .logging import set_access_log_identity

SERVICE_MTLS_ENABLED_ENV_VAR = "OPENMED_SERVICE_MTLS_ENABLED"
SERVICE_MTLS_CA_BUNDLE_ENV_VAR = "OPENMED_SERVICE_MTLS_CA_BUNDLE"
SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR = "OPENMED_SERVICE_MTLS_CLIENT_CERT_HEADER"
SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR = "OPENMED_SERVICE_MTLS_TRUSTED_PROXIES"
SERVICE_MTLS_PRINCIPALS_ENV_VAR = "OPENMED_SERVICE_MTLS_PRINCIPALS"
SERVICE_MTLS_CRL_FILES_ENV_VAR = "OPENMED_SERVICE_MTLS_CRL_FILES"
SERVICE_MTLS_CRL_GRACE_SECONDS_ENV_VAR = "OPENMED_SERVICE_MTLS_CRL_GRACE_SECONDS"
SERVICE_MTLS_CRL_FAIL_CLOSED_ENV_VAR = "OPENMED_SERVICE_MTLS_CRL_FAIL_CLOSED"

MAX_CLIENT_CERT_CHAIN_BYTES = 256 * 1024
MAX_CRL_FILE_BYTES = 64 * 1024 * 1024
# Profiles where CRL problems are reported instead of failing closed. Every
# other profile, including the default ``prod`` and custom profiles, fails
# closed.
CRL_REPORT_ONLY_PROFILES = frozenset({"dev", "test"})
_PEM_CRL_MARKER = b"-----BEGIN X509 CRL-----"
_LOGGER = logging.getLogger(__name__)
_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")

ErrorResponseFactory = Callable[..., Response]
CallNext = Callable[[Request], Awaitable[Response]]
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


@dataclass(frozen=True)
class MTLSPrincipalMapping:
    """Map exact certificate identities to a route-authorization principal."""

    identities: frozenset[str]
    principal: str
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MTLSClientIdentity:
    """Verified client certificate identity exposed to request handlers."""

    subject: str
    sans: tuple[str, ...]
    fingerprint_sha256: str

    @property
    def identifiers(self) -> tuple[str, ...]:
        """Return exact values eligible for configured principal mapping."""
        return (self.subject, *self.sans)


@dataclass(frozen=True)
class ServiceMTLSConfig:
    """Mutual-TLS settings for the REST service."""

    enabled: bool = False
    ca_bundle: Optional[Path] = None
    client_cert_header: Optional[str] = None
    trusted_proxies: tuple[IPNetwork, ...] = ()
    principals: tuple[MTLSPrincipalMapping, ...] = ()
    crl_files: tuple[Path, ...] = ()
    crl_grace_seconds: float = 0.0
    crl_fail_closed: bool = True


class MTLSAuthenticationError(ValueError):
    """mTLS authentication failure safe to expose through the API envelope.

    The code and message are fixed strings. They never contain certificate
    subjects, SANs, serial numbers, or file paths.
    """

    def __init__(self, *, code: str, message: str, status_code: int = 401) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def parse_service_mtls_config() -> ServiceMTLSConfig:
    """Read mTLS settings from the current process environment."""
    enabled = parse_bool(
        os.getenv(SERVICE_MTLS_ENABLED_ENV_VAR),
        env_var=SERVICE_MTLS_ENABLED_ENV_VAR,
        default=False,
    )
    if not enabled:
        return ServiceMTLSConfig(enabled=False)

    raw_ca_bundle = os.getenv(SERVICE_MTLS_CA_BUNDLE_ENV_VAR)
    if raw_ca_bundle is None or not raw_ca_bundle.strip():
        raise ValueError(
            f"{SERVICE_MTLS_CA_BUNDLE_ENV_VAR} is required when mTLS is enabled"
        )
    ca_bundle = Path(raw_ca_bundle.strip())
    if not ca_bundle.is_file():
        raise ValueError(f"{SERVICE_MTLS_CA_BUNDLE_ENV_VAR} must name a readable file")

    client_cert_header = _parse_client_cert_header(
        os.getenv(SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR)
    )
    trusted_proxies = parse_trusted_proxies(
        os.getenv(SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR)
    )
    if client_cert_header is not None and not trusted_proxies:
        raise ValueError(
            f"{SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR} is required when "
            f"{SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR} is configured"
        )

    return ServiceMTLSConfig(
        enabled=True,
        ca_bundle=ca_bundle,
        client_cert_header=client_cert_header,
        trusted_proxies=trusted_proxies,
        principals=parse_principal_mappings(os.getenv(SERVICE_MTLS_PRINCIPALS_ENV_VAR)),
        crl_files=parse_crl_files(os.getenv(SERVICE_MTLS_CRL_FILES_ENV_VAR)),
        crl_grace_seconds=parse_non_negative_float(
            os.getenv(SERVICE_MTLS_CRL_GRACE_SECONDS_ENV_VAR),
            env_var=SERVICE_MTLS_CRL_GRACE_SECONDS_ENV_VAR,
            default=0.0,
        ),
        crl_fail_closed=parse_crl_fail_closed(
            os.getenv(SERVICE_MTLS_CRL_FAIL_CLOSED_ENV_VAR),
            profile=os.getenv(PROFILE_ENV_VAR),
        ),
    )


def parse_crl_files(raw_value: Optional[str]) -> tuple[Path, ...]:
    """Parse comma-separated local CRL file paths.

    Args:
        raw_value: Comma-separated paths to PEM or DER CRL files, one CRL per
            file. Empty items are ignored.

    Returns:
        Configured CRL paths in order. Files are read and verified later by
        :class:`LocalCRLChecker`.
    """
    if raw_value is None or not raw_value.strip():
        return ()
    return tuple(Path(item.strip()) for item in raw_value.split(",") if item.strip())


def parse_crl_fail_closed(raw_value: Optional[str], *, profile: Optional[str]) -> bool:
    """Resolve whether CRL problems fail closed for the active profile.

    Args:
        raw_value: Optional boolean override from the environment.
        profile: Active ``OPENMED_PROFILE`` value. Missing means ``prod``.

    Returns:
        ``True`` when invalid or stale CRLs must reject requests. Profiles
        outside :data:`CRL_REPORT_ONLY_PROFILES` always fail closed.

    Raises:
        ValueError: If the override is not a boolean, or if it tries to
            disable fail-closed checks outside a development profile.
    """
    active_profile = (profile or "").strip() or "prod"
    production = active_profile not in CRL_REPORT_ONLY_PROFILES
    fail_closed = parse_bool(
        raw_value,
        env_var=SERVICE_MTLS_CRL_FAIL_CLOSED_ENV_VAR,
        default=production,
    )
    if production and not fail_closed:
        raise ValueError(
            f"{SERVICE_MTLS_CRL_FAIL_CLOSED_ENV_VAR} cannot be disabled outside "
            "development profiles"
        )
    return fail_closed


def parse_trusted_proxies(raw_value: Optional[str]) -> tuple[IPNetwork, ...]:
    """Parse comma-separated trusted proxy addresses and CIDR networks."""
    if raw_value is None or not raw_value.strip():
        return ()

    parsed: list[IPNetwork] = []
    for item in raw_value.split(","):
        value = item.strip()
        if not value:
            continue
        try:
            parsed.append(ipaddress.ip_network(value, strict=False))
        except ValueError as exc:
            raise ValueError(
                f"{SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR} contains an invalid "
                f"address or network"
            ) from exc
    return tuple(parsed)


def parse_principal_mappings(
    raw_value: Optional[str],
) -> tuple[MTLSPrincipalMapping, ...]:
    """Parse exact subject/SAN-to-principal mappings from JSON."""
    if raw_value is None or not raw_value.strip():
        return ()

    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{SERVICE_MTLS_PRINCIPALS_ENV_VAR} must be valid JSON"
        ) from exc
    if not isinstance(payload, list):
        raise ValueError(f"{SERVICE_MTLS_PRINCIPALS_ENV_VAR} must be a JSON list")

    mappings: list[MTLSPrincipalMapping] = []
    claimed_identities: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"{SERVICE_MTLS_PRINCIPALS_ENV_VAR}[{index}] must be a JSON object"
            )
        identities = _normalize_identities(
            item.get("identities", item.get("identity")),
            index=index,
        )
        duplicate = claimed_identities.intersection(identities)
        if duplicate:
            raise ValueError(
                f"{SERVICE_MTLS_PRINCIPALS_ENV_VAR} contains a duplicate identity"
            )
        claimed_identities.update(identities)

        principal = str(item.get("principal") or "").strip()
        if not principal:
            raise ValueError(
                f"{SERVICE_MTLS_PRINCIPALS_ENV_VAR}[{index}] requires principal"
            )
        mappings.append(
            MTLSPrincipalMapping(
                identities=frozenset(identities),
                principal=principal,
                scopes=_normalize_scopes(item.get("scopes", item.get("scope"))),
            )
        )
    return tuple(mappings)


def verify_client_certificate_chain(
    client_cert_chain: Sequence[str],
    ca_bundle: str | Path,
    *,
    crl_checker: Optional[LocalCRLChecker] = None,
) -> MTLSClientIdentity:
    """Verify a PEM client chain against a CA bundle and return its identity.

    Args:
        client_cert_chain: Leaf-first PEM client certificate chain.
        ca_bundle: PEM bundle containing one or more trusted CA certificates.
        crl_checker: Optional offline revocation checker applied to the
            verified chain before the identity is returned.

    Returns:
        Subject, SAN values, and SHA-256 fingerprint from the verified leaf.

    Raises:
        MTLSAuthenticationError: If a certificate in the verified chain is
            revoked, or a relevant CRL is stale and the checker fails closed.
        ValueError: If the chain or CA bundle is invalid or untrusted.
    """
    ca_path = Path(ca_bundle)
    try:
        ca_certificates = x509.load_pem_x509_certificates(ca_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("mTLS CA bundle is invalid") from exc
    if not ca_certificates:
        raise ValueError("mTLS CA bundle does not contain a certificate")
    return _verify_client_certificate_chain(
        client_cert_chain,
        Store(ca_certificates),
        crl_checker=crl_checker,
    )


@dataclass(frozen=True)
class _TrustedCRL:
    """A CRL whose signature was verified against a configured CA."""

    crl: x509.CertificateRevocationList
    issuer_public_key: bytes
    next_update: datetime


class LocalCRLChecker:
    """Offline revocation checks against operator-supplied CRL files.

    CRL files are read and their signatures are verified once, at
    construction. Each CRL must be signed by a CA certificate from the
    configured bundle. Revocation and ``next_update`` freshness are checked on
    every call to :meth:`check`, so a CRL that goes stale while the service is
    running is caught without a restart.

    In fail-closed mode an invalid CRL raises ``crl_invalid`` at construction
    and a stale CRL raises ``crl_stale`` per request. In report-only mode an
    invalid CRL is skipped and a stale CRL is still used; each problem is
    logged once as a value-free warning that carries only the code. A revoked
    certificate is rejected in both modes.

    Logs and errors never include certificate subjects, SANs, serial numbers,
    file paths, or underlying exception text.
    """

    def __init__(
        self,
        crl_files: Sequence[str | Path],
        ca_certificates: Sequence[x509.Certificate],
        *,
        fail_closed: bool = True,
        grace_seconds: float = 0.0,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        """Load and verify CRL files.

        Args:
            crl_files: PEM or DER CRL files, one CRL per file.
            ca_certificates: Trusted CA certificates allowed to sign CRLs.
            fail_closed: Reject on invalid or stale CRLs instead of reporting.
            grace_seconds: Seconds a CRL stays acceptable after ``next_update``.
            clock: Optional callable returning the current time. Naive values
                are treated as UTC. Defaults to the system UTC clock.

        Raises:
            MTLSAuthenticationError: With code ``crl_invalid`` when a CRL file
                cannot be read, parsed, or verified and ``fail_closed`` is set.
            ValueError: If ``grace_seconds`` is negative or not finite.
        """
        if not math.isfinite(grace_seconds) or grace_seconds < 0:
            raise ValueError("mTLS CRL grace period must be a non-negative number")
        self._fail_closed = fail_closed
        self._grace = timedelta(seconds=grace_seconds)
        self._clock = clock or _utc_now
        self._reported_codes: set[str] = set()
        trusted: list[_TrustedCRL] = []
        for position, crl_file in enumerate(crl_files, start=1):
            try:
                trusted.append(_load_trusted_crl(Path(crl_file), ca_certificates))
            except MTLSAuthenticationError as exc:
                if fail_closed:
                    raise crl_invalid(position=position) from None
                _LOGGER.warning(
                    "mTLS CRL file at position %d was rejected (%s); continuing "
                    "without it in report-only mode",
                    position,
                    exc.code,
                    extra={"openmed_code": exc.code, "openmed_crl_position": position},
                )
        self._crls = tuple(trusted)

    def check(self, chain: Sequence[x509.Certificate]) -> None:
        """Check a verified, leaf-first chain against the loaded CRLs.

        Args:
            chain: Verified chain from leaf to trust anchor. Each certificate
                is checked against CRLs signed by its issuer in the chain.

        Raises:
            MTLSAuthenticationError: With code ``client_certificate_revoked``
                when any certificate is revoked, or ``crl_stale`` when a
                relevant CRL is past ``next_update`` plus the grace period and
                the checker fails closed.
        """
        relevant: list[_TrustedCRL] = []
        for certificate, issuer in zip(chain, chain[1:]):
            issuer_public_key = _public_key_der(issuer)
            for trusted in self._crls:
                if (
                    trusted.issuer_public_key != issuer_public_key
                    or trusted.crl.issuer != certificate.issuer
                ):
                    continue
                relevant.append(trusted)
                revoked = trusted.crl.get_revoked_certificate_by_serial_number(
                    certificate.serial_number
                )
                if revoked is not None:
                    raise client_certificate_revoked()

        if not relevant:
            return
        now = self._now()
        if any(now > trusted.next_update + self._grace for trusted in relevant):
            if self._fail_closed:
                raise crl_stale()
            self._report_once("crl_stale")

    def _now(self) -> datetime:
        current = self._clock()
        if current.tzinfo is None:
            return current.replace(tzinfo=timezone.utc)
        return current.astimezone(timezone.utc)

    def _report_once(self, code: str) -> None:
        if code in self._reported_codes:
            return
        self._reported_codes.add(code)
        _LOGGER.warning(
            "mTLS revocation check reported %s; request allowed in report-only mode",
            code,
            extra={"openmed_code": code},
        )


class ServiceMTLS:
    """Verify and attach an mTLS client principal to each HTTP request."""

    def __init__(
        self,
        config: ServiceMTLSConfig,
        *,
        error_response: ErrorResponseFactory,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.config = config
        self._error_response = error_response
        self._store: Optional[Store] = None
        self._crl_checker: Optional[LocalCRLChecker] = None
        if config.enabled:
            if config.ca_bundle is None:
                raise ValueError("mTLS CA bundle is required")
            try:
                ca_certificates = x509.load_pem_x509_certificates(
                    config.ca_bundle.read_bytes()
                )
            except (OSError, ValueError) as exc:
                raise ValueError("mTLS CA bundle is invalid") from exc
            if not ca_certificates:
                raise ValueError("mTLS CA bundle does not contain a certificate")
            self._store = Store(ca_certificates)
            if config.crl_files:
                self._crl_checker = LocalCRLChecker(
                    config.crl_files,
                    ca_certificates,
                    fail_closed=config.crl_fail_closed,
                    grace_seconds=config.crl_grace_seconds,
                    clock=clock,
                )

    @property
    def enabled(self) -> bool:
        """Return whether mTLS authentication is active."""
        return self.config.enabled

    async def dispatch(self, request: Request, call_next: CallNext) -> Response:
        """Require a trusted client certificate and attach its principal."""
        if not self.enabled:
            return await call_next(request)

        try:
            chain = self._certificate_chain(request)
            if self._store is None:
                raise invalid_client_certificate()
            # Revocation is checked inside verification, before any principal
            # mapping below can run.
            identity = _verify_client_certificate_chain(
                chain,
                self._store,
                crl_checker=self._crl_checker,
            )
        except MTLSAuthenticationError as exc:
            return self._authentication_failure(exc)
        except ValueError:
            return self._authentication_failure(invalid_client_certificate())

        principal = self._principal_for(identity)
        request.state.mtls_identity = identity
        request.state.auth_principal = principal
        request.scope["openmed.mtls"] = identity
        request.scope["openmed.auth"] = principal
        set_access_log_identity(
            request,
            principal=principal.subject,
            credential_type=principal.credential_type,
        )
        return await call_next(request)

    def _certificate_chain(self, request: Request) -> tuple[str, ...]:
        extensions = request.scope.get("extensions")
        tls_extension: Any = None
        if isinstance(extensions, Mapping):
            tls_extension = extensions.get("tls")
        if isinstance(tls_extension, Mapping):
            if tls_extension.get("client_cert_error"):
                raise invalid_client_certificate()
            chain = tls_extension.get("client_cert_chain", ())
            normalized = _normalize_pem_chain(chain)
            if normalized:
                return normalized

        header_name = self.config.client_cert_header
        if header_name is not None:
            forwarded_certificate = request.headers.get(header_name)
            if forwarded_certificate is not None:
                if not _peer_is_trusted(request, self.config.trusted_proxies):
                    raise invalid_client_certificate()
                try:
                    decoded = unquote_to_bytes(forwarded_certificate).decode("ascii")
                except (UnicodeDecodeError, ValueError) as exc:
                    raise invalid_client_certificate() from exc
                return _normalize_pem_chain(decoded)

        raise client_certificate_required()

    def _principal_for(self, identity: MTLSClientIdentity) -> AuthPrincipal:
        identifiers = set(identity.identifiers)
        for mapping in self.config.principals:
            if identifiers.intersection(mapping.identities):
                principal = mapping.principal
                scopes = mapping.scopes
                break
        else:
            principal = f"mtls:{identity.fingerprint_sha256}"
            scopes = ()

        return AuthPrincipal(
            subject=principal,
            credential_type="mtls",
            scopes=scopes,
            claims={
                "certificate_subject": identity.subject,
                "certificate_sans": identity.sans,
                "certificate_sha256": identity.fingerprint_sha256,
            },
        )

    def _authentication_failure(self, exc: MTLSAuthenticationError) -> Response:
        return self._error_response(
            exc.status_code,
            exc.code,
            exc.message,
            details=None,
        )


def client_certificate_required() -> MTLSAuthenticationError:
    """Return a missing-client-certificate error."""
    return MTLSAuthenticationError(
        code="mtls_certificate_required",
        message="A verified client certificate is required",
    )


def invalid_client_certificate() -> MTLSAuthenticationError:
    """Return a generic invalid-client-certificate error."""
    return MTLSAuthenticationError(
        code="mtls_certificate_invalid",
        message="The client certificate is invalid or untrusted",
    )


def client_certificate_revoked() -> MTLSAuthenticationError:
    """Return a revoked-client-certificate error."""
    return MTLSAuthenticationError(
        code="client_certificate_revoked",
        message="The client certificate has been revoked",
    )


def crl_stale() -> MTLSAuthenticationError:
    """Return an out-of-date certificate revocation list error."""
    return MTLSAuthenticationError(
        code="crl_stale",
        message="Certificate revocation status is out of date",
        status_code=503,
    )


def crl_invalid(*, position: Optional[int] = None) -> MTLSAuthenticationError:
    """Return an invalid certificate revocation list error.

    Args:
        position: Optional 1-based position of the rejected file in the
            configured CRL list. The file path is never included.
    """
    message = "A configured certificate revocation list is invalid"
    if position is not None:
        message = f"{message} (file position {position})"
    return MTLSAuthenticationError(
        code="crl_invalid",
        message=message,
        status_code=503,
    )


def _verify_client_certificate_chain(
    client_cert_chain: Sequence[str],
    store: Store,
    *,
    crl_checker: Optional[LocalCRLChecker] = None,
) -> MTLSClientIdentity:
    chain = _load_client_certificates(client_cert_chain)
    try:
        verified = (
            PolicyBuilder()
            .store(store)
            .build_client_verifier()
            .verify(chain[0], list(chain[1:]))
        )
    except (VerificationError, x509.UnsupportedGeneralNameType) as exc:
        raise ValueError("client certificate chain is untrusted") from exc

    if crl_checker is not None:
        crl_checker.check(tuple(verified.chain))

    leaf = chain[0]
    der = leaf.public_bytes(serialization.Encoding.DER)
    return MTLSClientIdentity(
        subject=leaf.subject.rfc4514_string(),
        sans=_certificate_sans(leaf),
        fingerprint_sha256=hashlib.sha256(der).hexdigest(),
    )


def _load_trusted_crl(
    path: Path,
    ca_certificates: Sequence[x509.Certificate],
) -> _TrustedCRL:
    # Errors are raised "from None" so tracebacks cannot carry OS error text
    # that names the file path.
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_CRL_FILE_BYTES + 1)
    except OSError:
        raise crl_invalid() from None
    if not data or len(data) > MAX_CRL_FILE_BYTES:
        raise crl_invalid()

    is_pem = data.lstrip().startswith(b"-----BEGIN")
    if is_pem and data.count(_PEM_CRL_MARKER) != 1:
        # A second PEM block would be silently ignored by the parser.
        raise crl_invalid()
    try:
        crl = x509.load_pem_x509_crl(data) if is_pem else x509.load_der_x509_crl(data)
        next_update = crl.next_update_utc
        is_delta = _crl_has_extension(crl, x509.DeltaCRLIndicator)
    except ValueError:
        raise crl_invalid() from None
    # Delta CRLs list only changes since a base CRL; treating one as a complete
    # list would accept certificates revoked in the base CRL. Without
    # next_update, freshness cannot be enforced.
    if is_delta or next_update is None:
        raise crl_invalid()

    for ca_certificate in ca_certificates:
        if ca_certificate.subject != crl.issuer:
            continue
        if not _may_sign_crls(ca_certificate):
            continue
        if _crl_signature_is_valid(crl, ca_certificate):
            return _TrustedCRL(
                crl=crl,
                issuer_public_key=_public_key_der(ca_certificate),
                next_update=next_update,
            )
    raise crl_invalid()


def _crl_has_extension(
    crl: x509.CertificateRevocationList,
    extension_type: type[x509.ExtensionType],
) -> bool:
    try:
        crl.extensions.get_extension_for_class(extension_type)
    except x509.ExtensionNotFound:
        return False
    return True


def _may_sign_crls(certificate: x509.Certificate) -> bool:
    try:
        key_usage = certificate.extensions.get_extension_for_class(x509.KeyUsage).value
    except x509.ExtensionNotFound:
        return True
    except ValueError:
        return False
    return key_usage.crl_sign


def _crl_signature_is_valid(
    crl: x509.CertificateRevocationList,
    certificate: x509.Certificate,
) -> bool:
    try:
        public_key = certificate.public_key()
        return crl.is_signature_valid(public_key)  # type: ignore[arg-type]
    except (TypeError, ValueError, UnsupportedAlgorithm):
        return False


def _public_key_der(certificate: x509.Certificate) -> bytes:
    return certificate.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_client_certificates(
    client_cert_chain: Sequence[str],
) -> tuple[x509.Certificate, ...]:
    normalized = _normalize_pem_chain(client_cert_chain)
    if not normalized:
        raise ValueError("client certificate chain is empty")
    certificates: list[x509.Certificate] = []
    try:
        for pem in normalized:
            certificates.extend(x509.load_pem_x509_certificates(pem.encode("ascii")))
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("client certificate chain is invalid") from exc
    if not certificates:
        raise ValueError("client certificate chain is empty")
    return tuple(certificates)


def _normalize_pem_chain(raw_chain: Any) -> tuple[str, ...]:
    if raw_chain is None:
        return ()
    if isinstance(raw_chain, str):
        chain = (raw_chain,)
    elif isinstance(raw_chain, Sequence) and not isinstance(
        raw_chain, (bytes, bytearray)
    ):
        chain = tuple(str(item) for item in raw_chain)
    else:
        raise ValueError("client certificate chain is invalid")

    total_bytes = 0
    normalized: list[str] = []
    for certificate in chain:
        value = certificate.strip()
        if not value:
            continue
        try:
            total_bytes += len(value.encode("ascii"))
        except UnicodeEncodeError as exc:
            raise ValueError("client certificate chain is invalid") from exc
        if total_bytes > MAX_CLIENT_CERT_CHAIN_BYTES:
            raise ValueError("client certificate chain is too large")
        normalized.append(value)
    return tuple(normalized)


def _certificate_sans(certificate: x509.Certificate) -> tuple[str, ...]:
    try:
        extension = certificate.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
    except x509.ExtensionNotFound:
        return ()

    sans: list[str] = []
    for name in extension:
        if isinstance(name, x509.DNSName):
            sans.append(f"dns:{name.value}")
        elif isinstance(name, x509.UniformResourceIdentifier):
            sans.append(f"uri:{name.value}")
        elif isinstance(name, x509.IPAddress):
            sans.append(f"ip:{name.value}")
        elif isinstance(name, x509.RFC822Name):
            sans.append(f"email:{name.value}")
    return tuple(sans)


def _parse_client_cert_header(raw_value: Optional[str]) -> Optional[str]:
    if raw_value is None or not raw_value.strip():
        return None
    value = raw_value.strip()
    if not _HEADER_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"{SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR} must be a valid HTTP "
            "header name"
        )
    return value


def _peer_is_trusted(request: Request, networks: Sequence[IPNetwork]) -> bool:
    if request.client is None:
        return False
    try:
        peer = ipaddress.ip_address(request.client.host)
    except ValueError:
        return False
    return any(peer in network for network in networks)


def _normalize_identities(raw_value: Any, *, index: int) -> tuple[str, ...]:
    if isinstance(raw_value, str):
        values = (raw_value,)
    elif isinstance(raw_value, Sequence) and not isinstance(
        raw_value, (bytes, bytearray)
    ):
        values = tuple(str(item) for item in raw_value)
    else:
        raise ValueError(
            f"{SERVICE_MTLS_PRINCIPALS_ENV_VAR}[{index}] requires identity or identities"
        )
    normalized = tuple(value.strip() for value in values if value.strip())
    if not normalized:
        raise ValueError(
            f"{SERVICE_MTLS_PRINCIPALS_ENV_VAR}[{index}] requires identity or identities"
        )
    return normalized


def _normalize_scopes(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        values = raw_value.replace(",", " ").split()
    elif isinstance(raw_value, Sequence) and not isinstance(
        raw_value, (bytes, bytearray)
    ):
        values = [str(value).strip() for value in raw_value]
    else:
        raise ValueError("mTLS principal scopes must be a string or list")
    return tuple(dict.fromkeys(value for value in values if value))


__all__ = [
    "CRL_REPORT_ONLY_PROFILES",
    "MAX_CLIENT_CERT_CHAIN_BYTES",
    "MAX_CRL_FILE_BYTES",
    "LocalCRLChecker",
    "MTLSAuthenticationError",
    "MTLSClientIdentity",
    "MTLSPrincipalMapping",
    "SERVICE_MTLS_CA_BUNDLE_ENV_VAR",
    "SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR",
    "SERVICE_MTLS_CRL_FAIL_CLOSED_ENV_VAR",
    "SERVICE_MTLS_CRL_FILES_ENV_VAR",
    "SERVICE_MTLS_CRL_GRACE_SECONDS_ENV_VAR",
    "SERVICE_MTLS_ENABLED_ENV_VAR",
    "SERVICE_MTLS_PRINCIPALS_ENV_VAR",
    "SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR",
    "ServiceMTLS",
    "ServiceMTLSConfig",
    "client_certificate_revoked",
    "crl_invalid",
    "crl_stale",
    "parse_crl_fail_closed",
    "parse_crl_files",
    "parse_principal_mappings",
    "parse_service_mtls_config",
    "parse_trusted_proxies",
    "verify_client_certificate_chain",
]
