"""Mutual-TLS client certificate authentication for the REST service."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote_to_bytes

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.verification import (
    PolicyBuilder,
    Store,
    VerificationError,
)
from fastapi import Request
from starlette.responses import Response

from .auth import AuthPrincipal, parse_bool
from .logging import set_access_log_identity

SERVICE_MTLS_ENABLED_ENV_VAR = "OPENMED_SERVICE_MTLS_ENABLED"
SERVICE_MTLS_CA_BUNDLE_ENV_VAR = "OPENMED_SERVICE_MTLS_CA_BUNDLE"
SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR = "OPENMED_SERVICE_MTLS_CLIENT_CERT_HEADER"
SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR = "OPENMED_SERVICE_MTLS_TRUSTED_PROXIES"
SERVICE_MTLS_PRINCIPALS_ENV_VAR = "OPENMED_SERVICE_MTLS_PRINCIPALS"

MAX_CLIENT_CERT_CHAIN_BYTES = 256 * 1024
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


class MTLSAuthenticationError(ValueError):
    """mTLS authentication failure safe to expose through the API envelope."""

    def __init__(self, *, code: str, message: str) -> None:
        self.code = code
        self.message = message
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
    )


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
) -> MTLSClientIdentity:
    """Verify a PEM client chain against a CA bundle and return its identity.

    Args:
        client_cert_chain: Leaf-first PEM client certificate chain.
        ca_bundle: PEM bundle containing one or more trusted CA certificates.

    Returns:
        Subject, SAN values, and SHA-256 fingerprint from the verified leaf.

    Raises:
        ValueError: If the chain or CA bundle is invalid or untrusted.
    """
    ca_path = Path(ca_bundle)
    try:
        ca_certificates = x509.load_pem_x509_certificates(ca_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("mTLS CA bundle is invalid") from exc
    if not ca_certificates:
        raise ValueError("mTLS CA bundle does not contain a certificate")
    return _verify_client_certificate_chain(client_cert_chain, Store(ca_certificates))


class ServiceMTLS:
    """Verify and attach an mTLS client principal to each HTTP request."""

    def __init__(
        self,
        config: ServiceMTLSConfig,
        *,
        error_response: ErrorResponseFactory,
    ) -> None:
        self.config = config
        self._error_response = error_response
        self._store: Optional[Store] = None
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
            identity = _verify_client_certificate_chain(chain, self._store)
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
            401,
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


def _verify_client_certificate_chain(
    client_cert_chain: Sequence[str],
    store: Store,
) -> MTLSClientIdentity:
    chain = _load_client_certificates(client_cert_chain)
    try:
        PolicyBuilder().store(store).build_client_verifier().verify(
            chain[0], list(chain[1:])
        )
    except (VerificationError, x509.UnsupportedGeneralNameType) as exc:
        raise ValueError("client certificate chain is untrusted") from exc

    leaf = chain[0]
    der = leaf.public_bytes(serialization.Encoding.DER)
    return MTLSClientIdentity(
        subject=leaf.subject.rfc4514_string(),
        sans=_certificate_sans(leaf),
        fingerprint_sha256=hashlib.sha256(der).hexdigest(),
    )


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
    "MAX_CLIENT_CERT_CHAIN_BYTES",
    "MTLSAuthenticationError",
    "MTLSClientIdentity",
    "MTLSPrincipalMapping",
    "SERVICE_MTLS_CA_BUNDLE_ENV_VAR",
    "SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR",
    "SERVICE_MTLS_ENABLED_ENV_VAR",
    "SERVICE_MTLS_PRINCIPALS_ENV_VAR",
    "SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR",
    "ServiceMTLS",
    "ServiceMTLSConfig",
    "parse_principal_mappings",
    "parse_service_mtls_config",
    "parse_trusted_proxies",
    "verify_client_certificate_chain",
]
