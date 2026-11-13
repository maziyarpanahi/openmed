"""Listener policy and validated TLS/mTLS credentials for the gRPC service.

The gRPC server is loopback-only unless the caller supplies validated server
credentials. Plaintext binds on a non-loopback address need an explicit,
development-only override, and that override is refused outside the ``dev`` and
``test`` profiles.

Every refusal raises :class:`GrpcStartupError` carrying a stable reason code, an
optional credential category, and at most a certificate SHA-256 fingerprint.
Errors never include file paths, PEM bytes, private-key material, addresses, or
request content, and the underlying exception context is suppressed so that a
traceback cannot reveal them either.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import grpc
from cryptography import x509
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization

from openmed.core.config import PROFILE_ENV_VAR

StrPath = Union[str, "os.PathLike[str]"]

GRPC_REASON_ADDRESS_INVALID = "grpc_address_invalid"
GRPC_REASON_INSECURE_REMOTE_BIND = "grpc_insecure_remote_bind"
GRPC_REASON_INSECURE_DEV_OVERRIDE_REJECTED = "grpc_insecure_dev_override_rejected"
GRPC_REASON_TLS_INCOMPLETE = "grpc_tls_incomplete"
GRPC_REASON_TLS_CLIENT_AUTH_AMBIGUOUS = "grpc_tls_client_auth_ambiguous"
GRPC_REASON_TLS_UNREADABLE = "grpc_tls_unreadable"
GRPC_REASON_TLS_MALFORMED = "grpc_tls_malformed"
GRPC_REASON_TLS_KEY_MISMATCH = "grpc_tls_key_mismatch"
GRPC_REASON_TLS_CERTIFICATE_EXPIRED = "grpc_tls_certificate_expired"
GRPC_REASON_TLS_CERTIFICATE_NOT_YET_VALID = "grpc_tls_certificate_not_yet_valid"
GRPC_REASON_BIND_FAILED = "grpc_bind_failed"

GRPC_STARTUP_REASON_CODES = frozenset(
    {
        GRPC_REASON_ADDRESS_INVALID,
        GRPC_REASON_INSECURE_REMOTE_BIND,
        GRPC_REASON_INSECURE_DEV_OVERRIDE_REJECTED,
        GRPC_REASON_TLS_INCOMPLETE,
        GRPC_REASON_TLS_CLIENT_AUTH_AMBIGUOUS,
        GRPC_REASON_TLS_UNREADABLE,
        GRPC_REASON_TLS_MALFORMED,
        GRPC_REASON_TLS_KEY_MISMATCH,
        GRPC_REASON_TLS_CERTIFICATE_EXPIRED,
        GRPC_REASON_TLS_CERTIFICATE_NOT_YET_VALID,
        GRPC_REASON_BIND_FAILED,
    }
)

CATEGORY_CERTIFICATE_CHAIN = "certificate_chain"
CATEGORY_PRIVATE_KEY = "private_key"
CATEGORY_CLIENT_CA = "client_ca"
CATEGORY_CLIENT_AUTH = "client_auth"
CATEGORY_LISTENER = "listener"
CATEGORY_OVERRIDE = "override"

GRPC_CREDENTIAL_CATEGORIES = frozenset(
    {
        CATEGORY_CERTIFICATE_CHAIN,
        CATEGORY_PRIVATE_KEY,
        CATEGORY_CLIENT_CA,
        CATEGORY_CLIENT_AUTH,
        CATEGORY_LISTENER,
        CATEGORY_OVERRIDE,
    }
)

INSECURE_DEV_OVERRIDE_PROFILES = frozenset({"dev", "test"})
MAX_CREDENTIAL_FILE_BYTES = 1024 * 1024

TRANSPORT_PLAINTEXT = "plaintext"
TRANSPORT_TLS = "tls"
TRANSPORT_MTLS = "mtls"

_UNIX_PREFIXES = ("unix:", "unix-abstract:")
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
_DEFAULT_PROFILE = "prod"

logger = logging.getLogger(__name__)


class GrpcStartupError(RuntimeError):
    """Value-free refusal to start or bind a gRPC listener.

    Attributes:
        reason: Stable reason code from :data:`GRPC_STARTUP_REASON_CODES`.
        category: Controlled credential or listener category, when relevant.
        fingerprint_sha256: Lowercase SHA-256 fingerprint of the offending
            certificate, when one was parsed. Never key material.
    """

    def __init__(
        self,
        reason: str,
        *,
        category: Optional[str] = None,
        fingerprint_sha256: Optional[str] = None,
    ) -> None:
        self.reason = reason if reason in GRPC_STARTUP_REASON_CODES else "grpc_failed"
        self.category = category if category in GRPC_CREDENTIAL_CATEGORIES else None
        self.fingerprint_sha256 = (
            fingerprint_sha256
            if fingerprint_sha256 is not None
            and _FINGERPRINT_PATTERN.fullmatch(fingerprint_sha256)
            else None
        )
        detail = self.reason
        if self.category is not None:
            detail += f" [{self.category}]"
        if self.fingerprint_sha256 is not None:
            detail += f" (certificate sha256:{self.fingerprint_sha256})"
        super().__init__(f"gRPC startup refused: {detail}")


@dataclass(frozen=True, kw_only=True)
class GrpcTLSConfig:
    """File locations for gRPC server TLS and optional client verification.

    Attributes:
        certificate_chain_file: PEM server certificate, leaf first, optionally
            followed by intermediates.
        private_key_file: Unencrypted PEM private key matching the leaf.
        client_ca_file: PEM bundle of CAs trusted to issue client
            certificates. Supplying it enables mutual TLS.
        require_client_cert: Whether clients must present a certificate signed
            by ``client_ca_file``. ``None`` derives the value from whether a
            client CA is configured; ``True`` without a CA and ``False`` with a
            CA are both refused.
    """

    certificate_chain_file: Optional[StrPath]
    private_key_file: Optional[StrPath]
    client_ca_file: Optional[StrPath] = None
    require_client_cert: Optional[bool] = None


@dataclass(frozen=True)
class GrpcServerCredentials:
    """Validated gRPC server credentials plus value-free provenance.

    Attributes:
        server_credentials: grpcio credentials ready for ``add_secure_port``.
        certificate_sha256: Fingerprint of the served leaf certificate.
        certificate_not_valid_after: Expiry of the served leaf certificate.
        client_ca_sha256: Fingerprints of the trusted client CA certificates.
        require_client_cert: Whether the listener requires mutual TLS.
    """

    server_credentials: grpc.ServerCredentials = field(repr=False, compare=False)
    certificate_sha256: str
    certificate_not_valid_after: datetime
    client_ca_sha256: tuple[str, ...] = ()
    require_client_cert: bool = False

    @property
    def transport(self) -> str:
        """Return ``"mtls"`` when client certificates are required, else ``"tls"``."""
        return TRANSPORT_MTLS if self.require_client_cert else TRANSPORT_TLS


@dataclass(frozen=True)
class GrpcListener:
    """A validated listener decision that can be bound to a gRPC server.

    Attributes:
        address: The gRPC bind address.
        loopback: Whether the address is loopback or a local Unix socket.
        credentials: Validated TLS credentials, or ``None`` for plaintext.
        insecure_dev_override: Whether plaintext on a non-loopback address was
            allowed by the development override.
    """

    address: str = field(repr=False)
    loopback: bool
    credentials: Optional[GrpcServerCredentials] = None
    insecure_dev_override: bool = False

    @property
    def transport(self) -> str:
        """Return ``"plaintext"``, ``"tls"`` or ``"mtls"``."""
        if self.credentials is None:
            return TRANSPORT_PLAINTEXT
        return self.credentials.transport

    def report(self) -> dict[str, Any]:
        """Return a value-free summary suitable for logs and self-checks.

        The report contains controlled states and certificate fingerprints only;
        it never includes the address, file paths, or credential bytes.
        """
        credentials = self.credentials
        return {
            "loopback": self.loopback,
            "transport": self.transport,
            "insecure_dev_override": self.insecure_dev_override,
            "certificate_sha256": (
                credentials.certificate_sha256 if credentials is not None else None
            ),
            "certificate_not_valid_after": (
                credentials.certificate_not_valid_after.isoformat()
                if credentials is not None
                else None
            ),
            "client_ca_sha256": (
                list(credentials.client_ca_sha256) if credentials is not None else []
            ),
        }


def is_loopback_grpc_address(address: str) -> bool:
    """Return whether a gRPC bind address is reachable only from this host.

    Args:
        address: A ``host:port``, ``[ipv6]:port``, ``unix:`` or
            ``unix-abstract:`` address.

    Returns:
        ``True`` for loopback IP literals, ``localhost`` and Unix sockets;
        ``False`` for wildcard, routable or otherwise named hosts.

    Raises:
        GrpcStartupError: If the address cannot be parsed.
    """
    return _address_is_loopback(address)


def load_grpc_server_credentials(
    config: GrpcTLSConfig,
    *,
    now: Optional[datetime] = None,
) -> GrpcServerCredentials:
    """Load and validate server TLS material before any listener is bound.

    Args:
        config: Certificate chain, private key and optional client CA files.
        now: Reference time for certificate validity checks. Defaults to the
            current UTC time.

    Returns:
        Validated grpcio server credentials with value-free provenance.

    Raises:
        GrpcStartupError: If material is missing, unreadable, malformed,
            mismatched, outside its validity window, or the client
            certificate requirement is inconsistent with the client CA.
    """
    certificate_path = _configured_path(
        config.certificate_chain_file, CATEGORY_CERTIFICATE_CHAIN
    )
    key_path = _configured_path(config.private_key_file, CATEGORY_PRIVATE_KEY)
    client_ca_path = _configured_path(config.client_ca_file, CATEGORY_CLIENT_CA)
    if certificate_path is None:
        raise GrpcStartupError(
            GRPC_REASON_TLS_INCOMPLETE, category=CATEGORY_CERTIFICATE_CHAIN
        )
    if key_path is None:
        raise GrpcStartupError(
            GRPC_REASON_TLS_INCOMPLETE, category=CATEGORY_PRIVATE_KEY
        )
    require_client_cert = _require_client_cert(
        config.require_client_cert,
        has_client_ca=client_ca_path is not None,
    )

    certificate_bytes = _read_credential_file(
        certificate_path, CATEGORY_CERTIFICATE_CHAIN
    )
    chain = _load_certificates(certificate_bytes, CATEGORY_CERTIFICATE_CHAIN)
    leaf = chain[0]
    leaf_sha256 = certificate_sha256(leaf)
    _check_validity(leaf, leaf_sha256, now or datetime.now(timezone.utc))

    key_bytes = _read_credential_file(key_path, CATEGORY_PRIVATE_KEY)
    try:
        private_key = serialization.load_pem_private_key(key_bytes, password=None)
    except (TypeError, ValueError, UnsupportedAlgorithm):
        raise GrpcStartupError(
            GRPC_REASON_TLS_MALFORMED, category=CATEGORY_PRIVATE_KEY
        ) from None
    if _public_key_der(private_key.public_key()) != _public_key_der(leaf.public_key()):
        raise GrpcStartupError(
            GRPC_REASON_TLS_KEY_MISMATCH,
            category=CATEGORY_PRIVATE_KEY,
            fingerprint_sha256=leaf_sha256,
        )

    client_ca_bytes: Optional[bytes] = None
    client_ca_sha256: tuple[str, ...] = ()
    if client_ca_path is not None:
        client_ca_bytes = _read_credential_file(client_ca_path, CATEGORY_CLIENT_CA)
        client_ca_sha256 = tuple(
            certificate_sha256(certificate)
            for certificate in _load_certificates(client_ca_bytes, CATEGORY_CLIENT_CA)
        )

    try:
        server_credentials = grpc.ssl_server_credentials(
            [(key_bytes, certificate_bytes)],
            root_certificates=client_ca_bytes,
            require_client_auth=require_client_cert,
        )
    except Exception:
        raise GrpcStartupError(
            GRPC_REASON_TLS_MALFORMED,
            category=CATEGORY_CERTIFICATE_CHAIN,
            fingerprint_sha256=leaf_sha256,
        ) from None
    return GrpcServerCredentials(
        server_credentials=server_credentials,
        certificate_sha256=leaf_sha256,
        certificate_not_valid_after=leaf.not_valid_after_utc,
        client_ca_sha256=client_ca_sha256,
        require_client_cert=require_client_cert,
    )


def resolve_grpc_listener(
    address: str,
    *,
    tls: Union[GrpcTLSConfig, GrpcServerCredentials, None] = None,
    allow_insecure_remote_dev: bool = False,
    profile: Optional[str] = None,
) -> GrpcListener:
    """Decide whether and how a gRPC address may be bound, failing closed.

    Args:
        address: The gRPC bind address.
        tls: TLS file configuration to load and validate, or credentials that
            were already validated. ``None`` requests a plaintext listener.
        allow_insecure_remote_dev: Development-only override that permits a
            plaintext listener on a non-loopback address.
        profile: OpenMed profile name. Defaults to ``OPENMED_PROFILE``, which
            is treated as ``prod`` when unset.

    Returns:
        The validated listener decision.

    Raises:
        GrpcStartupError: If the address is invalid, the override is used
            outside the ``dev``/``test`` profiles, credentials are invalid, or
            a non-loopback plaintext listener is requested without the
            override.
    """
    if type(allow_insecure_remote_dev) is not bool:
        raise GrpcStartupError(
            GRPC_REASON_INSECURE_DEV_OVERRIDE_REJECTED, category=CATEGORY_OVERRIDE
        )
    loopback = _address_is_loopback(address)
    if (
        allow_insecure_remote_dev
        and _resolve_profile(profile) not in INSECURE_DEV_OVERRIDE_PROFILES
    ):
        raise GrpcStartupError(
            GRPC_REASON_INSECURE_DEV_OVERRIDE_REJECTED, category=CATEGORY_OVERRIDE
        )

    if isinstance(tls, GrpcServerCredentials):
        return GrpcListener(address=address, loopback=loopback, credentials=tls)
    if isinstance(tls, GrpcTLSConfig):
        return GrpcListener(
            address=address,
            loopback=loopback,
            credentials=load_grpc_server_credentials(tls),
        )
    if tls is not None:
        raise GrpcStartupError(
            GRPC_REASON_TLS_INCOMPLETE, category=CATEGORY_CERTIFICATE_CHAIN
        )
    if loopback:
        return GrpcListener(address=address, loopback=True)
    if not allow_insecure_remote_dev:
        raise GrpcStartupError(
            GRPC_REASON_INSECURE_REMOTE_BIND, category=CATEGORY_LISTENER
        )
    logger.warning(
        "grpc_insecure_dev_override_active",
        extra={"transport": TRANSPORT_PLAINTEXT, "loopback": False},
    )
    return GrpcListener(address=address, loopback=False, insecure_dev_override=True)


def bind_grpc_listener(server: grpc.Server, listener: GrpcListener) -> int:
    """Bind a validated listener and treat any non-binding as a startup error.

    Args:
        server: An unstarted grpcio server.
        listener: The decision returned by :func:`resolve_grpc_listener`.

    Returns:
        The bound port reported by grpcio (``1`` for Unix sockets).

    Raises:
        GrpcStartupError: If grpcio raises or reports that nothing was bound.
    """
    try:
        if listener.credentials is None:
            port = server.add_insecure_port(listener.address)
        else:
            port = server.add_secure_port(
                listener.address, listener.credentials.server_credentials
            )
    except Exception:
        raise GrpcStartupError(
            GRPC_REASON_BIND_FAILED, category=CATEGORY_LISTENER
        ) from None
    if type(port) is not int or port <= 0:
        raise GrpcStartupError(GRPC_REASON_BIND_FAILED, category=CATEGORY_LISTENER)
    return port


def certificate_sha256(certificate: x509.Certificate) -> str:
    """Return the lowercase SHA-256 fingerprint of a certificate's DER bytes.

    Args:
        certificate: The certificate to fingerprint.

    Returns:
        A 64-character lowercase hex digest, matching the REST mTLS identity
        fingerprint format.
    """
    der = certificate.public_bytes(serialization.Encoding.DER)
    return hashlib.sha256(der).hexdigest()


def _address_is_loopback(address: str) -> bool:
    if not isinstance(address, str) or not address or address != address.strip():
        raise GrpcStartupError(GRPC_REASON_ADDRESS_INVALID, category=CATEGORY_LISTENER)
    for prefix in _UNIX_PREFIXES:
        if address.startswith(prefix):
            if len(address) == len(prefix):
                raise GrpcStartupError(
                    GRPC_REASON_ADDRESS_INVALID, category=CATEGORY_LISTENER
                )
            return True

    if address.startswith("["):
        end = address.find("]")
        if end < 0 or address[end + 1 : end + 2] != ":":
            raise GrpcStartupError(
                GRPC_REASON_ADDRESS_INVALID, category=CATEGORY_LISTENER
            )
        host = address[1:end]
        port_text = address[end + 2 :]
        try:
            ipaddress.IPv6Address(host)
        except ValueError:
            raise GrpcStartupError(
                GRPC_REASON_ADDRESS_INVALID, category=CATEGORY_LISTENER
            ) from None
    else:
        host, separator, port_text = address.rpartition(":")
        if not separator or ":" in host or "/" in host:
            raise GrpcStartupError(
                GRPC_REASON_ADDRESS_INVALID, category=CATEGORY_LISTENER
            )

    if not port_text.isascii() or not port_text.isdigit() or int(port_text) > 65535:
        raise GrpcStartupError(GRPC_REASON_ADDRESS_INVALID, category=CATEGORY_LISTENER)
    return _is_loopback_host(host)


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
        return parsed.ipv4_mapped.is_loopback
    return parsed.is_loopback


def _resolve_profile(profile: Optional[str]) -> str:
    raw = profile if profile is not None else os.getenv(PROFILE_ENV_VAR)
    normalized = (raw or "").strip().lower()
    return normalized or _DEFAULT_PROFILE


def _configured_path(value: Optional[StrPath], category: str) -> Optional[Path]:
    if value is None:
        return None
    try:
        text = os.fspath(value)
    except TypeError:
        raise GrpcStartupError(GRPC_REASON_TLS_INCOMPLETE, category=category) from None
    if not isinstance(text, str) or not text.strip():
        return None
    return Path(text)


def _require_client_cert(requested: Optional[bool], *, has_client_ca: bool) -> bool:
    if requested is None:
        return has_client_ca
    if type(requested) is not bool:
        raise GrpcStartupError(
            GRPC_REASON_TLS_CLIENT_AUTH_AMBIGUOUS, category=CATEGORY_CLIENT_AUTH
        )
    if requested and not has_client_ca:
        raise GrpcStartupError(GRPC_REASON_TLS_INCOMPLETE, category=CATEGORY_CLIENT_CA)
    if not requested and has_client_ca:
        raise GrpcStartupError(
            GRPC_REASON_TLS_CLIENT_AUTH_AMBIGUOUS, category=CATEGORY_CLIENT_AUTH
        )
    return requested


def _read_credential_file(path: Path, category: str) -> bytes:
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_CREDENTIAL_FILE_BYTES + 1)
    except (OSError, ValueError):
        raise GrpcStartupError(GRPC_REASON_TLS_UNREADABLE, category=category) from None
    if not data.strip() or len(data) > MAX_CREDENTIAL_FILE_BYTES:
        raise GrpcStartupError(GRPC_REASON_TLS_MALFORMED, category=category)
    return data


def _load_certificates(data: bytes, category: str) -> list[x509.Certificate]:
    try:
        certificates = x509.load_pem_x509_certificates(data)
    except ValueError:
        raise GrpcStartupError(GRPC_REASON_TLS_MALFORMED, category=category) from None
    if not certificates:
        raise GrpcStartupError(GRPC_REASON_TLS_MALFORMED, category=category)
    return certificates


def _check_validity(
    certificate: x509.Certificate,
    fingerprint: str,
    now: datetime,
) -> None:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if now < certificate.not_valid_before_utc:
        raise GrpcStartupError(
            GRPC_REASON_TLS_CERTIFICATE_NOT_YET_VALID,
            category=CATEGORY_CERTIFICATE_CHAIN,
            fingerprint_sha256=fingerprint,
        )
    if now > certificate.not_valid_after_utc:
        raise GrpcStartupError(
            GRPC_REASON_TLS_CERTIFICATE_EXPIRED,
            category=CATEGORY_CERTIFICATE_CHAIN,
            fingerprint_sha256=fingerprint,
        )


def _public_key_der(public_key: Any) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


__all__ = [
    "CATEGORY_CERTIFICATE_CHAIN",
    "CATEGORY_CLIENT_AUTH",
    "CATEGORY_CLIENT_CA",
    "CATEGORY_LISTENER",
    "CATEGORY_OVERRIDE",
    "CATEGORY_PRIVATE_KEY",
    "GRPC_CREDENTIAL_CATEGORIES",
    "GRPC_REASON_ADDRESS_INVALID",
    "GRPC_REASON_BIND_FAILED",
    "GRPC_REASON_INSECURE_DEV_OVERRIDE_REJECTED",
    "GRPC_REASON_INSECURE_REMOTE_BIND",
    "GRPC_REASON_TLS_CERTIFICATE_EXPIRED",
    "GRPC_REASON_TLS_CERTIFICATE_NOT_YET_VALID",
    "GRPC_REASON_TLS_CLIENT_AUTH_AMBIGUOUS",
    "GRPC_REASON_TLS_INCOMPLETE",
    "GRPC_REASON_TLS_KEY_MISMATCH",
    "GRPC_REASON_TLS_MALFORMED",
    "GRPC_REASON_TLS_UNREADABLE",
    "GRPC_STARTUP_REASON_CODES",
    "GrpcListener",
    "GrpcServerCredentials",
    "GrpcStartupError",
    "GrpcTLSConfig",
    "INSECURE_DEV_OVERRIDE_PROFILES",
    "MAX_CREDENTIAL_FILE_BYTES",
    "TRANSPORT_MTLS",
    "TRANSPORT_PLAINTEXT",
    "TRANSPORT_TLS",
    "bind_grpc_listener",
    "certificate_sha256",
    "is_loopback_grpc_address",
    "load_grpc_server_credentials",
    "resolve_grpc_listener",
]
