"""Fail-closed egress policy for service-controlled outbound connections.

Every OpenMed adapter that can open an outbound connection asks this module
for an :class:`EgressDecision` before it serializes a request body or opens a
transport: webhook callbacks, KServe remote inference, SMART backend services
and FHIR Bulk Data, the OpenHIM mediator, the privacy gateway, the OTLP trace
exporter, and MCP OAuth metadata discovery (see :data:`ADAPTER_COVERAGE`).
A decision is scoped to one adapter, purpose, data class, credential class,
destination, deadline, and byte budget. Absence of a decision is denial.

Modes, in order of precedence:

* ``OPENMED_OFFLINE`` / ``local_only`` deny every outbound operation and
  override all allow rules, including on redirects and retries.
* ``deny`` refuses every outbound operation.
* ``enforce`` (the default for the ``prod`` profile and any unrecognized
  profile) allows only operations that match an operator rule.
* ``permissive`` (the default for the ``dev`` and ``test`` profiles) also
  allows HTTPS or gRPCS to public addresses without a rule.

Rules come from ``OPENMED_EGRESS_RULES`` (a JSON array) and/or the JSON file
named by ``OPENMED_EGRESS_RULES_FILE``. Each rule names its purposes and one
exact origin, for example::

    [{"purposes": ["webhook"], "destination": "https://hooks.example.org"}]

Destinations are resolved before connecting. Loopback, private, link-local,
multicast, reserved, unspecified, and cloud-metadata addresses are refused
unless the matching rule explicitly lists the range in ``cidrs`` (a cloud
metadata address must be listed as a single-address network). Clients built by
:func:`egress_http_client` and :func:`egress_async_http_client` connect only to
the exact addresses that passed this check, so a DNS answer that changes
between the check and the connection cannot move the socket.

Receipts carry only the policy version and digest, adapter, purpose, data and
credential classes, an opaque destination digest, the destination class,
attempt counts, byte totals, timing, and a controlled reason code. URLs, hosts,
query strings, headers, credentials, and payloads never enter receipts, logs,
or error messages.
"""

from __future__ import annotations

import functools
import hashlib
import http.client
import ipaddress
import json
import logging
import math
import os
import secrets
import socket
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlsplit

from openmed.core.offline import is_local_only
from openmed.mcp.upstream_endpoint_policy import (
    ResolvedAddressClass,
    UpstreamEndpointPolicyError,
    canonical_host,
    classify_address,
    resolve_hostname,
)

EGRESS_POLICY_VERSION = "1"
EGRESS_MODE_ENV_VAR = "OPENMED_EGRESS_MODE"
EGRESS_RULES_ENV_VAR = "OPENMED_EGRESS_RULES"
EGRESS_RULES_FILE_ENV_VAR = "OPENMED_EGRESS_RULES_FILE"
PROFILE_ENV_VAR = "OPENMED_PROFILE"
DEFAULT_MAX_REQUEST_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_TIMEOUT_SECONDS = 300.0
MAX_EGRESS_RULES = 256
MAX_EGRESS_RULES_BYTES = 256 * 1024
MAX_DESTINATION_LENGTH = 4096
MAX_RESOLVED_ADDRESSES = 64
PERMISSIVE_PROFILES = frozenset({"dev", "test"})

IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]
IPNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]
AddressResolver = Callable[[str, int], Iterable[Any]]

_LOGGER = logging.getLogger("openmed.service.egress")
_DESTINATION_DIGEST_DOMAIN = b"openmed.egress.destination.v1\0"
_RULE_DIGEST_DOMAIN = b"openmed.egress.rule.v1\0"
_POLICY_DIGEST_DOMAIN = b"openmed.egress.policy.v1\0"


class EgressMode(str, Enum):
    """How the policy treats operations that no rule matches."""

    DENY = "deny"
    ENFORCE = "enforce"
    PERMISSIVE = "permissive"


class EgressAdapter(str, Enum):
    """Outbound-capable adapter categories inventoried by the policy."""

    WEBHOOKS = "webhooks"
    REMOTE_INFERENCE = "remote_inference"
    SMART_BACKEND = "smart_backend"
    OPENHIM = "openhim"
    PRIVACY_GATEWAY = "privacy_gateway"
    TRACING = "tracing"
    MCP_METADATA = "mcp_metadata"


class EgressPurpose(str, Enum):
    """Declared purpose of one outbound operation."""

    WEBHOOK = "webhook"
    REMOTE_INFERENCE = "remote_inference"
    SMART_TOKEN = "smart_token"
    FHIR_BULK = "fhir_bulk"
    OPENHIM = "openhim"
    PRIVACY_GATEWAY = "privacy_gateway"
    TRACE_EXPORT = "trace_export"
    CONNECTOR_METADATA = "connector_metadata"


class EgressDataClass(str, Enum):
    """Classification of the data an outbound request body carries."""

    NONE = "none"
    METADATA = "metadata"
    TELEMETRY = "telemetry"
    DEIDENTIFIED_TEXT = "deidentified_text"
    TENSORS = "tensors"


class EgressCredential(str, Enum):
    """Credential class an outbound request forwards to its destination."""

    NONE = "none"
    SIGNATURE = "signature"
    BEARER = "bearer"
    BASIC = "basic"
    CLIENT_ASSERTION = "client_assertion"
    HEADER = "header"


ADAPTER_PURPOSES: Mapping[EgressAdapter, frozenset[EgressPurpose]] = {
    EgressAdapter.WEBHOOKS: frozenset({EgressPurpose.WEBHOOK}),
    EgressAdapter.REMOTE_INFERENCE: frozenset({EgressPurpose.REMOTE_INFERENCE}),
    EgressAdapter.SMART_BACKEND: frozenset(
        {EgressPurpose.SMART_TOKEN, EgressPurpose.FHIR_BULK}
    ),
    EgressAdapter.OPENHIM: frozenset({EgressPurpose.OPENHIM}),
    EgressAdapter.PRIVACY_GATEWAY: frozenset({EgressPurpose.PRIVACY_GATEWAY}),
    EgressAdapter.TRACING: frozenset({EgressPurpose.TRACE_EXPORT}),
    EgressAdapter.MCP_METADATA: frozenset({EgressPurpose.CONNECTOR_METADATA}),
}

PURPOSE_DATA_CLASSES: Mapping[EgressPurpose, frozenset[EgressDataClass]] = {
    EgressPurpose.WEBHOOK: frozenset({EgressDataClass.METADATA}),
    EgressPurpose.REMOTE_INFERENCE: frozenset({EgressDataClass.TENSORS}),
    EgressPurpose.SMART_TOKEN: frozenset({EgressDataClass.NONE}),
    EgressPurpose.FHIR_BULK: frozenset({EgressDataClass.NONE}),
    EgressPurpose.OPENHIM: frozenset({EgressDataClass.NONE, EgressDataClass.METADATA}),
    EgressPurpose.PRIVACY_GATEWAY: frozenset({EgressDataClass.DEIDENTIFIED_TEXT}),
    EgressPurpose.TRACE_EXPORT: frozenset({EgressDataClass.TELEMETRY}),
    EgressPurpose.CONNECTOR_METADATA: frozenset({EgressDataClass.NONE}),
}

_HTTP_SCHEMES = frozenset({"http", "https"})
PURPOSE_SCHEMES: Mapping[EgressPurpose, frozenset[str]] = {
    purpose: (
        frozenset({"http", "https", "grpc", "grpcs"})
        if purpose is EgressPurpose.REMOTE_INFERENCE
        else _HTTP_SCHEMES
    )
    for purpose in EgressPurpose
}

_DEFAULT_PORTS = {"http": 80, "https": 443, "grpc": 443, "grpcs": 443}
_SECURE_SCHEMES = frozenset({"https", "grpcs"})
_NON_CREDENTIALS = frozenset({EgressCredential.NONE, EgressCredential.SIGNATURE})

_ADDRESS_CLASS_REASONS = {
    ResolvedAddressClass.LOOPBACK: "loopback_address",
    ResolvedAddressClass.PRIVATE: "private_address",
    ResolvedAddressClass.LINK_LOCAL: "link_local_address",
    ResolvedAddressClass.MULTICAST: "multicast_address",
    ResolvedAddressClass.UNSPECIFIED: "unspecified_address",
    ResolvedAddressClass.CLOUD_METADATA: "cloud_metadata_address",
    ResolvedAddressClass.RESERVED: "reserved_address",
    ResolvedAddressClass.NON_GLOBAL: "non_global_address",
}

ALLOW_REASON_CODES = frozenset({"allowed_by_rule", "allowed_permissive"})
DENY_REASON_CODES = frozenset(
    {
        "address_outside_rule_networks",
        "budget_exceeded",
        "credential_escalation",
        "credentials_not_permitted",
        "data_class_not_permitted",
        "deadline_exceeded",
        "destination_mismatch",
        "destination_not_bound",
        "egress_disabled",
        "insecure_transport",
        "invalid_destination",
        "no_decision",
        "no_matching_rule",
        "offline_mode",
        "policy_unavailable",
        "purpose_not_permitted",
        "redirect_not_permitted",
        "resolution_failed",
        "unix_socket_not_permitted",
        "unknown_adapter",
        "unknown_credential",
        "unknown_data_class",
        "unknown_purpose",
        "unsupported_scheme",
        "userinfo_not_allowed",
        *_ADDRESS_CLASS_REASONS.values(),
    }
)
EGRESS_REASON_CODES = ALLOW_REASON_CODES | DENY_REASON_CODES

_CONFIG_REASON_CODES = frozenset(
    {
        "invalid_mode",
        "invalid_rules",
        "invalid_rule",
        "invalid_rule_destination",
        "invalid_rule_network",
        "invalid_rule_purpose",
        "invalid_rule_limit",
        "rules_file_unreadable",
        "too_many_rules",
        "pinning_unsupported",
    }
)


class EgressDenied(PermissionError):
    """Raised when the egress policy refuses an outbound operation.

    The message and attributes contain only a controlled reason code; the
    destination, request, and payload are never attached.
    """

    error_code = "egress_denied"

    def __init__(
        self,
        reason_code: str,
        *,
        receipt: Optional["EgressReceipt"] = None,
    ) -> None:
        safe_reason = reason_code if reason_code in DENY_REASON_CODES else "no_decision"
        super().__init__(f"Outbound connection denied by egress policy: {safe_reason}")
        self.reason_code = safe_reason
        self.receipt = receipt

    def to_dict(self) -> dict[str, str]:
        """Return a value-free structured error."""
        return {"code": self.error_code, "reason_code": self.reason_code}


class EgressConfigurationError(ValueError):
    """Raised when egress configuration is invalid; never echoes rule values."""

    def __init__(self, reason_code: str) -> None:
        safe_reason = (
            reason_code if reason_code in _CONFIG_REASON_CODES else "invalid_rules"
        )
        super().__init__(f"Egress policy configuration is invalid: {safe_reason}")
        self.reason_code = safe_reason


class EgressDestinationError(ValueError):
    """Raised when an outbound destination cannot be parsed safely."""

    def __init__(self, reason_code: str) -> None:
        safe_reason = (
            reason_code
            if reason_code
            in {"invalid_destination", "unsupported_scheme", "userinfo_not_allowed"}
            else "invalid_destination"
        )
        super().__init__(f"Outbound destination is invalid: {safe_reason}")
        self.reason_code = safe_reason


@dataclass(frozen=True)
class EgressDestination:
    """Canonical scheme, host, and port of one outbound destination.

    The representation omits the host so a destination never leaks through
    ``repr`` into logs or tracebacks.
    """

    scheme: str
    host: str = field(repr=False)
    port: int
    address: Optional[IPAddress] = field(default=None, repr=False)
    has_path: bool = field(default=False, repr=False)

    @property
    def secure(self) -> bool:
        """Return whether the scheme requires TLS."""
        return self.scheme in _SECURE_SCHEMES

    @property
    def origin(self) -> str:
        """Return the canonical ``scheme://host:port`` origin."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.scheme}://{host}:{self.port}"

    @property
    def digest(self) -> str:
        """Return an opaque digest of the origin for value-free receipts."""
        return hashlib.sha256(
            _DESTINATION_DIGEST_DOMAIN + self.origin.encode("utf-8")
        ).hexdigest()


def parse_destination(url: Any) -> EgressDestination:
    """Parse an outbound URL into a canonical destination.

    Args:
        url: Absolute ``http``, ``https``, ``grpc``, or ``grpcs`` URL.

    Returns:
        The canonical destination.

    Raises:
        EgressDestinationError: When the URL is malformed, uses another
            scheme, or carries user information. The error carries only a
            reason code.
    """

    if not isinstance(url, str) or not url or len(url) > MAX_DESTINATION_LENGTH:
        raise EgressDestinationError("invalid_destination")
    if any(char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F for char in url):
        raise EgressDestinationError("invalid_destination")
    try:
        parts = urlsplit(url)
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        raise EgressDestinationError("invalid_destination") from None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        raise EgressDestinationError("unsupported_scheme")
    if not parts.netloc:
        raise EgressDestinationError("invalid_destination")
    if "@" in parts.netloc:
        raise EgressDestinationError("userinfo_not_allowed")
    if hostname is None:
        raise EgressDestinationError("invalid_destination")
    try:
        host, address = canonical_host(hostname)
    except UpstreamEndpointPolicyError:
        raise EgressDestinationError("invalid_destination") from None
    resolved_port = port if port is not None else _DEFAULT_PORTS[scheme]
    if not 1 <= resolved_port <= 65535:
        raise EgressDestinationError("invalid_destination")
    return EgressDestination(
        scheme=scheme,
        host=host,
        port=resolved_port,
        address=address,
        has_path=bool(parts.path not in {"", "/"} or parts.query or parts.fragment),
    )


def _coerce_enum(enum_type: type[Enum], value: Any) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError:
        return None


def _address_in_network(address: IPAddress, network: IPNetwork) -> bool:
    candidate: IPAddress = address
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None and network.version == 4:
        candidate = mapped
    if candidate.version != network.version:
        return False
    return candidate in network


def _address_reason(address: IPAddress, networks: tuple[IPNetwork, ...]) -> str:
    """Return ``""`` when *address* is permitted, else a deny reason code."""

    address_class = classify_address(address)
    if networks:
        for network in networks:
            if not _address_in_network(address, network):
                continue
            if (
                address_class is ResolvedAddressClass.CLOUD_METADATA
                and network.num_addresses != 1
            ):
                continue
            return ""
        if address_class is ResolvedAddressClass.PUBLIC:
            return "address_outside_rule_networks"
        return _ADDRESS_CLASS_REASONS.get(
            address_class, "address_outside_rule_networks"
        )
    if address_class is ResolvedAddressClass.PUBLIC:
        return ""
    return _ADDRESS_CLASS_REASONS.get(address_class, "non_global_address")


def _coerce_address(value: Any) -> Optional[IPAddress]:
    if isinstance(value, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
        return value
    if isinstance(value, (tuple, list)):
        if len(value) >= 5 and isinstance(value[4], (tuple, list)) and value[4]:
            value = value[4][0]
        elif value:
            value = value[0]
        else:
            return None
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str) or "%" in value:
        return None
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class EgressRule:
    """One operator allow rule: purposes plus one exact destination origin.

    Args:
        purposes: Purposes this rule authorizes.
        destination: Exact scheme, host, and port the rule authorizes.
        networks: Resolved-address networks the destination may use. When
            empty, a hostname must resolve to public addresses only, and an
            IP-literal destination allows exactly that address.
        allow_credentials: Whether credential-bearing requests may be sent.
        allow_redirects: Whether a redirect back to this origin may be
            followed (each hop is re-authorized).
        max_request_bytes: Optional request-body budget for this rule.
        max_timeout_seconds: Optional per-request deadline ceiling.
    """

    purposes: frozenset[EgressPurpose]
    destination: EgressDestination
    networks: tuple[IPNetwork, ...] = ()
    allow_credentials: bool = False
    allow_redirects: bool = False
    max_request_bytes: Optional[int] = None
    max_timeout_seconds: Optional[float] = None

    _KEYS = frozenset(
        {
            "purposes",
            "destination",
            "cidrs",
            "allow_credentials",
            "allow_redirects",
            "max_request_bytes",
            "max_timeout_seconds",
        }
    )

    def __post_init__(self) -> None:
        if not self.purposes or not all(
            isinstance(item, EgressPurpose) for item in self.purposes
        ):
            raise EgressConfigurationError("invalid_rule_purpose")
        if not isinstance(self.destination, EgressDestination):
            raise EgressConfigurationError("invalid_rule_destination")
        if self.destination.has_path:
            raise EgressConfigurationError("invalid_rule_destination")
        for purpose in self.purposes:
            if self.destination.scheme not in PURPOSE_SCHEMES[purpose]:
                raise EgressConfigurationError("invalid_rule_destination")
        if any(network.prefixlen == 0 for network in self.networks):
            raise EgressConfigurationError("invalid_rule_network")
        if self.max_request_bytes is not None and (
            isinstance(self.max_request_bytes, bool)
            or not isinstance(self.max_request_bytes, int)
            or self.max_request_bytes < 0
        ):
            raise EgressConfigurationError("invalid_rule_limit")
        if self.max_timeout_seconds is not None and (
            isinstance(self.max_timeout_seconds, bool)
            or not isinstance(self.max_timeout_seconds, (int, float))
            or not math.isfinite(float(self.max_timeout_seconds))
            or float(self.max_timeout_seconds) <= 0
        ):
            raise EgressConfigurationError("invalid_rule_limit")
        if not isinstance(self.allow_credentials, bool) or not isinstance(
            self.allow_redirects, bool
        ):
            raise EgressConfigurationError("invalid_rule")

    @classmethod
    def from_mapping(cls, value: Any) -> "EgressRule":
        """Parse one rule from its JSON object form.

        Args:
            value: Mapping with ``purposes`` and ``destination`` plus the
                optional ``cidrs``, ``allow_credentials``, ``allow_redirects``,
                ``max_request_bytes``, and ``max_timeout_seconds`` keys.

        Returns:
            The validated rule.

        Raises:
            EgressConfigurationError: When the rule is malformed. The error
                carries only a reason code.
        """

        if not isinstance(value, Mapping) or not set(value) <= cls._KEYS:
            raise EgressConfigurationError("invalid_rule")
        raw_purposes = value.get("purposes")
        if (
            not isinstance(raw_purposes, list)
            or not raw_purposes
            or not all(isinstance(item, str) for item in raw_purposes)
        ):
            raise EgressConfigurationError("invalid_rule_purpose")
        purposes = set()
        for item in raw_purposes:
            purpose = _coerce_enum(EgressPurpose, item.strip())
            if purpose is None:
                raise EgressConfigurationError("invalid_rule_purpose")
            purposes.add(purpose)
        try:
            destination = parse_destination(value.get("destination"))
        except EgressDestinationError:
            raise EgressConfigurationError("invalid_rule_destination") from None
        raw_networks = value.get("cidrs", [])
        if not isinstance(raw_networks, list) or not all(
            isinstance(item, str) for item in raw_networks
        ):
            raise EgressConfigurationError("invalid_rule_network")
        networks: list[IPNetwork] = []
        for item in raw_networks:
            try:
                networks.append(ipaddress.ip_network(item.strip(), strict=False))
            except ValueError:
                raise EgressConfigurationError("invalid_rule_network") from None
        max_timeout = value.get("max_timeout_seconds")
        return cls(
            purposes=frozenset(purposes),
            destination=destination,
            networks=tuple(networks),
            allow_credentials=value.get("allow_credentials", False),
            allow_redirects=value.get("allow_redirects", False),
            max_request_bytes=value.get("max_request_bytes"),
            max_timeout_seconds=(
                float(max_timeout)
                if isinstance(max_timeout, (int, float))
                and not isinstance(max_timeout, bool)
                else max_timeout
            ),
        )

    @property
    def effective_networks(self) -> tuple[IPNetwork, ...]:
        """Return the networks resolved addresses must fall within."""
        if self.networks:
            return self.networks
        address = self.destination.address
        if address is not None:
            return (ipaddress.ip_network(address),)
        return ()

    @property
    def rule_id(self) -> str:
        """Return an opaque, stable identifier for receipts."""
        canonical = json.dumps(
            {
                "purposes": sorted(item.value for item in self.purposes),
                "origin": self.destination.origin,
                "cidrs": sorted(str(item) for item in self.networks),
                "allow_credentials": self.allow_credentials,
                "allow_redirects": self.allow_redirects,
                "max_request_bytes": self.max_request_bytes,
                "max_timeout_seconds": self.max_timeout_seconds,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(
            _RULE_DIGEST_DOMAIN + canonical.encode("utf-8")
        ).hexdigest()[:16]

    def matches(self, purpose: EgressPurpose, destination: EgressDestination) -> bool:
        """Return whether this rule covers *purpose* at *destination*."""
        return (
            purpose in self.purposes and self.destination.origin == destination.origin
        )


@dataclass(frozen=True)
class EgressRequest:
    """One outbound operation an adapter asks the policy to authorize.

    Args:
        adapter: Adapter category, one of :class:`EgressAdapter`.
        purpose: Declared purpose, one of :class:`EgressPurpose`.
        destination: Absolute destination URL. It is never logged or echoed.
        data_class: Classification of the request body.
        credential: Credential class forwarded to the destination.
        body_bytes: Serialized request-body size, when already known.
        timeout_seconds: Per-request deadline the adapter will apply.
        redirect: Whether this request follows a redirect.
        operation_id: Opaque logical-operation identifier for receipts.
        attempt: Attempt number within the logical operation.
    """

    adapter: str
    purpose: str
    destination: str = field(repr=False)
    data_class: str = EgressDataClass.NONE.value
    credential: str = EgressCredential.NONE.value
    body_bytes: int = 0
    timeout_seconds: Optional[float] = None
    redirect: bool = False
    operation_id: str = ""
    attempt: int = 1


@dataclass(frozen=True)
class EgressReceipt:
    """Value-free evidence for one allow or deny decision."""

    policy_version: str
    policy_digest: str
    mode: str
    stage: str
    decision: str
    reason_code: str
    adapter: str
    purpose: str
    data_class: str
    credential: str
    destination_digest: str
    destination_class: str
    rule_id: Optional[str]
    operation_id: str
    attempt: int
    request_bytes: int
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Return the receipt as a JSON-ready mapping."""
        return {
            "policy_version": self.policy_version,
            "policy_digest": self.policy_digest,
            "mode": self.mode,
            "stage": self.stage,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "adapter": self.adapter,
            "purpose": self.purpose,
            "data_class": self.data_class,
            "credential": self.credential,
            "destination_digest": self.destination_digest,
            "destination_class": self.destination_class,
            "rule_id": self.rule_id,
            "operation_id": self.operation_id,
            "attempt": self.attempt,
            "request_bytes": self.request_bytes,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class EgressDecision:
    """The policy's answer for one :class:`EgressRequest`.

    Only allowed decisions may open a transport. The destination is held
    privately for connection pinning and never appears in ``repr``.
    """

    allowed: bool
    reason_code: str
    adapter: str
    purpose: str
    data_class: str
    credential: str
    destination_digest: str
    destination_class: str
    mode: str
    policy_version: str
    policy_digest: str
    rule_id: Optional[str]
    max_request_bytes: int
    max_timeout_seconds: float
    operation_id: str = ""
    attempt: int = 1
    body_bytes: int = 0
    destination: Optional[EgressDestination] = field(
        default=None, repr=False, compare=False
    )
    networks: tuple[IPNetwork, ...] = field(default=(), repr=False, compare=False)
    policy: Optional["EgressPolicy"] = field(default=None, repr=False, compare=False)

    def receipt(
        self, stage: str = "authorize", duration_ms: float = 0.0
    ) -> EgressReceipt:
        """Return the value-free receipt for this decision."""
        return EgressReceipt(
            policy_version=self.policy_version,
            policy_digest=self.policy_digest,
            mode=self.mode,
            stage=stage,
            decision="allow" if self.allowed else "deny",
            reason_code=self.reason_code,
            adapter=self.adapter,
            purpose=self.purpose,
            data_class=self.data_class,
            credential=self.credential,
            destination_digest=self.destination_digest,
            destination_class=self.destination_class,
            rule_id=self.rule_id,
            operation_id=self.operation_id,
            attempt=self.attempt,
            request_bytes=self.body_bytes,
            duration_ms=round(max(duration_ms, 0.0), 3),
        )

    def require_body(self, body_bytes: int) -> "EgressDecision":
        """Check a serialized body against the decision's byte budget.

        Args:
            body_bytes: Size of the serialized request body.

        Returns:
            A copy of the decision carrying the body size for receipts.

        Raises:
            EgressDenied: With ``budget_exceeded`` when over budget, or
                ``no_decision`` when the decision does not allow egress.
        """

        if self.policy is None:
            raise EgressDenied("no_decision")
        return self.policy.check_body(self, body_bytes)

    def resolve(
        self,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> tuple[str, ...]:
        """Resolve and check the destination; return addresses to pin to.

        Args:
            host: Host the transport is about to connect to, if known.
            port: Port the transport is about to connect to, if known.

        Returns:
            Resolved IP address strings that passed the address policy.

        Raises:
            EgressDenied: When resolution fails, the connect target differs
                from the decision, or any address is not permitted.
        """

        if self.policy is None:
            raise EgressDenied("no_decision")
        return self.policy.resolve(self, host=host, port=port)

    def covers_url(self, url: str) -> bool:
        """Return whether *url* has the same origin as this decision."""
        if self.destination is None:
            return False
        try:
            return parse_destination(url).origin == self.destination.origin
        except EgressDestinationError:
            return False


def _log_receipt(receipt: EgressReceipt) -> None:
    level = logging.DEBUG if receipt.decision == "allow" else logging.WARNING
    if _LOGGER.isEnabledFor(level):
        _LOGGER.log(
            level,
            "egress_receipt %s",
            json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")),
        )


def _offline_active(local_only: bool) -> bool:
    """Return whether the policy flag or ``OPENMED_OFFLINE`` forbids egress.

    The environment is read on every call so offline mode applies to
    decisions, retries, and resolutions made after it is switched on.
    """

    return bool(local_only or is_local_only())


def default_mode_for_profile(profile: Optional[str]) -> EgressMode:
    """Return the default egress mode for an ``OPENMED_PROFILE`` value."""
    normalized = (profile or "prod").strip().lower() or "prod"
    if normalized in PERMISSIVE_PROFILES:
        return EgressMode.PERMISSIVE
    return EgressMode.ENFORCE


def _parse_rules_document(raw: str) -> list[EgressRule]:
    if len(raw.encode("utf-8")) > MAX_EGRESS_RULES_BYTES:
        raise EgressConfigurationError("too_many_rules")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        raise EgressConfigurationError("invalid_rules") from None
    if isinstance(payload, Mapping):
        if set(payload) != {"rules"}:
            raise EgressConfigurationError("invalid_rules")
        payload = payload["rules"]
    if not isinstance(payload, list):
        raise EgressConfigurationError("invalid_rules")
    if len(payload) > MAX_EGRESS_RULES:
        raise EgressConfigurationError("too_many_rules")
    return [EgressRule.from_mapping(item) for item in payload]


@dataclass(frozen=True)
class EgressPolicy:
    """Fail-closed policy shared by every outbound-capable adapter.

    Args:
        mode: How unmatched operations are treated.
        rules: Operator allow rules.
        max_request_bytes: Default request-body budget.
        max_timeout_seconds: Default per-request deadline ceiling.
        local_only: Treat the policy as offline regardless of environment.
        resolver: ``(host, port)`` resolver used before connecting. Tests and
            offline deployments inject deterministic answers.
        receipt_sink: Callable receiving each value-free receipt. Defaults to
            the ``openmed.service.egress`` logger.
    """

    mode: EgressMode = EgressMode.ENFORCE
    rules: tuple[EgressRule, ...] = ()
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_timeout_seconds: float = DEFAULT_MAX_TIMEOUT_SECONDS
    local_only: bool = False
    resolver: AddressResolver = field(
        default=resolve_hostname, repr=False, compare=False
    )
    receipt_sink: Optional[Callable[[EgressReceipt], None]] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        mode = _coerce_enum(EgressMode, self.mode)
        if mode is None:
            raise EgressConfigurationError("invalid_mode")
        object.__setattr__(self, "mode", mode)
        rules = tuple(self.rules)
        if len(rules) > MAX_EGRESS_RULES or not all(
            isinstance(rule, EgressRule) for rule in rules
        ):
            raise EgressConfigurationError("invalid_rules")
        object.__setattr__(self, "rules", rules)
        if (
            isinstance(self.max_request_bytes, bool)
            or not isinstance(self.max_request_bytes, int)
            or self.max_request_bytes < 0
            or isinstance(self.max_timeout_seconds, bool)
            or not math.isfinite(float(self.max_timeout_seconds))
            or float(self.max_timeout_seconds) <= 0
        ):
            raise EgressConfigurationError("invalid_rule_limit")

    @classmethod
    def from_rules(
        cls,
        rules: Iterable[Mapping[str, Any]],
        *,
        mode: Union[EgressMode, str] = EgressMode.ENFORCE,
        **kwargs: Any,
    ) -> "EgressPolicy":
        """Build a policy from rule mappings in their JSON object form.

        Args:
            rules: Rule mappings accepted by :meth:`EgressRule.from_mapping`.
            mode: Policy mode.
            **kwargs: Other :class:`EgressPolicy` fields.

        Returns:
            The validated policy.
        """

        return cls(
            mode=mode,  # type: ignore[arg-type]
            rules=tuple(EgressRule.from_mapping(rule) for rule in rules),
            **kwargs,
        )

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, Optional[str]]] = None,
        **kwargs: Any,
    ) -> "EgressPolicy":
        """Build the policy from ``OPENMED_EGRESS_*`` environment variables.

        Args:
            environ: Environment mapping; defaults to ``os.environ``.
            **kwargs: Other :class:`EgressPolicy` fields, such as a resolver.

        Returns:
            The validated policy.

        Raises:
            EgressConfigurationError: When the mode, rules, or rules file are
                invalid or unreadable.
        """

        env = os.environ if environ is None else environ
        raw_mode = (env.get(EGRESS_MODE_ENV_VAR) or "").strip().lower()
        if raw_mode:
            mode = _coerce_enum(EgressMode, raw_mode)
            if mode is None:
                raise EgressConfigurationError("invalid_mode")
        else:
            mode = default_mode_for_profile(env.get(PROFILE_ENV_VAR))
        rules: list[EgressRule] = []
        raw_rules = env.get(EGRESS_RULES_ENV_VAR)
        if raw_rules is not None and raw_rules.strip():
            rules.extend(_parse_rules_document(raw_rules))
        rules_file = (env.get(EGRESS_RULES_FILE_ENV_VAR) or "").strip()
        if rules_file:
            failed = False
            text = ""
            try:
                path = Path(rules_file)
                if path.stat().st_size > MAX_EGRESS_RULES_BYTES:
                    raise EgressConfigurationError("too_many_rules")
                text = path.read_text(encoding="utf-8")
            except EgressConfigurationError:
                raise
            except (OSError, UnicodeDecodeError):
                failed = True
            if failed:
                raise EgressConfigurationError("rules_file_unreadable")
            rules.extend(_parse_rules_document(text))
        if len(rules) > MAX_EGRESS_RULES:
            raise EgressConfigurationError("too_many_rules")
        return cls(mode=mode, rules=tuple(rules), **kwargs)

    @property
    def digest(self) -> str:
        """Return an opaque digest of the effective policy."""
        canonical = json.dumps(
            {
                "version": EGRESS_POLICY_VERSION,
                "mode": self.mode.value,
                "rules": [rule.rule_id for rule in self.rules],
                "max_request_bytes": self.max_request_bytes,
                "max_timeout_seconds": float(self.max_timeout_seconds),
                "local_only": self.local_only,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(
            _POLICY_DIGEST_DOMAIN + canonical.encode("utf-8")
        ).hexdigest()[:16]

    def to_safe_dict(self) -> dict[str, Any]:
        """Return value-free posture evidence for deployment profiles.

        The summary names the mode, version, digest, rule counts per purpose,
        and whether offline mode currently overrides the rules. It never
        includes destinations or networks.
        """

        purposes = {purpose.value: 0 for purpose in EgressPurpose}
        for rule in self.rules:
            for purpose in rule.purposes:
                purposes[purpose.value] += 1
        return {
            "policy_version": EGRESS_POLICY_VERSION,
            "policy_digest": self.digest,
            "mode": self.mode.value,
            "offline_override": _offline_active(self.local_only),
            "rule_count": len(self.rules),
            "rules_by_purpose": purposes,
            "credential_rules": sum(1 for rule in self.rules if rule.allow_credentials),
            "private_network_rules": sum(
                1 for rule in self.rules if rule.effective_networks
            ),
        }

    def emit(self, receipt: EgressReceipt) -> None:
        """Send a receipt to the configured sink without affecting decisions."""
        sink = self.receipt_sink or _log_receipt
        try:
            sink(receipt)
        except Exception:  # noqa: BLE001 - observability must not alter policy.
            pass

    def _decision(
        self,
        request: EgressRequest,
        reason_code: str,
        *,
        destination: Optional[EgressDestination] = None,
        rule: Optional[EgressRule] = None,
        destination_class: str = "unresolved",
    ) -> EgressDecision:
        max_bytes = self.max_request_bytes
        max_timeout = float(self.max_timeout_seconds)
        if rule is not None:
            if rule.max_request_bytes is not None:
                max_bytes = rule.max_request_bytes
            if rule.max_timeout_seconds is not None:
                max_timeout = float(rule.max_timeout_seconds)
        allowed = reason_code in ALLOW_REASON_CODES
        return EgressDecision(
            allowed=allowed,
            reason_code=reason_code,
            adapter=_safe_label(request.adapter, EgressAdapter),
            purpose=_safe_label(request.purpose, EgressPurpose),
            data_class=_safe_label(request.data_class, EgressDataClass),
            credential=_safe_label(request.credential, EgressCredential),
            destination_digest=destination.digest if destination else "",
            destination_class=destination_class,
            mode=self.mode.value,
            policy_version=EGRESS_POLICY_VERSION,
            policy_digest=self.digest,
            rule_id=rule.rule_id if rule is not None else None,
            max_request_bytes=max_bytes,
            max_timeout_seconds=max_timeout,
            operation_id=request.operation_id,
            attempt=request.attempt,
            body_bytes=_safe_count(request.body_bytes),
            destination=destination if allowed else None,
            networks=rule.effective_networks if (allowed and rule) else (),
            policy=self if allowed else None,
        )

    def deny(self, request: EgressRequest, reason_code: str) -> EgressDenied:
        """Record a denial receipt for *request* and return the exception.

        Args:
            request: The refused request.
            reason_code: Controlled deny reason.

        Returns:
            An :class:`EgressDenied` carrying the emitted receipt.
        """

        destination: Optional[EgressDestination]
        try:
            destination = parse_destination(request.destination)
        except EgressDestinationError:
            destination = None
        decision = self._decision(request, reason_code, destination=destination)
        receipt = decision.receipt("authorize")
        self.emit(receipt)
        return EgressDenied(decision.reason_code, receipt=receipt)

    def authorize(self, request: EgressRequest) -> EgressDecision:
        """Evaluate one request without network I/O.

        Literal-IP destinations are classified immediately. Hostnames are
        resolved and checked later by :meth:`resolve`, right before the
        transport connects.

        Args:
            request: The outbound operation to evaluate.

        Returns:
            An allowed or denied decision. A receipt is emitted either way.
        """

        started = time.perf_counter()
        decision = self._evaluate(request)
        self.emit(
            decision.receipt("authorize", (time.perf_counter() - started) * 1000.0)
        )
        return decision

    def require(self, request: EgressRequest) -> EgressDecision:
        """Return an allowed decision or raise :class:`EgressDenied`."""
        decision = self.authorize(request)
        if not decision.allowed:
            raise EgressDenied(decision.reason_code, receipt=decision.receipt())
        return decision

    def _evaluate(self, request: EgressRequest) -> EgressDecision:
        adapter = _coerce_enum(EgressAdapter, request.adapter)
        purpose = _coerce_enum(EgressPurpose, request.purpose)
        data_class = _coerce_enum(EgressDataClass, request.data_class)
        credential = _coerce_enum(EgressCredential, request.credential)
        if adapter is None:
            return self._decision(request, "unknown_adapter")
        if purpose is None:
            return self._decision(request, "unknown_purpose")
        if data_class is None:
            return self._decision(request, "unknown_data_class")
        if credential is None:
            return self._decision(request, "unknown_credential")
        if purpose not in ADAPTER_PURPOSES[adapter]:
            return self._decision(request, "purpose_not_permitted")
        if data_class not in PURPOSE_DATA_CLASSES[purpose]:
            return self._decision(request, "data_class_not_permitted")
        if _offline_active(self.local_only):
            return self._decision(request, "offline_mode")
        if self.mode is EgressMode.DENY:
            return self._decision(request, "egress_disabled")
        try:
            destination = parse_destination(request.destination)
        except EgressDestinationError as exc:
            return self._decision(request, exc.reason_code)

        def deny(reason: str) -> EgressDecision:
            return self._decision(request, reason, destination=destination)

        if destination.scheme not in PURPOSE_SCHEMES[purpose]:
            return deny("unsupported_scheme")
        rule = next(
            (item for item in self.rules if item.matches(purpose, destination)),
            None,
        )
        if rule is None:
            if self.mode is EgressMode.ENFORCE:
                return deny("no_matching_rule")
            if not destination.secure:
                return deny("insecure_transport")
        if request.redirect and (rule is None or not rule.allow_redirects):
            return deny("redirect_not_permitted")
        if (
            rule is not None
            and credential not in _NON_CREDENTIALS
            and not rule.allow_credentials
        ):
            return deny("credentials_not_permitted")
        max_bytes = (
            rule.max_request_bytes
            if rule is not None and rule.max_request_bytes is not None
            else self.max_request_bytes
        )
        if isinstance(request.body_bytes, bool) or not isinstance(
            request.body_bytes, int
        ):
            return deny("budget_exceeded")
        if request.body_bytes < 0 or request.body_bytes > max_bytes:
            return deny("budget_exceeded")
        max_timeout = (
            rule.max_timeout_seconds
            if rule is not None and rule.max_timeout_seconds is not None
            else self.max_timeout_seconds
        )
        if request.timeout_seconds is not None:
            try:
                timeout = float(request.timeout_seconds)
            except (TypeError, ValueError):
                return deny("deadline_exceeded")
            if not math.isfinite(timeout) or timeout <= 0 or timeout > max_timeout:
                return deny("deadline_exceeded")
        destination_class = "unresolved"
        if destination.address is not None:
            networks = rule.effective_networks if rule is not None else ()
            reason = _address_reason(destination.address, networks)
            destination_class = classify_address(destination.address).value
            if reason:
                return self._decision(
                    request,
                    reason,
                    destination=destination,
                    destination_class=destination_class,
                )
        return self._decision(
            request,
            "allowed_by_rule" if rule is not None else "allowed_permissive",
            destination=destination,
            rule=rule,
            destination_class=destination_class,
        )

    def check_body(self, decision: EgressDecision, body_bytes: int) -> EgressDecision:
        """Check a serialized body size against *decision*'s budget.

        Args:
            decision: An allowed decision from this policy.
            body_bytes: Serialized request-body size.

        Returns:
            A copy of the decision carrying the body size.

        Raises:
            EgressDenied: When the decision is not allowed or the body is
                over budget.
        """

        if not decision.allowed or decision.policy is not self:
            raise EgressDenied("no_decision")
        if (
            isinstance(body_bytes, bool)
            or not isinstance(body_bytes, int)
            or body_bytes < 0
            or body_bytes > decision.max_request_bytes
        ):
            denied = replace(
                decision,
                allowed=False,
                reason_code="budget_exceeded",
                body_bytes=body_bytes if isinstance(body_bytes, int) else 0,
            )
            receipt = denied.receipt("budget")
            self.emit(receipt)
            raise EgressDenied("budget_exceeded", receipt=receipt)
        checked = replace(decision, body_bytes=body_bytes)
        self.emit(checked.receipt("budget"))
        return checked

    def resolve(
        self,
        decision: EgressDecision,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> tuple[str, ...]:
        """Resolve *decision*'s host and check every address before connect.

        Args:
            decision: An allowed decision from this policy.
            host: Host the transport is about to connect to, if known.
            port: Port the transport is about to connect to, if known.

        Returns:
            Sorted IP address strings, all of which passed the policy.

        Raises:
            EgressDenied: On a missing decision, a connect target that differs
                from the decision, offline mode, resolution failure, or any
                prohibited address.
        """

        started = time.perf_counter()
        destination = decision.destination

        def fail(reason: str, destination_class: str = "unresolved") -> EgressDenied:
            denied = replace(
                decision,
                allowed=False,
                reason_code=reason,
                destination_class=destination_class,
            )
            receipt = denied.receipt(
                "resolve", (time.perf_counter() - started) * 1000.0
            )
            self.emit(receipt)
            return EgressDenied(reason, receipt=receipt)

        if not decision.allowed or decision.policy is not self or destination is None:
            raise EgressDenied("no_decision")
        if host is not None:
            try:
                canonical, _ = canonical_host(str(host))
            except UpstreamEndpointPolicyError:
                raise fail("destination_mismatch") from None
            if canonical != destination.host:
                raise fail("destination_mismatch")
        if port is not None and int(port) != destination.port:
            raise fail("destination_mismatch")
        if _offline_active(self.local_only):
            raise fail("offline_mode")
        if destination.address is not None:
            addresses: tuple[IPAddress, ...] = (destination.address,)
        else:
            addresses = self._resolve_addresses(destination)
            if not addresses:
                raise fail("resolution_failed")
        classes = {classify_address(address).value for address in addresses}
        destination_class = next(iter(classes)) if len(classes) == 1 else "mixed"
        for address in addresses:
            reason = _address_reason(address, decision.networks)
            if reason:
                raise fail(reason, destination_class)
        allowed = replace(decision, destination_class=destination_class)
        self.emit(allowed.receipt("resolve", (time.perf_counter() - started) * 1000.0))
        return tuple(str(address) for address in addresses)

    def _resolve_addresses(
        self, destination: EgressDestination
    ) -> tuple[IPAddress, ...]:
        try:
            result = self.resolver(destination.host, destination.port)
            values = list(result) if result is not None else []
        except Exception:  # noqa: BLE001 - resolver details never surface.
            return ()
        if len(values) > MAX_RESOLVED_ADDRESSES:
            return ()
        addresses: dict[str, IPAddress] = {}
        for value in values:
            address = _coerce_address(value)
            if address is None:
                return ()
            addresses[str(address)] = address
        return tuple(
            sorted(addresses.values(), key=lambda item: (item.version, int(item)))
        )


def _safe_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def _safe_label(value: Any, enum_type: type[Enum]) -> str:
    coerced = _coerce_enum(enum_type, value)
    return coerced.value if coerced is not None else "unknown"


_POLICY_ENV_KEYS = (
    EGRESS_MODE_ENV_VAR,
    EGRESS_RULES_ENV_VAR,
    EGRESS_RULES_FILE_ENV_VAR,
    PROFILE_ENV_VAR,
)


@functools.lru_cache(maxsize=16)
def _cached_policy(
    values: tuple[Optional[str], ...],
    file_signature: tuple[int, int],
) -> EgressPolicy:
    del file_signature
    return EgressPolicy.from_env(dict(zip(_POLICY_ENV_KEYS, values)))


def current_egress_policy() -> EgressPolicy:
    """Return the process egress policy derived from the environment.

    The parsed policy is cached per environment value and rules-file
    signature, so changing ``OPENMED_EGRESS_*`` or editing the rules file is
    picked up on the next call. Offline mode is evaluated on every decision.

    Raises:
        EgressConfigurationError: When the configuration is invalid.
    """

    values = tuple(os.environ.get(key) for key in _POLICY_ENV_KEYS)
    signature = (0, 0)
    rules_file = (values[2] or "").strip()
    if rules_file:
        try:
            stat = Path(rules_file).stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            signature = (-1, -1)
    return _cached_policy(values, signature)


def validate_egress_configuration() -> dict[str, Any]:
    """Validate the environment policy and return its value-free posture.

    Raises:
        EgressConfigurationError: When the configuration is invalid.
    """

    return current_egress_policy().to_safe_dict()


class EgressOperation:
    """One logical outbound operation whose retries share one policy context.

    The policy is captured when the operation starts, so a retry or redirect
    is evaluated under the same rules, and offline mode is re-checked on every
    attempt. Destinations outside ``bound_destinations`` are refused, and the
    first credential class used for a purpose at a destination cannot be
    swapped for another one later in the operation.

    Args:
        adapter: Adapter category, one of :class:`EgressAdapter`.
        policy: Policy to use; defaults to :func:`current_egress_policy`.
        bound_destinations: Destinations this operation may contact. When
            empty, any destination the policy allows may be contacted.
    """

    def __init__(
        self,
        adapter: Union[EgressAdapter, str],
        *,
        policy: Optional[EgressPolicy] = None,
        bound_destinations: Iterable[str] = (),
    ) -> None:
        self.adapter = adapter.value if isinstance(adapter, EgressAdapter) else adapter
        self.policy = policy if policy is not None else current_egress_policy()
        self.operation_id = secrets.token_hex(8)
        bound: set[str] = set()
        self._bound_invalid = False
        for item in bound_destinations:
            try:
                bound.add(parse_destination(item).origin)
            except EgressDestinationError:
                self._bound_invalid = True
        self._bound = frozenset(bound)
        self._attempts: dict[tuple[str, str], int] = {}
        self._credentials: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def authorize(
        self,
        destination: str,
        *,
        purpose: Union[EgressPurpose, str],
        data_class: Union[EgressDataClass, str] = EgressDataClass.NONE,
        credential: Union[EgressCredential, str] = EgressCredential.NONE,
        timeout_seconds: Optional[float] = None,
        body_bytes: int = 0,
        redirect: bool = False,
    ) -> EgressDecision:
        """Authorize one attempt of this operation before serialization.

        Args:
            destination: Absolute destination URL.
            purpose: Declared purpose.
            data_class: Classification of the request body.
            credential: Credential class forwarded to the destination.
            timeout_seconds: Per-request deadline the adapter will apply.
            body_bytes: Serialized body size, when already known.
            redirect: Whether this attempt follows a redirect.

        Returns:
            An allowed decision.

        Raises:
            EgressDenied: When the policy or the operation binding refuses it.
        """

        purpose_value = _enum_value(purpose)
        credential_value = _enum_value(credential)
        try:
            origin = parse_destination(destination).origin
        except EgressDestinationError:
            origin = ""
        key = (purpose_value, origin)
        with self._lock:
            attempt = self._attempts.get(key, 0) + 1
            self._attempts[key] = attempt
            bound_credential = self._credentials.get(key)
        request = EgressRequest(
            adapter=self.adapter,
            purpose=purpose_value,
            destination=destination,
            data_class=_enum_value(data_class),
            credential=credential_value,
            body_bytes=body_bytes,
            timeout_seconds=timeout_seconds,
            redirect=redirect,
            operation_id=self.operation_id,
            attempt=attempt,
        )
        if origin and (
            self._bound_invalid or (self._bound and origin not in self._bound)
        ):
            raise self.policy.deny(request, "destination_not_bound")
        if (
            bound_credential is not None
            and credential_value != EgressCredential.NONE.value
            and credential_value != bound_credential
        ):
            raise self.policy.deny(request, "credential_escalation")
        decision = self.policy.require(request)
        if credential_value != EgressCredential.NONE.value:
            with self._lock:
                self._credentials.setdefault(key, credential_value)
        return decision


def _enum_value(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


_ACTIVE_DECISION: ContextVar[Optional[EgressDecision]] = ContextVar(
    "openmed_egress_decision",
    default=None,
)


@contextmanager
def egress_scope(decision: EgressDecision) -> Iterator[EgressDecision]:
    """Make *decision* the active decision for transports in this context.

    Guarded transports refuse to open a connection unless the active decision
    allows the request's origin.

    Args:
        decision: An allowed decision.

    Raises:
        EgressDenied: When *decision* is not an allowed decision.
    """

    if not isinstance(decision, EgressDecision) or not decision.allowed:
        raise EgressDenied("no_decision")
    token = _ACTIVE_DECISION.set(decision)
    try:
        yield decision
    finally:
        _ACTIVE_DECISION.reset(token)


def active_egress_decision() -> Optional[EgressDecision]:
    """Return the decision active in the current context, if any."""
    return _ACTIVE_DECISION.get()


def _unscoped_denial(reason: str) -> EgressDenied:
    receipt = EgressReceipt(
        policy_version=EGRESS_POLICY_VERSION,
        policy_digest="",
        mode="unknown",
        stage="connect",
        decision="deny",
        reason_code=reason,
        adapter="unknown",
        purpose="unknown",
        data_class="unknown",
        credential="unknown",
        destination_digest="",
        destination_class="unresolved",
        rule_id=None,
        operation_id="",
        attempt=0,
        request_bytes=0,
        duration_ms=0.0,
    )
    _log_receipt(receipt)
    return EgressDenied(reason, receipt=receipt)


def _require_decision_for(url: str) -> EgressDecision:
    decision = _ACTIVE_DECISION.get()
    if decision is None or not decision.allowed or decision.destination is None:
        raise _unscoped_denial("no_decision")
    if not decision.covers_url(url):
        policy = decision.policy
        denied = replace(decision, allowed=False, reason_code="destination_mismatch")
        receipt = denied.receipt("connect")
        if policy is not None:
            policy.emit(receipt)
        raise EgressDenied("destination_mismatch", receipt=receipt)
    return decision


def _require_connect_decision() -> EgressDecision:
    decision = _ACTIVE_DECISION.get()
    if decision is None or not decision.allowed:
        raise _unscoped_denial("no_decision")
    return decision


@functools.lru_cache(maxsize=1)
def _httpx_types() -> dict[str, type]:
    import httpcore
    import httpx

    class EgressHTTPTransport(httpx.BaseTransport):
        """Sync transport that refuses requests without an active decision."""

        def __init__(self, inner: httpx.BaseTransport) -> None:
            self._inner = inner

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            _require_decision_for(str(request.url))
            return self._inner.handle_request(request)

        def close(self) -> None:
            self._inner.close()

    class EgressAsyncHTTPTransport(httpx.AsyncBaseTransport):
        """Async transport that refuses requests without an active decision."""

        def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
            self._inner = inner

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            _require_decision_for(str(request.url))
            return await self._inner.handle_async_request(request)

        async def aclose(self) -> None:
            await self._inner.aclose()

    class PinnedNetworkBackend(httpcore.NetworkBackend):
        """Connect only to addresses that passed the active decision."""

        def __init__(self, inner: httpcore.NetworkBackend) -> None:
            self._inner = inner

        def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: Optional[float] = None,
            local_address: Optional[str] = None,
            socket_options: Optional[Iterable[Any]] = None,
        ) -> httpcore.NetworkStream:
            decision = _require_connect_decision()
            addresses = decision.resolve(host=host, port=port)
            error: Optional[Exception] = None
            for address in addresses:
                try:
                    return self._inner.connect_tcp(
                        address,
                        port,
                        timeout=timeout,
                        local_address=local_address,
                        socket_options=socket_options,
                    )
                except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                    error = exc
            assert error is not None
            raise error

        def connect_unix_socket(self, *args: Any, **kwargs: Any) -> Any:
            raise _unscoped_denial("unix_socket_not_permitted")

        def sleep(self, seconds: float) -> None:
            self._inner.sleep(seconds)

    class PinnedAsyncNetworkBackend(httpcore.AsyncNetworkBackend):
        """Async variant of :class:`PinnedNetworkBackend`."""

        def __init__(self, inner: httpcore.AsyncNetworkBackend) -> None:
            self._inner = inner

        async def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: Optional[float] = None,
            local_address: Optional[str] = None,
            socket_options: Optional[Iterable[Any]] = None,
        ) -> httpcore.AsyncNetworkStream:
            import anyio.to_thread

            decision = _require_connect_decision()
            addresses = await anyio.to_thread.run_sync(
                functools.partial(decision.resolve, host=host, port=port)
            )
            error: Optional[Exception] = None
            for address in addresses:
                try:
                    return await self._inner.connect_tcp(
                        address,
                        port,
                        timeout=timeout,
                        local_address=local_address,
                        socket_options=socket_options,
                    )
                except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                    error = exc
            assert error is not None
            raise error

        async def connect_unix_socket(self, *args: Any, **kwargs: Any) -> Any:
            raise _unscoped_denial("unix_socket_not_permitted")

        async def sleep(self, seconds: float) -> None:
            await self._inner.sleep(seconds)

    return {
        "sync_transport": EgressHTTPTransport,
        "async_transport": EgressAsyncHTTPTransport,
        "sync_backend": PinnedNetworkBackend,
        "async_backend": PinnedAsyncNetworkBackend,
    }


def _install_pinned_backend(transport: Any, backend_type: type) -> None:
    pool = getattr(transport, "_pool", None)
    inner = getattr(pool, "_network_backend", None)
    if pool is None or inner is None:
        raise EgressConfigurationError("pinning_unsupported")
    pool._network_backend = backend_type(inner)


def egress_http_transport(*, verify: Any = True, cert: Any = None) -> Any:
    """Return an ``httpx`` transport that enforces decisions and pins sockets.

    Args:
        verify: TLS verification setting passed to ``httpx.HTTPTransport``.
        cert: Optional client certificate passed to ``httpx.HTTPTransport``.

    Returns:
        An ``httpx.BaseTransport`` that refuses requests outside the active
        :func:`egress_scope` and connects only to checked addresses.
    """

    import httpx

    types = _httpx_types()
    inner = httpx.HTTPTransport(verify=verify, cert=cert)
    _install_pinned_backend(inner, types["sync_backend"])
    return types["sync_transport"](inner)


def egress_async_http_transport(*, verify: Any = True, cert: Any = None) -> Any:
    """Return the async variant of :func:`egress_http_transport`."""

    import httpx

    types = _httpx_types()
    inner = httpx.AsyncHTTPTransport(verify=verify, cert=cert)
    _install_pinned_backend(inner, types["async_backend"])
    return types["async_transport"](inner)


def egress_http_client(
    *,
    timeout: Any,
    verify: Any = True,
    cert: Any = None,
) -> Any:
    """Return an ``httpx.Client`` whose every request needs a decision.

    The client never follows redirects, ignores proxy and ``.netrc``
    environment settings, and connects only to checked addresses.

    Args:
        timeout: Default request timeout.
        verify: TLS verification setting.
        cert: Optional client certificate.
    """

    import httpx

    return httpx.Client(
        transport=egress_http_transport(verify=verify, cert=cert),
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
    )


def egress_async_http_client(
    *,
    timeout: Any,
    verify: Any = True,
    cert: Any = None,
    transport: Any = None,
) -> Any:
    """Return an ``httpx.AsyncClient`` whose every request needs a decision.

    Args:
        timeout: Default request timeout.
        verify: TLS verification setting.
        cert: Optional client certificate.
        transport: Optional caller-supplied transport used instead of the
            pinned transport (for example a test transport). Adapter-level
            decisions still apply; socket pinning is then the caller's.
    """

    import httpx

    return httpx.AsyncClient(
        transport=(
            transport
            if transport is not None
            else egress_async_http_transport(verify=verify, cert=cert)
        ),
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
    )


def pinned_grpc_target(decision: EgressDecision) -> tuple[str, list[tuple[str, str]]]:
    """Resolve *decision* and return a pinned gRPC target plus channel options.

    The target lists only checked addresses of one IP family, and the channel
    options keep TLS hostname verification and ``:authority`` on the original
    host.

    Args:
        decision: An allowed decision for a ``grpc`` or ``grpcs`` destination.

    Returns:
        ``(target, options)`` for ``grpc.insecure_channel`` or
        ``grpc.secure_channel``.
    """

    addresses = decision.resolve()
    destination = decision.destination
    assert destination is not None
    family = ipaddress.ip_address(addresses[0]).version
    selected = [
        item for item in addresses if ipaddress.ip_address(item).version == family
    ]
    if family == 6:
        target = "ipv6:" + ",".join(f"[{item}]:{destination.port}" for item in selected)
    else:
        target = "ipv4:" + ",".join(f"{item}:{destination.port}" for item in selected)
    options: list[tuple[str, str]] = []
    if destination.address is None:
        options = [
            ("grpc.ssl_target_name_override", destination.host),
            ("grpc.default_authority", f"{destination.host}:{destination.port}"),
        ]
    return target, options


def _pinned_create_connection(
    address: tuple[str, int],
    timeout: Any = socket._GLOBAL_DEFAULT_TIMEOUT,  # type: ignore[attr-defined]
    source_address: Optional[tuple[str, int]] = None,
) -> socket.socket:
    decision = _require_connect_decision()
    host, port = address
    addresses = decision.resolve(host=host, port=port)
    error: Optional[OSError] = None
    for item in addresses:
        try:
            return socket.create_connection((item, port), timeout, source_address)
        except OSError as exc:
            error = exc
    assert error is not None
    raise error


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _pinned_create_connection  # type: ignore[assignment]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _pinned_create_connection  # type: ignore[assignment]


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req: urllib.request.Request) -> Any:
        _require_decision_for(req.full_url)
        return self.do_open(_PinnedHTTPConnection, req)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request) -> Any:
        _require_decision_for(req.full_url)
        return self.do_open(
            _PinnedHTTPSConnection,
            req,
            context=getattr(self, "_context", None),
        )


class _EgressRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(
        self,
        operation: EgressOperation,
        *,
        purpose: Union[EgressPurpose, str],
        data_class: Union[EgressDataClass, str],
        credential: Union[EgressCredential, str],
        timeout_seconds: Optional[float],
    ) -> None:
        super().__init__()
        self._operation = operation
        self._purpose = purpose
        self._data_class = data_class
        self._credential = credential
        self._timeout_seconds = timeout_seconds

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Optional[urllib.request.Request]:
        decision = self._operation.authorize(
            newurl,
            purpose=self._purpose,
            data_class=self._data_class,
            credential=self._credential,
            timeout_seconds=self._timeout_seconds,
            redirect=True,
        )
        new_request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_request is not None:
            _ACTIVE_DECISION.set(decision)
        return new_request


def egress_urlopen(
    operation: EgressOperation,
    request: urllib.request.Request,
    *,
    purpose: Union[EgressPurpose, str],
    data_class: Union[EgressDataClass, str] = EgressDataClass.NONE,
    credential: Union[EgressCredential, str] = EgressCredential.NONE,
    timeout: float,
) -> Any:
    """Open a ``urllib`` request under one egress decision.

    Environment proxies are ignored, sockets connect only to checked
    addresses, and every redirect hop is re-authorized under *operation*.

    Args:
        operation: Logical operation the request belongs to.
        request: Prepared ``urllib.request.Request``.
        purpose: Declared purpose.
        data_class: Classification of the request body.
        credential: Credential class forwarded to the destination.
        timeout: Socket timeout in seconds.

    Returns:
        The ``urllib`` response object.

    Raises:
        EgressDenied: When the request or a redirect hop is refused.
    """

    decision = operation.authorize(
        request.full_url,
        purpose=purpose,
        data_class=data_class,
        credential=credential,
        timeout_seconds=timeout,
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _PinnedHTTPHandler(),
        _PinnedHTTPSHandler(),
        _EgressRedirectHandler(
            operation,
            purpose=purpose,
            data_class=data_class,
            credential=credential,
            timeout_seconds=timeout,
        ),
    )
    with egress_scope(decision):
        return opener.open(request, timeout=timeout)


@dataclass(frozen=True)
class AdapterCoverage:
    """One row of the outbound adapter coverage matrix."""

    module: str
    adapter: EgressAdapter
    purposes: frozenset[EgressPurpose]
    decision_point: str
    socket_pinning: str


ADAPTER_COVERAGE: tuple[AdapterCoverage, ...] = (
    AdapterCoverage(
        "openmed.service.webhooks",
        EgressAdapter.WEBHOOKS,
        frozenset({EgressPurpose.WEBHOOK}),
        "job submission and every delivery attempt, before body serialization",
        "pinned httpx transport",
    ),
    AdapterCoverage(
        "openmed.service.backends.remote_inference",
        EgressAdapter.REMOTE_INFERENCE,
        frozenset({EgressPurpose.REMOTE_INFERENCE}),
        "transport construction and every inference call",
        "pinned httpx transport; gRPC channel targets checked addresses",
    ),
    AdapterCoverage(
        "openmed.service.smart_backend",
        EgressAdapter.SMART_BACKEND,
        frozenset({EgressPurpose.SMART_TOKEN, EgressPurpose.FHIR_BULK}),
        "job submission plus token, kickoff, status, and file requests",
        "pinned httpx transport",
    ),
    AdapterCoverage(
        "openmed.service.bulk_data",
        EgressAdapter.SMART_BACKEND,
        frozenset({EgressPurpose.SMART_TOKEN, EgressPurpose.FHIR_BULK}),
        "job submission; transfers run through openmed.service.smart_backend",
        "pinned httpx transport (via smart_backend)",
    ),
    AdapterCoverage(
        "openmed.service.openhim_mediator",
        EgressAdapter.OPENHIM,
        frozenset({EgressPurpose.OPENHIM}),
        "every authentication, registration, and heartbeat request",
        "pinned httpx transport",
    ),
    AdapterCoverage(
        "openmed.service.privacy_gateway",
        EgressAdapter.PRIVACY_GATEWAY,
        frozenset({EgressPurpose.PRIVACY_GATEWAY}),
        "every forwarded prompt, before body serialization",
        "pinned httpx transport",
    ),
    AdapterCoverage(
        "openmed.service.tracing",
        EgressAdapter.TRACING,
        frozenset({EgressPurpose.TRACE_EXPORT}),
        "exporter construction and every export batch",
        "resolution checked before each batch; the OTLP exporter owns the socket",
    ),
    AdapterCoverage(
        "openmed.service.security",
        EgressAdapter.MCP_METADATA,
        frozenset({EgressPurpose.CONNECTOR_METADATA}),
        "every metadata fetch and redirect hop",
        "pinned urllib connection",
    ),
)


__all__ = [
    "ADAPTER_COVERAGE",
    "ADAPTER_PURPOSES",
    "ALLOW_REASON_CODES",
    "AdapterCoverage",
    "DEFAULT_MAX_REQUEST_BYTES",
    "DEFAULT_MAX_TIMEOUT_SECONDS",
    "DENY_REASON_CODES",
    "EGRESS_MODE_ENV_VAR",
    "EGRESS_POLICY_VERSION",
    "EGRESS_REASON_CODES",
    "EGRESS_RULES_ENV_VAR",
    "EGRESS_RULES_FILE_ENV_VAR",
    "EgressAdapter",
    "EgressConfigurationError",
    "EgressCredential",
    "EgressDataClass",
    "EgressDecision",
    "EgressDenied",
    "EgressDestination",
    "EgressDestinationError",
    "EgressMode",
    "EgressOperation",
    "EgressPolicy",
    "EgressPurpose",
    "EgressReceipt",
    "EgressRequest",
    "EgressRule",
    "PURPOSE_DATA_CLASSES",
    "PURPOSE_SCHEMES",
    "active_egress_decision",
    "current_egress_policy",
    "default_mode_for_profile",
    "egress_async_http_client",
    "egress_async_http_transport",
    "egress_http_client",
    "egress_http_transport",
    "egress_scope",
    "egress_urlopen",
    "parse_destination",
    "pinned_grpc_target",
    "validate_egress_configuration",
]
