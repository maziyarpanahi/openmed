"""Pure, SSRF-safe policy checks for MCP upstream endpoints.

The policy is intentionally separate from any HTTP client.  Callers validate
an endpoint immediately before handing it to a requester, and validate every
redirect target before following it.  DNS resolution is injected so tests and
offline deployments can use deterministic synthetic answers.

No exception raised by this module contains the endpoint, its query string,
resolver details, or an underlying exception message.  Validated endpoint
objects retain the original URL for the requester, but their representation
and safe metadata omit it.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar
from urllib.parse import SplitResult, urljoin, urlsplit

MAX_ENDPOINT_LENGTH = 4096

AddressResolver = Callable[[str, int], Iterable[Any]]
RequestResult = TypeVar("RequestResult")


class ResolvedAddressClass(str, Enum):
    """Classification used by the fail-closed address policy."""

    PUBLIC = "public"
    LOOPBACK = "loopback"
    PRIVATE = "private"
    LINK_LOCAL = "link_local"
    MULTICAST = "multicast"
    UNSPECIFIED = "unspecified"
    CLOUD_METADATA = "cloud_metadata"
    RESERVED = "reserved"
    NON_GLOBAL = "non_global"


_SAFE_ERROR_MESSAGES = {
    "endpoint_rejected": "The upstream endpoint was rejected by policy.",
    "invalid_endpoint": "The upstream endpoint is invalid.",
    "endpoint_too_long": "The upstream endpoint exceeds the configured limit.",
    "unsupported_scheme": "The upstream endpoint scheme is not allowed.",
    "missing_scheme": "The upstream endpoint must include a scheme.",
    "missing_host": "The upstream endpoint must include a host.",
    "userinfo_not_allowed": "Upstream endpoint user information is not allowed.",
    "invalid_host": "The upstream endpoint host is invalid.",
    "invalid_port": "The upstream endpoint port is invalid.",
    "non_https_origin": "Remote upstream endpoints must use HTTPS.",
    "origin_not_allowed": "The upstream endpoint origin is not approved.",
    "loopback_not_allowed": "Loopback upstream endpoints require development policy.",
    "loopback_only_violation": "A development endpoint did not resolve to loopback only.",
    "invalid_allowed_origin": "An approved upstream origin is invalid.",
    "invalid_allowed_origin_scheme": "Approved upstream origins must use HTTPS.",
    "invalid_allowed_origin_path": "Approved upstream origins must not include a path.",
    "invalid_resolver": "The upstream endpoint resolver is invalid.",
    "dns_resolution_failed": "The upstream endpoint could not be resolved safely.",
    "no_resolved_addresses": "The upstream endpoint returned no usable addresses.",
    "invalid_resolved_address": "The upstream endpoint returned an invalid address.",
    "cloud_metadata_address": "The upstream endpoint resolved to a cloud metadata address.",
    "private_address": "The upstream endpoint resolved to a private address.",
    "link_local_address": "The upstream endpoint resolved to a link-local address.",
    "multicast_address": "The upstream endpoint resolved to a multicast address.",
    "unspecified_address": "The upstream endpoint resolved to an unspecified address.",
    "loopback_address": "The upstream endpoint resolved to a loopback address.",
    "reserved_address": "The upstream endpoint resolved to a reserved address.",
    "non_global_address": "The upstream endpoint resolved to a non-global address.",
    "mixed_public_prohibited_addresses": (
        "The upstream endpoint returned mixed public and prohibited addresses."
    ),
    "mixed_prohibited_addresses": (
        "The upstream endpoint returned multiple prohibited address classes."
    ),
    "redirect_mode_change": "A redirect changed between remote and loopback policy.",
    "invalid_redirect_chain": "The upstream redirect chain is invalid.",
}

_LOOPBACK_HOSTNAMES = frozenset({"localhost"})
_CLOUD_METADATA_NETWORKS = (
    ipaddress.ip_network("169.254.169.254/32"),
    ipaddress.ip_network("169.254.170.2/32"),
    ipaddress.ip_network("100.100.100.200/32"),
    ipaddress.ip_network("fd00:ec2::254/128"),
)
_RESOLUTION_FAILED = object()
_ADDRESS_REASON_CODES = {
    ResolvedAddressClass.CLOUD_METADATA: "cloud_metadata_address",
    ResolvedAddressClass.PRIVATE: "private_address",
    ResolvedAddressClass.LINK_LOCAL: "link_local_address",
    ResolvedAddressClass.MULTICAST: "multicast_address",
    ResolvedAddressClass.UNSPECIFIED: "unspecified_address",
    ResolvedAddressClass.LOOPBACK: "loopback_address",
    ResolvedAddressClass.RESERVED: "reserved_address",
    ResolvedAddressClass.NON_GLOBAL: "non_global_address",
}


class UpstreamEndpointPolicyError(ValueError):
    """Base class for typed, redacted endpoint-policy failures."""

    error_code = "upstream_endpoint_policy_error"

    def __init__(self, reason_code: str = "endpoint_rejected") -> None:
        safe_reason = (
            reason_code if reason_code in _SAFE_ERROR_MESSAGES else "endpoint_rejected"
        )
        self.reason_code = safe_reason
        self.reason = safe_reason
        super().__init__(_SAFE_ERROR_MESSAGES[safe_reason])

    @property
    def code(self) -> str:
        """Return the stable machine-readable error code."""

        return self.error_code

    @property
    def safe_message(self) -> str:
        """Return the static message safe for a client or operational log."""

        return str(self)

    def to_dict(self) -> dict[str, str]:
        """Return a PHI- and credential-safe structured error."""

        return {
            "code": self.error_code,
            "reason_code": self.reason_code,
            "message": self.safe_message,
        }


class UpstreamEndpointConfigurationError(UpstreamEndpointPolicyError):
    """Raised when an endpoint policy is configured incorrectly."""

    error_code = "upstream_endpoint_policy_configuration_error"


class UpstreamEndpointResolutionError(UpstreamEndpointPolicyError):
    """Raised when resolution is unavailable or returns a prohibited answer."""

    error_code = "upstream_endpoint_resolution_error"


class UpstreamEndpointRedirectError(UpstreamEndpointPolicyError):
    """Raised when a redirect would escape the validated policy boundary."""

    error_code = "upstream_endpoint_redirect_error"


@dataclass(frozen=True)
class _ParsedEndpoint:
    """Normalized URL components used internally by the policy."""

    url: str = field(repr=False)
    split: SplitResult = field(repr=False)
    scheme: str
    host: str = field(repr=False)
    port: int
    origin: str
    ip_address: ipaddress.IPv4Address | ipaddress.IPv6Address | None = field(repr=False)


@dataclass(frozen=True)
class ValidatedUpstreamEndpoint:
    """An endpoint that passed scheme, origin, DNS, and address checks.

    ``url`` is retained for the requester and may include a sensitive query
    string.  Treat it as request data; use :meth:`to_safe_dict` for logs or
    audit records.
    """

    url: str = field(repr=False)
    origin: str
    host: str = field(repr=False)
    port: int
    addresses: tuple[str, ...] = field(repr=False)
    address_classes: tuple[ResolvedAddressClass, ...] = field(repr=False)
    is_loopback: bool

    @property
    def resolved_addresses(self) -> tuple[str, ...]:
        """Return the canonical addresses checked by the policy."""

        return self.addresses

    def to_safe_dict(self) -> dict[str, Any]:
        """Return metadata that omits the URL, host, and resolved addresses."""

        return {
            "origin": self.origin,
            "port": self.port,
            "resolved_address_count": len(self.addresses),
            "loopback": self.is_loopback,
        }


def resolve_hostname(host: str, port: int) -> tuple[str, ...]:
    """Resolve a hostname to socket address strings.

    This is the production default.  Tests and offline callers should inject
    a deterministic resolver instead of making DNS calls.
    """

    records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addresses: list[str] = []
    for record in records:
        if len(record) < 5:
            continue
        sockaddr = record[4]
        if isinstance(sockaddr, (tuple, list)) and sockaddr:
            address = sockaddr[0]
            if isinstance(address, str):
                addresses.append(address)
    return tuple(addresses)


def _canonical_origin(scheme: str, host: str, port: int) -> str:
    formatted_host = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80
    suffix = "" if port == default_port else f":{port}"
    return f"{scheme}://{formatted_host}{suffix}"


def _canonical_host(
    host: str,
) -> tuple[str, ipaddress.IPv4Address | ipaddress.IPv6Address | None]:
    if not host or any(char.isspace() for char in host) or "%" in host:
        raise UpstreamEndpointPolicyError("invalid_host")
    host = host.rstrip(".")
    if not host or any(char in host for char in "*/\\"):
        raise UpstreamEndpointPolicyError("invalid_host")

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None

    if address is not None:
        return str(address), address
    if ":" in host or len(host) > 253:
        raise UpstreamEndpointPolicyError("invalid_host")
    try:
        ascii_host = host.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        raise UpstreamEndpointPolicyError("invalid_host") from None
    labels = ascii_host.split(".")
    if (
        not ascii_host
        or any(not label or len(label) > 63 for label in labels)
        or any(label.startswith("-") or label.endswith("-") for label in labels)
    ):
        raise UpstreamEndpointPolicyError("invalid_host")
    return ascii_host, None


def _parse_endpoint(endpoint: Any) -> _ParsedEndpoint:
    if not isinstance(endpoint, str):
        raise UpstreamEndpointPolicyError("invalid_endpoint")
    if not endpoint:
        raise UpstreamEndpointPolicyError("invalid_endpoint")
    if len(endpoint) > MAX_ENDPOINT_LENGTH:
        raise UpstreamEndpointPolicyError("endpoint_too_long")
    if any(
        char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F for char in endpoint
    ):
        raise UpstreamEndpointPolicyError("invalid_endpoint")

    try:
        parsed = urlsplit(endpoint)
    except (TypeError, ValueError):
        raise UpstreamEndpointPolicyError("invalid_endpoint") from None

    scheme = parsed.scheme.lower()
    if not scheme:
        raise UpstreamEndpointPolicyError("missing_scheme")
    if scheme not in {"http", "https"}:
        raise UpstreamEndpointPolicyError("unsupported_scheme")
    if not parsed.netloc:
        raise UpstreamEndpointPolicyError("missing_host")
    if (
        "@" in parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise UpstreamEndpointPolicyError("userinfo_not_allowed")
    try:
        raw_host = parsed.hostname
    except ValueError:
        raise UpstreamEndpointPolicyError("invalid_host") from None
    if raw_host is None:
        raise UpstreamEndpointPolicyError("missing_host")
    try:
        host, ip_address = _canonical_host(raw_host)
    except UpstreamEndpointPolicyError:
        raise
    try:
        parsed_port = parsed.port
    except ValueError:
        raise UpstreamEndpointPolicyError("invalid_port") from None
    port = (
        parsed_port if parsed_port is not None else (443 if scheme == "https" else 80)
    )
    if not 1 <= port <= 65535:
        raise UpstreamEndpointPolicyError("invalid_port")
    return _ParsedEndpoint(
        url=endpoint,
        split=parsed,
        scheme=scheme,
        host=host,
        port=port,
        origin=_canonical_origin(scheme, host, port),
        ip_address=ip_address,
    )


def _normalize_allowed_origin(origin: Any) -> str:
    parsed = _parse_endpoint(origin)
    if parsed.scheme != "https":
        raise UpstreamEndpointConfigurationError("invalid_allowed_origin_scheme")
    if (
        parsed.split.path not in {"", "/"}
        or parsed.split.query
        or parsed.split.fragment
    ):
        raise UpstreamEndpointConfigurationError("invalid_allowed_origin_path")
    return parsed.origin


def _iter_resolver_values(result: Any) -> Iterable[Any]:
    if result is None:
        return ()
    if isinstance(result, (str, bytes, ipaddress.IPv4Address, ipaddress.IPv6Address)):
        return (result,)
    if isinstance(result, (tuple, list)) and len(result) >= 5:
        if isinstance(result[0], int) and isinstance(result[4], (tuple, list)):
            return (result,)
    if (
        isinstance(result, (tuple, list))
        and len(result) == 2
        and isinstance(result[1], int)
        and isinstance(
            result[0], (str, bytes, ipaddress.IPv4Address, ipaddress.IPv6Address)
        )
    ):
        return (result,)
    try:
        return tuple(result)
    except Exception:
        return (result,)


def _coerce_address(value: Any) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if isinstance(value, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
        return value
    if isinstance(value, (tuple, list)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def _address_class(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ResolvedAddressClass:
    mapped = getattr(address, "ipv4_mapped", None)
    classification_address = mapped or address
    if any(classification_address in network for network in _CLOUD_METADATA_NETWORKS):
        return ResolvedAddressClass.CLOUD_METADATA
    if classification_address.is_loopback:
        return ResolvedAddressClass.LOOPBACK
    if classification_address.is_link_local:
        return ResolvedAddressClass.LINK_LOCAL
    if classification_address.is_multicast:
        return ResolvedAddressClass.MULTICAST
    if classification_address.is_unspecified:
        return ResolvedAddressClass.UNSPECIFIED
    if classification_address.is_private:
        return ResolvedAddressClass.PRIVATE
    if classification_address.is_reserved:
        return ResolvedAddressClass.RESERVED
    if not classification_address.is_global:
        return ResolvedAddressClass.NON_GLOBAL
    return ResolvedAddressClass.PUBLIC


def _sort_addresses(
    addresses: Iterable[ipaddress.IPv4Address | ipaddress.IPv6Address],
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    unique = {str(address): address for address in addresses}
    return tuple(
        sorted(unique.values(), key=lambda address: (address.version, int(address)))
    )


def _resolve_addresses(
    parsed: _ParsedEndpoint,
    resolver: AddressResolver,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    if parsed.ip_address is not None:
        return (parsed.ip_address,)
    try:
        result = resolver(parsed.host, parsed.port)
    except Exception:
        result = _RESOLUTION_FAILED
    if result is _RESOLUTION_FAILED:
        raise UpstreamEndpointResolutionError("dns_resolution_failed")
    try:
        values = _iter_resolver_values(result)
    except Exception:
        raise UpstreamEndpointResolutionError("dns_resolution_failed") from None
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for value in values:
        address = value
        if isinstance(value, (tuple, list)) and len(value) >= 5:
            address = value[4]
        coerced = _coerce_address(address)
        if coerced is None:
            raise UpstreamEndpointResolutionError("invalid_resolved_address")
        addresses.append(coerced)
    if not addresses:
        raise UpstreamEndpointResolutionError("no_resolved_addresses")
    return _sort_addresses(addresses)


def _validate_address_classes(
    classes: tuple[ResolvedAddressClass, ...],
    *,
    loopback_only: bool,
) -> None:
    distinct = set(classes)
    if loopback_only:
        if distinct == {ResolvedAddressClass.LOOPBACK}:
            return
        raise UpstreamEndpointResolutionError("loopback_only_violation")
    if distinct == {ResolvedAddressClass.PUBLIC}:
        return
    if ResolvedAddressClass.PUBLIC in distinct:
        raise UpstreamEndpointResolutionError("mixed_public_prohibited_addresses")
    if len(distinct) > 1:
        raise UpstreamEndpointResolutionError("mixed_prohibited_addresses")
    reason_code = _ADDRESS_REASON_CODES[next(iter(distinct))]
    raise UpstreamEndpointResolutionError(reason_code)


def _is_loopback_host(
    host: str,
    ip_address: ipaddress.IPv4Address | ipaddress.IPv6Address | None,
) -> bool:
    if host in _LOOPBACK_HOSTNAMES:
        return True
    if ip_address is None:
        return False
    return _address_class(ip_address) == ResolvedAddressClass.LOOPBACK


@dataclass(frozen=True)
class UpstreamEndpointPolicy:
    """Fail-closed policy for operator-approved MCP upstream origins.

    Args:
        allowed_origins: Exact HTTPS origins approved for remote upstreams.
            Paths, queries, fragments, wildcards, and user information are not
            accepted in this allowlist.
        resolver: Callable receiving ``(hostname, port)`` and returning IP
            strings or ``getaddrinfo``-style records.
        allow_loopback: Explicit development-only mode for ``localhost`` and
            literal loopback URLs.  Every answer must be loopback, and remote
            origins remain subject to the HTTPS allowlist.
    """

    allowed_origins: frozenset[str] = field(default_factory=frozenset)
    resolver: AddressResolver = field(
        default=resolve_hostname, repr=False, compare=False
    )
    allow_loopback: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_loopback, bool) or not callable(self.resolver):
            raise UpstreamEndpointConfigurationError("invalid_resolver")
        if isinstance(self.allowed_origins, str):
            raise UpstreamEndpointConfigurationError("invalid_allowed_origin")
        try:
            values = tuple(self.allowed_origins)
        except TypeError:
            raise UpstreamEndpointConfigurationError("invalid_allowed_origin") from None
        normalized: set[str] = set()
        for value in values:
            try:
                normalized.add(_normalize_allowed_origin(value))
            except UpstreamEndpointPolicyError as exc:
                if isinstance(exc, UpstreamEndpointConfigurationError):
                    raise
                raise UpstreamEndpointConfigurationError(
                    "invalid_allowed_origin"
                ) from None
        object.__setattr__(self, "allowed_origins", frozenset(normalized))

    @classmethod
    def development_loopback(
        cls,
        *,
        resolver: AddressResolver = resolve_hostname,
    ) -> "UpstreamEndpointPolicy":
        """Build an explicit policy for synthetic/local development endpoints."""

        return cls(resolver=resolver, allow_loopback=True)

    @property
    def approved_origins(self) -> frozenset[str]:
        """Return the normalized exact origin allowlist."""

        return self.allowed_origins

    def validate(self, endpoint: str) -> ValidatedUpstreamEndpoint:
        """Validate one endpoint before a requester is called.

        The resolver is consulted for every hostname on every call.  This is
        deliberate: a prior successful lookup must not authorize a later DNS
        answer that points at a prohibited address.
        """

        parsed = _parse_endpoint(endpoint)
        loopback_host = _is_loopback_host(parsed.host, parsed.ip_address)
        if parsed.scheme == "http" and not loopback_host:
            raise UpstreamEndpointPolicyError("non_https_origin")
        if loopback_host:
            if not self.allow_loopback:
                raise UpstreamEndpointPolicyError("loopback_not_allowed")
        elif parsed.origin not in self.allowed_origins:
            raise UpstreamEndpointPolicyError("origin_not_allowed")

        addresses = _resolve_addresses(parsed, self.resolver)
        classes = tuple(_address_class(address) for address in addresses)
        _validate_address_classes(classes, loopback_only=loopback_host)
        return ValidatedUpstreamEndpoint(
            url=parsed.url,
            origin=parsed.origin,
            host=parsed.host,
            port=parsed.port,
            addresses=tuple(str(address) for address in addresses),
            address_classes=classes,
            is_loopback=loopback_host,
        )

    def authorize(self, endpoint: str) -> str:
        """Validate an endpoint and return its original URL for the requester."""

        return self.validate(endpoint).url

    def validate_redirect(
        self,
        redirect_target: str,
        *,
        base_url: str | None = None,
    ) -> ValidatedUpstreamEndpoint:
        """Validate an absolute or relative redirect target.

        A base URL is validated again before a relative target is joined.  A
        redirect cannot switch between the remote and loopback policy modes;
        this prevents a permitted remote service from redirecting a development
        request into a local service, or vice versa.
        """

        if base_url is None:
            return self.validate(redirect_target)
        base = self.validate(base_url)
        if not isinstance(redirect_target, str):
            raise UpstreamEndpointRedirectError("invalid_endpoint")
        try:
            target = urljoin(base.url, redirect_target)
        except (TypeError, ValueError):
            raise UpstreamEndpointRedirectError("invalid_endpoint") from None
        candidate = self.validate(target)
        if candidate.is_loopback != base.is_loopback:
            raise UpstreamEndpointRedirectError("redirect_mode_change")
        return candidate

    def validate_redirect_chain(
        self,
        endpoint: str,
        redirect_targets: Iterable[str],
    ) -> tuple[ValidatedUpstreamEndpoint, ...]:
        """Validate an initial endpoint and each redirect in sequence."""

        if isinstance(redirect_targets, (str, bytes)):
            raise UpstreamEndpointRedirectError("invalid_redirect_chain")
        try:
            current = self.validate(endpoint)
            approvals = [current]
            for redirect_target in redirect_targets:
                current = self.validate_redirect(redirect_target, base_url=current.url)
                approvals.append(current)
        except TypeError:
            raise UpstreamEndpointRedirectError("invalid_redirect_chain") from None
        return tuple(approvals)

    def call(
        self,
        endpoint: str,
        requester: Callable[[str], RequestResult],
    ) -> RequestResult:
        """Validate an endpoint immediately before invoking ``requester``.

        The callback receives the original URL only after policy validation.
        Redirect handling remains the requester's responsibility through
        :meth:`validate_redirect` or :meth:`validate_redirect_chain`.
        """

        url = self.authorize(endpoint)
        return requester(url)


def validate_upstream_endpoint(
    endpoint: str,
    *,
    allowed_origins: Iterable[str] = (),
    resolver: AddressResolver = resolve_hostname,
    allow_loopback: bool = False,
) -> ValidatedUpstreamEndpoint:
    """Validate one endpoint using a one-shot policy."""

    return UpstreamEndpointPolicy(
        allowed_origins=allowed_origins,
        resolver=resolver,
        allow_loopback=allow_loopback,
    ).validate(endpoint)


def authorize_upstream_endpoint(
    endpoint: str,
    *,
    allowed_origins: Iterable[str] = (),
    resolver: AddressResolver = resolve_hostname,
    allow_loopback: bool = False,
) -> str:
    """Return an endpoint for a requester only after policy validation."""

    return UpstreamEndpointPolicy(
        allowed_origins=allowed_origins,
        resolver=resolver,
        allow_loopback=allow_loopback,
    ).authorize(endpoint)


EndpointApproval = ValidatedUpstreamEndpoint
EndpointPolicyError = UpstreamEndpointPolicyError


__all__ = [
    "AddressResolver",
    "EndpointApproval",
    "EndpointPolicyError",
    "MAX_ENDPOINT_LENGTH",
    "ResolvedAddressClass",
    "UpstreamEndpointConfigurationError",
    "UpstreamEndpointPolicy",
    "UpstreamEndpointPolicyError",
    "UpstreamEndpointRedirectError",
    "UpstreamEndpointResolutionError",
    "ValidatedUpstreamEndpoint",
    "authorize_upstream_endpoint",
    "resolve_hostname",
    "validate_upstream_endpoint",
]
