"""Security boundaries shared by the OpenMed MCP gateway and adapters.

The module is deliberately dependency-light.  Local stdio mode can import it
without an HTTP client or an identity-provider SDK, while remote deployments
can opt into the MCP OAuth metadata and resource-indicator flow.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request as URLRequest
from urllib.request import urlopen

_LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_URL_SCHEMES = frozenset({"http", "https"})
_DEFAULT_MAX_PAYLOAD_BYTES = 256 * 1024
_DEFAULT_MAX_STRING_LENGTH = 64 * 1024
_DEFAULT_MAX_ARRAY_ITEMS = 256
_DEFAULT_MAX_OBJECT_KEYS = 128
_DEFAULT_MAX_NESTING = 12
_DEFAULT_MAX_NODES = 2048
_METADATA_MAX_BYTES = 1024 * 1024

MCP_AUTH_ENABLED_ENV_VAR = "OPENMED_MCP_AUTH_ENABLED"
MCP_RESOURCE_URL_ENV_VAR = "OPENMED_MCP_RESOURCE_URL"
MCP_AUTHORIZATION_SERVER_URL_ENV_VAR = "OPENMED_MCP_AUTHORIZATION_SERVER_URL"
MCP_REQUIRED_SCOPES_ENV_VAR = "OPENMED_MCP_REQUIRED_SCOPES"
MCP_ALLOW_INSECURE_LOCALHOST_ENV_VAR = "OPENMED_MCP_ALLOW_INSECURE_LOCALHOST"
MCP_STATE_CHANGE_SCOPES_ENV_VAR = "OPENMED_MCP_STATE_CHANGE_SCOPES"

_PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b"
        r".{0,100}\b(?:previous|prior|above|system|developer|safety|guardrail)"
        r"\b.{0,60}\b(?:instruction|prompt|rule|message)s?\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:system|developer)\s+(?:message|prompt|instruction)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|exfiltrate|disclose|print|dump|show)\b"
        r".{0,100}\b(?:token|secret|credential|password|prompt|key)s?\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\b(?:jailbreak|prompt\s+injection)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:execute|invoke|call|run)\b\s+(?:the\s+)?"
        r"(?:tool|function|command)\b",
        re.IGNORECASE,
    ),
)

_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "client_assertion",
        "client_secret",
        "cookie",
        "id_token",
        "mcp_token",
        "password",
        "proxy_authorization",
        "refresh_token",
        "secret",
        "token",
    }
)
_PAYLOAD_FIELD_NAMES = frozenset(
    {
        "arguments",
        "content",
        "document",
        "document_text",
        "input",
        "payload",
        "resource_text",
        "response_body",
        "request_body",
        "raw_text",
        "text",
        "tool_arguments",
        "tool_payload",
    }
)
_BEARER_PATTERN = re.compile(r"(?i)(\bbearer\s+)[^\s,;]+")
_SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(\b(?:access[_-]?token|refresh[_-]?token|client[_-]?secret|"
    r"client[_-]?assertion|authorization|api[_-]?key|password|secret)\b"
    r"\s*[:=]\s*)([^\s,;]+)"
)
_JWT_PATTERN = re.compile(
    r"\b[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
)


class MCPGatewaySecurityError(ValueError):
    """Base error with a stable, safe message for a gateway boundary."""

    code = "security_error"
    safe_message = "The MCP security policy rejected the request."

    def __init__(self, message: Optional[str] = None) -> None:
        # Callers may use a custom message for local diagnostics, but the
        # public message remains fixed so credentials and tool data cannot be
        # reflected by accident.
        del message
        super().__init__(self.safe_message)


class MCPConfigurationError(MCPGatewaySecurityError):
    """Raised when remote MCP authorization is incomplete or unsafe."""

    code = "invalid_security_configuration"
    safe_message = "The MCP security configuration is invalid."


class InsecureRemoteURLError(MCPGatewaySecurityError):
    """Raised when a non-local remote URL does not use HTTPS."""

    code = "insecure_remote_url"
    safe_message = "Remote MCP authorization URLs must use HTTPS."


class MetadataDiscoveryError(MCPGatewaySecurityError):
    """Raised when protected-resource or authorization-server metadata is bad."""

    code = "metadata_discovery_failed"
    safe_message = "MCP authorization metadata is unavailable or invalid."


class MCPAuthorizationError(MCPGatewaySecurityError):
    """Raised when a token cannot authorize a protected MCP operation."""

    code = "invalid_token"
    safe_message = "The MCP access token is invalid for this resource."


class MissingScopeError(MCPAuthorizationError):
    """Raised when a verified token lacks a tool scope."""

    code = "insufficient_scope"
    safe_message = "The MCP access token lacks the required tool scope."


class TokenPassthroughError(MCPGatewaySecurityError):
    """Raised when inbound MCP credentials are headed to an upstream service."""

    code = "token_passthrough_rejected"
    safe_message = "Inbound MCP credentials cannot be passed to an upstream service."


class InvalidRedirectURIError(MCPGatewaySecurityError):
    """Raised when an OAuth redirect URI is not an exact registered URI."""

    code = "invalid_redirect_uri"
    safe_message = "The OAuth redirect URI is not registered for this client."


class PKCEError(MCPGatewaySecurityError):
    """Raised when an authorization-code request does not use S256 PKCE."""

    code = "invalid_pkce"
    safe_message = "Authorization-code clients must use S256 PKCE."


class ToolPolicyError(MCPGatewaySecurityError):
    """Raised when a tool call violates schema, bounds, or authorization policy."""

    code = "tool_policy_rejected"
    safe_message = "The MCP tool call was rejected by policy."


class PromptInjectionError(ToolPolicyError):
    """Raised when untrusted tool content contains instruction-like payloads."""

    code = "prompt_injection_detected"
    safe_message = "Untrusted document content cannot issue tool instructions."


class PayloadBoundsError(ToolPolicyError):
    """Raised when a tool payload exceeds a configured resource bound."""

    code = "payload_too_large"
    safe_message = "The MCP tool payload exceeds the configured safety bounds."


class StateChangePermissionError(ToolPolicyError):
    """Raised when a state-changing tool lacks explicit permission."""

    code = "state_change_not_permitted"
    safe_message = "This state-changing MCP tool requires explicit permission."


def is_localhost_url(url: str) -> bool:
    """Return whether *url* names an explicit loopback development host."""

    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    return host is not None and host.lower().rstrip(".") in _LOCALHOST_HOSTS


def validate_remote_url(
    url: str,
    *,
    allow_insecure_localhost: bool = False,
) -> str:
    """Validate and normalize an MCP resource or authorization URL.

    HTTP is accepted only for an explicitly enabled localhost development
    target.  User information, queries, and fragments are rejected because
    they are not part of a stable OAuth resource identifier.
    """

    if not isinstance(url, str) or not url.strip():
        raise InsecureRemoteURLError()
    value = url.strip()
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise InsecureRemoteURLError() from None

    if (
        parsed.scheme.lower() not in _URL_SCHEMES
        or hostname is None
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port is None
        and ":" in parsed.netloc.rsplit("]", 1)[-1]
    ):
        raise InsecureRemoteURLError()

    normalized_host = hostname.lower().rstrip(".")
    local_http = (
        parsed.scheme.lower() == "http"
        and normalized_host in _LOCALHOST_HOSTS
        and allow_insecure_localhost
    )
    if parsed.scheme.lower() != "https" and not local_http:
        raise InsecureRemoteURLError()

    if ":" in hostname and not hostname.startswith("["):
        normalized_host = f"[{normalized_host}]"
    normalized_netloc = normalized_host
    if port is not None:
        normalized_netloc = f"{normalized_netloc}:{port}"
    return urlunsplit(
        (
            parsed.scheme.lower(),
            normalized_netloc,
            parsed.path,
            "",
            "",
        )
    )


def _same_resource(left: str, right: str) -> bool:
    """Compare URL resource identifiers with case-normalized host components."""

    try:
        return validate_remote_url(
            left, allow_insecure_localhost=True
        ) == validate_remote_url(
            right,
            allow_insecure_localhost=True,
        )
    except InsecureRemoteURLError:
        return False


def build_protected_resource_metadata_url(resource_url: str) -> str:
    """Build the RFC 9728 protected-resource metadata endpoint URL."""

    resource = validate_remote_url(resource_url, allow_insecure_localhost=True)
    parsed = urlsplit(resource)
    resource_path = parsed.path if parsed.path != "/" else ""
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            "/.well-known/oauth-protected-resource" + resource_path,
            "",
            "",
        )
    )


def build_authorization_server_metadata_url(issuer_url: str) -> str:
    """Build the RFC 8414 authorization-server metadata endpoint URL."""

    issuer = validate_remote_url(issuer_url, allow_insecure_localhost=True)
    return issuer.rstrip("/") + "/.well-known/oauth-authorization-server"


def _normalized_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = [str(item).strip() for item in value]
    else:
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return tuple(result)


@dataclass(frozen=True)
class ProtectedResourceMetadata:
    """Validated RFC 9728 metadata for an MCP protected resource."""

    resource: str
    authorization_servers: tuple[str, ...]
    scopes_supported: tuple[str, ...] = ()
    bearer_methods_supported: tuple[str, ...] = ("header",)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_resource: Optional[str] = None,
        allow_insecure_localhost: bool = False,
    ) -> "ProtectedResourceMetadata":
        """Parse metadata and require an exact resource binding."""

        if not isinstance(payload, Mapping):
            raise MetadataDiscoveryError()
        raw_resource = payload.get("resource")
        if not isinstance(raw_resource, str):
            raise MetadataDiscoveryError()
        resource = validate_remote_url(
            raw_resource,
            allow_insecure_localhost=allow_insecure_localhost,
        )
        if expected_resource is not None:
            expected = validate_remote_url(
                expected_resource,
                allow_insecure_localhost=allow_insecure_localhost,
            )
            if not _same_resource(resource, expected):
                raise MCPAuthorizationError()

        authorization_servers = _normalized_strings(
            payload.get("authorization_servers")
        )
        if not authorization_servers:
            raise MetadataDiscoveryError()
        normalized_servers = tuple(
            validate_remote_url(
                server,
                allow_insecure_localhost=allow_insecure_localhost,
            )
            for server in authorization_servers
        )
        bearer_methods = _normalized_strings(payload.get("bearer_methods_supported"))
        if bearer_methods and "header" not in bearer_methods:
            raise MetadataDiscoveryError()
        return cls(
            resource=resource,
            authorization_servers=normalized_servers,
            scopes_supported=_normalized_strings(payload.get("scopes_supported")),
            bearer_methods_supported=bearer_methods or ("header",),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready metadata document."""

        return {
            "resource": self.resource,
            "authorization_servers": list(self.authorization_servers),
            "scopes_supported": list(self.scopes_supported) or None,
            "bearer_methods_supported": list(self.bearer_methods_supported),
        }


@dataclass(frozen=True)
class AuthorizationServerMetadata:
    """Validated RFC 8414 authorization-server metadata."""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    code_challenge_methods_supported: tuple[str, ...]
    scopes_supported: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_issuer: Optional[str] = None,
        allow_insecure_localhost: bool = False,
    ) -> "AuthorizationServerMetadata":
        """Parse metadata and require S256 PKCE support."""

        if not isinstance(payload, Mapping):
            raise MetadataDiscoveryError()
        fields = (
            payload.get("issuer"),
            payload.get("authorization_endpoint"),
            payload.get("token_endpoint"),
        )
        if not all(isinstance(value, str) for value in fields):
            raise MetadataDiscoveryError()
        issuer, authorization_endpoint, token_endpoint = (
            validate_remote_url(
                value,
                allow_insecure_localhost=allow_insecure_localhost,
            )
            for value in fields
        )
        if expected_issuer is not None and not _same_resource(
            issuer,
            validate_remote_url(
                expected_issuer,
                allow_insecure_localhost=allow_insecure_localhost,
            ),
        ):
            raise MCPAuthorizationError()
        methods = _normalized_strings(payload.get("code_challenge_methods_supported"))
        if "S256" not in methods:
            raise PKCEError()
        return cls(
            issuer=issuer,
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint,
            code_challenge_methods_supported=methods,
            scopes_supported=_normalized_strings(payload.get("scopes_supported")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready metadata document."""

        return {
            "issuer": self.issuer,
            "authorization_endpoint": self.authorization_endpoint,
            "token_endpoint": self.token_endpoint,
            "code_challenge_methods_supported": list(
                self.code_challenge_methods_supported
            ),
            "scopes_supported": list(self.scopes_supported) or None,
        }


@dataclass(frozen=True)
class MCPAuthorizationMetadata:
    """The protected-resource and authorization-server metadata pair."""

    protected_resource: ProtectedResourceMetadata
    authorization_server: AuthorizationServerMetadata

    @property
    def resource(self) -> str:
        """Return the resource identifier advertised by the server."""

        return self.protected_resource.resource


MetadataFetcher = Callable[[str], Any]


def _decode_metadata_response(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return dict(response)
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            payload = json_method()
        except Exception as exc:  # noqa: BLE001 - keep errors secret-safe.
            raise MetadataDiscoveryError() from exc
        if isinstance(payload, Mapping):
            return dict(payload)
    raise MetadataDiscoveryError()


def _default_metadata_fetcher(url: str) -> Mapping[str, Any]:
    request = URLRequest(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310 - URL is validated.
            final_url = response.geturl()
            if not _same_resource(final_url, url):
                raise MetadataDiscoveryError()
            body = response.read(_METADATA_MAX_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise MetadataDiscoveryError() from exc
    if len(body) > _METADATA_MAX_BYTES:
        raise MetadataDiscoveryError()
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataDiscoveryError() from exc
    if not isinstance(payload, Mapping):
        raise MetadataDiscoveryError()
    return dict(payload)


def _fetch_metadata(
    url: str, fetch_json: Optional[MetadataFetcher]
) -> Mapping[str, Any]:
    fetcher = fetch_json or _default_metadata_fetcher
    try:
        response = fetcher(url)
    except Exception as exc:  # noqa: BLE001 - do not expose transport details.
        raise MetadataDiscoveryError() from exc
    if inspect.isawaitable(response):
        raise MetadataDiscoveryError()
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and status_code >= 400:
        raise MetadataDiscoveryError()
    return _decode_metadata_response(response)


def discover_protected_resource_metadata(
    resource_url: str,
    *,
    fetch_json: Optional[MetadataFetcher] = None,
    allow_insecure_localhost: bool = False,
) -> ProtectedResourceMetadata:
    """Discover and validate RFC 9728 metadata for *resource_url*."""

    resource = validate_remote_url(
        resource_url,
        allow_insecure_localhost=allow_insecure_localhost,
    )
    metadata_url = build_protected_resource_metadata_url(resource)
    payload = _fetch_metadata(metadata_url, fetch_json)
    return ProtectedResourceMetadata.from_mapping(
        payload,
        expected_resource=resource,
        allow_insecure_localhost=allow_insecure_localhost,
    )


def discover_authorization_server_metadata(
    issuer_url: str,
    *,
    fetch_json: Optional[MetadataFetcher] = None,
    allow_insecure_localhost: bool = False,
) -> AuthorizationServerMetadata:
    """Discover and validate RFC 8414 metadata for *issuer_url*."""

    issuer = validate_remote_url(
        issuer_url,
        allow_insecure_localhost=allow_insecure_localhost,
    )
    metadata_url = build_authorization_server_metadata_url(issuer)
    payload = _fetch_metadata(metadata_url, fetch_json)
    return AuthorizationServerMetadata.from_mapping(
        payload,
        expected_issuer=issuer,
        allow_insecure_localhost=allow_insecure_localhost,
    )


class MCPMetadataClient:
    """Dependency-light client for the MCP OAuth metadata discovery flow."""

    def __init__(
        self,
        *,
        fetch_json: Optional[MetadataFetcher] = None,
        allow_insecure_localhost: bool = False,
    ) -> None:
        self.fetch_json = fetch_json
        self.allow_insecure_localhost = allow_insecure_localhost

    def discover(self, resource_url: str) -> MCPAuthorizationMetadata:
        """Discover protected-resource metadata, then its first AS metadata."""

        protected = discover_protected_resource_metadata(
            resource_url,
            fetch_json=self.fetch_json,
            allow_insecure_localhost=self.allow_insecure_localhost,
        )
        issuer = protected.authorization_servers[0]
        authorization_server = discover_authorization_server_metadata(
            issuer,
            fetch_json=self.fetch_json,
            allow_insecure_localhost=self.allow_insecure_localhost,
        )
        return MCPAuthorizationMetadata(protected, authorization_server)


def discover_mcp_authorization_metadata(
    resource_url: str,
    *,
    fetch_json: Optional[MetadataFetcher] = None,
    allow_insecure_localhost: bool = False,
) -> MCPAuthorizationMetadata:
    """Discover both metadata documents required by a remote MCP client."""

    return MCPMetadataClient(
        fetch_json=fetch_json,
        allow_insecure_localhost=allow_insecure_localhost,
    ).discover(resource_url)


def validate_redirect_uri(
    redirect_uri: str,
    registered_redirect_uris: Sequence[str],
) -> str:
    """Require an exact, fragment-free match to a registered redirect URI."""

    if not isinstance(redirect_uri, str) or not redirect_uri.strip():
        raise InvalidRedirectURIError()
    if redirect_uri != redirect_uri.strip():
        raise InvalidRedirectURIError()
    candidate = redirect_uri
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        raise InvalidRedirectURIError() from None
    if (
        not parsed.scheme
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or parsed.scheme.lower() in {"data", "file", "javascript"}
        or "*" in candidate
    ):
        raise InvalidRedirectURIError()
    if parsed.scheme.lower() == "https" and not parsed.netloc:
        raise InvalidRedirectURIError()
    if parsed.scheme.lower() == "http" and (
        not parsed.netloc or not is_localhost_url(candidate)
    ):
        raise InvalidRedirectURIError()
    normalized_registered = []
    for registered in registered_redirect_uris:
        if not isinstance(registered, str) or not registered.strip():
            raise InvalidRedirectURIError()
        if registered != registered.strip():
            raise InvalidRedirectURIError()
        value = registered
        try:
            registered_parts = urlsplit(value)
        except ValueError:
            raise InvalidRedirectURIError() from None
        if (
            not registered_parts.scheme
            or registered_parts.fragment
            or registered_parts.username is not None
            or registered_parts.password is not None
            or registered_parts.scheme.lower() in {"data", "file", "javascript"}
            or "*" in value
        ):
            raise InvalidRedirectURIError()
        if registered_parts.scheme.lower() == "https" and not registered_parts.netloc:
            raise InvalidRedirectURIError()
        if registered_parts.scheme.lower() == "http" and (
            not registered_parts.netloc or not is_localhost_url(value)
        ):
            raise InvalidRedirectURIError()
        normalized_registered.append(value)
    if candidate not in normalized_registered:
        raise InvalidRedirectURIError()
    return candidate


def validate_registered_redirect_uris(redirect_uris: Sequence[str]) -> tuple[str, ...]:
    """Validate and normalize a client's complete redirect URI set."""

    if isinstance(redirect_uris, (str, bytes, bytearray)) or not redirect_uris:
        raise InvalidRedirectURIError()
    values = tuple(str(uri) for uri in redirect_uris)
    for value in values:
        validate_redirect_uri(value, values)
    return values


def validate_pkce(code_challenge: str, code_challenge_method: str = "S256") -> str:
    """Require a valid RFC 7636 S256 code challenge."""

    if (
        code_challenge_method != "S256"
        or not isinstance(code_challenge, str)
        or not 43 <= len(code_challenge) <= 128
        or re.fullmatch(r"[A-Za-z0-9._~-]+", code_challenge) is None
    ):
        raise PKCEError()
    return code_challenge


def create_pkce_challenge(verifier: str) -> str:
    """Return the S256 challenge for a client-generated verifier."""

    if (
        not isinstance(verifier, str)
        or not 43 <= len(verifier) <= 128
        or re.fullmatch(r"[A-Za-z0-9._~-]+", verifier) is None
    ):
        raise PKCEError()
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def build_authorization_url(
    metadata: AuthorizationServerMetadata | MCPAuthorizationMetadata,
    *,
    client_id: str,
    redirect_uri: str,
    registered_redirect_uris: Sequence[str],
    code_challenge: str,
    state: str,
    scopes: Sequence[str] = (),
    resource: Optional[str] = None,
) -> str:
    """Build a resource-indicating authorization-code request with PKCE."""

    if isinstance(metadata, MCPAuthorizationMetadata):
        authorization_metadata = metadata.authorization_server
        if resource is None:
            resource = metadata.resource
        elif not _same_resource(resource, metadata.resource):
            raise MCPAuthorizationError()
    else:
        authorization_metadata = metadata
    if not isinstance(client_id, str) or not client_id.strip():
        raise MCPAuthorizationError()
    if not isinstance(state, str) or not state:
        raise MCPAuthorizationError()
    validate_redirect_uri(redirect_uri, registered_redirect_uris)
    validate_pkce(code_challenge)
    if resource is None:
        raise MCPAuthorizationError()
    resource = validate_remote_url(resource, allow_insecure_localhost=True)
    validate_remote_url(
        authorization_metadata.authorization_endpoint,
        allow_insecure_localhost=True,
    )
    if "S256" not in authorization_metadata.code_challenge_methods_supported:
        raise PKCEError()
    normalized_scopes = _normalized_strings(scopes)
    if authorization_metadata.scopes_supported and not set(normalized_scopes).issubset(
        authorization_metadata.scopes_supported
    ):
        raise MissingScopeError()
    query = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    if normalized_scopes:
        query["scope"] = " ".join(normalized_scopes)
    query["resource"] = resource
    return authorization_metadata.authorization_endpoint + "?" + urlencode(query)


@dataclass(frozen=True)
class ValidatedAccessToken:
    """A resource-bound token projection safe for authorization decisions."""

    token: str = field(repr=False)
    client_id: str
    scopes: tuple[str, ...]
    resource: str
    expires_at: Optional[float] = None
    subject: Optional[str] = None
    claims: Mapping[str, Any] = field(default_factory=dict, repr=False)


def _token_value(token: Any, name: str, default: Any = None) -> Any:
    if isinstance(token, Mapping):
        return token.get(name, default)
    return getattr(token, name, default)


def _token_claims(token: Any) -> Mapping[str, Any]:
    claims = _token_value(token, "claims")
    if isinstance(claims, Mapping):
        return dict(claims)
    if isinstance(token, Mapping):
        return dict(token)
    return {}


def _token_audiences(token: Any, claims: Mapping[str, Any]) -> tuple[str, ...]:
    audience = claims.get("aud")
    if audience is None:
        audience = _token_value(token, "aud")
    if audience is None:
        audience = _token_value(token, "audience")
    return _normalized_strings(audience)


def _scopes_satisfy(granted: Sequence[str], required: Sequence[str]) -> bool:
    granted_set = set(granted)
    if "*" in granted_set:
        return True
    for scope in required:
        if scope in granted_set:
            continue
        namespace = scope.split(":", 1)[0]
        if f"{namespace}:*" in granted_set:
            continue
        return False
    return True


def validate_token_binding(
    token: Any,
    *,
    resource_url: str,
    required_scopes: Sequence[str] = (),
    issuer_url: Optional[str] = None,
    now: Optional[float] = None,
) -> ValidatedAccessToken:
    """Validate issuer, RFC 8707 resource binding, and required scopes."""

    expected_resource = validate_remote_url(
        resource_url,
        allow_insecure_localhost=True,
    )
    claims = _token_claims(token)
    if issuer_url is not None:
        expected_issuer = validate_remote_url(
            issuer_url,
            allow_insecure_localhost=True,
        )
        token_issuer = claims.get("iss")
        if token_issuer is not None and (
            not isinstance(token_issuer, str)
            or not _same_resource(token_issuer, expected_issuer)
        ):
            raise MCPAuthorizationError()
    token_resource = _token_value(token, "resource")
    if token_resource is None:
        token_resource = claims.get("resource")
    audiences = _token_audiences(token, claims)
    if token_resource is None and not audiences:
        raise MCPAuthorizationError()
    if token_resource is not None:
        if not isinstance(token_resource, str) or not _same_resource(
            token_resource,
            expected_resource,
        ):
            raise MCPAuthorizationError()
    if audiences:
        normalized_audiences = tuple(
            validate_remote_url(audience, allow_insecure_localhost=True)
            for audience in audiences
        )
        if set(normalized_audiences) != {expected_resource}:
            raise MCPAuthorizationError()
    raw_scopes = _token_value(token, "scopes")
    if raw_scopes is None:
        raw_scopes = claims.get("scope")
    scopes = _normalized_strings(raw_scopes)
    if not _scopes_satisfy(scopes, _normalized_strings(required_scopes)):
        raise MissingScopeError()

    expires_at = _token_value(token, "expires_at")
    if expires_at is None:
        expires_at = claims.get("exp")
    try:
        normalized_expiry = float(expires_at) if expires_at is not None else None
    except (TypeError, ValueError):
        raise MCPAuthorizationError() from None
    if normalized_expiry is not None and normalized_expiry <= (
        time.time() if now is None else now
    ):
        raise MCPAuthorizationError()

    raw_token = _token_value(token, "token", "")
    client_id = _token_value(token, "client_id", "")
    subject = _token_value(token, "subject")
    if not isinstance(raw_token, str) or not raw_token:
        raise MCPAuthorizationError()
    if not isinstance(client_id, str) or not client_id:
        raise MCPAuthorizationError()
    return ValidatedAccessToken(
        token=raw_token,
        client_id=client_id,
        scopes=scopes,
        resource=expected_resource,
        expires_at=normalized_expiry,
        subject=subject if isinstance(subject, str) else None,
        claims=claims,
    )


async def _resolve_token(
    resolver: Callable[[str], Any],
    token: str,
) -> Any:
    result = resolver(token)
    if inspect.isawaitable(result):
        return await result
    return result


class MCPTokenVerifier:
    """MCP SDK token verifier with strict resource and scope checks.

    A resolver or an in-memory mapping must be supplied.  The default never
    decodes or accepts opaque bearer strings, so a remote server cannot
    accidentally treat an unverified inbound token as trusted.
    """

    def __init__(
        self,
        *,
        resource_url: str,
        token_store: Optional[Mapping[str, Any]] = None,
        resolver: Optional[Callable[[str], Any]] = None,
        required_scopes: Sequence[str] = (),
        issuer_url: Optional[str] = None,
    ) -> None:
        self.resource_url = validate_remote_url(
            resource_url,
            allow_insecure_localhost=True,
        )
        if token_store is not None and resolver is not None:
            raise MCPConfigurationError()
        self._token_store = dict(token_store or {})
        self._resolver = resolver
        self.required_scopes = _normalized_strings(required_scopes)
        self.issuer_url = (
            validate_remote_url(issuer_url, allow_insecure_localhost=True)
            if issuer_url is not None
            else None
        )

    async def verify_token(self, token: str) -> Any:
        """Return an MCP SDK access token only when every binding passes."""

        if not isinstance(token, str) or not token.strip():
            return None
        try:
            raw = (
                await _resolve_token(self._resolver, token)
                if self._resolver is not None
                else self._token_store.get(token)
            )
            if raw is None:
                return None
            if isinstance(raw, Mapping):
                raw = dict(raw)
                raw.setdefault("token", token)
            validated = validate_token_binding(
                raw,
                resource_url=self.resource_url,
                required_scopes=self.required_scopes,
                issuer_url=self.issuer_url,
            )
            from mcp.server.auth.provider import AccessToken

            return AccessToken(
                token=validated.token,
                client_id=validated.client_id,
                scopes=list(validated.scopes),
                expires_at=(
                    int(validated.expires_at)
                    if validated.expires_at is not None
                    else None
                ),
                resource=validated.resource,
                subject=validated.subject,
                claims=dict(validated.claims),
            )
        except Exception:  # noqa: BLE001 - verifier failures are always invalid.
            return None


@dataclass(frozen=True)
class MCPAuthorizationConfig:
    """Configuration for authenticated remote MCP mode."""

    enabled: bool = False
    resource_url: Optional[str] = None
    authorization_server_url: Optional[str] = None
    required_scopes: tuple[str, ...] = ()
    tool_scopes: Mapping[str, Sequence[str]] = field(default_factory=dict)
    state_change_scopes: tuple[str, ...] = ("mcp:state:write",)
    allow_insecure_localhost: bool = False
    max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES
    max_string_length: int = _DEFAULT_MAX_STRING_LENGTH
    max_array_items: int = _DEFAULT_MAX_ARRAY_ITEMS
    max_object_keys: int = _DEFAULT_MAX_OBJECT_KEYS
    max_nesting: int = _DEFAULT_MAX_NESTING
    max_nodes: int = _DEFAULT_MAX_NODES

    def __post_init__(self) -> None:
        if self.enabled and (
            not self.resource_url or not self.authorization_server_url
        ):
            raise MCPConfigurationError()
        if self.resource_url:
            resource = validate_remote_url(
                self.resource_url,
                allow_insecure_localhost=self.allow_insecure_localhost,
            )
            object.__setattr__(self, "resource_url", resource)
        if self.authorization_server_url:
            issuer = validate_remote_url(
                self.authorization_server_url,
                allow_insecure_localhost=self.allow_insecure_localhost,
            )
            object.__setattr__(self, "authorization_server_url", issuer)
        object.__setattr__(
            self, "required_scopes", _normalized_strings(self.required_scopes)
        )
        object.__setattr__(
            self, "state_change_scopes", _normalized_strings(self.state_change_scopes)
        )
        object.__setattr__(
            self,
            "tool_scopes",
            {
                str(name): _normalized_strings(scopes)
                for name, scopes in self.tool_scopes.items()
            },
        )
        bounds = (
            self.max_payload_bytes,
            self.max_string_length,
            self.max_array_items,
            self.max_object_keys,
            self.max_nesting,
            self.max_nodes,
        )
        if any(not isinstance(value, int) or value <= 0 for value in bounds):
            raise MCPConfigurationError()

    @classmethod
    def from_env(cls) -> "MCPAuthorizationConfig":
        """Read opt-in remote MCP authorization settings from the environment."""

        enabled = _parse_bool(os.getenv(MCP_AUTH_ENABLED_ENV_VAR), default=False)
        allow_localhost = _parse_bool(
            os.getenv(MCP_ALLOW_INSECURE_LOCALHOST_ENV_VAR),
            default=False,
        )
        required = _normalized_strings(os.getenv(MCP_REQUIRED_SCOPES_ENV_VAR))
        state_scopes = _normalized_strings(
            os.getenv(MCP_STATE_CHANGE_SCOPES_ENV_VAR)
        ) or ("mcp:state:write",)
        return cls(
            enabled=enabled,
            resource_url=os.getenv(MCP_RESOURCE_URL_ENV_VAR) or None,
            authorization_server_url=(
                os.getenv(MCP_AUTHORIZATION_SERVER_URL_ENV_VAR) or None
            ),
            required_scopes=required,
            state_change_scopes=state_scopes,
            allow_insecure_localhost=allow_localhost,
        )

    def required_scopes_for_tool(self, tool_name: str) -> tuple[str, ...]:
        """Return the configured scope set for one registered tool."""

        configured = self.tool_scopes.get(tool_name)
        if configured:
            return tuple(configured)
        return (f"mcp:tool:{tool_name}",)

    def auth_settings(self) -> Any:
        """Return MCP SDK ``AuthSettings`` for this configuration."""

        if (
            not self.enabled
            or not self.resource_url
            or not self.authorization_server_url
        ):
            return None
        from mcp.server.auth.settings import AuthSettings

        return AuthSettings(
            issuer_url=self.authorization_server_url,
            resource_server_url=self.resource_url,
            required_scopes=list(self.required_scopes) or None,
        )


def _parse_bool(value: Optional[str], *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    raise MCPConfigurationError()


class SecureOAuthAuthorizationServerProvider:
    """Validate MCP OAuth provider inputs before delegating to an AS store."""

    def __init__(
        self,
        provider: Any,
        *,
        resource_url: str,
        issuer_url: Optional[str] = None,
        allow_insecure_localhost: bool = False,
    ) -> None:
        self.provider = provider
        self.resource_url = validate_remote_url(
            resource_url,
            allow_insecure_localhost=allow_insecure_localhost,
        )
        self.issuer_url = (
            validate_remote_url(
                issuer_url,
                allow_insecure_localhost=allow_insecure_localhost,
            )
            if issuer_url is not None
            else None
        )

    async def get_client(self, client_id: str) -> Any:
        client = await self.provider.get_client(client_id)
        if client is not None:
            self._validate_client(client)
        return client

    async def register_client(self, client_info: Any) -> None:
        self._validate_client(client_info)
        await self.provider.register_client(client_info)

    async def authorize(self, client: Any, params: Any) -> str:
        self._validate_client(client)
        validate_pkce(params.code_challenge)
        validate_redirect_uri(
            str(params.redirect_uri),
            tuple(str(uri) for uri in (client.redirect_uris or ())),
        )
        requested_resource = getattr(params, "resource", None)
        if requested_resource is not None and not _same_resource(
            str(requested_resource),
            self.resource_url,
        ):
            raise MCPAuthorizationError()
        return await self.provider.authorize(client, params)

    async def load_access_token(self, token: str) -> Any:
        raw = await self.provider.load_access_token(token)
        if raw is None:
            return None
        try:
            validate_token_binding(
                raw,
                resource_url=self.resource_url,
                issuer_url=self.issuer_url,
            )
        except MCPGatewaySecurityError:
            return None
        return raw

    def _validate_client(self, client: Any) -> None:
        redirect_uris = getattr(client, "redirect_uris", None)
        if not redirect_uris:
            raise InvalidRedirectURIError()
        validate_registered_redirect_uris(tuple(str(uri) for uri in redirect_uris))
        grant_types = _normalized_strings(getattr(client, "grant_types", ()))
        if grant_types and "authorization_code" not in grant_types:
            raise MCPAuthorizationError()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)


def _walk_payload(
    value: Any,
    *,
    depth: int,
    counters: dict[str, int],
    max_string_length: int,
    max_array_items: int,
    max_object_keys: int,
    max_nesting: int,
    max_nodes: int,
) -> None:
    counters["nodes"] += 1
    if counters["nodes"] > max_nodes or depth > max_nesting:
        raise PayloadBoundsError()
    if isinstance(value, str):
        if len(value) > max_string_length:
            raise PayloadBoundsError()
        return
    if isinstance(value, Mapping):
        if len(value) > max_object_keys:
            raise PayloadBoundsError()
        for key, child in value.items():
            if not isinstance(key, str) or len(key) > max_string_length:
                raise PayloadBoundsError()
            _walk_payload(
                child,
                depth=depth + 1,
                counters=counters,
                max_string_length=max_string_length,
                max_array_items=max_array_items,
                max_object_keys=max_object_keys,
                max_nesting=max_nesting,
                max_nodes=max_nodes,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) > max_array_items:
            raise PayloadBoundsError()
        for child in value:
            _walk_payload(
                child,
                depth=depth + 1,
                counters=counters,
                max_string_length=max_string_length,
                max_array_items=max_array_items,
                max_object_keys=max_object_keys,
                max_nesting=max_nesting,
                max_nodes=max_nodes,
            )


def validate_payload_bounds(
    payload: Any,
    *,
    max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
    max_string_length: int = _DEFAULT_MAX_STRING_LENGTH,
    max_array_items: int = _DEFAULT_MAX_ARRAY_ITEMS,
    max_object_keys: int = _DEFAULT_MAX_OBJECT_KEYS,
    max_nesting: int = _DEFAULT_MAX_NESTING,
    max_nodes: int = _DEFAULT_MAX_NODES,
) -> None:
    """Reject oversized or excessively nested JSON-like tool arguments."""

    try:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        raise PayloadBoundsError() from None
    if len(encoded.encode("utf-8")) > max_payload_bytes:
        raise PayloadBoundsError()
    _walk_payload(
        payload,
        depth=0,
        counters={"nodes": 0},
        max_string_length=max_string_length,
        max_array_items=max_array_items,
        max_object_keys=max_object_keys,
        max_nesting=max_nesting,
        max_nodes=max_nodes,
    )


def contains_prompt_injection(value: Any) -> bool:
    """Return whether untrusted nested content resembles an instruction attack."""

    if isinstance(value, str):
        return any(
            pattern.search(value) is not None for pattern in _PROMPT_INJECTION_PATTERNS
        )
    if isinstance(value, Mapping):
        return any(
            contains_prompt_injection(key) or contains_prompt_injection(child)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return any(contains_prompt_injection(child) for child in value)
    return False


def validate_untrusted_tool_content(value: Any) -> None:
    """Reject instruction-like content embedded in a document/tool payload."""

    if contains_prompt_injection(value):
        raise PromptInjectionError()


class MCPToolPolicy:
    """Schema, content, resource, and state-change policy for MCP tools."""

    def __init__(
        self,
        *,
        required_scopes: Optional[Mapping[str, Sequence[str]]] = None,
        state_change_scopes: Sequence[str] = ("mcp:state:write",),
        require_authentication: bool = False,
        allow_local_state_changes: bool = True,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
        max_string_length: int = _DEFAULT_MAX_STRING_LENGTH,
        max_array_items: int = _DEFAULT_MAX_ARRAY_ITEMS,
        max_object_keys: int = _DEFAULT_MAX_OBJECT_KEYS,
        max_nesting: int = _DEFAULT_MAX_NESTING,
        max_nodes: int = _DEFAULT_MAX_NODES,
    ) -> None:
        self.required_scopes = {
            str(name): _normalized_strings(scopes)
            for name, scopes in (required_scopes or {}).items()
        }
        self.state_change_scopes = _normalized_strings(state_change_scopes)
        self.require_authentication = require_authentication
        self.allow_local_state_changes = allow_local_state_changes
        self.bounds = {
            "max_payload_bytes": max_payload_bytes,
            "max_string_length": max_string_length,
            "max_array_items": max_array_items,
            "max_object_keys": max_object_keys,
            "max_nesting": max_nesting,
            "max_nodes": max_nodes,
        }

    def required_scopes_for_tool(self, tool_name: str) -> tuple[str, ...]:
        """Return configured or conservative per-tool scopes."""

        configured = self.required_scopes.get(tool_name)
        if configured:
            return configured
        return (f"mcp:tool:{tool_name}",)

    def has_state_change_permission(self, granted_scopes: Sequence[str]) -> bool:
        """Return whether a principal has the separate state-change grant."""

        return _scopes_satisfy(granted_scopes, self.state_change_scopes)

    def validate_tool_call(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        granted_scopes: Sequence[str] = (),
        permission_granted: bool = False,
    ) -> dict[str, Any]:
        """Validate one call without logging or returning its payload."""

        if not isinstance(arguments, Mapping):
            raise ToolPolicyError()
        validate_payload_bounds(arguments, **self.bounds)
        validate_untrusted_tool_content(arguments)
        from openmed.mcp.tool_registry import (
            TOOL_REGISTRY,
            validate_registered_tool_input,
        )

        spec = TOOL_REGISTRY.get(tool_name)
        validated = validate_registered_tool_input(tool_name, arguments)
        required = self.required_scopes_for_tool(tool_name)
        if self.require_authentication and not granted_scopes:
            raise MCPAuthorizationError()
        if self.require_authentication and not _scopes_satisfy(
            granted_scopes,
            required,
        ):
            raise MissingScopeError()
        if not spec.read_only_hint and not self.allow_local_state_changes:
            if not permission_granted and not self.has_state_change_permission(
                granted_scopes
            ):
                raise StateChangePermissionError()
        return validated


def assert_no_token_passthrough(
    payload: Any,
    *,
    inbound_token: Optional[str] = None,
) -> None:
    """Reject authorization material in an upstream request payload or headers."""

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized_key = str(key).strip().lower().replace("-", "_")
                if normalized_key in _SENSITIVE_FIELD_NAMES:
                    raise TokenPassthroughError()
                walk(child)
            return
        if isinstance(value, Sequence) and not isinstance(
            value,
            (bytes, bytearray, str),
        ):
            for child in value:
                walk(child)
            return
        if inbound_token and isinstance(value, str) and inbound_token in value:
            raise TokenPassthroughError()

    walk(payload)


def build_upstream_headers(
    headers: Optional[Mapping[str, str]] = None,
    *,
    inbound_token: Optional[str] = None,
) -> dict[str, str]:
    """Return allowlisted upstream headers after rejecting credential forwarding."""

    if headers is None:
        return {}
    assert_no_token_passthrough(headers, inbound_token=inbound_token)
    allowed = {"accept", "content-type", "if-match", "if-none-match"}
    return {
        str(key): str(value)
        for key, value in headers.items()
        if str(key).lower() in allowed
    }


def redact_sensitive_text(value: Any, *, secrets: Sequence[str] = ()) -> str:
    """Redact bearer credentials, JWTs, and secret assignments from text."""

    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "[REDACTED]")
    text = _BEARER_PATTERN.sub(r"\1[REDACTED]", text)
    text = _SENSITIVE_ASSIGNMENT_PATTERN.sub(r"\1[REDACTED]", text)
    return _JWT_PATTERN.sub("[REDACTED]", text)


def redact_log_value(value: Any, *, key: Optional[str] = None) -> Any:
    """Recursively redact credentials and raw tool/document payloads for logs."""

    normalized_key = (key or "").lower().replace("-", "_")
    if normalized_key in _SENSITIVE_FIELD_NAMES:
        return "[REDACTED]"
    if normalized_key in _PAYLOAD_FIELD_NAMES:
        return "[REDACTED_PAYLOAD]"
    if isinstance(value, Mapping):
        return {
            str(child_key): redact_log_value(child, key=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [redact_log_value(child) for child in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value


class MCPLogFilter(logging.Filter):
    """Keep MCP logger messages free of credentials and tool payloads."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_sensitive_text(record.msg)
        if record.args:
            if isinstance(record.args, Mapping):
                record.args = redact_log_value(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(redact_log_value(item) for item in record.args)
            else:
                record.args = redact_log_value(record.args)
        for name, value in tuple(record.__dict__.items()):
            if name.startswith("_") or name in {"msg", "args"}:
                continue
            if isinstance(value, (Mapping, list, tuple)):
                setattr(record, name, redact_log_value(value, key=name))
        return True


def install_mcp_log_filter() -> MCPLogFilter:
    """Install the redaction filter on OpenMed and MCP logger namespaces."""

    security_filter = MCPLogFilter()
    for logger_name in ("openmed.mcp", "mcp"):
        logger = logging.getLogger(logger_name)
        if not any(isinstance(item, MCPLogFilter) for item in logger.filters):
            logger.addFilter(security_filter)
    return security_filter


def safe_error_payload(error: BaseException) -> dict[str, Any]:
    """Return a stable error envelope without exception text or input data."""

    if isinstance(error, MCPGatewaySecurityError):
        code = error.code
        message = error.safe_message
    else:
        code = "execution_error"
        message = "The MCP tool could not complete the request."
    return {"error": {"code": code, "message": message}, "is_error": True}


__all__ = [
    "AuthorizationServerMetadata",
    "InvalidRedirectURIError",
    "InsecureRemoteURLError",
    "MCPAuthConfig",
    "MCPAuthorizationConfig",
    "MCPAuthorizationError",
    "MCPAuthorizationMetadata",
    "MCPAuthorizationProvider",
    "MCPConfigurationError",
    "MCPGatewaySecurityError",
    "MCPLogFilter",
    "MCPMetadataClient",
    "MCPTokenVerifier",
    "MCPToolPolicy",
    "MetadataDiscoveryError",
    "MissingScopeError",
    "OAuthAuthorizationServerMetadata",
    "PKCEError",
    "PayloadBoundsError",
    "PromptInjectionError",
    "ProtectedResourceMetadata",
    "SecureOAuthAuthorizationServerProvider",
    "StateChangePermissionError",
    "TokenPassthroughError",
    "ValidatedAccessToken",
    "assert_no_token_passthrough",
    "build_authorization_server_metadata_url",
    "build_authorization_url",
    "build_protected_resource_metadata_url",
    "build_upstream_headers",
    "contains_prompt_injection",
    "create_pkce_challenge",
    "discover_authorization_server_metadata",
    "discover_mcp_authorization_metadata",
    "discover_protected_resource_metadata",
    "install_mcp_log_filter",
    "is_localhost_url",
    "redact_log_value",
    "redact_sensitive_text",
    "safe_error_payload",
    "validate_payload_bounds",
    "validate_pkce",
    "validate_redirect_uri",
    "validate_registered_redirect_uris",
    "validate_remote_url",
    "validate_token_binding",
    "validate_untrusted_tool_content",
]

# Short aliases keep the boundary easy to discover for integrations that use
# ``auth`` rather than ``authorization`` in their configuration vocabulary.
MCPAuthConfig = MCPAuthorizationConfig
OAuthAuthorizationServerMetadata = AuthorizationServerMetadata
MCPAuthorizationProvider = SecureOAuthAuthorizationServerProvider
