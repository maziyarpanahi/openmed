"""Authentication middleware for the OpenMed REST service."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import Request
from starlette.responses import Response

from .throttle import TokenBucketRateLimiter, client_identity, format_retry_after

SERVICE_AUTH_ENABLED_ENV_VAR = "OPENMED_SERVICE_AUTH_ENABLED"
SERVICE_AUTH_DENY_BY_DEFAULT_ENV_VAR = "OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT"
SERVICE_AUTH_API_KEYS_ENV_VAR = "OPENMED_SERVICE_AUTH_API_KEYS"
SERVICE_AUTH_JWKS_ENV_VAR = "OPENMED_SERVICE_AUTH_JWKS"
SERVICE_AUTH_JWKS_FILE_ENV_VAR = "OPENMED_SERVICE_AUTH_JWKS_FILE"
SERVICE_AUTH_JWT_ISSUER_ENV_VAR = "OPENMED_SERVICE_AUTH_JWT_ISSUER"
SERVICE_AUTH_JWT_AUDIENCE_ENV_VAR = "OPENMED_SERVICE_AUTH_JWT_AUDIENCE"
SERVICE_AUTH_JWT_LEEWAY_ENV_VAR = "OPENMED_SERVICE_AUTH_JWT_LEEWAY_SECONDS"
SERVICE_AUTH_ROUTE_SCOPES_ENV_VAR = "OPENMED_SERVICE_AUTH_ROUTE_SCOPES"
SERVICE_AUTH_FAILURE_RPS_ENV_VAR = "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS"
SERVICE_AUTH_FAILURE_BURST_ENV_VAR = "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST"
SERVICE_AUTH_FAILURE_KEY_ENV_VAR = "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_KEY"

DEFAULT_AUTH_FAILURE_RATE_LIMIT_RPS = 5.0
DEFAULT_AUTH_FAILURE_RATE_LIMIT_BURST = 10
DEFAULT_AUTH_EXEMPT_PATHS = frozenset(
    {
        "/health",
        "/livez",
        "/readyz",
        "/metrics",
        "/openapi.json",
        "/docs",
        "/redoc",
    }
)
DEFAULT_ROUTE_SCOPES = {
    ("GET", "/models/loaded"): ("models:read",),
    ("POST", "/models/unload"): ("models:write",),
    ("POST", "/analyze"): ("analyze:write",),
    ("POST", "/pii/extract"): ("pii:read",),
    ("POST", "/pii/deidentify"): ("pii:write",),
}

_BOOLEAN_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_BOOLEAN_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
_SUPPORTED_JWT_ALGORITHMS = {"HS256", "RS256"}
_SHA256_DIGEST_INFO_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")

ErrorResponseFactory = Callable[..., Response]
CallNext = Callable[[Request], Awaitable[Response]]


@dataclass(frozen=True)
class APIKeyCredential:
    """Static API-key credential configured by hash."""

    key_id: str
    key_hash: str
    principal: str
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuthPrincipal:
    """Authenticated caller attached to request context."""

    subject: str
    credential_type: str
    scopes: tuple[str, ...] = ()
    key_id: Optional[str] = None
    claims: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ServiceAuthConfig:
    """Authentication settings for the REST service."""

    enabled: bool = False
    deny_by_default: bool = True
    api_keys: tuple[APIKeyCredential, ...] = ()
    jwks: tuple[Mapping[str, Any], ...] = ()
    jwt_issuer: Optional[str] = None
    jwt_audiences: tuple[str, ...] = ()
    jwt_leeway_seconds: float = 0.0
    route_scopes: Mapping[tuple[str, str], tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_ROUTE_SCOPES)
    )
    exempt_paths: frozenset[str] = DEFAULT_AUTH_EXEMPT_PATHS
    failure_rate_limit_rps: float = DEFAULT_AUTH_FAILURE_RATE_LIMIT_RPS
    failure_rate_limit_burst: int = DEFAULT_AUTH_FAILURE_RATE_LIMIT_BURST
    failure_rate_limit_key_by: str = "peer"

    @property
    def failure_rate_limit_enabled(self) -> bool:
        """Return whether failed auth attempts are rate limited."""
        return self.failure_rate_limit_rps > 0 and self.failure_rate_limit_burst > 0


class AuthError(ValueError):
    """Authentication failure safe to expose through the API envelope."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        challenge: str,
    ) -> None:
        self.code = code
        self.message = message
        self.challenge = challenge
        super().__init__(message)


def hash_api_key(api_key: str) -> str:
    """Return the supported SHA-256 hash representation for an API key."""
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def parse_service_auth_config() -> ServiceAuthConfig:
    """Read authentication settings from the current process environment."""
    enabled = parse_bool(
        os.getenv(SERVICE_AUTH_ENABLED_ENV_VAR),
        env_var=SERVICE_AUTH_ENABLED_ENV_VAR,
        default=False,
    )
    deny_by_default = parse_bool(
        os.getenv(SERVICE_AUTH_DENY_BY_DEFAULT_ENV_VAR),
        env_var=SERVICE_AUTH_DENY_BY_DEFAULT_ENV_VAR,
        default=True,
    )
    route_scopes = parse_route_scopes(os.getenv(SERVICE_AUTH_ROUTE_SCOPES_ENV_VAR))
    failure_key_by = parse_failure_rate_limit_key(
        os.getenv(SERVICE_AUTH_FAILURE_KEY_ENV_VAR)
    )

    if not enabled:
        return ServiceAuthConfig(
            enabled=False,
            deny_by_default=deny_by_default,
            route_scopes=route_scopes,
            failure_rate_limit_key_by=failure_key_by,
        )

    return ServiceAuthConfig(
        enabled=True,
        deny_by_default=deny_by_default,
        api_keys=parse_api_key_credentials(os.getenv(SERVICE_AUTH_API_KEYS_ENV_VAR)),
        jwks=parse_jwks(
            os.getenv(SERVICE_AUTH_JWKS_ENV_VAR),
            os.getenv(SERVICE_AUTH_JWKS_FILE_ENV_VAR),
        ),
        jwt_issuer=_optional_string(os.getenv(SERVICE_AUTH_JWT_ISSUER_ENV_VAR)),
        jwt_audiences=parse_audiences(os.getenv(SERVICE_AUTH_JWT_AUDIENCE_ENV_VAR)),
        jwt_leeway_seconds=parse_non_negative_float(
            os.getenv(SERVICE_AUTH_JWT_LEEWAY_ENV_VAR),
            env_var=SERVICE_AUTH_JWT_LEEWAY_ENV_VAR,
            default=0.0,
        ),
        route_scopes=route_scopes,
        failure_rate_limit_rps=parse_non_negative_float(
            os.getenv(SERVICE_AUTH_FAILURE_RPS_ENV_VAR),
            env_var=SERVICE_AUTH_FAILURE_RPS_ENV_VAR,
            default=DEFAULT_AUTH_FAILURE_RATE_LIMIT_RPS,
        ),
        failure_rate_limit_burst=parse_non_negative_int(
            os.getenv(SERVICE_AUTH_FAILURE_BURST_ENV_VAR),
            env_var=SERVICE_AUTH_FAILURE_BURST_ENV_VAR,
            default=DEFAULT_AUTH_FAILURE_RATE_LIMIT_BURST,
        ),
        failure_rate_limit_key_by=failure_key_by,
    )


def parse_bool(raw_value: Optional[str], *, env_var: str, default: bool) -> bool:
    """Parse a service boolean environment value."""
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if not normalized:
        return default
    if normalized in _BOOLEAN_TRUE_VALUES:
        return True
    if normalized in _BOOLEAN_FALSE_VALUES:
        return False
    raise ValueError(f"{env_var} must be a boolean value like 'true' or 'false'")


def parse_non_negative_float(
    raw_value: Optional[str],
    *,
    env_var: str,
    default: float,
) -> float:
    """Parse a non-negative float environment value."""
    if raw_value is None or not raw_value.strip():
        return default
    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be a non-negative number") from exc
    if parsed < 0:
        raise ValueError(f"{env_var} must be greater than or equal to 0")
    return parsed


def parse_non_negative_int(
    raw_value: Optional[str],
    *,
    env_var: str,
    default: int,
) -> int:
    """Parse a non-negative integer environment value."""
    if raw_value is None or not raw_value.strip():
        return default
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{env_var} must be greater than or equal to 0")
    return parsed


def parse_api_key_credentials(raw_value: Optional[str]) -> tuple[APIKeyCredential, ...]:
    """Parse hashed static API-key credentials from JSON."""
    if raw_value is None or not raw_value.strip():
        return ()

    payload = _load_json(raw_value, env_var=SERVICE_AUTH_API_KEYS_ENV_VAR)
    if isinstance(payload, Mapping):
        credentials = payload.get("keys", ())
    else:
        credentials = payload
    if not isinstance(credentials, Sequence) or isinstance(credentials, (str, bytes)):
        raise ValueError(f"{SERVICE_AUTH_API_KEYS_ENV_VAR} must be a JSON list")

    parsed: list[APIKeyCredential] = []
    for index, item in enumerate(credentials):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"{SERVICE_AUTH_API_KEYS_ENV_VAR}[{index}] must be a JSON object"
            )
        key_hash = _normalize_api_key_hash(
            item.get("key_hash") or item.get("sha256") or item.get("hash")
        )
        key_id = str(item.get("id") or item.get("key_id") or f"key-{index + 1}")
        principal = str(item.get("principal") or item.get("subject") or key_id)
        parsed.append(
            APIKeyCredential(
                key_id=key_id,
                key_hash=key_hash,
                principal=principal,
                scopes=_normalize_scopes(item.get("scopes") or item.get("scope")),
            )
        )
    return tuple(parsed)


def parse_jwks(
    inline_jwks: Optional[str],
    jwks_file: Optional[str],
) -> tuple[Mapping[str, Any], ...]:
    """Parse JWT verification keys from inline JSON and optional JWKS file."""
    keys: list[Mapping[str, Any]] = []
    for raw_value, env_var in (
        (_read_optional_file(jwks_file), SERVICE_AUTH_JWKS_FILE_ENV_VAR),
        (inline_jwks, SERVICE_AUTH_JWKS_ENV_VAR),
    ):
        if raw_value is None or not raw_value.strip():
            continue
        payload = _load_json(raw_value, env_var=env_var)
        keys.extend(_extract_jwks_keys(payload, env_var=env_var))
    return tuple(keys)


def parse_audiences(raw_value: Optional[str]) -> tuple[str, ...]:
    """Parse comma-separated JWT audiences."""
    return _normalize_scopes(raw_value)


def parse_route_scopes(
    raw_value: Optional[str],
) -> Mapping[tuple[str, str], tuple[str, ...]]:
    """Parse route scope overrides from JSON."""
    route_scopes = dict(DEFAULT_ROUTE_SCOPES)
    if raw_value is None or not raw_value.strip():
        return route_scopes

    payload = _load_json(raw_value, env_var=SERVICE_AUTH_ROUTE_SCOPES_ENV_VAR)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{SERVICE_AUTH_ROUTE_SCOPES_ENV_VAR} must be a JSON object")

    for route, scopes in payload.items():
        method, path = _parse_route_key(str(route))
        route_scopes[(method, path)] = _normalize_scopes(scopes)
    return route_scopes


def parse_failure_rate_limit_key(raw_value: Optional[str]) -> str:
    """Parse the client identity source for auth failure rate limits."""
    if raw_value is None or not raw_value.strip():
        return "peer"

    normalized = raw_value.strip().lower()
    aliases = {
        "global": "global",
        "process": "global",
        "xff": "x-forwarded-for",
        "x-forwarded-for": "x-forwarded-for",
        "peer": "peer",
        "client": "peer",
        "remote": "peer",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"{SERVICE_AUTH_FAILURE_KEY_ENV_VAR} must be one of: global, "
            "x-forwarded-for, or peer"
        ) from exc


def authenticate_api_key(
    api_key: str,
    credentials: Sequence[APIKeyCredential],
) -> AuthPrincipal:
    """Authenticate one raw API key against configured hashes."""
    key_hash = hash_api_key(api_key)
    for credential in credentials:
        if hmac.compare_digest(key_hash, credential.key_hash):
            return AuthPrincipal(
                subject=credential.principal,
                credential_type="api_key",
                scopes=credential.scopes,
                key_id=credential.key_id,
            )
    raise invalid_credentials()


def authenticate_jwt(
    token: str,
    config: ServiceAuthConfig,
    *,
    now: Optional[float] = None,
) -> AuthPrincipal:
    """Authenticate a JWT bearer token against configured JWKS keys."""
    if not config.jwks:
        raise invalid_credentials()

    header, payload, signature, signing_input = _decode_jwt(token)
    alg = header.get("alg")
    if alg not in _SUPPORTED_JWT_ALGORITHMS:
        raise invalid_credentials()

    jwk = _select_jwk(config.jwks, header, alg=str(alg))
    if alg == "HS256":
        _verify_hs256(jwk, signing_input, signature)
    elif alg == "RS256":
        _verify_rs256(jwk, signing_input, signature)
    else:
        raise invalid_credentials()

    _validate_jwt_claims(payload, config, now=time.time() if now is None else now)
    subject = payload.get("sub") or payload.get("client_id") or "jwt"
    return AuthPrincipal(
        subject=str(subject),
        credential_type="jwt",
        scopes=_jwt_scopes(payload),
        key_id=_optional_string(header.get("kid")),
        claims=payload,
    )


class ServiceAuth:
    """FastAPI middleware dispatcher for service authentication."""

    def __init__(
        self,
        config: ServiceAuthConfig,
        *,
        error_response: ErrorResponseFactory,
    ) -> None:
        self.config = config
        self._error_response = error_response
        self._failure_limiter = TokenBucketRateLimiter(
            requests_per_second=config.failure_rate_limit_rps,
            burst_size=config.failure_rate_limit_burst,
        )

    @property
    def enabled(self) -> bool:
        """Return whether authentication is active."""
        return self.config.enabled

    async def dispatch(self, request: Request, call_next: CallNext) -> Response:
        """Authenticate and authorize one HTTP request."""
        if not self.enabled:
            return await call_next(request)

        required_scopes = self._required_scopes(request)
        if required_scopes is None:
            return await call_next(request)

        try:
            principal = self._authenticate_request(request)
        except AuthError as exc:
            return await self._authentication_failure(request, exc)

        request.state.auth_principal = principal
        request.scope["openmed.auth"] = principal
        if not scopes_satisfy(principal.scopes, required_scopes):
            return self._forbidden(required_scopes)
        return await call_next(request)

    def _required_scopes(self, request: Request) -> Optional[tuple[str, ...]]:
        path = request.url.path
        if path in self.config.exempt_paths:
            return None
        route_key = (request.method.upper(), path)
        if route_key in self.config.route_scopes:
            return self.config.route_scopes[route_key]
        if self.config.deny_by_default:
            return ()
        return None

    def _authenticate_request(self, request: Request) -> AuthPrincipal:
        api_key = request.headers.get("x-api-key")
        if api_key:
            return authenticate_api_key(api_key, self.config.api_keys)

        authorization = request.headers.get("authorization")
        if not authorization:
            raise missing_credentials()

        scheme, _, value = authorization.strip().partition(" ")
        if not value:
            raise missing_credentials()
        scheme = scheme.lower()
        if scheme == "bearer":
            return authenticate_jwt(value.strip(), self.config)
        if scheme in {"apikey", "api-key"}:
            return authenticate_api_key(value.strip(), self.config.api_keys)
        raise invalid_credentials()

    async def _authentication_failure(
        self,
        request: Request,
        exc: AuthError,
    ) -> Response:
        if self.config.failure_rate_limit_enabled:
            key = client_identity(request, self.config.failure_rate_limit_key_by)
            decision = await self._failure_limiter.consume(key)
            if not decision.allowed:
                response = self._error_response(
                    429,
                    "auth_rate_limited",
                    "Too many failed authentication attempts",
                    details={"retry_after_seconds": decision.retry_after_seconds},
                )
                response.headers["Retry-After"] = format_retry_after(
                    decision.retry_after_seconds
                )
                return response

        response = self._error_response(
            401,
            exc.code,
            exc.message,
            details=None,
        )
        response.headers["WWW-Authenticate"] = exc.challenge
        return response

    def _forbidden(self, required_scopes: Sequence[str]) -> Response:
        challenge = 'Bearer error="insufficient_scope"'
        if required_scopes:
            challenge += f', scope="{" ".join(required_scopes)}"'
        response = self._error_response(
            403,
            "forbidden",
            "Authenticated principal does not have the required scope",
            details={"required_scopes": list(required_scopes)},
        )
        response.headers["WWW-Authenticate"] = challenge
        return response


def missing_credentials() -> AuthError:
    """Return a missing-credentials auth error."""
    return AuthError(
        code="authentication_required",
        message="Authentication credentials were not provided",
        challenge="Bearer, ApiKey",
    )


def invalid_credentials() -> AuthError:
    """Return an invalid-credentials auth error."""
    return AuthError(
        code="invalid_credentials",
        message="Authentication credentials are invalid",
        challenge='Bearer error="invalid_token", ApiKey',
    )


def scopes_satisfy(granted: Sequence[str], required: Sequence[str]) -> bool:
    """Return whether granted scopes satisfy every required scope."""
    if not required:
        return True
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


def _normalize_api_key_hash(raw_value: Any) -> str:
    if raw_value is None:
        raise ValueError("API key credential requires key_hash")
    value = str(raw_value).strip().lower()
    if value.startswith("sha256:"):
        digest = value.removeprefix("sha256:")
    else:
        digest = value
        value = f"sha256:{digest}"
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("API key hash must be a SHA-256 hex digest")
    return value


def _normalize_scopes(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        items = raw_value.replace(",", " ").split()
    elif isinstance(raw_value, Sequence):
        items = [str(item) for item in raw_value]
    else:
        raise ValueError("Scopes must be a string or list")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        scope = item.strip()
        if not scope or scope in seen:
            continue
        normalized.append(scope)
        seen.add(scope)
    return tuple(normalized)


def _optional_string(raw_value: Any) -> Optional[str]:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    return value or None


def _load_json(raw_value: str, *, env_var: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{env_var} must contain valid JSON") from exc


def _read_optional_file(raw_path: Optional[str]) -> Optional[str]:
    if raw_path is None or not raw_path.strip():
        return None
    return Path(raw_path).read_text(encoding="utf-8")


def _extract_jwks_keys(payload: Any, *, env_var: str) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        keys = payload.get("keys", ())
    else:
        keys = payload
    if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)):
        raise ValueError(f"{env_var} must be a JWKS object or JSON key list")

    parsed: list[Mapping[str, Any]] = []
    for index, key in enumerate(keys):
        if not isinstance(key, Mapping):
            raise ValueError(f"{env_var} key {index} must be a JSON object")
        parsed.append(dict(key))
    return parsed


def _parse_route_key(route: str) -> tuple[str, str]:
    method, separator, path = route.strip().partition(" ")
    if not separator:
        raise ValueError(
            f"{SERVICE_AUTH_ROUTE_SCOPES_ENV_VAR} route keys must look like "
            "'POST /analyze'"
        )
    method = method.upper()
    path = path.strip()
    if not method or not path.startswith("/"):
        raise ValueError(
            f"{SERVICE_AUTH_ROUTE_SCOPES_ENV_VAR} route keys must include "
            "an HTTP method and absolute path"
        )
    return method, path


def _decode_jwt(
    token: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], bytes, bytes]:
    parts = token.split(".")
    if len(parts) != 3:
        raise invalid_credentials()

    header_segment, payload_segment, signature_segment = parts
    header = _decode_json_segment(header_segment)
    payload = _decode_json_segment(payload_segment)
    signature = _b64url_decode(signature_segment)
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    return header, payload, signature, signing_input


def _decode_json_segment(segment: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(_b64url_decode(segment).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise invalid_credentials() from None
    if not isinstance(payload, Mapping):
        raise invalid_credentials()
    return payload


def _b64url_decode(value: str) -> bytes:
    try:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode((value + padding).encode("ascii"))
    except (binascii.Error, UnicodeEncodeError):
        raise invalid_credentials() from None


def _select_jwk(
    keys: Sequence[Mapping[str, Any]],
    header: Mapping[str, Any],
    *,
    alg: str,
) -> Mapping[str, Any]:
    kid = _optional_string(header.get("kid"))
    candidates = []
    for key in keys:
        if kid is not None and _optional_string(key.get("kid")) != kid:
            continue
        if _jwk_matches_algorithm(key, alg):
            candidates.append(key)

    if len(candidates) != 1:
        raise invalid_credentials()
    return candidates[0]


def _jwk_matches_algorithm(key: Mapping[str, Any], alg: str) -> bool:
    key_alg = _optional_string(key.get("alg"))
    if key_alg is not None and key_alg != alg:
        return False
    kty = _optional_string(key.get("kty"))
    if alg == "HS256":
        return kty == "oct"
    if alg == "RS256":
        return kty == "RSA"
    return False


def _verify_hs256(
    jwk: Mapping[str, Any], signing_input: bytes, signature: bytes
) -> None:
    encoded_secret = _optional_string(jwk.get("k"))
    if encoded_secret is None:
        raise invalid_credentials()
    secret = _b64url_decode(encoded_secret)
    expected = hmac.new(secret, signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise invalid_credentials()


def _verify_rs256(
    jwk: Mapping[str, Any], signing_input: bytes, signature: bytes
) -> None:
    modulus = _jwk_int(jwk, "n")
    exponent = _jwk_int(jwk, "e")
    key_size = (modulus.bit_length() + 7) // 8
    if key_size <= 0 or len(signature) != key_size:
        raise invalid_credentials()

    signature_int = int.from_bytes(signature, "big")
    encoded = pow(signature_int, exponent, modulus).to_bytes(key_size, "big")
    digest = hashlib.sha256(signing_input).digest()
    if not _verify_pkcs1v15_sha256(encoded, digest):
        raise invalid_credentials()


def _jwk_int(jwk: Mapping[str, Any], name: str) -> int:
    encoded = _optional_string(jwk.get(name))
    if encoded is None:
        raise invalid_credentials()
    value = int.from_bytes(_b64url_decode(encoded), "big")
    if value <= 0:
        raise invalid_credentials()
    return value


def _verify_pkcs1v15_sha256(encoded: bytes, digest: bytes) -> bool:
    if not encoded.startswith(b"\x00\x01"):
        return False
    separator_index = encoded.find(b"\x00", 2)
    if separator_index < 10:
        return False
    padding = encoded[2:separator_index]
    if not padding or any(byte != 0xFF for byte in padding):
        return False
    expected = _SHA256_DIGEST_INFO_PREFIX + digest
    return hmac.compare_digest(encoded[separator_index + 1 :], expected)


def _validate_jwt_claims(
    payload: Mapping[str, Any],
    config: ServiceAuthConfig,
    *,
    now: float,
) -> None:
    exp = _numeric_claim(payload, "exp", required=True)
    if exp is None:
        raise invalid_credentials()
    if now > exp + config.jwt_leeway_seconds:
        raise invalid_credentials()

    nbf = _numeric_claim(payload, "nbf", required=False)
    if nbf is not None and now + config.jwt_leeway_seconds < nbf:
        raise invalid_credentials()

    iat = _numeric_claim(payload, "iat", required=False)
    if iat is not None and now + config.jwt_leeway_seconds < iat:
        raise invalid_credentials()

    if config.jwt_issuer is not None and payload.get("iss") != config.jwt_issuer:
        raise invalid_credentials()

    if config.jwt_audiences:
        token_audiences = _claim_audiences(payload.get("aud"))
        if not token_audiences.intersection(config.jwt_audiences):
            raise invalid_credentials()


def _numeric_claim(
    payload: Mapping[str, Any],
    name: str,
    *,
    required: bool,
) -> Optional[float]:
    value = payload.get(name)
    if value is None:
        if required:
            raise invalid_credentials()
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise invalid_credentials() from None


def _claim_audiences(raw_value: Any) -> set[str]:
    if raw_value is None:
        return set()
    if isinstance(raw_value, str):
        return {raw_value}
    if isinstance(raw_value, Sequence):
        return {str(item) for item in raw_value}
    return set()


def _jwt_scopes(payload: Mapping[str, Any]) -> tuple[str, ...]:
    for claim_name in ("scope", "scp", "scopes"):
        value = payload.get(claim_name)
        if value:
            return _normalize_scopes(value)
    return ()
