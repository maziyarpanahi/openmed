"""Authentication, per-RPC authorization and resource budgets for gRPC.

Every OpenMed RPC passes through :class:`GrpcSecurityInterceptor` before the
servicer runs. The interceptor authenticates an API key or JWT from call
metadata, or a verified mutual-TLS peer certificate, into the same
:class:`~openmed.service.auth.AuthPrincipal` the REST service uses. It then
checks the scopes of the REST route each RPC mirrors, takes a rate-limit token
and a concurrency slot, and bounds the call by the smaller of the client
deadline and the server execution budget.

Every refusal maps to a canonical gRPC status and carries a stable, value-free
reason code in the ``openmed-reason`` trailing metadata. Status details are
fixed strings: metadata, tokens, request text, chunks, certificate contents and
rejected values are never reflected.
"""

from __future__ import annotations

import math
import os
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, NoReturn, Optional
from urllib.parse import unquote

import grpc
from cryptography import x509
from cryptography.x509.verification import Store

from openmed.guard.operational_limits import OperationalLimits

from .auth import (
    ROUTE_POLICIES,
    AuthError,
    AuthPrincipal,
    ServiceAuthConfig,
    authenticate_api_key,
    authenticate_jwt,
    parse_service_auth_config,
    scopes_satisfy,
)
from .grpc_security import (
    CATEGORY_LISTENER,
    GRPC_REASON_AUTH_REQUIRED,
    GrpcListener,
    GrpcStartupError,
)
from .limits import DEFAULT_MAX_TEXT_LENGTH, get_max_text_length
from .mtls import (
    SERVICE_MTLS_PRINCIPALS_ENV_VAR,
    MTLSPrincipalMapping,
    parse_principal_mappings,
    principal_for_mtls_identity,
    verify_client_certificate_chain,
)
from .proto.generated import openmed_pb2
from .request_limits import request_limits_from_env
from .runtime import (
    ServiceRuntime,
    ServiceThrottleConfig,
    parse_service_throttle_config,
)
from .throttle import (
    GLOBAL_THROTTLE_KEY,
    BlockingConcurrencyGate,
    BlockingConcurrencyPermit,
    TokenBucketRateLimiter,
)

GRPC_SERVICE_NAME = "openmed.service.v1.OpenMedService"
OPENMED_REASON_METADATA_KEY = "openmed-reason"

CATEGORY_UNAUTHENTICATED = "unauthenticated"
CATEGORY_PERMISSION_DENIED = "permission_denied"
CATEGORY_INVALID_ARGUMENT = "invalid_argument"
CATEGORY_RESOURCE_EXHAUSTED = "resource_exhausted"
CATEGORY_DEADLINE_EXCEEDED = "deadline_exceeded"
CATEGORY_INTERNAL = "internal"

REASON_AUTHENTICATION_REQUIRED = "authentication_required"
REASON_INVALID_CREDENTIALS = "invalid_credentials"
REASON_CREDENTIALS_MALFORMED = "credentials_malformed"
REASON_INSUFFICIENT_SCOPE = "insufficient_scope"
REASON_AUTH_RATE_LIMITED = "auth_rate_limited"
REASON_RATE_LIMITED = "rate_limited"
REASON_CONCURRENCY_LIMITED = "concurrency_limited"
REASON_TEXT_TOO_LONG = "text_too_long"
REASON_STREAM_TOO_MANY_CHUNKS = "stream_too_many_chunks"
REASON_STREAM_CHUNK_TOO_LARGE = "stream_chunk_too_large"
REASON_STREAM_BUFFER_TOO_LARGE = "stream_buffer_too_large"
REASON_INVALID_REQUEST = "invalid_request"
REASON_MODEL_NOT_SERVED = "model_not_served"
REASON_DEADLINE_EXCEEDED = "deadline_exceeded"
REASON_INTERNAL_ERROR = "internal_error"

DEFAULT_GRPC_MAX_SEND_MESSAGE_BYTES = 64 * 1024 * 1024
DEFAULT_GRPC_MAX_STREAM_CHUNKS = 4096
DEFAULT_GRPC_MAX_STREAM_CHUNK_BYTES = 128 * 1024
# Matches the REST ``/pii/deidentify/stream`` carry-over buffer.
DEFAULT_GRPC_STREAM_BUFFER_CHARS = 4096
DEFAULT_GRPC_MAX_STREAM_BUFFER_CHARS = 64 * 1024
DEFAULT_GRPC_MAX_RPC_SECONDS = 300.0
MAX_CREDENTIAL_METADATA_CHARS = 8192

_GRPC_MAX_OPTION_VALUE = 2**31 - 1
_API_KEY_METADATA = "x-api-key"
_AUTHORIZATION_METADATA = "authorization"
_FORWARDED_FOR_METADATA = "x-forwarded-for"
_CREDENTIAL_METADATA = frozenset({_API_KEY_METADATA, _AUTHORIZATION_METADATA})
_MTLS_PEM_PROPERTY = "x509_pem_cert"
_ADMISSION_CLASSES = frozenset({"model", "heavy"})


@dataclass(frozen=True)
class GrpcFailure:
    """Canonical category, gRPC status and fixed message for one reason code.

    Attributes:
        reason: Stable reason code sent in ``openmed-reason`` trailing metadata.
        category: Transport-neutral failure category.
        status: Canonical gRPC status code.
        message: Fixed, value-free status details.
    """

    reason: str
    category: str
    status: grpc.StatusCode
    message: str


def _failures(*items: GrpcFailure) -> Mapping[str, GrpcFailure]:
    return MappingProxyType({item.reason: item for item in items})


GRPC_FAILURES: Mapping[str, GrpcFailure] = _failures(
    GrpcFailure(
        REASON_AUTHENTICATION_REQUIRED,
        CATEGORY_UNAUTHENTICATED,
        grpc.StatusCode.UNAUTHENTICATED,
        "Authentication credentials were not provided",
    ),
    GrpcFailure(
        REASON_INVALID_CREDENTIALS,
        CATEGORY_UNAUTHENTICATED,
        grpc.StatusCode.UNAUTHENTICATED,
        "Authentication credentials are invalid",
    ),
    GrpcFailure(
        REASON_CREDENTIALS_MALFORMED,
        CATEGORY_UNAUTHENTICATED,
        grpc.StatusCode.UNAUTHENTICATED,
        "Authentication metadata is malformed",
    ),
    GrpcFailure(
        REASON_INSUFFICIENT_SCOPE,
        CATEGORY_PERMISSION_DENIED,
        grpc.StatusCode.PERMISSION_DENIED,
        "Authenticated principal does not have the required scope",
    ),
    GrpcFailure(
        REASON_AUTH_RATE_LIMITED,
        CATEGORY_RESOURCE_EXHAUSTED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        "Too many failed authentication attempts",
    ),
    GrpcFailure(
        REASON_RATE_LIMITED,
        CATEGORY_RESOURCE_EXHAUSTED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        "Request rate limit exceeded",
    ),
    GrpcFailure(
        REASON_CONCURRENCY_LIMITED,
        CATEGORY_RESOURCE_EXHAUSTED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        "Service is busy; retry later",
    ),
    GrpcFailure(
        REASON_TEXT_TOO_LONG,
        CATEGORY_RESOURCE_EXHAUSTED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        "Request text exceeds the server limit",
    ),
    GrpcFailure(
        REASON_STREAM_TOO_MANY_CHUNKS,
        CATEGORY_RESOURCE_EXHAUSTED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        "Stream chunk count exceeds the server limit",
    ),
    GrpcFailure(
        REASON_STREAM_CHUNK_TOO_LARGE,
        CATEGORY_RESOURCE_EXHAUSTED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        "Stream chunk exceeds the server limit",
    ),
    GrpcFailure(
        REASON_STREAM_BUFFER_TOO_LARGE,
        CATEGORY_RESOURCE_EXHAUSTED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        "Stream buffer exceeds the server limit",
    ),
    GrpcFailure(
        REASON_INVALID_REQUEST,
        CATEGORY_INVALID_ARGUMENT,
        grpc.StatusCode.INVALID_ARGUMENT,
        "Invalid request",
    ),
    GrpcFailure(
        REASON_MODEL_NOT_SERVED,
        CATEGORY_INVALID_ARGUMENT,
        grpc.StatusCode.INVALID_ARGUMENT,
        "model_not_served",
    ),
    GrpcFailure(
        REASON_DEADLINE_EXCEEDED,
        CATEGORY_DEADLINE_EXCEEDED,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        "Request exceeded its deadline",
    ),
    GrpcFailure(
        REASON_INTERNAL_ERROR,
        CATEGORY_INTERNAL,
        grpc.StatusCode.INTERNAL,
        "Internal server error",
    ),
)
GRPC_RPC_REASON_CODES = frozenset(GRPC_FAILURES)

#: The REST route each RPC mirrors. Scopes and admission come from that route's
#: policy, including any ``OPENMED_SERVICE_AUTH_ROUTE_SCOPES`` override.
GRPC_RPC_ROUTES: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        f"/{GRPC_SERVICE_NAME}/Analyze": ("POST", "/analyze"),
        f"/{GRPC_SERVICE_NAME}/Extract": ("POST", "/pii/extract"),
        f"/{GRPC_SERVICE_NAME}/Deidentify": ("POST", "/pii/deidentify"),
        f"/{GRPC_SERVICE_NAME}/StreamDeidentify": ("POST", "/pii/deidentify/stream"),
    }
)


class GrpcPolicyError(Exception):
    """Value-free RPC refusal carrying a stable reason code.

    Attributes:
        reason: A code from :data:`GRPC_RPC_REASON_CODES`; unknown codes are
            replaced by ``internal_error`` so free text never reaches a client.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason if reason in GRPC_FAILURES else REASON_INTERNAL_ERROR
        super().__init__(self.reason)

    @property
    def failure(self) -> GrpcFailure:
        """Return the canonical status mapping for this refusal."""
        return GRPC_FAILURES[self.reason]


def abort_rpc(context: grpc.ServicerContext, reason: str) -> NoReturn:
    """Fail the RPC with its canonical status and an ``openmed-reason`` trailer.

    Args:
        context: The servicer context of the RPC to fail.
        reason: A code from :data:`GRPC_RPC_REASON_CODES`.

    Raises:
        Exception: Always; ``context.abort`` raises to unwind the handler.
    """
    failure = GRPC_FAILURES.get(reason) or GRPC_FAILURES[REASON_INTERNAL_ERROR]
    context.set_trailing_metadata(((OPENMED_REASON_METADATA_KEY, failure.reason),))
    context.abort(failure.status, failure.message)
    raise RuntimeError(REASON_INTERNAL_ERROR)  # pragma: no cover - abort raises


@dataclass(frozen=True, kw_only=True)
class GrpcLimits:
    """Server-side size, stream and time budgets for the gRPC service.

    Attributes:
        max_receive_message_bytes: Largest request message grpcio accepts.
            Larger messages are refused by the transport before they reach
            the application. Defaults to the REST request-byte limit.
        max_send_message_bytes: Largest response message grpcio sends.
        max_text_chars: Largest request text, and largest total of all stream
            chunks, in characters. Defaults to ``OPENMED_SERVICE_MAX_TEXT_LENGTH``.
        max_stream_chunks: Most chunks one ``StreamDeidentify`` call may carry
            or be split into.
        max_stream_chunk_bytes: Largest UTF-8 size of one stream chunk.
        default_stream_buffer_chars: Carry-over buffer used when the caller
            leaves ``max_buffer`` unset.
        max_stream_buffer_chars: Largest carry-over buffer a caller may request.
        max_rpc_seconds: Server execution budget per RPC. The effective
            deadline is the smaller of this and the client deadline.
    """

    max_receive_message_bytes: int = OperationalLimits().max_request_bytes
    max_send_message_bytes: int = DEFAULT_GRPC_MAX_SEND_MESSAGE_BYTES
    max_text_chars: int = DEFAULT_MAX_TEXT_LENGTH
    max_stream_chunks: int = DEFAULT_GRPC_MAX_STREAM_CHUNKS
    max_stream_chunk_bytes: int = DEFAULT_GRPC_MAX_STREAM_CHUNK_BYTES
    default_stream_buffer_chars: int = DEFAULT_GRPC_STREAM_BUFFER_CHARS
    max_stream_buffer_chars: int = DEFAULT_GRPC_MAX_STREAM_BUFFER_CHARS
    max_rpc_seconds: float = DEFAULT_GRPC_MAX_RPC_SECONDS

    def __post_init__(self) -> None:
        for name in (
            "max_receive_message_bytes",
            "max_send_message_bytes",
            "max_text_chars",
            "max_stream_chunks",
            "max_stream_chunk_bytes",
            "default_stream_buffer_chars",
            "max_stream_buffer_chars",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= _GRPC_MAX_OPTION_VALUE:
                raise ValueError(f"gRPC limit {name} must be a positive integer")
        if self.default_stream_buffer_chars > self.max_stream_buffer_chars:
            raise ValueError("gRPC default stream buffer exceeds its maximum")
        seconds = self.max_rpc_seconds
        if (
            type(seconds) not in {int, float}
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("gRPC limit max_rpc_seconds must be positive and finite")
        object.__setattr__(self, "max_rpc_seconds", float(seconds))

    @classmethod
    def from_env(cls, *, runtime: Optional[ServiceRuntime] = None) -> "GrpcLimits":
        """Build limits that share the REST service's configured budgets.

        Args:
            runtime: Service runtime whose ``config.timeout`` becomes the RPC
                execution budget, as it bounds REST streaming.

        Returns:
            Limits using ``OPENMED_SERVICE_MAX_REQUEST_BYTES`` and
            ``OPENMED_SERVICE_MAX_TEXT_LENGTH`` where they are set.
        """
        timeout = 0.0
        if runtime is not None:
            raw_timeout = getattr(runtime.config, "timeout", 0)
            if type(raw_timeout) in {int, float}:
                timeout = float(raw_timeout)
        return cls(
            max_receive_message_bytes=request_limits_from_env().max_request_bytes,
            max_text_chars=get_max_text_length(),
            max_rpc_seconds=(
                timeout
                if math.isfinite(timeout) and timeout > 0
                else DEFAULT_GRPC_MAX_RPC_SECONDS
            ),
        )

    @property
    def default_stream_chunk_chars(self) -> int:
        """Return the split size used when a caller leaves ``chunk_size`` unset.

        Four UTF-8 bytes per character is the worst case, so these chunks
        always fit within :attr:`max_stream_chunk_bytes`.
        """
        return max(1, self.max_stream_chunk_bytes // 4)

    def server_options(self) -> tuple[tuple[str, int], ...]:
        """Return grpcio server options enforcing the message-size limits."""
        return (
            ("grpc.max_receive_message_length", self.max_receive_message_bytes),
            ("grpc.max_send_message_length", self.max_send_message_bytes),
        )

    def check_text(self, text: str) -> None:
        """Refuse request text longer than :attr:`max_text_chars`.

        Raises:
            GrpcPolicyError: ``text_too_long``.
        """
        if len(text) > self.max_text_chars:
            raise GrpcPolicyError(REASON_TEXT_TOO_LONG)

    def stream_chunks(self, request: Any) -> tuple[str, ...]:
        """Return the bounded chunks of one ``DeidentifyStreamRequest``.

        Chunk count, per-chunk size and the aggregate text are checked before
        any chunk is copied or joined.

        Args:
            request: A ``DeidentifyStreamRequest`` message.

        Returns:
            Explicit ``chunks``, or ``request.request.text`` split by
            ``chunk_size``.

        Raises:
            GrpcPolicyError: ``stream_too_many_chunks``,
                ``stream_chunk_too_large`` or ``text_too_long``.
            ValueError: If ``chunk_size`` is negative.
        """
        if request.chunks:
            if len(request.chunks) > self.max_stream_chunks:
                raise GrpcPolicyError(REASON_STREAM_TOO_MANY_CHUNKS)
            total = 0
            for chunk in request.chunks:
                self._check_chunk(chunk)
                total += len(chunk)
                if total > self.max_text_chars:
                    raise GrpcPolicyError(REASON_TEXT_TOO_LONG)
            return tuple(str(chunk) for chunk in request.chunks)

        text = request.request.text
        self.check_text(text)
        chunk_size = int(request.chunk_size or 0)
        if chunk_size < 0:
            raise ValueError("Invalid chunk_size")
        if chunk_size == 0:
            chunk_size = self.default_stream_chunk_chars
        if chunk_size >= len(text):
            self._check_chunk(text)
            return (text,)
        if math.ceil(len(text) / chunk_size) > self.max_stream_chunks:
            raise GrpcPolicyError(REASON_STREAM_TOO_MANY_CHUNKS)
        chunks = tuple(
            text[start : start + chunk_size]
            for start in range(0, len(text), chunk_size)
        )
        for chunk in chunks:
            self._check_chunk(chunk)
        return chunks

    def stream_buffer(self, requested: int) -> int:
        """Return the server-capped carry-over buffer for a stream.

        Args:
            requested: The caller's ``max_buffer``; ``0`` selects the default.

        Returns:
            The buffer size to pass to the streaming de-identifier.

        Raises:
            GrpcPolicyError: ``stream_buffer_too_large`` above the cap.
            ValueError: If ``requested`` is negative.
        """
        value = int(requested or 0)
        if value < 0:
            raise ValueError("Invalid max_buffer")
        if value == 0:
            return self.default_stream_buffer_chars
        if value > self.max_stream_buffer_chars:
            raise GrpcPolicyError(REASON_STREAM_BUFFER_TOO_LARGE)
        return value

    def _check_chunk(self, chunk: str) -> None:
        limit = self.max_stream_chunk_bytes
        if len(chunk) > limit:
            raise GrpcPolicyError(REASON_STREAM_CHUNK_TOO_LARGE)
        if len(chunk) * 4 > limit and len(chunk.encode("utf-8")) > limit:
            raise GrpcPolicyError(REASON_STREAM_CHUNK_TOO_LARGE)


@dataclass(frozen=True)
class GrpcRpcPolicy:
    """Authorization and admission decision for one RPC method.

    Attributes:
        method: Full gRPC method name, such as
            ``/openmed.service.v1.OpenMedService/Analyze``.
        scopes: Scopes the authenticated principal must hold.
        requires_authentication: Whether authentication applies when enabled.
        admission: Whether the RPC takes rate and concurrency permits.
    """

    method: str
    scopes: tuple[str, ...]
    requires_authentication: bool = True
    admission: bool = True


def grpc_rpc_policies(auth: ServiceAuthConfig) -> dict[str, GrpcRpcPolicy]:
    """Return the per-RPC policy table derived from the REST route policies.

    Args:
        auth: Service authentication settings; their ``route_scopes`` carry any
            operator scope overrides, so REST and gRPC stay aligned.

    Returns:
        A policy for each RPC in :data:`GRPC_RPC_ROUTES`.
    """
    policies: dict[str, GrpcRpcPolicy] = {}
    for method, route in GRPC_RPC_ROUTES.items():
        route_policy = ROUTE_POLICIES[route]
        policies[method] = GrpcRpcPolicy(
            method=method,
            scopes=tuple(auth.route_scopes.get(route, route_policy.scopes)),
            requires_authentication=True,
            admission=route_policy.admission in _ADMISSION_CLASSES,
        )
    return policies


def validate_grpc_rpc_policies() -> None:
    """Fail closed when an ``OpenMedService`` RPC has no declared policy.

    Raises:
        ValueError: If the protobuf service declares an undeclared method.
    """
    service = openmed_pb2.DESCRIPTOR.services_by_name["OpenMedService"]
    for method in service.methods:
        name = f"/{service.full_name}/{method.name}"
        if name not in GRPC_RPC_ROUTES:
            raise ValueError("gRPC method has no declared security policy")


@dataclass(frozen=True, kw_only=True)
class GrpcServicePolicy:
    """Authentication, throttling and budget settings for the gRPC service.

    Attributes:
        auth: The REST authentication settings: API keys, JWKS, route scopes,
            deny-by-default and the failed-authentication rate limit.
        mtls_principals: Certificate identity mappings shared with REST mTLS.
        throttle: The REST rate and concurrency settings.
        limits: Message, stream and deadline budgets.
    """

    auth: ServiceAuthConfig = field(default_factory=ServiceAuthConfig)
    mtls_principals: tuple[MTLSPrincipalMapping, ...] = ()
    throttle: ServiceThrottleConfig = field(default_factory=ServiceThrottleConfig)
    limits: GrpcLimits = field(default_factory=GrpcLimits)

    @classmethod
    def from_env(
        cls, *, runtime: Optional[ServiceRuntime] = None
    ) -> "GrpcServicePolicy":
        """Read the gRPC policy from the same environment as the REST service.

        Args:
            runtime: Service runtime supplying throttle settings and the
                execution budget; the environment is read when omitted.

        Returns:
            The policy for :class:`GrpcSecurityInterceptor`.
        """
        return cls(
            auth=parse_service_auth_config(),
            mtls_principals=parse_principal_mappings(
                os.getenv(SERVICE_MTLS_PRINCIPALS_ENV_VAR)
            ),
            throttle=(
                runtime.throttle
                if runtime is not None
                else parse_service_throttle_config()
            ),
            limits=GrpcLimits.from_env(runtime=runtime),
        )


def require_authenticated_listener(
    listener: GrpcListener, policy: GrpcServicePolicy
) -> None:
    """Refuse anonymous RPCs on a listener that other hosts can reach.

    Anonymous calls stay available on loopback listeners and under the
    development-only plaintext override. Any other listener needs service
    authentication or a mutual-TLS listener whose client CAs are known.

    Args:
        listener: The validated listener decision.
        policy: The policy the server will enforce.

    Raises:
        GrpcStartupError: ``grpc_auth_required`` before anything is served.
    """
    if listener.loopback or listener.insecure_dev_override or policy.auth.enabled:
        return
    if _listener_client_cas(listener):
        return
    raise GrpcStartupError(GRPC_REASON_AUTH_REQUIRED, category=CATEGORY_LISTENER)


def listener_client_ca_certificates(
    listener: Optional[GrpcListener],
) -> tuple[x509.Certificate, ...]:
    """Return client CA certificates for a mutual-TLS listener, else ``()``."""
    if listener is None:
        return ()
    return _listener_client_cas(listener)


def grpc_peer_host(peer: Optional[str]) -> Optional[str]:
    """Return the host part of a grpcio ``context.peer()`` string.

    Args:
        peer: A value such as ``ipv4:127.0.0.1:5000`` or ``ipv6:[::1]:5000``.

    Returns:
        The host, ``"local"`` for Unix sockets, or ``None`` when unknown.
    """
    if not isinstance(peer, str) or ":" not in peer:
        return None
    kind, _, rest = peer.partition(":")
    if kind in {"unix", "unix-abstract"}:
        return "local"
    if kind not in {"ipv4", "ipv6"}:
        return None
    host = unquote(rest).rpartition(":")[0]
    host = host.removeprefix("[").removesuffix("]")
    return host or None


class GrpcSecurityInterceptor(grpc.ServerInterceptor):
    """Enforce authentication, scopes, admission and deadlines for each RPC.

    Checks run in order: authentication, authorization, rate limit,
    concurrency, deadline. All of them complete before the servicer starts,
    so a refused call never reaches model work. Permits are released exactly
    once, when the RPC's work settles: after a unary handler returns or
    raises, and when a stream is exhausted, fails, is cancelled, or passes its
    deadline. A stream stops at the next fragment boundary after it is
    cancelled; a unary model call that is already running is not pre-empted.
    """

    def __init__(
        self,
        policy: Optional[GrpcServicePolicy] = None,
        *,
        client_ca_certificates: Sequence[x509.Certificate] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the interceptor.

        Args:
            policy: The policy to enforce; defaults to an anonymous loopback
                policy with default budgets.
            client_ca_certificates: CAs trusted for mutual-TLS client
                certificates. Supplying them enables certificate principals.
            clock: Monotonic clock for rate limits and deadlines.
        """
        self.policy = policy if policy is not None else GrpcServicePolicy()
        self._clock = clock
        certificates = list(client_ca_certificates)
        self._mtls_store: Optional[Store] = (
            Store(certificates) if certificates else None
        )
        self._rpc_policies = grpc_rpc_policies(self.policy.auth)
        throttle = self.policy.throttle
        self.rate_limiter = TokenBucketRateLimiter(
            requests_per_second=throttle.rate_limit_rps,
            burst_size=throttle.rate_limit_burst,
            clock=clock,
        )
        self.concurrency_gate = BlockingConcurrencyGate(
            max_concurrency=throttle.max_concurrency,
            wait_seconds=throttle.concurrency_wait_seconds,
        )
        auth = self.policy.auth
        self._failure_limiter = TokenBucketRateLimiter(
            requests_per_second=auth.failure_rate_limit_rps if auth.enabled else 0.0,
            burst_size=auth.failure_rate_limit_burst if auth.enabled else 0,
            clock=clock,
        )

    @property
    def authentication_enabled(self) -> bool:
        """Return whether RPCs must carry credentials or a client certificate."""
        return self.policy.auth.enabled or self._mtls_store is not None

    def rpc_policy(self, method: str) -> GrpcRpcPolicy:
        """Return the policy for a full gRPC method name.

        Methods outside :data:`GRPC_RPC_ROUTES` follow ``deny_by_default``:
        they need an authenticated principal (with no particular scope) unless
        deny-by-default is switched off.
        """
        policy = self._rpc_policies.get(method)
        if policy is not None:
            return policy
        return GrpcRpcPolicy(
            method=method,
            scopes=(),
            requires_authentication=self.policy.auth.deny_by_default,
            admission=False,
        )

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], Optional[Any]],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> Optional[Any]:
        """Wrap the next handler with this interceptor's policy checks."""
        handler = continuation(handler_call_details)
        if handler is None:
            return None
        return self._wrap(handler, self.rpc_policy(handler_call_details.method))

    def authenticate(self, context: grpc.ServicerContext) -> Optional[AuthPrincipal]:
        """Authenticate one RPC into the shared service principal.

        When service authentication is enabled, metadata credentials are
        checked first, as REST checks headers: ``x-api-key``, or
        ``authorization`` with a ``Bearer`` JWT or an ``ApiKey`` value.
        Exactly one credential entry is accepted; duplicates, both kinds at
        once, ``-bin`` variants and non-printable values are malformed.
        Without metadata credentials, a verified mutual-TLS peer certificate
        is mapped through the shared mTLS principal mappings.

        Args:
            context: The servicer context of the RPC.

        Returns:
            The principal, or ``None`` when authentication is not enabled.

        Raises:
            GrpcPolicyError: ``authentication_required``,
                ``invalid_credentials`` or ``credentials_malformed``.
        """
        if not self.authentication_enabled:
            return None
        auth = self.policy.auth
        api_key: Optional[str] = None
        authorization: Optional[str] = None
        if auth.enabled:
            api_key, authorization = _credential_metadata(context.invocation_metadata())
        try:
            if api_key is not None:
                return authenticate_api_key(api_key, auth.api_keys)
            if authorization is not None:
                scheme, _, value = authorization.partition(" ")
                value = value.strip()
                if not value:
                    raise GrpcPolicyError(REASON_CREDENTIALS_MALFORMED)
                scheme = scheme.lower()
                if scheme == "bearer":
                    return authenticate_jwt(value, auth)
                if scheme in {"apikey", "api-key"}:
                    return authenticate_api_key(value, auth.api_keys)
                raise GrpcPolicyError(REASON_INVALID_CREDENTIALS)
        except GrpcPolicyError:
            raise
        except AuthError as exc:
            raise GrpcPolicyError(_auth_error_reason(exc)) from None
        except Exception:
            raise GrpcPolicyError(REASON_INVALID_CREDENTIALS) from None

        principal = self._mtls_principal(context)
        if principal is None:
            raise GrpcPolicyError(REASON_AUTHENTICATION_REQUIRED)
        return principal

    def _wrap(self, handler: Any, rpc: GrpcRpcPolicy) -> Any:
        options = {
            "request_deserializer": handler.request_deserializer,
            "response_serializer": handler.response_serializer,
        }
        if handler.request_streaming and handler.response_streaming:
            return grpc.stream_stream_rpc_method_handler(
                self._guard_stream(rpc, handler.stream_stream), **options
            )
        if handler.request_streaming:
            return grpc.stream_unary_rpc_method_handler(
                self._guard_unary(rpc, handler.stream_unary), **options
            )
        if handler.response_streaming:
            return grpc.unary_stream_rpc_method_handler(
                self._guard_stream(rpc, handler.unary_stream), **options
            )
        return grpc.unary_unary_rpc_method_handler(
            self._guard_unary(rpc, handler.unary_unary), **options
        )

    def _guard_unary(
        self, rpc: GrpcRpcPolicy, behavior: Callable[[Any, Any], Any]
    ) -> Callable[[Any, grpc.ServicerContext], Any]:
        def guarded(request: Any, context: grpc.ServicerContext) -> Any:
            admission = self._admit_or_abort(rpc, context)
            try:
                response = behavior(request, context)
                if admission.expired():
                    abort_rpc(context, REASON_DEADLINE_EXCEEDED)
                return response
            finally:
                admission.release()

        return guarded

    def _guard_stream(
        self, rpc: GrpcRpcPolicy, behavior: Callable[[Any, Any], Any]
    ) -> Callable[[Any, grpc.ServicerContext], Iterator[Any]]:
        def guarded(request: Any, context: grpc.ServicerContext) -> Iterator[Any]:
            admission = self._admit_or_abort(rpc, context)
            try:
                stream = _GuardedStream(iter(behavior(request, context)), admission)
            except BaseException:
                admission.release()
                raise
            stream.attach(context)
            return stream

        return guarded

    def _admit_or_abort(
        self, rpc: GrpcRpcPolicy, context: grpc.ServicerContext
    ) -> "_RpcAdmission":
        try:
            return self._admit(rpc, context)
        except GrpcPolicyError as exc:
            abort_rpc(context, exc.reason)

    def _admit(
        self, rpc: GrpcRpcPolicy, context: grpc.ServicerContext
    ) -> "_RpcAdmission":
        deadline = self._deadline(context)
        if rpc.requires_authentication and self.authentication_enabled:
            try:
                principal = self.authenticate(context)
            except GrpcPolicyError as exc:
                raise self._authentication_failure(context, exc) from None
            if principal is None or not scopes_satisfy(principal.scopes, rpc.scopes):
                raise GrpcPolicyError(REASON_INSUFFICIENT_SCOPE)

        permit: Optional[BlockingConcurrencyPermit] = None
        if rpc.admission:
            key = self._client_key(context, self.policy.throttle.key_by)
            if not self.rate_limiter.try_consume(key).allowed:
                raise GrpcPolicyError(REASON_RATE_LIMITED)
            if self.concurrency_gate.enabled:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise GrpcPolicyError(REASON_DEADLINE_EXCEEDED)
                permit = self.concurrency_gate.acquire(key, timeout=remaining)
                if permit is None:
                    raise GrpcPolicyError(
                        REASON_DEADLINE_EXCEEDED
                        if self._clock() >= deadline
                        else REASON_CONCURRENCY_LIMITED
                    )

        admission = _RpcAdmission(permit, deadline, self._clock)
        if admission.expired() or not context.is_active():
            admission.release()
            raise GrpcPolicyError(REASON_DEADLINE_EXCEEDED)
        return admission

    def _deadline(self, context: grpc.ServicerContext) -> float:
        budget = self.policy.limits.max_rpc_seconds
        remaining = context.time_remaining()
        if isinstance(remaining, (int, float)) and math.isfinite(remaining):
            budget = min(budget, max(0.0, float(remaining)))
        return self._clock() + budget

    def _authentication_failure(
        self, context: grpc.ServicerContext, error: GrpcPolicyError
    ) -> GrpcPolicyError:
        if self._failure_limiter.enabled:
            key = self._client_key(context, self.policy.auth.failure_rate_limit_key_by)
            if not self._failure_limiter.try_consume(key).allowed:
                return GrpcPolicyError(REASON_AUTH_RATE_LIMITED)
        return error

    def _client_key(self, context: grpc.ServicerContext, key_by: str) -> str:
        if key_by == "global":
            return GLOBAL_THROTTLE_KEY
        if key_by == "x-forwarded-for":
            forwarded = _metadata_values(
                context.invocation_metadata(), _FORWARDED_FOR_METADATA
            )
            if forwarded and isinstance(forwarded[0], str):
                first_hop = forwarded[0].split(",", 1)[0].strip()
                if first_hop:
                    return first_hop
        return grpc_peer_host(context.peer()) or "unknown"

    def _mtls_principal(self, context: grpc.ServicerContext) -> Optional[AuthPrincipal]:
        if self._mtls_store is None:
            return None
        try:
            pem_values = tuple(context.auth_context().get(_MTLS_PEM_PROPERTY) or ())
        except Exception:
            raise GrpcPolicyError(REASON_INVALID_CREDENTIALS) from None
        if not pem_values:
            return None
        if len(pem_values) != 1:
            raise GrpcPolicyError(REASON_INVALID_CREDENTIALS)
        try:
            pem = bytes(pem_values[0]).decode("ascii")
            identity = verify_client_certificate_chain((pem,), self._mtls_store)
        except Exception:
            raise GrpcPolicyError(REASON_INVALID_CREDENTIALS) from None
        return principal_for_mtls_identity(identity, self.policy.mtls_principals)


class _RpcAdmission:
    """One RPC's held concurrency permit and effective deadline."""

    def __init__(
        self,
        permit: Optional[BlockingConcurrencyPermit],
        deadline: float,
        clock: Callable[[], float],
    ) -> None:
        self._permit = permit
        self.deadline = deadline
        self._clock = clock
        self._lock = threading.Lock()
        self._released = False

    def expired(self) -> bool:
        return self._clock() >= self.deadline

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        if self._permit is not None:
            self._permit.release()


class _GuardedStream(Iterator[Any]):
    """Response iterator that enforces the deadline and releases permits once.

    grpcio pulls fragments on a worker thread and runs termination callbacks
    on its polling thread. Model bookkeeping is thread-affine, so a stream
    that has started is only ever closed on the worker thread that ran it:
    after its in-flight step returns, or when grpcio drops the iterator once
    the RPC is cancelled. A stream that never started is closed by the
    termination callback directly. The callback holds only a weak reference,
    so it never keeps a finished stream alive.
    """

    def __init__(self, inner: Iterator[Any], admission: _RpcAdmission) -> None:
        self._inner = inner
        self._admission = admission
        self._context: Optional[grpc.ServicerContext] = None
        self._lock = threading.Lock()
        self._started = False
        self._terminated = False
        self._closed = False

    def attach(self, context: grpc.ServicerContext) -> None:
        self._context = context
        reference = weakref.ref(self)

        def on_terminated() -> None:
            stream = reference()
            if stream is not None:
                stream.terminate()

        if not context.add_callback(on_terminated):
            self.terminate()

    def __iter__(self) -> "_GuardedStream":
        return self

    def __next__(self) -> Any:
        with self._lock:
            if self._closed or self._terminated:
                self._close()
                raise StopIteration
            self._started = True
            self._check_deadline()
            try:
                item = next(self._inner)
            except BaseException:
                self._close()
                raise
            if self._terminated:
                self._close()
                raise StopIteration
            self._check_deadline()
            return item

    def __del__(self) -> None:
        try:
            self._close()
        except Exception:  # pragma: no cover - finalizers must not raise
            pass

    def terminate(self) -> None:
        """Mark the RPC ended; close now only if the stream never started."""
        self._terminated = True
        if self._started:
            return
        if self._lock.acquire(blocking=False):
            try:
                if not self._started:
                    self._close()
            finally:
                self._lock.release()

    def close(self) -> None:
        """Close the wrapped iterator and release the RPC's permits."""
        with self._lock:
            self._close()

    def _check_deadline(self) -> None:
        if self._admission.expired():
            self._close()
            if self._context is not None:
                abort_rpc(self._context, REASON_DEADLINE_EXCEEDED)
            raise StopIteration

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._inner, "close", None)
            if callable(close):
                close()
        finally:
            self._admission.release()


def _credential_metadata(
    metadata: Optional[Sequence[Any]],
) -> tuple[Optional[str], Optional[str]]:
    api_keys = _metadata_values(metadata, _API_KEY_METADATA)
    authorizations = _metadata_values(metadata, _AUTHORIZATION_METADATA)
    if len(api_keys) > 1 or len(authorizations) > 1 or (api_keys and authorizations):
        raise GrpcPolicyError(REASON_CREDENTIALS_MALFORMED)
    for value in (*api_keys, *authorizations):
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > MAX_CREDENTIAL_METADATA_CHARS
            or not value.isascii()
            or not value.isprintable()
        ):
            raise GrpcPolicyError(REASON_CREDENTIALS_MALFORMED)
    return (
        api_keys[0] if api_keys else None,
        authorizations[0] if authorizations else None,
    )


def _metadata_values(metadata: Optional[Sequence[Any]], name: str) -> list[Any]:
    values: list[Any] = []
    binary_name = f"{name}-bin"
    for item in metadata or ():
        try:
            key, value = item
        except (TypeError, ValueError):
            raise GrpcPolicyError(REASON_CREDENTIALS_MALFORMED) from None
        key = str(key).lower()
        if key == binary_name and name in _CREDENTIAL_METADATA:
            raise GrpcPolicyError(REASON_CREDENTIALS_MALFORMED)
        if key == name:
            values.append(value)
    return values


def _auth_error_reason(error: AuthError) -> str:
    if error.code == REASON_AUTHENTICATION_REQUIRED:
        return REASON_AUTHENTICATION_REQUIRED
    return REASON_INVALID_CREDENTIALS


def _listener_client_cas(listener: GrpcListener) -> tuple[x509.Certificate, ...]:
    credentials = listener.credentials
    if credentials is None or not credentials.require_client_cert:
        return ()
    return tuple(credentials.client_ca_certificates)


__all__ = [
    "CATEGORY_DEADLINE_EXCEEDED",
    "CATEGORY_INTERNAL",
    "CATEGORY_INVALID_ARGUMENT",
    "CATEGORY_PERMISSION_DENIED",
    "CATEGORY_RESOURCE_EXHAUSTED",
    "CATEGORY_UNAUTHENTICATED",
    "DEFAULT_GRPC_MAX_RPC_SECONDS",
    "DEFAULT_GRPC_MAX_SEND_MESSAGE_BYTES",
    "DEFAULT_GRPC_MAX_STREAM_BUFFER_CHARS",
    "DEFAULT_GRPC_MAX_STREAM_CHUNKS",
    "DEFAULT_GRPC_MAX_STREAM_CHUNK_BYTES",
    "DEFAULT_GRPC_STREAM_BUFFER_CHARS",
    "GRPC_FAILURES",
    "GRPC_RPC_REASON_CODES",
    "GRPC_RPC_ROUTES",
    "GRPC_SERVICE_NAME",
    "GrpcFailure",
    "GrpcLimits",
    "GrpcPolicyError",
    "GrpcRpcPolicy",
    "GrpcSecurityInterceptor",
    "GrpcServicePolicy",
    "MAX_CREDENTIAL_METADATA_CHARS",
    "OPENMED_REASON_METADATA_KEY",
    "REASON_AUTHENTICATION_REQUIRED",
    "REASON_AUTH_RATE_LIMITED",
    "REASON_CONCURRENCY_LIMITED",
    "REASON_CREDENTIALS_MALFORMED",
    "REASON_DEADLINE_EXCEEDED",
    "REASON_INSUFFICIENT_SCOPE",
    "REASON_INTERNAL_ERROR",
    "REASON_INVALID_CREDENTIALS",
    "REASON_INVALID_REQUEST",
    "REASON_MODEL_NOT_SERVED",
    "REASON_RATE_LIMITED",
    "REASON_STREAM_BUFFER_TOO_LARGE",
    "REASON_STREAM_CHUNK_TOO_LARGE",
    "REASON_STREAM_TOO_MANY_CHUNKS",
    "REASON_TEXT_TOO_LONG",
    "abort_rpc",
    "grpc_peer_host",
    "grpc_rpc_policies",
    "listener_client_ca_certificates",
    "require_authenticated_listener",
    "validate_grpc_rpc_policies",
]
