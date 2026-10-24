"""Bridge text decision requests to a local System One or Decisions server.

Many decision models ship their own serving stack (CUDA runtimes, custom heads,
pinned engines) and answer ``POST /v1/systemone`` or the OpenAI-compatible
``POST /v1/decisions`` route. :class:`DecisionServerBackend` is a specialist
decision backend that sends text requests to such a server and parses the
answers with :mod:`openmed.structured.decision_interop`, so OpenMed applies its
own calibration and abstention to the server's probabilities.

The endpoint policy is local-first:

* loopback endpoints (``127.0.0.0/8``, ``::1`` or ``localhost`` resolving only
  to loopback) and Unix domain sockets are accepted by default;
* a private-network origin is reachable only when the operator names it in
  ``allowed_private_origins`` *and* declares ``contains_phi=False``; requests
  marked as containing PHI never leave loopback, and offline mode
  (``OPENMED_OFFLINE``) keeps them on loopback too;
* public hosts are always refused.

Endpoint, redirect and DNS checks reuse
:class:`~openmed.mcp.upstream_endpoint_policy.UpstreamEndpointPolicy`. The
client connects only to the addresses that policy approved for the call, so a
redirect or retry can never resend a request body to a different host. Only
same-origin ``307``/``308`` redirects are followed.

Transport errors, refusals and malformed bodies become typed ``failure`` or
``unsupported`` decision states with controlled codes, never a fabricated
answer. Exceptions and log records carry codes and counts only; they never
contain the endpoint URL or query, the request input, instructions, options or
credentials. Inputs are text (or JSON text); media inputs are not sent.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import socket
import stat
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final
from urllib.parse import urljoin, urlsplit

from openmed.core.offline import is_local_only
from openmed.mcp.upstream_endpoint_policy import (
    AddressResolver,
    UpstreamEndpointPolicy,
    UpstreamEndpointPolicyError,
    resolve_hostname,
)

from .decision import (
    DecisionBackendIdentity,
    DecisionBackendKind,
    DecisionBackendOutput,
    DecisionConflictError,
    DecisionError,
    DecisionMode,
    DecisionRequest,
    DecisionUnsupportedError,
)
from .decision_interop import (
    from_openai_decisions_response,
    from_system_one_response,
    to_openai_decisions_request,
    to_system_one_request,
)

SYSTEM_ONE_PATH: Final = "/v1/systemone"
OPENAI_DECISIONS_PATH: Final = "/v1/decisions"
DEFAULT_MAX_RESPONSE_BYTES: Final = 1_048_576
MAX_RESPONSE_BYTES_LIMIT: Final = 16_777_216
DEFAULT_MAX_REDIRECTS: Final = 2
MAX_REDIRECTS_LIMIT: Final = 5
DEFAULT_MAX_RETRIES: Final = 1
MAX_RETRIES_LIMIT: Final = 3
MAX_BEARER_TOKEN_CHARS: Final = 4096
MAX_UNIX_SOCKET_PATH_BYTES: Final = 107
_UNIX_SOCKET_ORIGIN: Final = "http://localhost"
_USER_AGENT: Final = "openmed-decision-client"
_RETRY_BACKOFF_SECONDS: Final = 0.05
_LOGGER = logging.getLogger(__name__)


class DecisionServerProtocol(str, Enum):
    """Wire protocol spoken by a decision server."""

    SYSTEM_ONE = "system_one"
    OPENAI_DECISIONS = "openai_decisions"


class DecisionServerScope(str, Enum):
    """Where a decision server is reached; recorded in the backend runtime."""

    LOOPBACK = "loopback"
    UNIX_SOCKET = "unix_socket"
    PRIVATE_NETWORK = "private_network"


_DEFAULT_PATHS: Final = {
    DecisionServerProtocol.SYSTEM_ONE: SYSTEM_ONE_PATH,
    DecisionServerProtocol.OPENAI_DECISIONS: OPENAI_DECISIONS_PATH,
}
_SAFE_MESSAGES: Final = {
    "decision_server_failure": "The decision server request failed.",
    "decision_server_endpoint_refused": (
        "The decision server endpoint was refused by policy."
    ),
    "decision_server_unreachable": "The decision server could not be reached.",
    "decision_server_unavailable": "The decision server is unavailable.",
    "decision_server_unauthorized": "The decision server refused the credentials.",
    "decision_server_redirect_refused": "A decision server redirect was refused.",
    "decision_server_response_invalid": (
        "The decision server returned an invalid response."
    ),
    "decision_server_response_too_large": (
        "The decision server response exceeds the configured limit."
    ),
    "decision_server_input_rejected": "The decision server rejected the request.",
    "decision_server_route_unsupported": (
        "The decision server does not serve this protocol route."
    ),
    "decision_server_model_mismatch": (
        "The decision server reported a different model than declared."
    ),
    "backend_timeout": "The decision server did not answer within the timeout.",
}
_ENDPOINT_REASONS: Final = frozenset(
    {
        "endpoint_rejected",
        "invalid_endpoint",
        "invalid_private_origin",
        "invalid_unix_socket",
        "unix_socket_unsupported",
        "phi_requires_loopback",
        "offline_mode",
    }
)


class DecisionServerError(DecisionError):
    """Base class for redacted decision-server failures.

    The exception class selects the decision state (``failure`` here,
    ``unsupported``, ``conflict`` or a timeout in subclasses) and ``code`` is
    the controlled result code. Messages are static and never contain the
    endpoint, request input, instructions, options or credentials.
    """

    default_code = "decision_server_failure"

    def __init__(self, code: str | None = None) -> None:
        safe = code if code in _SAFE_MESSAGES else self.default_code
        self.code = safe
        super().__init__(_SAFE_MESSAGES[safe])

    @property
    def decision_code(self) -> str:
        """Return the controlled code reported in the decision result."""

        return self.code

    def to_dict(self) -> dict[str, str]:
        """Return a PHI- and credential-safe structured error."""

        return {"code": self.code, "message": str(self)}


class DecisionServerEndpointError(DecisionServerError):
    """Raised when an endpoint is refused at construction or call time.

    ``reason_code`` is a static policy reason such as ``loopback_only_violation``
    or ``phi_requires_loopback``.
    """

    default_code = "decision_server_endpoint_refused"

    def __init__(self, reason_code: str = "endpoint_rejected") -> None:
        super().__init__("decision_server_endpoint_refused")
        self.reason_code = (
            reason_code if reason_code in _ENDPOINT_REASONS else "endpoint_rejected"
        )

    @classmethod
    def from_policy(
        cls,
        error: UpstreamEndpointPolicyError,
    ) -> DecisionServerEndpointError:
        """Carry the static reason code of an endpoint-policy refusal."""

        refused = cls()
        refused.reason_code = error.reason_code
        return refused

    def to_dict(self) -> dict[str, str]:
        """Return a PHI- and credential-safe structured error."""

        return {**super().to_dict(), "reason_code": self.reason_code}


class DecisionServerUnsupportedError(DecisionServerError, DecisionUnsupportedError):
    """Raised when the server refuses the request or does not serve the route."""

    default_code = "decision_server_input_rejected"


class DecisionServerConflictError(DecisionServerError, DecisionConflictError):
    """Raised when the server reports a model other than the declared one."""

    default_code = "decision_server_model_mismatch"


class DecisionServerTimeoutError(DecisionServerError, TimeoutError):
    """Raised when the server does not answer within the request timeout."""

    default_code = "backend_timeout"


class _RetryableTransportError(Exception):
    """Internal marker for connection-level failures on the approved target."""


@dataclass(frozen=True, slots=True)
class _Target:
    """One call's approved destination; addresses are pinned for the call."""

    scope: DecisionServerScope
    origin: str = field(repr=False)
    request_target: str = field(repr=False)
    host_header: str = field(repr=False)
    host: str = field(repr=False)
    port: int
    addresses: tuple[str, ...] = field(repr=False)
    unix_socket: str | None = field(repr=False)
    tls: bool


class DecisionServerBackend:
    """Specialist decision backend served by a local decision-model server.

    The backend implements :class:`~openmed.structured.decision.DecisionBackend`
    and :class:`~openmed.structured.decision.DecisionQuestionSetBackend`, so a
    question set is sent as one request and every question is calibrated and
    abstains on its own inside :func:`~openmed.structured.decision.decide`.

    Args:
        endpoint: ``http`` or ``https`` URL of the server. A URL without a path
            uses the protocol's default route. Exactly one of ``endpoint`` and
            ``unix_socket`` is required.
        unix_socket: Absolute path of a Unix domain socket the server listens
            on. Requests are sent with ``Host: localhost``.
        protocol: ``system_one`` (``POST /v1/systemone``) or
            ``openai_decisions`` (``POST /v1/decisions``).
        model_id: Declared model identifier, recorded in the backend identity.
        revision: Declared immutable model revision, recorded in the identity.
        license_id: Declared SPDX license of the served weights. Undeclared or
            non-permissive licenses are rejected, as for every backend.
        backend_id: Identity ``backend_id``; derived from the protocol by
            default.
        served_model: Name the server knows the model by. It is sent as
            ``model`` (the OpenAI-compatible route always sends one and falls
            back to ``model_id``) and accepted when the server reports it.
        path: Request path for ``unix_socket`` servers; the protocol's default
            route when omitted.
        contains_phi: Whether requests through this backend may contain PHI.
            ``True`` by default, which keeps every request on loopback or a
            Unix socket.
        allowed_private_origins: Exact private-network origins (for example
            ``http://10.0.0.5:8000``) the operator approves. Used only when
            ``contains_phi`` is ``False``.
        resolver: Hostname resolver for the endpoint policy; inject a
            deterministic resolver in tests.
        max_retries: Connection-level retries to the same approved addresses.
        max_redirects: Same-origin ``307``/``308`` redirects to follow.
        max_response_bytes: Largest response body accepted.
        bearer_token: Optional key sent as ``Authorization: Bearer``. It never
            appears in representations, errors or logs.
        verify_served_model: Reject answers whose reported ``model`` names a
            different model than ``model_id`` or ``served_model``.

    Raises:
        DecisionServerEndpointError: If the endpoint is refused by policy.
        DecisionError: If the identity or a bound is invalid, including an
            undeclared or non-permissive license.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        unix_socket: str | os.PathLike[str] | None = None,
        protocol: DecisionServerProtocol | str = DecisionServerProtocol.SYSTEM_ONE,
        model_id: str,
        revision: str,
        license_id: str,
        backend_id: str | None = None,
        served_model: str | None = None,
        path: str | None = None,
        contains_phi: bool = True,
        allowed_private_origins: Iterable[str] = (),
        resolver: AddressResolver = resolve_hostname,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        bearer_token: str | None = None,
        verify_served_model: bool = True,
    ) -> None:
        self._protocol = DecisionServerProtocol(protocol)
        if (endpoint is None) == (unix_socket is None):
            raise DecisionError("exactly one of endpoint and unix_socket is required")
        if not isinstance(contains_phi, bool):
            raise TypeError("contains_phi must be a boolean")
        if not isinstance(verify_served_model, bool):
            raise TypeError("verify_served_model must be a boolean")
        self._contains_phi = contains_phi
        self._verify_served_model = verify_served_model
        self._max_retries = _bounded_int(max_retries, "max_retries", MAX_RETRIES_LIMIT)
        self._max_redirects = _bounded_int(
            max_redirects, "max_redirects", MAX_REDIRECTS_LIMIT
        )
        self._max_response_bytes = _bounded_int(
            max_response_bytes,
            "max_response_bytes",
            MAX_RESPONSE_BYTES_LIMIT,
            minimum=1,
        )
        self._served_model = (
            None if served_model is None else _model_name(served_model, "served_model")
        )
        self._bearer_token = _bearer_token(bearer_token)
        try:
            self._policy = UpstreamEndpointPolicy(
                resolver=resolver,
                allow_loopback=True,
                allowed_private_origins=frozenset(
                    _origin_values(allowed_private_origins)
                ),
            )
        except UpstreamEndpointPolicyError as error:
            raise DecisionServerEndpointError.from_policy(error) from None
        self._endpoint = None if endpoint is None else _endpoint_text(endpoint)
        self._unix_socket = None if unix_socket is None else _socket_path(unix_socket)
        self._path: str | None
        if self._unix_socket is not None:
            self._path = _request_path(
                _DEFAULT_PATHS[self._protocol] if path is None else path
            )
        elif path is not None:
            raise DecisionError("path applies to unix_socket servers only")
        else:
            self._path = None
        scope = self._approve(validate_socket=False).scope
        self._scope = scope
        self._digest = _endpoint_digest(self._endpoint or self._unix_socket or "")
        self.identity = DecisionBackendIdentity(
            backend_id=backend_id or f"openmed.decision_server.{self._protocol.value}",
            kind=DecisionBackendKind.SPECIALIST,
            runtime=f"{self._protocol.value}_server:{scope.value}",
            model_id=model_id,
            revision=revision,
            license_id=license_id,
        )

    @property
    def protocol(self) -> DecisionServerProtocol:
        """Return the wire protocol."""

        return self._protocol

    @property
    def scope(self) -> DecisionServerScope:
        """Return where the server is reached."""

        return self._scope

    @property
    def contains_phi(self) -> bool:
        """Return whether requests are treated as containing PHI."""

        return self._contains_phi

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(protocol={self._protocol.value!r}, "
            f"scope={self._scope.value!r}, model_id={self.identity.model_id!r})"
        )

    def safe_metadata(self) -> dict[str, Any]:
        """Return log-safe configuration metadata.

        The endpoint, socket path and credentials are represented by a digest
        and flags only.
        """

        return {
            "protocol": self._protocol.value,
            "scope": self._scope.value,
            "backend_id": self.identity.backend_id,
            "model_id": self.identity.model_id,
            "revision": self.identity.revision,
            "license_id": self.identity.license_id,
            "contains_phi": self._contains_phi,
            "endpoint_digest": self._digest,
            "authenticated": self._bearer_token is not None,
            "max_retries": self._max_retries,
            "max_redirects": self._max_redirects,
        }

    def score(self, request: DecisionRequest) -> DecisionBackendOutput:
        """Score one single-question request on the server.

        Args:
            request: A single-question decision request.

        Returns:
            Raw per-option (or per-level) probabilities for :func:`decide`.

        Raises:
            DecisionUnsupportedError: If the mode has no wire equivalent or the
                server refuses the request.
            DecisionServerError: On transport, status or response failures.
        """

        if not isinstance(request, DecisionRequest):
            raise TypeError("request must be a DecisionRequest")
        if request.mode is DecisionMode.QUESTION_SET:
            raise DecisionUnsupportedError(
                "question sets are scored through score_questions"
            )
        (question,) = request.as_questions()
        return self._exchange(request)[question.question_id]

    def score_questions(
        self,
        request: DecisionRequest,
    ) -> Mapping[str, DecisionBackendOutput]:
        """Score every question of a request in one server call.

        Args:
            request: A decision request; a question set is sent as one body.

        Returns:
            One backend output per question identifier.

        Raises:
            DecisionUnsupportedError: If a question has no wire equivalent or
                the server refuses the request.
            DecisionServerError: On transport, status or response failures.
        """

        if not isinstance(request, DecisionRequest):
            raise TypeError("request must be a DecisionRequest")
        return self._exchange(request)

    def _exchange(self, request: DecisionRequest) -> dict[str, DecisionBackendOutput]:
        started = time.monotonic()
        deadline = started + request.timeout_ms / 1000.0
        body = self._render(request)
        payload = json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        try:
            target = self._approve(validate_socket=True)
            data = self._post(target, payload, deadline)
            document = _decode_document(data)
            self._check_model(document)
            outputs = self._parse(request, document)
        except DecisionServerError as error:
            _LOGGER.debug(
                "decision server call ended with %s (protocol=%s, scope=%s, "
                "questions=%d, elapsed_ms=%d)",
                error.code,
                self._protocol.value,
                self._scope.value,
                len(request.as_questions()),
                int((time.monotonic() - started) * 1000),
            )
            raise
        _LOGGER.debug(
            "decision server call answered (protocol=%s, scope=%s, questions=%d, "
            "elapsed_ms=%d)",
            self._protocol.value,
            self._scope.value,
            len(outputs),
            int((time.monotonic() - started) * 1000),
        )
        return outputs

    def _render(self, request: DecisionRequest) -> dict[str, Any]:
        if self._protocol is DecisionServerProtocol.SYSTEM_ONE:
            body = to_system_one_request(request)
            if self._served_model is not None:
                body["model"] = self._served_model
            return body
        return to_openai_decisions_request(
            request,
            model=self._served_model or self.identity.model_id,
        )

    def _parse(
        self,
        request: DecisionRequest,
        document: Mapping[str, Any],
    ) -> dict[str, DecisionBackendOutput]:
        try:
            if self._protocol is DecisionServerProtocol.SYSTEM_ONE:
                return from_system_one_response(request, document)
            return from_openai_decisions_response(request, document)
        except (DecisionError, TypeError, ValueError):
            raise DecisionServerError("decision_server_response_invalid") from None

    def _check_model(self, document: Mapping[str, Any]) -> None:
        reported = document.get("model")
        if reported is None or not self._verify_served_model:
            return
        declared = {_model_key(self.identity.model_id)}
        if self._served_model is not None:
            declared.add(_model_key(self._served_model))
        if not isinstance(reported, str) or _model_key(reported) not in declared:
            raise DecisionServerConflictError("decision_server_model_mismatch")

    def _approve(self, *, validate_socket: bool) -> _Target:
        """Validate the endpoint and pin the approved addresses for one call."""

        if self._unix_socket is not None:
            if getattr(socket, "AF_UNIX", None) is None:
                raise DecisionServerEndpointError("unix_socket_unsupported")
            if validate_socket:
                try:
                    mode = os.stat(self._unix_socket).st_mode
                except OSError:
                    raise DecisionServerError("decision_server_unreachable") from None
                if not stat.S_ISSOCK(mode):
                    raise DecisionServerError("decision_server_unreachable")
            return _Target(
                scope=DecisionServerScope.UNIX_SOCKET,
                origin=_UNIX_SOCKET_ORIGIN,
                request_target=str(self._path),
                host_header="localhost",
                host="localhost",
                port=80,
                addresses=(),
                unix_socket=self._unix_socket,
                tls=False,
            )
        try:
            approval = self._policy.validate(str(self._endpoint))
        except UpstreamEndpointPolicyError as error:
            raise DecisionServerEndpointError.from_policy(error) from None
        if approval.is_loopback:
            scope = DecisionServerScope.LOOPBACK
        elif approval.is_private_network:
            if self._contains_phi:
                raise DecisionServerEndpointError("phi_requires_loopback")
            if is_local_only():
                raise DecisionServerEndpointError("offline_mode")
            scope = DecisionServerScope.PRIVATE_NETWORK
        else:  # pragma: no cover - the policy approves no other origins
            raise DecisionServerEndpointError("endpoint_rejected")
        parts = urlsplit(approval.url)
        return _Target(
            scope=scope,
            origin=approval.origin,
            request_target=_target_from_parts(
                parts.path, parts.query, _DEFAULT_PATHS[self._protocol]
            ),
            host_header=parts.netloc,
            host=approval.host,
            port=approval.port,
            addresses=approval.addresses,
            unix_socket=None,
            tls=parts.scheme.lower() == "https",
        )

    def _post(self, target: _Target, payload: bytes, deadline: float) -> bytes:
        request_target = target.request_target
        retries = 0
        redirects = 0
        while True:
            try:
                status, location, data = self._send_once(
                    target, request_target, payload, deadline
                )
            except _RetryableTransportError:
                retries += 1
                remaining = deadline - time.monotonic()
                if retries > self._max_retries or remaining <= 0:
                    raise DecisionServerError("decision_server_unreachable") from None
                # Retries reuse the pinned addresses of the approved target.
                time.sleep(min(_RETRY_BACKOFF_SECONDS * retries, remaining / 2))
                continue
            if status in (307, 308):
                redirects += 1
                if redirects > self._max_redirects:
                    raise DecisionServerError("decision_server_redirect_refused")
                request_target = self._redirect_target(target, request_target, location)
                continue
            return _status_body(status, data)

    def _redirect_target(
        self,
        target: _Target,
        current: str,
        location: str | None,
    ) -> str:
        """Return the request target of a same-origin redirect, or refuse it."""

        if (
            not isinstance(location, str)
            or not location
            or not location.isascii()
            or _has_unsafe_characters(location)
        ):
            raise DecisionServerError("decision_server_redirect_refused")
        if target.unix_socket is not None:
            # The socket is the only destination; the target must stay on the
            # synthetic ``http://localhost`` origin the request was sent to.
            try:
                joined = urlsplit(urljoin(target.origin + current, location))
                same_origin = (
                    joined.scheme == "http"
                    and joined.hostname == "localhost"
                    and joined.port in (None, 80)
                    and "@" not in joined.netloc
                )
            except ValueError:
                same_origin = False
            if not same_origin:
                raise DecisionServerError("decision_server_redirect_refused")
            return _target_from_parts(joined.path, joined.query, "/")
        try:
            candidate = self._policy.validate_redirect(
                location, base_url=target.origin + current
            )
        except UpstreamEndpointPolicyError:
            raise DecisionServerError("decision_server_redirect_refused") from None
        if candidate.origin != target.origin:
            raise DecisionServerError("decision_server_redirect_refused")
        parts = urlsplit(candidate.url)
        return _target_from_parts(parts.path, parts.query, "/")

    def _send_once(
        self,
        target: _Target,
        request_target: str,
        payload: bytes,
        deadline: float,
    ) -> tuple[int, str | None, bytes]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DecisionServerTimeoutError()
        connection = _connection(target, remaining)
        watchdog = _Watchdog(connection, remaining)
        headers = {
            "Host": target.host_header,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
            "User-Agent": _USER_AGENT,
        }
        if self._bearer_token is not None:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        try:
            connection.connect()
            remaining = deadline - time.monotonic()
            if watchdog.fired or remaining <= 0 or connection.sock is None:
                raise DecisionServerTimeoutError()
            # Reads after a slow connect get only the time left in the budget.
            connection.sock.settimeout(remaining)
            connection.request("POST", request_target, body=payload, headers=headers)
            response = connection.getresponse()
            status = response.status
            location = response.getheader("Location")
            data = (
                _read_bounded(response, self._max_response_bytes)
                if status == 200
                else b""
            )
        except DecisionServerError:
            raise
        except TimeoutError:
            raise DecisionServerTimeoutError() from None
        except ConnectionError:
            # Refused, reset or closed before an answer: retry the same target.
            if watchdog.fired or time.monotonic() >= deadline:
                raise DecisionServerTimeoutError() from None
            raise _RetryableTransportError() from None
        except http.client.HTTPException:
            if watchdog.fired or time.monotonic() >= deadline:
                raise DecisionServerTimeoutError() from None
            raise DecisionServerError("decision_server_response_invalid") from None
        except Exception:
            if watchdog.fired or time.monotonic() >= deadline:
                raise DecisionServerTimeoutError() from None
            raise DecisionServerError("decision_server_unreachable") from None
        finally:
            watchdog.cancel()
            connection.close()
        if watchdog.fired:
            raise DecisionServerTimeoutError()
        return status, location, data


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that dials only policy-approved addresses."""

    def __init__(self, addresses: tuple[str, ...], port: int, timeout: float):
        super().__init__(addresses[0], port, timeout=timeout)
        self._addresses = addresses

    def connect(self) -> None:
        self.sock = _open_tcp(self._addresses, self.port, self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection to approved addresses, verified against the hostname."""

    def __init__(
        self,
        addresses: tuple[str, ...],
        port: int,
        server_hostname: str,
        timeout: float,
    ):
        import ssl

        context = ssl.create_default_context()
        super().__init__(addresses[0], port, timeout=timeout, context=context)
        self._addresses = addresses
        self._server_hostname = server_hostname
        self._tls_context = context

    def connect(self) -> None:
        sock = _open_tcp(self._addresses, self.port, self.timeout)
        try:
            self.sock = self._tls_context.wrap_socket(
                sock, server_hostname=self._server_hostname
            )
        except BaseException:
            sock.close()
            raise


class _UnixSocketConnection(http.client.HTTPConnection):
    """HTTP connection over a Unix domain socket."""

    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.timeout)
            sock.connect(self._socket_path)
        except BaseException:
            sock.close()
            raise
        self.sock = sock


class _Watchdog:
    """Close a connection's socket when the request deadline passes.

    Socket timeouts bound each read; the watchdog also bounds the total time
    of a server that trickles bytes just under the per-read timeout.
    """

    def __init__(self, connection: http.client.HTTPConnection, seconds: float):
        self.fired = False
        self._connection = connection
        self._timer = threading.Timer(seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.fired = True
        sock = getattr(self._connection, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def cancel(self) -> None:
        self._timer.cancel()


def _connection(target: _Target, timeout: float) -> http.client.HTTPConnection:
    if target.unix_socket is not None:
        return _UnixSocketConnection(target.unix_socket, timeout)
    if target.tls:
        return _PinnedHTTPSConnection(
            target.addresses, target.port, target.host, timeout
        )
    return _PinnedHTTPConnection(target.addresses, target.port, timeout)


def _open_tcp(addresses: tuple[str, ...], port: int, timeout: Any) -> socket.socket:
    last_error: OSError | None = None
    for address in addresses:
        try:
            sock = socket.create_connection((address, port), timeout)
        except OSError as error:
            last_error = error
            continue
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        return sock
    raise last_error or ConnectionRefusedError()


def _read_bounded(response: http.client.HTTPResponse, limit: int) -> bytes:
    declared = response.getheader("Content-Length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError:
            raise DecisionServerError("decision_server_response_invalid") from None
        if length > limit:
            raise DecisionServerError("decision_server_response_too_large")
    data = response.read(limit + 1)
    if len(data) > limit:
        raise DecisionServerError("decision_server_response_too_large")
    return data


def _status_body(status: int, data: bytes) -> bytes:
    if status == 200:
        return data
    if status in (400, 413, 422):
        raise DecisionServerUnsupportedError("decision_server_input_rejected")
    if status in (404, 405, 501):
        raise DecisionServerUnsupportedError("decision_server_route_unsupported")
    if status in (401, 403, 407):
        raise DecisionServerError("decision_server_unauthorized")
    if status in (408, 504):
        raise DecisionServerTimeoutError()
    if 300 <= status < 400:
        raise DecisionServerError("decision_server_redirect_refused")
    if status == 429 or status >= 500:
        raise DecisionServerError("decision_server_unavailable")
    raise DecisionServerError("decision_server_response_invalid")


def _decode_document(data: bytes) -> Mapping[str, Any]:
    try:
        document = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise DecisionServerError("decision_server_response_invalid") from None
    if not isinstance(document, dict):
        raise DecisionServerError("decision_server_response_invalid")
    return document


def _reject_json_constant(_value: str) -> Any:
    raise ValueError("non-finite JSON constants are not accepted")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object keys are not accepted")
        result[key] = value
    return result


def _model_key(value: str) -> str:
    """Compare model names without an organisation prefix or letter case."""

    return value.strip().rstrip("/").rsplit("/", 1)[-1].casefold()


def _target_from_parts(path: str, query: str, default: str) -> str:
    target = path if path not in ("", "/") else default
    return f"{target}?{query}" if query else target


def _endpoint_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_int(value: Any, name: str, maximum: int, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise DecisionError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _model_name(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise DecisionError(f"{name} must be a bounded identifier")
    return value


def _bearer_token(value: Any) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= MAX_BEARER_TOKEN_CHARS
        or any(not 0x21 <= ord(char) <= 0x7E for char in value)
    ):
        raise DecisionError("bearer_token must be printable ASCII without spaces")
    return value


def _origin_values(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise DecisionServerEndpointError("invalid_private_origin")
    try:
        return tuple(values)
    except TypeError:
        raise DecisionServerEndpointError("invalid_private_origin") from None


def _endpoint_text(value: Any) -> str:
    if not isinstance(value, str) or not value.isascii():
        raise DecisionServerEndpointError("invalid_endpoint")
    return value


def _socket_path(value: Any) -> str:
    try:
        path = os.fspath(value)
    except TypeError:
        raise DecisionServerEndpointError("invalid_unix_socket") from None
    if (
        not isinstance(path, str)
        or not os.path.isabs(path)
        or "\x00" in path
        or len(path.encode("utf-8", "surrogateescape")) > MAX_UNIX_SOCKET_PATH_BYTES
    ):
        raise DecisionServerEndpointError("invalid_unix_socket")
    return path


def _has_unsafe_characters(value: str) -> bool:
    return any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)


def _request_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value.startswith("//")
        or not value.isascii()
        or _has_unsafe_characters(value)
        or "#" in value
    ):
        raise DecisionError("path must be an absolute ASCII request path")
    return value


__all__ = [
    "DEFAULT_MAX_REDIRECTS",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_MAX_RETRIES",
    "OPENAI_DECISIONS_PATH",
    "SYSTEM_ONE_PATH",
    "DecisionServerBackend",
    "DecisionServerConflictError",
    "DecisionServerEndpointError",
    "DecisionServerError",
    "DecisionServerProtocol",
    "DecisionServerScope",
    "DecisionServerTimeoutError",
    "DecisionServerUnsupportedError",
]
