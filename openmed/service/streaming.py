"""NDJSON streaming helpers for REST PII extraction and de-identification.

Both NDJSON routes share one set of bounds: the request deadline (profile
``timeout``), a write-idle bound for consumers that stop reading, and a
maximum event count. Every bound ends the stream with at most one value-free
``type: "error"`` line carrying a controlled code; nothing follows it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Collection,
    Iterator,
    Mapping,
    MutableMapping,
)
from typing import Any, Callable, Optional

import anyio
from pydantic import Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from openmed.core.policy import canonical_policy_name, load_policy
from openmed.core.streaming import deidentify_stream

from .runtime import (
    DEFAULT_SERVICE_STREAM_MAX_EVENTS,
    DEFAULT_SERVICE_STREAM_WRITE_IDLE_SECONDS,
    ServiceRuntime,
    StreamDeadlineExceeded,
)
from .schemas import PIIDeidentifyRequest, PIIExtractStreamRequest
from .throttle import RequestWork, current_request_work

DEFAULT_DEIDENTIFY_STREAM_CHUNK_SIZE = 1024
MAX_DEIDENTIFY_STREAM_CHUNK_SIZE = 32768
DEFAULT_DEIDENTIFY_STREAM_MAX_BUFFER = 4096

NDJSON_MEDIA_TYPE = "application/x-ndjson"
STREAM_ROUTE_PATHS = frozenset({"/pii/extract/stream", "/pii/deidentify/stream"})
STREAM_WRITE_GUARD_SCOPE_KEY = "openmed.stream_write_guard"
STREAM_TERMINAL_GRACE_SECONDS = 1.0
STREAM_WRITE_IDLE_CODE = "stream_write_idle"
STREAM_EVENT_LIMIT_CODE = "stream_event_limit"

_END_OF_STREAM = object()
_LOGGER = logging.getLogger(__name__)


class StreamWriteGuard:
    """Bound how long one NDJSON response may stay blocked on a client write.

    The guard wraps an ASGI ``send`` callable. When one body write stays
    blocked for longer than ``idle_seconds`` the write is abandoned, a
    value-free ``stream_write_idle`` error line is offered to the client for a
    short grace period, and every later message is dropped so the producer
    stops at its next event. If the client still is not reading, the response
    is left incomplete and the ASGI server closes the connection.

    Attributes:
        idle_seconds: Longest time one body write may stay blocked.
        expired: Whether the bound has fired for this response.
        installed: Whether a ``send`` callable is already wrapped.
    """

    def __init__(
        self,
        idle_seconds: float = DEFAULT_SERVICE_STREAM_WRITE_IDLE_SECONDS,
    ) -> None:
        self.idle_seconds = float(idle_seconds)
        self.expired = False
        self.installed = False

    @classmethod
    def for_scope(
        cls,
        scope: MutableMapping[str, Any],
        idle_seconds: float,
    ) -> "StreamWriteGuard":
        """Return the request's guard, configured with *idle_seconds*.

        Args:
            scope: ASGI scope of the streaming request.
            idle_seconds: Write-idle bound to apply.

        Returns:
            The guard installed by :class:`StreamWriteIdleMiddleware`, or a new
            uninstalled guard that the response wraps around its own ``send``.
        """
        guard = scope.get(STREAM_WRITE_GUARD_SCOPE_KEY)
        if not isinstance(guard, cls):
            guard = cls(idle_seconds)
            scope[STREAM_WRITE_GUARD_SCOPE_KEY] = guard
        else:
            guard.idle_seconds = float(idle_seconds)
        return guard

    def wrap(self, send: Send) -> Send:
        """Return *send* bounded by this guard.

        Args:
            send: ASGI send callable to bound.

        Returns:
            A send callable that bounds every response-body write.
        """
        self.installed = True

        async def bounded_send(message: Message) -> None:
            if self.expired:
                return
            if message.get("type") != "http.response.body":
                await send(message)
                return
            with anyio.move_on_after(self.idle_seconds) as write:
                await send(message)
            if write.cancelled_caught:
                await self._expire(send, body_done=not message.get("more_body"))

        return bounded_send

    async def _expire(self, send: Send, *, body_done: bool) -> None:
        self.expired = True
        _LOGGER.warning("NDJSON stream stopped: %s", STREAM_WRITE_IDLE_CODE)
        grace = min(STREAM_TERMINAL_GRACE_SECONDS, self.idle_seconds)
        with anyio.move_on_after(grace):
            if not body_done:
                # Only a write that still carried events is replaced by the
                # terminal line; a stalled end-of-body write already handed
                # off every event, so it is only retried.
                line = _ndjson_error(
                    STREAM_WRITE_IDLE_CODE,
                    "Stream consumer did not read within the write-idle bound",
                    {"write_idle_seconds": self.idle_seconds},
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": line.encode("utf-8"),
                        "more_body": True,
                    }
                )
            await send({"type": "http.response.body", "body": b"", "more_body": False})


class StreamWriteIdleMiddleware:
    """Apply the NDJSON write-idle bound at the outermost ASGI layer.

    Bounding the server's own ``send`` (rather than a send nested inside
    buffering middleware) is what lets a stalled consumer be disconnected:
    after the bound fires, inner layers drain immediately and the ASGI call
    returns, so the server can close the connection.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        paths: Collection[str] = STREAM_ROUTE_PATHS,
    ) -> None:
        self.app = app
        self._paths = frozenset(paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") not in self._paths:
            await self.app(scope, receive, send)
            return
        guard = StreamWriteGuard()
        scope = dict(scope)
        scope[STREAM_WRITE_GUARD_SCOPE_KEY] = guard
        await self.app(scope, receive, guard.wrap(send))


class NDJSONStreamControls:
    """Per-request bounds and admission ownership for one NDJSON stream.

    Creating the controls takes a hold on the request's admission work, so the
    concurrency permit and drain token stay held while the body streams
    instead of returning as soon as headers are sent.

    Attributes:
        write_guard: Write-idle guard for this response.
        max_events: Largest number of lines the stream may emit before it
            ends with a ``stream_event_limit`` error line.
    """

    def __init__(
        self,
        *,
        write_guard: StreamWriteGuard,
        max_events: int,
        work: Optional[RequestWork],
    ) -> None:
        self.write_guard = write_guard
        self.max_events = int(max_events)
        self._work = work
        self._release_hold: Optional[Callable[[], None]] = (
            work.hold() if work is not None else None
        )

    @classmethod
    def for_request(
        cls,
        scope: MutableMapping[str, Any],
        runtime: ServiceRuntime,
    ) -> "NDJSONStreamControls":
        """Create the controls for a streaming request.

        Call this only after request validation succeeds and immediately before
        returning :meth:`response`, so the admission hold is always released.

        Args:
            scope: ASGI scope of the streaming request.
            runtime: Service runtime supplying the configured bounds.

        Returns:
            Controls bound to the current request's admission work.
        """
        idle_seconds = getattr(
            runtime,
            "stream_write_idle_seconds",
            DEFAULT_SERVICE_STREAM_WRITE_IDLE_SECONDS,
        )
        max_events = getattr(
            runtime, "stream_max_events", DEFAULT_SERVICE_STREAM_MAX_EVENTS
        )
        return cls(
            write_guard=StreamWriteGuard.for_scope(scope, idle_seconds),
            max_events=max_events,
            work=current_request_work(),
        )

    def retain(self, future: asyncio.Future[Any]) -> None:
        """Keep admission held until abandoned model work settles.

        Args:
            future: Worker future the stream stopped awaiting while it runs.
        """
        if self._work is not None:
            self._work.track(future)

    def release(self) -> None:
        """Release the admission hold once; retained work keeps its own claim."""
        release, self._release_hold = self._release_hold, None
        if release is not None:
            release()

    def response(self, lines: AsyncIterator[str]) -> "BoundedNDJSONResponse":
        """Wrap NDJSON *lines* in a bounded streaming response.

        Args:
            lines: Async generator of complete NDJSON lines.

        Returns:
            A response that enforces the event cap and write-idle bound and
            releases admission when the body ends.
        """
        return BoundedNDJSONResponse(self._bounded_lines(lines), controls=self)

    async def _bounded_lines(
        self, lines: AsyncIterator[str]
    ) -> AsyncGenerator[str, None]:
        emitted = 0
        try:
            async for line in lines:
                if self.write_guard.expired:
                    return
                if emitted >= self.max_events:
                    _LOGGER.warning(
                        "NDJSON stream stopped: %s", STREAM_EVENT_LIMIT_CODE
                    )
                    yield _ndjson_error(
                        STREAM_EVENT_LIMIT_CODE,
                        "Stream exceeded the configured event limit",
                        {"max_events": self.max_events},
                    )
                    return
                emitted += 1
                yield line
                if self.write_guard.expired:
                    return
        finally:
            aclose = getattr(lines, "aclose", None)
            if aclose is not None:
                await aclose()


class BoundedNDJSONResponse(StreamingResponse):
    """NDJSON response that ends deterministically and releases admission.

    Starlette leaves an abandoned body generator to garbage collection; this
    response closes it explicitly so model references are released at the end
    of the response, then releases the stream's admission hold.
    """

    media_type = NDJSON_MEDIA_TYPE

    def __init__(
        self,
        content: AsyncGenerator[str, None],
        *,
        controls: NDJSONStreamControls,
    ) -> None:
        super().__init__(content, media_type=NDJSON_MEDIA_TYPE)
        self._content = content
        self._controls = controls

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Stream the body under the write-idle bound, then release admission."""
        guard = self._controls.write_guard
        if not guard.installed:
            send = guard.wrap(send)
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self._content.aclose()
            finally:
                self._controls.release()


def pii_extract_ndjson_stream(
    payload: PIIExtractStreamRequest,
    runtime: ServiceRuntime,
    *,
    retain_abandoned_work: Optional[Callable[[asyncio.Future[Any]], None]] = None,
) -> AsyncIterator[str]:
    """Return an async NDJSON iterator for ``/pii/extract/stream``.

    One request deadline (the profile ``timeout``) spans every extraction
    window. When it passes, the stream ends with one controlled ``timeout``
    error line and nothing after it; model errors end it with a value-free
    ``bad_request`` or ``internal_error`` line.

    Args:
        payload: Validated streaming extraction request.
        runtime: Shared service runtime.
        retain_abandoned_work: Optional callback that keeps admission held for
            a window still running after the stream stops waiting for it.

    Returns:
        An async generator of complete NDJSON lines.
    """
    return _iterate_pii_extract_ndjson(payload, runtime, retain_abandoned_work)


async def _iterate_pii_extract_ndjson(
    payload: PIIExtractStreamRequest,
    runtime: ServiceRuntime,
    retain_abandoned_work: Optional[Callable[[asyncio.Future[Any]], None]],
) -> AsyncGenerator[str, None]:
    timeout_seconds = float(getattr(runtime.config, "timeout", 0) or 0)
    events = runtime.stream_pii_extract(
        text=payload.text,
        model_name=payload.model_name,
        confidence_threshold=payload.confidence_threshold,
        use_smart_merging=payload.use_smart_merging,
        lang=payload.lang,
        normalize_accents=payload.normalize_accents,
        keep_alive=payload.keep_alive,
        chunk_size=payload.chunk_size,
        window_chars=payload.window_chars,
        tokenizer_context_chars=payload.tokenizer_context_chars,
        max_entity_chars=payload.max_entity_chars,
        timeout_seconds=timeout_seconds,
        retain_abandoned_work=retain_abandoned_work,
    )
    try:
        try:
            async for event in events:
                event_payload = event.to_dict(include_text=payload.include_text)
                event_payload["audit"] = event.to_audit_dict()
                yield json.dumps(event_payload, sort_keys=True) + "\n"
        finally:
            await events.aclose()
    except StreamDeadlineExceeded as exc:
        yield _ndjson_error(
            "timeout",
            f"Request exceeded configured timeout of {exc.timeout_seconds:g} seconds",
            {"timeout_seconds": exc.timeout_seconds},
        )
    except asyncio.CancelledError:
        raise
    except ValueError:
        yield _ndjson_error(
            "bad_request", "Streaming PII extraction request failed", None
        )
    except Exception:
        yield _ndjson_error("internal_error", "Internal server error", None)


class PIIDeidentifyStreamRequest(PIIDeidentifyRequest):
    """Request schema for ``/pii/deidentify/stream``."""

    chunk_size: int = Field(
        default=DEFAULT_DEIDENTIFY_STREAM_CHUNK_SIZE,
        ge=1,
        le=MAX_DEIDENTIFY_STREAM_CHUNK_SIZE,
    )


def deidentify_ndjson_stream(
    payload: PIIDeidentifyStreamRequest,
    runtime: ServiceRuntime,
) -> AsyncIterator[str]:
    """Return an async NDJSON iterator backed by core streaming de-identification.

    Policy resolution happens before the response starts so invalid policies keep
    the REST service's normal validation/error-envelope behavior.
    """

    policy_name = canonical_policy_name(payload.policy) if payload.policy else None
    policy_profile = load_policy(policy_name) if policy_name is not None else None
    keep_mapping = payload.keep_mapping or bool(
        policy_profile is not None and policy_profile.keep_mapping
    )
    return _iterate_deidentify_ndjson(
        payload,
        runtime,
        policy_name=policy_name,
        keep_mapping=keep_mapping,
    )


async def _iterate_deidentify_ndjson(
    payload: PIIDeidentifyStreamRequest,
    runtime: ServiceRuntime,
    *,
    policy_name: str | None,
    keep_mapping: bool,
) -> AsyncIterator[str]:
    model_key: str | None = None
    error: BaseException | None = None
    iterator: Iterator[Any] | None = None
    chunk_index = 0
    timeout_seconds = float(getattr(runtime.config, "timeout", 0) or 0)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds if timeout_seconds > 0 else None

    # Reversible placeholders must be numbered across the whole document, just
    # as they are for the single-shot endpoint. Keeping that opt-in request in
    # one core window preserves its exact mapping semantics.
    max_buffer = DEFAULT_DEIDENTIFY_STREAM_MAX_BUFFER
    if keep_mapping:
        max_buffer = max(max_buffer, len(payload.text))

    try:
        model_key = runtime.begin_model_request(payload.model_name)
        iterator = iter(
            deidentify_stream(
                _text_chunks(payload.text, payload.chunk_size),
                max_buffer=max_buffer,
                method=payload.method,
                model_name=payload.model_name,
                confidence_threshold=payload.confidence_threshold,
                keep_year=payload.keep_year,
                shift_dates=payload.shift_dates,
                date_shift_days=payload.date_shift_days,
                keep_mapping=keep_mapping,
                config=runtime.config,
                use_smart_merging=payload.use_smart_merging,
                lang=payload.lang,
                normalize_accents=payload.normalize_accents,
                use_safety_sweep=payload.use_safety_sweep,
                loader=runtime.get_loader(),
                policy=policy_name,
            )
        )

        while True:
            event = await _next_with_deadline(iterator, deadline)
            if event is _END_OF_STREAM:
                break
            if bool(getattr(event, "final", False)):
                final_payload: dict[str, Any] = {
                    "type": "final",
                    "audit": dict(getattr(event, "audit_record", None) or {}),
                    "spans": [
                        _span_to_dict(span)
                        for span in tuple(getattr(event, "spans", ()) or ())
                    ],
                }
                if keep_mapping:
                    mapping = dict(getattr(event, "mapping", None) or {})
                    if mapping:
                        final_payload["mapping"] = mapping
                yield _ndjson(final_payload)
                continue

            redacted_text = str(getattr(event, "redacted_text", ""))
            if redacted_text:
                yield _ndjson(
                    {
                        "type": "chunk",
                        "index": chunk_index,
                        "redacted_text": redacted_text,
                    }
                )
                chunk_index += 1
    except asyncio.TimeoutError as exc:
        error = exc
        yield _ndjson(
            {
                "type": "error",
                "error": {
                    "code": "timeout",
                    "message": (
                        "Request exceeded configured timeout of "
                        f"{timeout_seconds:g} seconds"
                    ),
                    "details": {"timeout_seconds": timeout_seconds},
                },
            }
        )
    except asyncio.CancelledError as exc:
        error = exc
        raise
    except ValueError as exc:
        error = exc
        yield _ndjson(
            {
                "type": "error",
                "error": {
                    "code": "bad_request",
                    "message": "Streaming de-identification request failed",
                    "details": None,
                },
            }
        )
    except Exception as exc:
        error = exc
        yield _ndjson(
            {
                "type": "error",
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error",
                    "details": None,
                },
            }
        )
    finally:
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        if model_key is not None:
            runtime.finish_model_request(model_key, payload.keep_alive, error)


async def _next_with_deadline(
    iterator: Iterator[Any],
    deadline: float | None,
) -> Any:
    operation = run_in_threadpool(_next_or_end, iterator)
    if deadline is None:
        return await operation

    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise asyncio.TimeoutError
    return await asyncio.wait_for(operation, timeout=remaining)


def _next_or_end(iterator: Iterator[Any]) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _END_OF_STREAM


def _text_chunks(text: str, chunk_size: int) -> Iterator[str]:
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


def _span_to_dict(span: Any) -> Mapping[str, Any]:
    to_dict = getattr(span, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if isinstance(span, Mapping):
        return dict(span)
    raise TypeError("Streaming span must be a mapping or expose to_dict()")


def _ndjson(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"


def _ndjson_error(
    code: str,
    message: str,
    details: Optional[Mapping[str, Any]],
) -> str:
    return _ndjson(
        {
            "type": "error",
            "error": {
                "code": code,
                "message": message,
                "details": dict(details) if details is not None else None,
            },
        }
    )


__all__ = [
    "DEFAULT_DEIDENTIFY_STREAM_CHUNK_SIZE",
    "STREAM_EVENT_LIMIT_CODE",
    "STREAM_ROUTE_PATHS",
    "STREAM_WRITE_IDLE_CODE",
    "BoundedNDJSONResponse",
    "NDJSONStreamControls",
    "PIIDeidentifyStreamRequest",
    "StreamWriteGuard",
    "StreamWriteIdleMiddleware",
    "deidentify_ndjson_stream",
    "pii_extract_ndjson_stream",
]
