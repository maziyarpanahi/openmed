"""OpenAI-compatible local endpoint with request-scoped PHI redaction.

The proxy accepts the common ``/v1/chat/completions`` request shape, redacts
text locally, and passes only the redacted JSON document to an injected
transport.  Placeholder mappings stay in the request call stack and are
never included in transport metadata, responses, or logs.  A second route
without the ``/v1`` prefix is provided for clients whose base URL already
contains the version path.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import AsyncIterable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..privacy_gateway import (
    DEFAULT_PRIVACY_GATEWAY_MIN_CONFIDENCE,
    PrivacyGatewayError,
    PrivacyGatewayPolicy,
    PrivacyGatewayTransportResponse,
    PrivacyTripwireViolation,
    coerce_gateway_entities,
    redact_text,
    reidentify_placeholders,
    safety_sweep_tripwire,
    sha256_text,
)

DEFAULT_MODEL = "openmed-local"
DEFAULT_REDACTION_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
CHAT_COMPLETIONS_COMPATIBILITY_PATH = "/chat/completions"
_REQUEST_ID_HEADER = "x-request-id"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_PLACEHOLDER_PREFIX = "<<OPENMED_PHI_"
_DONE = object()

EntityExtractor = Callable[..., Any]
ChatTransport = Callable[..., Any]


class PrivacyProxyError(RuntimeError):
    """Base class for safe, PHI-free proxy errors."""

    status_code = 502
    error_code = "privacy_proxy_error"
    reason_code = "privacy_proxy_error"

    def __init__(
        self,
        message: str = "Privacy proxy request failed",
        *,
        reason_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        if reason_code is not None:
            self.reason_code = reason_code


class PrivacyProxyRequestError(PrivacyProxyError):
    """Raised when a compatibility request is not valid."""

    status_code = 400
    error_code = "privacy_proxy_invalid_request"
    reason_code = "invalid_request"


class PrivacyProxyConfigurationError(PrivacyProxyError):
    """Raised when no local transport has been injected."""

    status_code = 503
    error_code = "privacy_proxy_not_configured"
    reason_code = "missing_transport"


class PrivacyProxyRedactionError(PrivacyProxyError):
    """Raised when local redaction or its outbound tripwire fails."""

    status_code = 400
    error_code = "privacy_proxy_redaction_failed"
    reason_code = "redaction_failed"


class PrivacyProxyTransportError(PrivacyProxyError):
    """Raised when the injected transport cannot produce a response."""

    status_code = 502
    error_code = "privacy_proxy_transport_failed"
    reason_code = "transport_failed"


class PrivacyProxyResponseError(PrivacyProxyError):
    """Raised when a transport response cannot be restored safely."""

    status_code = 502
    error_code = "privacy_proxy_response_rejected"
    reason_code = "response_rejected"


class _MissingTransport:
    """Fail-closed transport used by the import-safe module-level app."""

    def __call__(self, *_: Any, **__: Any) -> Any:
        raise PrivacyProxyConfigurationError()


@dataclass(frozen=True)
class RedactedChatRequest:
    """Request-scoped redacted payload and in-memory restoration state."""

    payload: Mapping[str, Any]
    request_id: str
    completion_id: str
    created: int
    model: str
    stream: bool
    placeholder_map: Mapping[str, str] = field(repr=False)


class _IncrementalRestorer:
    """Restore text chunks while retaining a placeholder split across chunks."""

    def __init__(self, placeholder_map: Mapping[str, str]) -> None:
        self._placeholder_map = placeholder_map
        self._buffer = ""

    def feed(self, text: str) -> str:
        """Return the safe portion of a non-final stream chunk."""
        self._buffer += text
        candidate_start = self._buffer.rfind(_PLACEHOLDER_PREFIX)
        if candidate_start >= 0 and ">>" not in self._buffer[candidate_start:]:
            safe = self._buffer[:candidate_start]
            self._buffer = self._buffer[candidate_start:]
            return reidentify_placeholders(safe, self._placeholder_map)

        hold = _placeholder_prefix_suffix_length(self._buffer)
        if hold:
            safe = self._buffer[:-hold]
            self._buffer = self._buffer[-hold:]
        else:
            safe = self._buffer
            self._buffer = ""
        return reidentify_placeholders(safe, self._placeholder_map)

    def finish(self) -> str:
        """Restore the final buffered text and reject unknown placeholders."""
        safe = reidentify_placeholders(self._buffer, self._placeholder_map)
        self._buffer = ""
        return safe


class _StreamState:
    """Mutable state used only while one streaming response is encoded."""

    def __init__(self, prepared: RedactedChatRequest) -> None:
        self.prepared = prepared
        self.raw_restorer = _IncrementalRestorer(prepared.placeholder_map)
        self.mapping_restorers: dict[tuple[str, ...], _IncrementalRestorer] = {}
        self.role_sent = False
        self.finish_sent = False
        self.done_sent = False
        self.emitted = False


class PrivacyProxy:
    """Redact chat requests before an injected transport and restore locally."""

    def __init__(
        self,
        *,
        transport: Optional[ChatTransport] = None,
        extractor: Optional[EntityExtractor] = None,
        tripwire_extractor: Optional[EntityExtractor] = None,
        policy: Optional[PrivacyGatewayPolicy] = None,
        redaction_model: str = DEFAULT_REDACTION_MODEL,
    ) -> None:
        self.transport: ChatTransport = transport or _MissingTransport()
        self.extractor = extractor or _default_extractor
        self.tripwire_extractor = tripwire_extractor or safety_sweep_tripwire
        self.policy = policy or PrivacyGatewayPolicy(
            min_confidence=DEFAULT_PRIVACY_GATEWAY_MIN_CONFIDENCE
        )
        self.redaction_model = redaction_model

    @property
    def configured(self) -> bool:
        """Return whether a transport other than the fail-closed default exists."""
        return not isinstance(self.transport, _MissingTransport)

    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        request_id_header: Optional[str] = None,
    ) -> RedactedChatRequest:
        """Validate and redact one JSON chat-completion request."""
        normalized = _validate_request_payload(payload)
        request_id = _stable_request_id(normalized, request_id_header)
        placeholder_map: dict[str, str] = {}

        def _redact(value: str, path: tuple[str, ...]) -> str:
            return self._redact_value(
                value,
                path=path,
                request_id=request_id,
                placeholder_map=placeholder_map,
            )

        redacted_payload = _walk_json_strings(normalized, _redact)
        model = str(normalized.get("model", DEFAULT_MODEL))
        completion_digest = sha256_text(f"{request_id}:{model}")
        return RedactedChatRequest(
            payload=redacted_payload,
            request_id=request_id,
            completion_id=f"chatcmpl-{completion_digest[:24]}",
            created=int(completion_digest[:8], 16),
            model=model,
            stream=bool(normalized.get("stream", False)),
            placeholder_map=placeholder_map,
        )

    async def call(
        self,
        prepared: RedactedChatRequest,
        *,
        transport: Optional[ChatTransport] = None,
    ) -> Any:
        """Call the configured transport with only the redacted request."""
        active_transport = transport or self.transport
        if isinstance(active_transport, _MissingTransport):
            raise PrivacyProxyConfigurationError()
        try:
            result = _invoke_transport(active_transport, prepared)
            if inspect.isawaitable(result):
                result = await result
            return result
        except PrivacyProxyError:
            raise
        except Exception:
            raise PrivacyProxyTransportError() from None

    def restore_completion(
        self,
        response: Any,
        prepared: RedactedChatRequest,
    ) -> dict[str, Any]:
        """Restore a non-streaming transport response into OpenAI shape."""
        try:
            restored = _restore_json_value(response, prepared.placeholder_map)
        except PrivacyProxyError:
            raise
        except Exception:
            raise PrivacyProxyResponseError() from None

        if isinstance(restored, Mapping) and _has_choices(restored):
            result = dict(restored)
            result.setdefault("id", prepared.completion_id)
            result.setdefault("object", "chat.completion")
            result.setdefault("created", prepared.created)
            result.setdefault("model", prepared.model)
            return result
        try:
            content = _transport_content(restored)
        except PrivacyProxyError:
            raise
        except Exception:
            raise PrivacyProxyResponseError() from None
        return _completion_response(
            content,
            completion_id=prepared.completion_id,
            created=prepared.created,
            model=prepared.model,
        )

    async def stream_events(
        self,
        response: Any,
        prepared: RedactedChatRequest,
    ) -> AsyncIterable[dict[str, Any] | object]:
        """Yield restored OpenAI-compatible chunks and a final ``[DONE]`` marker."""
        state = _StreamState(prepared)
        try:
            async for item in _iterate_response(response):
                for event in _stream_item_events(item, state):
                    if event is _DONE:
                        state.done_sent = True
                    else:
                        state.emitted = True
                        if _event_finishes(event):
                            state.finish_sent = True
                    yield event

            tail = state.raw_restorer.finish()
            if tail:
                yield _raw_stream_chunk(tail, state)
            for restorer in state.mapping_restorers.values():
                tail = restorer.finish()
                if tail:
                    yield _raw_stream_chunk(tail, state)
            if not state.finish_sent:
                state.finish_sent = True
                yield _finish_stream_chunk(state)
            if not state.done_sent:
                state.done_sent = True
                yield _DONE
        except PrivacyProxyError:
            raise
        except Exception:
            raise PrivacyProxyResponseError() from None

    def _redact_value(
        self,
        text: str,
        *,
        path: tuple[str, ...],
        request_id: str,
        placeholder_map: dict[str, str],
    ) -> str:
        """Redact one text leaf and merge its mapping into this request."""
        if not text:
            return text
        safe_path = ".".join(path) or "root"
        field_request_id = f"{request_id}:{sha256_text(safe_path)[:8]}"
        try:
            detected = _call_detector(
                self.extractor,
                text,
                model_name=self.redaction_model,
                confidence_threshold=self.policy.detector_confidence_floor,
                use_smart_merging=True,
                lang="en",
                normalize_accents=None,
            )
            entities = coerce_gateway_entities(detected, text)
            self.policy.enforce(entities)
            session = redact_text(text, entities, request_id=field_request_id)
            residual = _call_detector(
                self.tripwire_extractor,
                session.redacted_text,
                model_name=self.redaction_model,
                confidence_threshold=0.0,
                use_smart_merging=True,
                lang="en",
                normalize_accents=None,
            )
            residual_entities = coerce_gateway_entities(
                residual,
                session.redacted_text,
            )
            if residual_entities:
                raise PrivacyTripwireViolation(
                    "Outbound tripwire detected residual PHI",
                    reason_code="outbound_tripwire_detected",
                )
        except PrivacyGatewayError as exc:
            raise PrivacyProxyRedactionError(
                reason_code=getattr(exc, "reason_code", "redaction_failed")
            ) from None
        except Exception:
            raise PrivacyProxyRedactionError() from None

        for placeholder, original in session.placeholder_map.items():
            if placeholder in placeholder_map:
                raise PrivacyProxyRedactionError(reason_code="placeholder_collision")
            placeholder_map[placeholder] = original
        return session.redacted_text


def create_app(
    *,
    transport: Optional[ChatTransport] = None,
    extractor: Optional[EntityExtractor] = None,
    tripwire_extractor: Optional[EntityExtractor] = None,
    policy: Optional[PrivacyGatewayPolicy] = None,
    redaction_model: str = DEFAULT_REDACTION_MODEL,
) -> FastAPI:
    """Create an import-safe local privacy-proxy FastAPI application.

    ``transport`` receives a copied, JSON-compatible request mapping and may
    be synchronous or asynchronous.  For streaming requests it may return a
    synchronous or asynchronous iterable of text chunks or OpenAI-style
    response chunks.  Omitting it leaves the app healthy but makes completion
    requests fail closed without attempting any network call.
    """
    fastapi_app = FastAPI(
        title="OpenMed Local Privacy Proxy",
        description="OpenAI-compatible local redaction boundary.",
        version="1",
    )
    proxy = PrivacyProxy(
        transport=transport,
        extractor=extractor,
        tripwire_extractor=tripwire_extractor,
        policy=policy,
        redaction_model=redaction_model,
    )
    fastapi_app.state.privacy_proxy = proxy
    fastapi_app.state.privacy_proxy_transport = transport

    @fastapi_app.get("/health")
    async def health() -> dict[str, Any]:
        """Return local readiness without invoking a model or transport."""
        return {
            "status": "ok",
            "transport_configured": getattr(
                fastapi_app.state,
                "privacy_proxy_transport",
                None,
            )
            is not None,
        }

    async def chat_completions(request: Request) -> Any:
        try:
            try:
                payload = await request.json()
            except (TypeError, ValueError):
                raise PrivacyProxyRequestError(reason_code="invalid_json") from None
            if not isinstance(payload, Mapping):
                raise PrivacyProxyRequestError()
            active_proxy: PrivacyProxy = request.app.state.privacy_proxy
            active_transport = getattr(
                request.app.state,
                "privacy_proxy_transport",
                active_proxy.transport,
            )
            if active_transport is None:
                raise PrivacyProxyConfigurationError()
            prepared = active_proxy.prepare(
                payload,
                request_id_header=request.headers.get(_REQUEST_ID_HEADER),
            )
            result = await active_proxy.call(
                prepared,
                transport=active_transport,
            )
            if prepared.stream:
                return StreamingResponse(
                    _stream_body(active_proxy, result, prepared),
                    media_type="text/event-stream",
                    headers={
                        "cache-control": "no-cache",
                        "x-privacy-proxy": "local",
                        "x-request-id": prepared.request_id,
                    },
                )
            response = active_proxy.restore_completion(result, prepared)
            return JSONResponse(
                response,
                headers={
                    "x-privacy-proxy": "local",
                    "x-request-id": prepared.request_id,
                },
            )
        except PrivacyProxyError as exc:
            return _error_response(exc)
        except Exception:
            return _error_response(PrivacyProxyError())

    fastapi_app.add_api_route(
        CHAT_COMPLETIONS_PATH,
        chat_completions,
        methods=["POST"],
        name="chat_completions_v1",
    )
    fastapi_app.add_api_route(
        CHAT_COMPLETIONS_COMPATIBILITY_PATH,
        chat_completions,
        methods=["POST"],
        name="chat_completions_compatibility",
        include_in_schema=False,
    )
    return fastapi_app


async def _stream_body(
    proxy: PrivacyProxy,
    response: Any,
    prepared: RedactedChatRequest,
) -> AsyncIterable[bytes]:
    """Encode restored response chunks as server-sent events."""
    try:
        async for event in proxy.stream_events(response, prepared):
            if event is _DONE:
                yield b"data: [DONE]\n\n"
            else:
                yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode(
                    "utf-8"
                )
    except PrivacyProxyError as exc:
        error = _error_payload(exc)
        yield f"data: {json.dumps(error, separators=(',', ':'))}\n\n".encode("utf-8")


def _validate_request_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PrivacyProxyRequestError()
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise PrivacyProxyRequestError(reason_code="messages_required")
    for message in messages:
        if not isinstance(message, Mapping):
            raise PrivacyProxyRequestError(reason_code="invalid_message")
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise PrivacyProxyRequestError(reason_code="message_role_required")
        if "content" not in message and not any(
            key in message for key in ("tool_calls", "function_call")
        ):
            raise PrivacyProxyRequestError(reason_code="message_content_required")
    model = payload.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or not model.strip():
        raise PrivacyProxyRequestError(reason_code="model_invalid")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise PrivacyProxyRequestError(reason_code="stream_invalid")
    return dict(payload)


def _stable_request_id(
    payload: Mapping[str, Any],
    request_id_header: Optional[str],
) -> str:
    if request_id_header is not None:
        if not _REQUEST_ID_PATTERN.fullmatch(request_id_header):
            raise PrivacyProxyRequestError(reason_code="request_id_invalid")
        source = f"header:{request_id_header}"
    else:
        source = _canonical_json(payload)
    return sha256_text(source)[:32]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _walk_json_strings(
    value: Any,
    redact: Callable[[str, tuple[str, ...]], str],
    path: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            child_path = (*path, key_text)
            if key_text in {"model", "role", "type"}:
                result[key_text] = child
            else:
                result[key_text] = _walk_json_strings(child, redact, child_path)
        return result
    if isinstance(value, list):
        return [
            _walk_json_strings(child, redact, (*path, str(index)))
            for index, child in enumerate(value)
        ]
    if isinstance(value, str):
        return redact(value, path)
    return value


def _call_detector(detector: EntityExtractor, text: str, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(detector)
    except (TypeError, ValueError):
        return detector(text, **kwargs)
    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return detector(text, **kwargs)
    accepted = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return detector(text, **accepted)


def _default_extractor(text: str, **kwargs: Any) -> Any:
    import openmed

    return openmed.extract_pii(text, **kwargs)


def _invoke_transport(transport: ChatTransport, prepared: RedactedChatRequest) -> Any:
    target = transport
    if prepared.stream and callable(getattr(transport, "stream", None)):
        target = transport.stream
    elif not callable(target) and callable(getattr(transport, "complete", None)):
        target = transport.complete
    if not callable(target):
        raise PrivacyProxyTransportError(reason_code="transport_not_callable")
    metadata = {
        "request_id": prepared.request_id,
        "model": prepared.model,
        "stream": prepared.stream,
        "placeholder_count": len(prepared.placeholder_map),
    }
    kwargs = {
        "request_id": prepared.request_id,
        "stream": prepared.stream,
        "metadata": metadata,
    }
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return target(prepared.payload, **kwargs)
    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return target(prepared.payload, **kwargs)
    accepted = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return target(prepared.payload, **accepted)


async def _iterate_response(response: Any) -> AsyncIterable[Any]:
    if inspect.isawaitable(response):
        response = await response
    if hasattr(response, "__aiter__"):
        async for item in response:
            yield item
        return
    if isinstance(response, (str, bytes, Mapping, PrivacyGatewayTransportResponse)):
        yield response
        return
    try:
        iterator = iter(response)
    except TypeError:
        yield response
        return
    for item in iterator:
        if inspect.isawaitable(item):
            item = await item
        yield item


def _restore_json_value(value: Any, placeholder_map: Mapping[str, str]) -> Any:
    if isinstance(value, PrivacyGatewayTransportResponse):
        return _restore_json_value(value.text, placeholder_map)
    if isinstance(value, str):
        return reidentify_placeholders(value, placeholder_map)
    if isinstance(value, Mapping):
        return {
            str(key): _restore_json_value(child, placeholder_map)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_restore_json_value(child, placeholder_map) for child in value]
    if isinstance(value, tuple):
        return [_restore_json_value(child, placeholder_map) for child in value]
    return value


def _transport_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, bytes):
        return response.decode("utf-8", errors="replace")
    if isinstance(response, PrivacyGatewayTransportResponse):
        return response.text
    if isinstance(response, Mapping):
        for key in ("content", "text", "response", "completion", "output"):
            if isinstance(response.get(key), str):
                return str(response[key])
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping) and isinstance(
                    message.get("content"), str
                ):
                    return str(message["content"])
                if isinstance(first.get("text"), str):
                    return str(first["text"])
    raise PrivacyProxyTransportError(reason_code="invalid_transport_response")


def _completion_response(
    content: str,
    *,
    completion_id: str,
    created: int,
    model: str,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _has_choices(response: Mapping[str, Any]) -> bool:
    return isinstance(response.get("choices"), list)


def _stream_item_events(
    item: Any, state: _StreamState
) -> list[dict[str, Any] | object]:
    if isinstance(item, bytes):
        item = item.decode("utf-8", errors="replace")
    if isinstance(item, str):
        sse_events = _parse_sse_item(item)
        if sse_events is not None:
            events: list[dict[str, Any] | object] = []
            for parsed in sse_events:
                if parsed is _DONE:
                    events.append(_DONE)
                else:
                    events.extend(_stream_mapping_events(parsed, state))
            return events
        if item.strip() == "[DONE]":
            return [_DONE]
        restored = state.raw_restorer.feed(item)
        return [_raw_stream_chunk(restored, state)] if restored else []
    if isinstance(item, Mapping):
        return _stream_mapping_events(item, state)
    if isinstance(item, PrivacyGatewayTransportResponse):
        restored = state.raw_restorer.feed(item.text)
        return [_raw_stream_chunk(restored, state)] if restored else []
    raise PrivacyProxyResponseError(reason_code="invalid_stream_chunk")


def _stream_mapping_events(
    item: Mapping[str, Any],
    state: _StreamState,
) -> list[dict[str, Any] | object]:
    try:
        restored = _restore_stream_value(item, state)
    except PrivacyGatewayError:
        raise PrivacyProxyResponseError(reason_code="mangled_placeholder") from None
    if not isinstance(restored, Mapping):
        raise PrivacyProxyResponseError(reason_code="invalid_stream_chunk")
    if not _has_choices(restored):
        content = _transport_content(restored)
        return [_raw_stream_chunk(content, state)] if content else []
    result = dict(restored)
    result.setdefault("id", state.prepared.completion_id)
    result["object"] = "chat.completion.chunk"
    result.setdefault("created", state.prepared.created)
    result.setdefault("model", state.prepared.model)
    choices = result.get("choices")
    if not isinstance(choices, list):
        raise PrivacyProxyResponseError(reason_code="invalid_stream_chunk")
    normalized_choices: list[dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, Mapping):
            raise PrivacyProxyResponseError(reason_code="invalid_stream_chunk")
        normalized = dict(choice)
        normalized.setdefault("index", 0)
        if "delta" not in normalized:
            message = normalized.pop("message", {})
            normalized["delta"] = dict(message) if isinstance(message, Mapping) else {}
        normalized_choices.append(normalized)
        if normalized.get("finish_reason") is not None:
            state.finish_sent = True
    result["choices"] = normalized_choices
    state.role_sent = True
    return [result]


def _restore_stream_value(
    value: Any,
    state: _StreamState,
    path: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, str):
        if path and path[-1] == "content":
            restorer = state.mapping_restorers.setdefault(
                path,
                _IncrementalRestorer(state.prepared.placeholder_map),
            )
            return restorer.feed(value)
        return reidentify_placeholders(value, state.prepared.placeholder_map)
    if isinstance(value, Mapping):
        return {
            str(key): _restore_stream_value(child, state, (*path, str(key)))
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _restore_stream_value(child, state, (*path, str(index)))
            for index, child in enumerate(value)
        ]
    return value


def _raw_stream_chunk(text: str, state: _StreamState) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if not state.role_sent:
        delta["role"] = "assistant"
        state.role_sent = True
    if text:
        delta["content"] = text
    return {
        "id": state.prepared.completion_id,
        "object": "chat.completion.chunk",
        "created": state.prepared.created,
        "model": state.prepared.model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


def _finish_stream_chunk(state: _StreamState) -> dict[str, Any]:
    return {
        "id": state.prepared.completion_id,
        "object": "chat.completion.chunk",
        "created": state.prepared.created,
        "model": state.prepared.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }


def _event_finishes(event: dict[str, Any]) -> bool:
    choices = event.get("choices")
    return bool(
        isinstance(choices, list)
        and choices
        and isinstance(choices[0], Mapping)
        and choices[0].get("finish_reason") is not None
    )


def _parse_sse_item(item: str) -> Optional[list[dict[str, Any] | object]]:
    if not item.lstrip().startswith("data:"):
        return None
    events: list[dict[str, Any] | object] = []
    for line in item.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            events.append(_DONE)
            continue
        try:
            parsed = json.loads(data)
        except (TypeError, ValueError):
            events.append({"content": data})
        else:
            if isinstance(parsed, Mapping):
                events.append(dict(parsed))
            else:
                events.append({"content": str(parsed)})
    return events


def _placeholder_prefix_suffix_length(text: str) -> int:
    max_length = min(len(text), len(_PLACEHOLDER_PREFIX) - 1)
    for length in range(max_length, 0, -1):
        if text.endswith(_PLACEHOLDER_PREFIX[:length]):
            return length
    return 0


def _error_payload(exc: PrivacyProxyError) -> dict[str, Any]:
    return {
        "error": {
            "code": exc.error_code,
            "message": "Privacy proxy request failed",
            "details": {"reason": exc.reason_code},
        }
    }


def _error_response(exc: PrivacyProxyError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=_error_payload(exc))


app = create_app()


__all__ = [
    "CHAT_COMPLETIONS_COMPATIBILITY_PATH",
    "CHAT_COMPLETIONS_PATH",
    "DEFAULT_MODEL",
    "DEFAULT_REDACTION_MODEL",
    "PrivacyProxy",
    "PrivacyProxyConfigurationError",
    "PrivacyProxyError",
    "PrivacyProxyRedactionError",
    "PrivacyProxyRequestError",
    "PrivacyProxyResponseError",
    "PrivacyProxyTransportError",
    "RedactedChatRequest",
    "app",
    "create_app",
]
