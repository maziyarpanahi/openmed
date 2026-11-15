"""Anthropic Messages API request adapter for the local privacy proxy.

Validates a ``POST /v1/messages`` request and redacts its ``system`` text and
``text`` content blocks through the same detector, tripwire and placeholder
map as the chat-completions route.  ``model``, ``role``, ``type``,
``stop_sequences`` and numeric fields are forwarded unchanged; other string
fields such as ``metadata`` are redacted like the chat route.  Non-text
content blocks are refused until their redaction rules exist, so no
unredacted image, document, tool or thinking payload reaches a transport.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..privacy_gateway import sha256_text
from .app import (
    PrivacyProxy,
    PrivacyProxyRequestError,
    RedactedChatRequest,
    _transport_content,
    _walk_json_strings,
)

MESSAGES_PATH = "/v1/messages"
MESSAGES_COMPATIBILITY_PATH = "/messages"
MESSAGES_API = "messages"
_ROLES = frozenset({"user", "assistant"})
# Forwarded verbatim; every other top-level string leaf is redacted.
_UNTOUCHED_FIELDS = frozenset({"model", "stop_sequences", "stream"})


def validate_messages_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of a valid Messages request or raise a coded 400."""
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise PrivacyProxyRequestError(reason_code="model_invalid")
    max_tokens = payload.get("max_tokens")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
    ):
        raise PrivacyProxyRequestError(reason_code="max_tokens_invalid")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise PrivacyProxyRequestError(reason_code="stream_invalid")
    if stream:
        # Messages streaming restoration is tracked separately; fail closed.
        raise PrivacyProxyRequestError(reason_code="stream_unsupported")

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise PrivacyProxyRequestError(reason_code="messages_required")
    for message in messages:
        if not isinstance(message, Mapping):
            raise PrivacyProxyRequestError(reason_code="invalid_message")
        if message.get("role") not in _ROLES:
            raise PrivacyProxyRequestError(reason_code="message_role_invalid")
        if "content" not in message:
            raise PrivacyProxyRequestError(reason_code="message_content_required")
        _validate_content(message["content"])

    if "system" in payload:
        _validate_content(payload["system"], field="system")
    return dict(payload)


def prepare_messages_request(
    proxy: PrivacyProxy,
    payload: Mapping[str, Any],
    *,
    request_id: str,
) -> RedactedChatRequest:
    """Validate and redact one Messages request for the injected transport."""
    normalized = validate_messages_payload(payload)
    placeholder_map: dict[str, str] = {}

    def redact(value: str, path: tuple[str, ...]) -> str:
        return proxy._redact_value(
            value,
            path=path,
            request_id=request_id,
            placeholder_map=placeholder_map,
        )

    redacted = {
        key: value
        if key in _UNTOUCHED_FIELDS or key in {"system", "messages"}
        else _walk_json_strings(value, redact, (key,))
        for key, value in normalized.items()
    }
    if "system" in normalized:
        redacted["system"] = _redact_content(normalized["system"], ("system",), redact)
    redacted["messages"] = [
        {
            **message,
            "content": _redact_content(
                message["content"], ("messages", str(index), "content"), redact
            ),
        }
        for index, message in enumerate(normalized["messages"])
    ]

    model = str(normalized["model"])
    digest = sha256_text(f"{request_id}:{model}")
    return RedactedChatRequest(
        payload=redacted,
        request_id=request_id,
        completion_id=f"msg_{digest[:24]}",
        created=int(digest[:8], 16),
        model=model,
        stream=False,
        placeholder_map=placeholder_map,
        api=MESSAGES_API,
    )


def messages_response(restored: Any, prepared: RedactedChatRequest) -> Any:
    """Return a restored Messages response, wrapping bare text transports."""
    if isinstance(restored, Mapping):
        return dict(restored)
    return {
        "id": prepared.completion_id,
        "type": "message",
        "role": "assistant",
        "model": prepared.model,
        "content": [{"type": "text", "text": _transport_content(restored)}],
        "stop_reason": "end_turn",
    }


def _validate_content(content: Any, *, field: str = "content") -> None:
    if isinstance(content, str):
        return
    if not isinstance(content, list) or not content:
        raise PrivacyProxyRequestError(reason_code=f"{field}_invalid")
    for block in content:
        if not isinstance(block, Mapping) or not isinstance(block.get("type"), str):
            raise PrivacyProxyRequestError(reason_code="content_block_invalid")
        if block["type"] != "text":
            raise PrivacyProxyRequestError(reason_code="content_block_unsupported")
        if not isinstance(block.get("text"), str):
            raise PrivacyProxyRequestError(reason_code="content_block_invalid")


def _redact_content(
    content: Any,
    path: tuple[str, ...],
    redact: Callable[[str, tuple[str, ...]], str],
) -> Any:
    if isinstance(content, str):
        return redact(content, path)
    return [
        {**block, "text": redact(block["text"], (*path, str(index), "text"))}
        for index, block in enumerate(content)
    ]


__all__ = [
    "MESSAGES_COMPATIBILITY_PATH",
    "MESSAGES_PATH",
    "messages_response",
    "prepare_messages_request",
    "validate_messages_payload",
]
