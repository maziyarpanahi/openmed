"""Tests for the Anthropic Messages request shape at the privacy proxy."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from openmed.service.privacy_proxy.app import create_app

SYNTHETIC_NAME = "Avery Example"
SYNTHETIC_PHONE = "555-0100"
SYNTHETIC_MRN = "MRN-0042"
API_KEY_SENTINEL = "sk-ant-sentinel-key"
AUTH_SENTINEL = "Bearer sentinel-token"


def _entity(text: str, value: str, label: str) -> dict[str, Any]:
    start = text.index(value)
    return {
        "label": label,
        "start": start,
        "end": start + len(value),
        "confidence": 0.99,
    }


def _extractor(text: str, **_: Any) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for value, label in (
        (SYNTHETIC_NAME, "NAME"),
        (SYNTHETIC_PHONE, "PHONE"),
        (SYNTHETIC_MRN, "ID_NUM"),
    ):
        if value in text:
            entities.append(_entity(text, value, label))
    return entities


class RecordingTransport:
    def __init__(self, reply: Any = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.reply = reply

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if self.reply is not None:
            return self.reply
        payload = args[0]
        echoed = payload["messages"][0]["content"]
        text = echoed if isinstance(echoed, str) else echoed[0]["text"]
        return {
            "id": "msg_fixture",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
        }

    @property
    def payload(self) -> dict[str, Any]:
        return self.calls[0][0][0]


def _client(transport: Any) -> TestClient:
    return TestClient(
        create_app(
            transport=transport,
            extractor=_extractor,
            tripwire_extractor=lambda *_args, **_kwargs: [],
        ),
        base_url="http://127.0.0.1",
    )


def _request(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "claude-fixture",
        "max_tokens": 256,
        "system": f"The patient is {SYNTHETIC_NAME}.",
        "stop_sequences": ["\n\nHuman:"],
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Call {SYNTHETIC_NAME} at {SYNTHETIC_PHONE}.",
                    }
                ],
            },
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": f"Chart number {SYNTHETIC_MRN}."},
        ],
    }
    body.update(overrides)
    return body


@pytest.mark.parametrize("path", ["/v1/messages", "/messages"])
def test_messages_request_is_redacted_before_transport(path: str) -> None:
    transport = RecordingTransport()
    with _client(transport) as client:
        response = client.post(path, json=_request())

    assert response.status_code == 200
    forwarded = json.dumps(transport.payload, sort_keys=True)
    for value in (SYNTHETIC_NAME, SYNTHETIC_PHONE, SYNTHETIC_MRN):
        assert value not in forwarded
    assert "<<OPENMED_PHI_" in transport.payload["system"]
    assert "<<OPENMED_PHI_" in transport.payload["messages"][0]["content"][0]["text"]
    assert "<<OPENMED_PHI_" in transport.payload["messages"][2]["content"]


def test_messages_untouched_fields_and_api_metadata() -> None:
    transport = RecordingTransport()
    with _client(transport) as client:
        client.post("/v1/messages", json=_request())

    payload = transport.payload
    assert payload["model"] == "claude-fixture"
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0.2
    assert payload["stop_sequences"] == ["\n\nHuman:"]
    assert [m["role"] for m in payload["messages"]] == ["user", "assistant", "user"]
    assert payload["messages"][0]["content"][0]["type"] == "text"
    assert payload["messages"][1]["content"] == "Noted."
    assert transport.calls[0][1]["metadata"]["api"] == "messages"


def test_messages_response_is_restored_locally() -> None:
    transport = RecordingTransport()
    with _client(transport) as client:
        response = client.post("/v1/messages", json=_request())

    body = response.json()
    assert body["content"][0]["text"] == f"Call {SYNTHETIC_NAME} at {SYNTHETIC_PHONE}."
    assert response.headers["x-privacy-proxy"] == "local"


def test_bare_text_transport_reply_is_wrapped_as_a_message() -> None:
    transport = RecordingTransport(reply="All set.")
    with _client(transport) as client:
        body = client.post("/v1/messages", json=_request()).json()

    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "claude-fixture"
    assert body["content"] == [{"type": "text", "text": "All set."}]


def test_inbound_credentials_never_reach_the_transport() -> None:
    transport = RecordingTransport()
    with _client(transport) as client:
        response = client.post(
            "/v1/messages",
            json=_request(),
            headers={"x-api-key": API_KEY_SENTINEL, "Authorization": AUTH_SENTINEL},
        )

    assert response.status_code == 200
    seen = repr(transport.calls)
    assert API_KEY_SENTINEL not in seen
    assert "sentinel-token" not in seen


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"max_tokens": None}, "max_tokens_invalid"),
        ({"max_tokens": 0}, "max_tokens_invalid"),
        ({"max_tokens": True}, "max_tokens_invalid"),
        ({"max_tokens": "256"}, "max_tokens_invalid"),
        ({"model": ""}, "model_invalid"),
        ({"messages": f"Hi {SYNTHETIC_NAME}"}, "messages_required"),
        ({"messages": []}, "messages_required"),
        (
            {"messages": [{"role": "system", "content": f"Hi {SYNTHETIC_NAME}"}]},
            "message_role_invalid",
        ),
        ({"messages": [{"role": "user"}]}, "message_content_required"),
        ({"system": 42}, "system_invalid"),
        ({"system": [{"type": "text", "text": 7}]}, "content_block_invalid"),
        ({"stream": "yes"}, "stream_invalid"),
        ({"stream": True}, "stream_unsupported"),
    ],
)
def test_invalid_messages_requests_return_coded_400_without_text(
    overrides: dict[str, Any], reason: str
) -> None:
    body = _request(**overrides)
    if overrides.get("max_tokens", 1) is None:
        body.pop("max_tokens")
    transport = RecordingTransport()
    with _client(transport) as client:
        response = client.post("/v1/messages", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["details"]["reason"] == reason
    for value in (SYNTHETIC_NAME, SYNTHETIC_PHONE, SYNTHETIC_MRN):
        assert value not in response.text
    assert transport.calls == []


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
        {"type": "tool_result", "tool_use_id": "t1", "content": SYNTHETIC_NAME},
        {"type": "document", "source": {"type": "text", "data": SYNTHETIC_NAME}},
    ],
)
def test_non_text_content_blocks_fail_closed(block: dict[str, Any]) -> None:
    transport = RecordingTransport()
    messages = [{"role": "user", "content": [block]}]
    with _client(transport) as client:
        response = client.post("/v1/messages", json=_request(messages=messages))

    assert response.status_code == 400
    assert response.json()["error"]["details"]["reason"] == "content_block_unsupported"
    assert SYNTHETIC_NAME not in response.text
    assert transport.calls == []


def test_messages_without_transport_fail_closed() -> None:
    with TestClient(create_app(extractor=_extractor)) as client:
        response = client.post("/v1/messages", json=_request())

    assert response.status_code == 503
    assert response.json()["error"]["details"]["reason"] == "missing_transport"


def test_other_top_level_string_fields_are_redacted() -> None:
    transport = RecordingTransport()
    body = _request(metadata={"user_id": f"{SYNTHETIC_NAME} {SYNTHETIC_MRN}"})
    with _client(transport) as client:
        client.post("/v1/messages", json=body)

    assert SYNTHETIC_NAME not in json.dumps(transport.payload)
    assert transport.payload["metadata"]["user_id"].startswith("<<OPENMED_PHI_")
