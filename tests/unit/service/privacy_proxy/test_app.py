"""Focused tests for the OpenAI-compatible local privacy proxy."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from openmed.service.privacy_proxy.app import create_app

SYNTHETIC_NAME = "Avery Example"
SYNTHETIC_PHONE = "555-0100"
QUOTED_NAME = 'Avery "Ace" Example'


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
    if QUOTED_NAME in text:
        entities.append(_entity(text, QUOTED_NAME, "NAME"))
    if SYNTHETIC_NAME in text:
        entities.append(_entity(text, SYNTHETIC_NAME, "NAME"))
    if SYNTHETIC_PHONE in text:
        entities.append(_entity(text, SYNTHETIC_PHONE, "PHONE"))
    return entities


def _client(transport: Any) -> TestClient:
    return TestClient(
        create_app(
            transport=transport,
            extractor=_extractor,
            tripwire_extractor=lambda *_args, **_kwargs: [],
        ),
        base_url="http://127.0.0.1",
    )


def _tool_call_request(arguments: Any) -> dict[str, Any]:
    return {
        "model": "local-fixture",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": SYNTHETIC_NAME,
                        "type": "function",
                        "function": {
                            "name": SYNTHETIC_PHONE,
                            "arguments": arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": SYNTHETIC_PHONE,
                "content": "tool result",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": SYNTHETIC_PHONE,
                    "parameters": {"type": "object"},
                },
            }
        ],
    }


def test_chat_completion_redacts_payload_before_transport_and_restores_response():
    seen_payloads: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        seen_payloads.append(payload)
        redacted = payload["messages"][0]["content"]
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": redacted},
                    "finish_reason": "stop",
                }
            ]
        }

    prompt = f"Please call {SYNTHETIC_NAME} at {SYNTHETIC_PHONE}."
    with _client(transport) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-fixture",
                "messages": [{"role": "user", "content": prompt}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == prompt
    assert "mapping" not in response.json()
    assert seen_payloads
    forwarded = json.dumps(seen_payloads[0], sort_keys=True)
    assert SYNTHETIC_NAME not in forwarded
    assert SYNTHETIC_PHONE not in forwarded
    assert "OPENMED_PHI" in forwarded


def test_tool_call_arguments_are_redacted_as_json_with_identifiers_intact():
    seen_payloads: list[dict[str, Any]] = []
    original_arguments = {
        "name": QUOTED_NAME,
        "phone": SYNTHETIC_PHONE,
        "type": SYNTHETIC_NAME,
        "nested": {"contact": SYNTHETIC_NAME},
    }

    def transport(payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        seen_payloads.append(payload)
        tool_call = payload["messages"][0]["tool_calls"][0]
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_call["function"]["name"],
                                    "arguments": tool_call["function"]["arguments"],
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    with _client(transport) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_tool_call_request(json.dumps(original_arguments)),
        )

    assert response.status_code == 200
    forwarded = seen_payloads[0]
    forwarded_call = forwarded["messages"][0]["tool_calls"][0]
    forwarded_arguments_text = forwarded_call["function"]["arguments"]
    forwarded_arguments = json.loads(forwarded_arguments_text)

    assert forwarded_call["id"] == SYNTHETIC_NAME
    assert forwarded_call["function"]["name"] == SYNTHETIC_PHONE
    assert forwarded["messages"][1]["tool_call_id"] == SYNTHETIC_PHONE
    assert forwarded["tools"][0]["function"]["name"] == SYNTHETIC_PHONE
    assert forwarded_arguments.keys() == original_arguments.keys()
    assert forwarded_arguments["name"] != QUOTED_NAME
    assert forwarded_arguments["phone"] != SYNTHETIC_PHONE
    assert forwarded_arguments["type"] != SYNTHETIC_NAME
    assert forwarded_arguments["nested"]["contact"] != SYNTHETIC_NAME
    assert "OPENMED_PHI" in forwarded_arguments_text
    assert QUOTED_NAME not in forwarded_arguments_text
    assert SYNTHETIC_PHONE not in forwarded_arguments_text

    restored_call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert restored_call["id"] == SYNTHETIC_NAME
    assert restored_call["function"]["name"] == SYNTHETIC_PHONE
    assert json.loads(restored_call["function"]["arguments"]) == original_arguments


def test_legacy_function_call_preserves_name_and_redacts_json_arguments():
    seen_payloads: list[dict[str, Any]] = []
    original_arguments = {"patient": SYNTHETIC_NAME, "phone": SYNTHETIC_PHONE}

    def transport(payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        seen_payloads.append(payload)
        function_call = payload["messages"][0]["function_call"]
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": function_call,
                    },
                    "finish_reason": "function_call",
                }
            ]
        }

    with _client(transport) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-fixture",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": SYNTHETIC_PHONE,
                            "arguments": json.dumps(original_arguments),
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    forwarded_call = seen_payloads[0]["messages"][0]["function_call"]
    assert forwarded_call["name"] == SYNTHETIC_PHONE
    assert SYNTHETIC_NAME not in forwarded_call["arguments"]
    assert SYNTHETIC_PHONE not in forwarded_call["arguments"]

    restored_call = response.json()["choices"][0]["message"]["function_call"]
    assert restored_call["name"] == SYNTHETIC_PHONE
    assert json.loads(restored_call["arguments"]) == original_arguments


@pytest.mark.parametrize(
    "arguments",
    [
        f'{{"patient": "{SYNTHETIC_NAME}"',
        {"patient": SYNTHETIC_NAME},
    ],
    ids=["malformed-json", "not-encoded-json"],
)
def test_tool_call_arguments_must_be_json(arguments: Any):
    def transport(*_args: Any, **_kwargs: Any) -> str:
        pytest.fail("Invalid tool arguments reached the transport")

    with _client(transport) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_tool_call_request(arguments),
        )

    assert response.status_code == 400
    assert response.json()["error"]["details"]["reason"] == "tool_arguments_invalid"
    assert SYNTHETIC_NAME not in response.text
    assert SYNTHETIC_PHONE not in response.text
    assert "patient" not in response.text


def test_non_json_tool_arguments_in_response_fail_closed():
    invalid_response_arguments = f'{{"patient": "{SYNTHETIC_NAME}"'

    def transport(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": invalid_response_arguments,
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    with _client(transport) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_tool_call_request("{}"),
        )

    assert response.status_code == 502
    assert response.json()["error"]["details"]["reason"] == "tool_arguments_invalid"
    assert SYNTHETIC_NAME not in response.text
    assert "patient" not in response.text


def test_chat_completion_rejects_duplicated_response_placeholders():
    def transport(payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        token = payload["messages"][0]["content"]
        return {"content": f"{token} {token}"}

    with _client(transport) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-fixture",
                "messages": [{"role": "user", "content": SYNTHETIC_NAME}],
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["details"]["reason"] == "duplicate_placeholder"
    assert SYNTHETIC_NAME not in response.text


def test_streaming_completion_restores_split_placeholders_without_mapping():
    seen_payloads: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any], **_: Any):
        seen_payloads.append(payload)
        redacted = payload["messages"][0]["content"]
        token = redacted.split(" ", 1)[-1]
        yield "The contact is "
        yield token[: len(token) // 2]
        yield token[len(token) // 2 :]

    prompt = f"Contact {SYNTHETIC_NAME}"
    with _client(transport) as client:
        with client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": "local-fixture",
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            },
        ) as response:
            assert response.status_code == 200
            lines = list(response.iter_lines())
            stream_text = "\n".join(lines)
            events = [
                json.loads(line[6:]) for line in lines if line.startswith("data: {")
            ]
            assert stream_text.rstrip().endswith("data: [DONE]")

    content = "".join(
        event["choices"][0]["delta"].get("content", "") for event in events
    )
    assert SYNTHETIC_NAME in content
    assert "OPENMED_PHI" not in content
    assert "mapping" not in stream_text
    assert seen_payloads
    assert SYNTHETIC_NAME not in json.dumps(seen_payloads[0])


def test_streaming_completion_stops_before_restoring_a_duplicate_placeholder():
    def transport(payload: dict[str, Any], **_: Any):
        token = payload["messages"][0]["content"]
        yield token
        yield token

    with _client(transport) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-fixture",
                "stream": True,
                "messages": [{"role": "user", "content": SYNTHETIC_NAME}],
            },
        )

    assert response.status_code == 200
    assert response.text.count(SYNTHETIC_NAME) == 1
    assert '"reason":"duplicate_placeholder"' in response.text
    assert "data: [DONE]" not in response.text


def test_streaming_openai_chunks_restore_placeholder_split_across_events():
    def transport(payload: dict[str, Any], **_: Any):
        redacted = payload["messages"][0]["content"]
        token = redacted.split(" ", 1)[-1]
        yield {"choices": [{"delta": {"content": "Answer "}}]}
        yield {"choices": [{"delta": {"content": token[:8]}}]}
        yield {"choices": [{"delta": {"content": token[8:]}}]}

    with _client(transport) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-fixture",
                "stream": True,
                "messages": [{"role": "user", "content": SYNTHETIC_NAME}],
            },
        )

    assert response.status_code == 200
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    content = "".join(
        event["choices"][0]["delta"].get("content", "") for event in events
    )
    assert content == f"Answer {SYNTHETIC_NAME}"
    assert "OPENMED_PHI" not in response.text


def test_same_request_has_deterministic_redacted_payload_and_response_id():
    seen_payloads: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any], **_: Any) -> str:
        seen_payloads.append(payload)
        return payload["messages"][0]["content"]

    request = {
        "model": "local-fixture",
        "messages": [{"role": "user", "content": f"Hi {SYNTHETIC_NAME}"}],
    }
    with _client(transport) as client:
        first = client.post("/v1/chat/completions", json=request)
        second = client.post("/v1/chat/completions", json=request)

    assert first.status_code == second.status_code == 200
    assert seen_payloads[0] == seen_payloads[1]
    assert first.json()["id"] == second.json()["id"]


def test_unconfigured_proxy_is_local_and_fails_closed_without_echoing_input():
    prompt = f"Contact {SYNTHETIC_NAME}"
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        health = client.get("/health")
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-fixture",
                "messages": [{"role": "user", "content": prompt}],
            },
        )

    assert health.status_code == 200
    assert health.json()["transport_configured"] is False
    assert response.status_code == 503
    assert prompt not in response.text


@pytest.mark.parametrize(
    "content_type",
    ["application/json; charset=utf-8", "Application/JSON; Charset=UTF-8"],
)
def test_chat_completion_accepts_json_content_type_with_charset(content_type: str):
    seen_payloads: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any], **_: Any) -> str:
        seen_payloads.append(payload)
        return payload["messages"][0]["content"]

    prompt = f"Contact {SYNTHETIC_NAME} at {SYNTHETIC_PHONE}."
    body = json.dumps(
        {
            "model": "local-fixture",
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")

    with _client(transport) as client:
        response = client.post(
            "/v1/chat/completions",
            content=body,
            headers={"Content-Type": content_type},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == prompt
    assert len(seen_payloads) == 1


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/chat/completions"])
@pytest.mark.parametrize(
    "headers",
    [
        [("Content-Type", "text/plain")],
        [],
        [("Content-Type", "application/json"), ("Content-Type", "text/plain")],
    ],
    ids=["non-json", "missing", "conflicting"],
)
def test_chat_completion_rejects_unsupported_media_type_without_reading_body(
    path: str, headers: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
):
    def read_body(_request: Request) -> None:
        pytest.fail("Unsupported media type reached the request body reader")

    def transport(*_args: Any, **_kwargs: Any) -> str:
        pytest.fail("Unsupported media type reached the transport")

    monkeypatch.setattr(Request, "stream", read_body)
    prompt = f"Contact {SYNTHETIC_NAME} at {SYNTHETIC_PHONE}."
    body = json.dumps(
        {
            "model": "local-fixture",
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")

    with _client(transport) as client:
        response = client.post(path, content=body, headers=headers)

    assert response.status_code == 415
    assert response.json() == {
        "error": {
            "code": "privacy_proxy_invalid_request",
            "message": "Privacy proxy request failed",
            "details": {"reason": "unsupported_content_type"},
        }
    }
    assert SYNTHETIC_NAME not in response.text
    assert SYNTHETIC_PHONE not in response.text
    assert prompt not in response.text


def test_chat_completion_returns_safe_400_for_malformed_json():
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/v1/chat/completions",
            content=f'{{"messages": "{SYNTHETIC_NAME}',
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["details"]["reason"] == "invalid_json"
    assert SYNTHETIC_NAME not in response.text
