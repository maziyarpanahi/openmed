"""Focused tests for the OpenAI-compatible local privacy proxy."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from openmed.service.privacy_proxy.app import create_app

SYNTHETIC_NAME = "Avery Example"
SYNTHETIC_PHONE = "555-0100"


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
