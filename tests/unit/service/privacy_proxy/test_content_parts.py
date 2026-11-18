"""Content-part refusal at the local privacy proxy.

Parts the text detector cannot inspect (images, audio, files or anything
unrecognized) must refuse the whole request before any extractor, tripwire or
transport call. All values are synthetic.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from openmed.service.privacy_proxy.app import (
    PrivacyProxy,
    PrivacyProxyLimits,
    PrivacyProxyRequestError,
    create_app,
)

SYNTHETIC_NAME = "Avery Example"
# Planted in media payloads and in malicious ``type`` values; it must never
# appear in a response, an exception or a log record.
SENTINEL = "SENTINEL5097QZ"
DATA_URL = f"data:image/png;base64,{SENTINEL}AAAA"

MEDIA_PARTS: dict[str, dict[str, Any]] = {
    "image_url": {"type": "image_url", "image_url": {"url": DATA_URL}},
    "input_audio": {
        "type": "input_audio",
        "input_audio": {"data": SENTINEL, "format": "wav"},
    },
    "file": {
        "type": "file",
        "file": {"file_data": SENTINEL, "filename": f"{SENTINEL}.pdf"},
    },
}


class Recorder:
    """Injected extractor, tripwire and transport that record every call."""

    def __init__(self) -> None:
        self.extracted: list[str] = []
        self.tripwire: list[str] = []
        self.forwarded: list[dict[str, Any]] = []

    def extractor(self, text: str, **_: Any) -> list[dict[str, Any]]:
        self.extracted.append(text)
        if SYNTHETIC_NAME not in text:
            return []
        start = text.index(SYNTHETIC_NAME)
        return [
            {
                "label": "NAME",
                "start": start,
                "end": start + len(SYNTHETIC_NAME),
                "confidence": 0.99,
            }
        ]

    def tripwire_extractor(self, text: str, **_: Any) -> list[Any]:
        self.tripwire.append(text)
        return []

    def transport(self, payload: dict[str, Any], **_: Any) -> Any:
        self.forwarded.append(payload)
        if payload.get("stream"):
            return iter(["Done."])
        return "Done."

    def assert_untouched(self) -> None:
        assert self.extracted == []
        assert self.tripwire == []
        assert self.forwarded == []


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


def _client(recorder: Recorder, **kwargs: Any) -> TestClient:
    return TestClient(
        create_app(
            transport=recorder.transport,
            extractor=recorder.extractor,
            tripwire_extractor=recorder.tripwire_extractor,
            **kwargs,
        ),
        base_url="http://127.0.0.1",
    )


def _request(*messages: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"model": "local-fixture", "messages": list(messages), **extra}


def _user(*parts: Any) -> dict[str, Any]:
    return {"role": "user", "content": list(parts)}


def _text(value: str = f"Summarize the note for {SYNTHETIC_NAME}.") -> dict[str, Any]:
    return {"type": "text", "text": value}


def _refusal(part_type: str, message_index: int) -> dict[str, Any]:
    return {
        "error": {
            "code": "privacy_proxy_invalid_request",
            "message": "Privacy proxy request failed",
            "details": {
                "reason": "unsupported_content_part",
                "part_type": part_type,
                "message_index": message_index,
            },
        }
    }


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/chat/completions"])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_media_parts_are_refused_before_redaction_or_transport(
    path: str, stream: bool, subtests: Any
) -> None:
    for part_type, part in MEDIA_PARTS.items():
        with subtests.test(part_type=part_type):
            recorder = Recorder()
            with _client(recorder) as client:
                response = client.post(
                    path, json=_request(_user(_text(), part), stream=stream)
                )

            assert response.status_code == 400
            assert response.headers["content-type"].startswith("application/json")
            assert response.json() == _refusal(part_type, 0)
            assert SENTINEL not in response.text
            assert SYNTHETIC_NAME not in response.text
            recorder.assert_untouched()


def test_part_in_a_later_message_refuses_the_whole_request(recorder: Recorder):
    request = _request(
        {"role": "system", "content": f"Patient is {SYNTHETIC_NAME}."},
        _user(_text()),
        {"role": "assistant", "content": "Noted."},
        _user(_text("And this one?"), MEDIA_PARTS["image_url"]),
    )
    with _client(recorder) as client:
        response = client.post("/v1/chat/completions", json=request)

    assert response.status_code == 400
    assert response.json() == _refusal("image_url", 3)
    # The earlier text was not redacted, which would have shown a partial pass.
    recorder.assert_untouched()


def test_first_refused_part_is_reported(recorder: Recorder):
    request = _request(
        _user(_text()),
        _user(MEDIA_PARTS["file"], MEDIA_PARTS["image_url"]),
        _user(MEDIA_PARTS["input_audio"]),
    )
    with _client(recorder) as client:
        response = client.post("/v1/chat/completions", json=request)

    assert response.json() == _refusal("file", 1)
    recorder.assert_untouched()


@pytest.mark.parametrize(
    ("part", "reported"),
    [
        ({"type": "video_url", "video_url": {"url": DATA_URL}}, "unknown"),
        ({"type": f"x-{SENTINEL}", "data": SENTINEL}, "unknown"),
        # Responses API text types are not chat-completions parts.
        ({"type": "input_text", "text": SENTINEL}, "unknown"),
        ({"type": "output_text", "text": SENTINEL}, "unknown"),
        ({"type": "Text", "text": SENTINEL}, "unknown"),
        ({"type": "", "text": SENTINEL}, "unknown"),
        ({"text": SENTINEL}, "invalid"),
        ({"type": None, "text": SENTINEL}, "invalid"),
        ({"type": 7, "text": SENTINEL}, "invalid"),
        ({"type": True, "text": SENTINEL}, "invalid"),
        ({"type": {"name": SENTINEL}, "text": SENTINEL}, "invalid"),
        ({"type": [SENTINEL], "text": SENTINEL}, "invalid"),
        (SENTINEL, "invalid"),
        ([SENTINEL], "invalid"),
        (None, "invalid"),
        ({"type": "refusal", "refusal": SENTINEL}, "refusal"),
    ],
    ids=[
        "unknown-type",
        "sentinel-type",
        "input_text",
        "output_text",
        "wrong-case",
        "empty-type",
        "missing-type",
        "null-type",
        "number-type",
        "bool-type",
        "object-type",
        "array-type",
        "string-part",
        "array-part",
        "null-part",
        "user-refusal",
    ],
)
def test_unknown_and_malformed_parts_are_refused_with_safe_markers(
    recorder: Recorder,
    caplog: pytest.LogCaptureFixture,
    part: Any,
    reported: str,
):
    caplog.set_level(logging.DEBUG)
    with _client(recorder) as client:
        response = client.post(
            "/v1/chat/completions", json=_request(_user(_text(), part))
        )

    assert response.status_code == 400
    assert response.json() == _refusal(reported, 0)
    assert SENTINEL not in response.text
    assert SENTINEL not in caplog.text
    recorder.assert_untouched()


@pytest.mark.parametrize(
    "content",
    [MEDIA_PARTS["image_url"], _text(), 7, True],
    ids=["media-object", "text-object", "number", "bool"],
)
def test_content_that_is_not_text_or_a_part_list_is_refused(
    recorder: Recorder, content: Any
):
    with _client(recorder) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_request({"role": "user", "content": content}),
        )

    assert response.status_code == 400
    assert response.json()["error"]["details"] == {"reason": "message_content_invalid"}
    assert SENTINEL not in response.text
    recorder.assert_untouched()


def test_text_parts_and_assistant_refusals_are_redacted_and_forwarded(
    recorder: Recorder,
):
    request = _request(
        {"role": "system", "content": [_text("You are a careful assistant.")]},
        {"role": "developer", "content": [_text("Answer briefly.")]},
        _user(_text(), _text("Thanks.")),
        {
            "role": "assistant",
            "content": [
                _text("Partial answer."),
                {"type": "refusal", "refusal": f"Cannot discuss {SYNTHETIC_NAME}."},
            ],
        },
        _user(_text("Try again.")),
    )
    original = copy.deepcopy(request)
    with _client(recorder) as client:
        response = client.post("/v1/chat/completions", json=request)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Done."
    assert request == original
    assert len(recorder.forwarded) == 1
    forwarded = recorder.forwarded[0]
    assert [
        [part["type"] for part in message["content"]]
        for message in forwarded["messages"]
    ] == [["text"], ["text"], ["text", "text"], ["text", "refusal"], ["text"]]
    assert SYNTHETIC_NAME not in json.dumps(forwarded)
    assert "OPENMED_PHI" in forwarded["messages"][2]["content"][0]["text"]
    assert "OPENMED_PHI" in forwarded["messages"][3]["content"][1]["refusal"]
    assert any(SYNTHETIC_NAME in text for text in recorder.extracted)


def test_tool_turns_and_type_keys_outside_content_are_not_parts(recorder: Recorder):
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"type": "image_url"}'},
    }
    request = _request(
        {"role": "user", "content": "Look up the record."},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "assistant", "tool_calls": [{**tool_call, "id": "call_2"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Found."},
        {"role": "tool", "tool_call_id": "call_2", "content": [_text("Found.")]},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"type": {"type": "string"}},
                    },
                },
            }
        ],
        response_format={"type": "json_object"},
    )
    with _client(recorder) as client:
        response = client.post("/v1/chat/completions", json=request)

    assert response.status_code == 200
    assert len(recorder.forwarded) == 1
    forwarded = recorder.forwarded[0]
    assert forwarded["messages"][1]["tool_calls"][0]["type"] == "function"
    assert forwarded["response_format"] == {"type": "json_object"}


def test_streaming_text_part_request_still_streams(recorder: Recorder):
    with _client(recorder) as client:
        response = client.post(
            "/chat/completions",
            json=_request(_user(_text()), stream=True),
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.rstrip().endswith("data: [DONE]")
    assert len(recorder.forwarded) == 1
    assert SYNTHETIC_NAME not in json.dumps(recorder.forwarded[0])


def test_existing_limit_and_request_id_errors_keep_precedence(recorder: Recorder):
    oversized = _request(*[_user(_text("Hi."))] * 3, _user(MEDIA_PARTS["file"]))
    with _client(recorder, limits=PrivacyProxyLimits(max_messages=2)) as client:
        limited = client.post("/v1/chat/completions", json=oversized)
    with _client(recorder) as client:
        bad_id = client.post(
            "/v1/chat/completions",
            json=_request(_user(MEDIA_PARTS["file"])),
            headers={"x-request-id": "bad id"},
        )

    assert limited.status_code == 413
    assert limited.json()["error"]["details"] == {"reason": "message_count_exceeded"}
    assert bad_id.status_code == 400
    assert bad_id.json()["error"]["details"] == {"reason": "request_id_invalid"}
    recorder.assert_untouched()


def test_direct_prepare_refuses_without_redacting_or_changing_input(
    recorder: Recorder,
):
    proxy = PrivacyProxy(
        transport=recorder.transport,
        extractor=recorder.extractor,
        tripwire_extractor=recorder.tripwire_extractor,
    )
    payload = _request(_user(_text()), _user(MEDIA_PARTS["input_audio"]))
    original = copy.deepcopy(payload)

    with pytest.raises(PrivacyProxyRequestError) as raised:
        proxy.prepare(payload)

    error = raised.value
    assert error.status_code == 400
    assert error.reason_code == "unsupported_content_part"
    assert (error.part_type, error.message_index) == ("input_audio", 1)
    assert SENTINEL not in f"{error!s} {error!r}"
    assert payload == original
    recorder.assert_untouched()
