"""Focused tests for the outbound message privacy boundary."""

from __future__ import annotations

import json
from typing import Any

import pytest

from openmed.service.privacy_proxy.outbound import (
    OutboundRequestPrivacyFilter,
    RedactionError,
    RedactionResult,
    RedactorRequiredError,
    ReplacementStateLimitError,
    RequestStateStore,
    UnsupportedContentTypeError,
    UnsupportedRequestBodyError,
)

SYNTHETIC_NAME = "Synthetic Patient"
SYNTHETIC_PHONE = "555-0101"
NAME_TOKEN = "<NAME>"
PHONE_TOKEN = "<PHONE>"


def synthetic_redactor(text: str) -> RedactionResult:
    replacements: dict[str, str] = {}
    redacted = text
    if SYNTHETIC_NAME in redacted:
        replacements[NAME_TOKEN] = SYNTHETIC_NAME
        redacted = redacted.replace(SYNTHETIC_NAME, NAME_TOKEN)
    if SYNTHETIC_PHONE in redacted:
        replacements[PHONE_TOKEN] = SYNTHETIC_PHONE
        redacted = redacted.replace(SYNTHETIC_PHONE, PHONE_TOKEN)
    return RedactionResult(redacted, replacements)


def test_transform_redacts_message_content_and_keeps_request_state() -> None:
    body = {
        "model": "local-test-model",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Use concise language."},
            {
                "role": "user",
                "content": f"Contact {SYNTHETIC_NAME} at {SYNTHETIC_PHONE}.",
            },
        ],
    }
    original_body = json.loads(json.dumps(body))
    privacy_filter = OutboundRequestPrivacyFilter(synthetic_redactor)

    prepared = privacy_filter.transform(body, request_id="request-907")

    assert body == original_body
    assert prepared.request_id == "request-907"
    assert prepared.body["messages"][1]["content"] == (
        f"Contact {NAME_TOKEN} at {PHONE_TOKEN}."
    )
    assert dict(prepared.replacements) == {
        NAME_TOKEN: SYNTHETIC_NAME,
        PHONE_TOKEN: SYNTHETIC_PHONE,
    }
    assert privacy_filter.get_state("request-907") == prepared.state
    assert prepared.state.to_metadata() == {
        "request_id": "request-907",
        "replacement_count": 2,
        "message_count": 2,
        "redacted_field_count": 1,
    }


def test_transform_is_deterministic_with_a_caller_owned_request_id() -> None:
    body = {"messages": [{"role": "user", "content": f"Hi {SYNTHETIC_NAME}."}]}

    first = OutboundRequestPrivacyFilter(synthetic_redactor).transform(
        body,
        request_id="stable-request",
    )
    second = OutboundRequestPrivacyFilter(synthetic_redactor).transform(
        body,
        request_id="stable-request",
    )

    assert first.body == second.body
    assert first.state.to_metadata() == second.state.to_metadata()


def test_json_text_and_typed_text_parts_are_supported() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Name: {SYNTHETIC_NAME}"},
                    {"type": "input_text", "text": "No sensitive value."},
                ],
            }
        ]
    }
    privacy_filter = OutboundRequestPrivacyFilter(synthetic_redactor)

    prepared = privacy_filter.transform(
        json.dumps(body),
        content_type="application/json; charset=utf-8",
        request_id="json-request",
    )

    assert isinstance(prepared.body, str)
    payload = json.loads(prepared.body)
    assert payload["messages"][0]["content"][0]["text"] == f"Name: {NAME_TOKEN}"
    assert prepared.content_type == "application/json"


def test_unsupported_content_type_fails_before_redactor_or_dispatch() -> None:
    calls: list[str] = []

    def redactor(text: str) -> str:
        calls.append(text)
        return text

    privacy_filter = OutboundRequestPrivacyFilter(redactor)

    with pytest.raises(UnsupportedContentTypeError) as exc_info:
        privacy_filter.transform(
            {"messages": [{"role": "user", "content": SYNTHETIC_NAME}]},
            content_type="text/plain",
            request_id="unsupported-content-type",
        )

    assert calls == []
    assert SYNTHETIC_NAME not in str(exc_info.value)
    assert len(privacy_filter.state_store) == 0


def test_unsupported_message_parts_fail_closed_without_storing_state() -> None:
    privacy_filter = OutboundRequestPrivacyFilter(synthetic_redactor)
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Name: {SYNTHETIC_NAME}"},
                    {"type": "image_url", "image_url": {"url": "local"}},
                ],
            }
        ]
    }

    with pytest.raises(UnsupportedRequestBodyError):
        privacy_filter.transform(body, request_id="unsupported-part")

    assert len(privacy_filter.state_store) == 0


def test_invalid_json_and_non_message_shapes_fail_closed() -> None:
    privacy_filter = OutboundRequestPrivacyFilter(synthetic_redactor)

    with pytest.raises(UnsupportedRequestBodyError):
        privacy_filter.transform("not json", request_id="invalid-json")
    with pytest.raises(UnsupportedRequestBodyError):
        privacy_filter.transform({"prompt": SYNTHETIC_NAME}, request_id="not-chat")

    assert len(privacy_filter.state_store) == 0


def test_redactor_failures_do_not_echo_raw_message_content() -> None:
    def failing_redactor(text: str) -> None:
        raise RuntimeError(f"unexpected redactor input: {text}")

    privacy_filter = OutboundRequestPrivacyFilter(failing_redactor)

    with pytest.raises(RedactionError) as exc_info:
        privacy_filter.transform(
            {"messages": [{"role": "user", "content": SYNTHETIC_NAME}]},
            request_id="redactor-error",
        )

    assert SYNTHETIC_NAME not in str(exc_info.value)
    assert len(privacy_filter.state_store) == 0


def test_incomplete_redaction_is_rejected_without_raw_error_details() -> None:
    def incomplete_redactor(text: str) -> RedactionResult:
        return RedactionResult(text, {NAME_TOKEN: SYNTHETIC_NAME})

    privacy_filter = OutboundRequestPrivacyFilter(incomplete_redactor)

    with pytest.raises(RedactionError) as exc_info:
        privacy_filter.transform(
            {"messages": [{"role": "user", "content": SYNTHETIC_NAME}]},
            request_id="incomplete-redaction",
        )

    assert SYNTHETIC_NAME not in str(exc_info.value)
    assert len(privacy_filter.state_store) == 0


def test_safe_representations_and_metadata_do_not_record_raw_values() -> None:
    privacy_filter = OutboundRequestPrivacyFilter(synthetic_redactor)
    prepared = privacy_filter.transform(
        {"messages": [{"role": "user", "content": SYNTHETIC_NAME}]},
        request_id="safe-representation",
    )

    rendered = repr(prepared) + repr(prepared.state) + repr(prepared.replacements)
    rendered += repr(prepared.to_metadata())

    assert SYNTHETIC_NAME not in rendered
    assert NAME_TOKEN in prepared.body["messages"][0]["content"]


def test_state_store_is_bounded_and_state_can_be_consumed_once() -> None:
    store = RequestStateStore(max_entries=1)
    privacy_filter = OutboundRequestPrivacyFilter(
        synthetic_redactor,
        state_store=store,
    )
    privacy_filter.transform(
        {"messages": [{"role": "user", "content": SYNTHETIC_NAME}]},
        request_id="first-request",
    )

    with pytest.raises(ReplacementStateLimitError):
        privacy_filter.transform(
            {"messages": [{"role": "user", "content": SYNTHETIC_PHONE}]},
            request_id="second-request",
        )

    state = privacy_filter.consume_state("first-request")
    assert state.request_id == "first-request"
    assert not store.contains("first-request")


def test_missing_redactor_is_rejected_before_request_processing() -> None:
    with pytest.raises(RedactorRequiredError):
        OutboundRequestPrivacyFilter()


@pytest.mark.parametrize("body", [b'{"messages": []}', {"messages": []}])
def test_empty_message_request_is_safe_and_preserves_encoding(body: Any) -> None:
    privacy_filter = OutboundRequestPrivacyFilter(synthetic_redactor)

    prepared = privacy_filter.transform(body, request_id="empty-request")

    if isinstance(body, bytes):
        assert isinstance(prepared.body, bytes)
        assert json.loads(prepared.body) == {"messages": []}
    else:
        assert prepared.body == {"messages": []}
    assert prepared.state.to_metadata()["replacement_count"] == 0
