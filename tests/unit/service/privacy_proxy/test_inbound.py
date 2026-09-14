"""Focused tests for local inbound placeholder restoration."""

from __future__ import annotations

from typing import Any

import pytest

from openmed.service.privacy_proxy.inbound import (
    DuplicateMappingError,
    DuplicatePlaceholderError,
    InboundRestorationPolicy,
    InboundRestorationState,
    InboundRestorationStore,
    MalformedPlaceholderError,
    RestorationLimitError,
    UnknownPlaceholderError,
    UnknownRequestStateError,
    UnsupportedResponseError,
    restore_structured_response,
    restore_text,
)

NAME_PLACEHOLDER = "<<OPENMED_PHI_NAME_DEADBEEF_000001>>"
PHONE_PLACEHOLDER = "<<OPENMED_PHI_PHONE_DEADBEEF_000002>>"
NAME_VALUE = "Synthetic Patient"
PHONE_VALUE = "555-0100"
MAPPING = {
    NAME_PLACEHOLDER: NAME_VALUE,
    PHONE_PLACEHOLDER: PHONE_VALUE,
}


def test_restore_text_is_exact_and_deterministic():
    response = f"Contact {NAME_PLACEHOLDER} at {PHONE_PLACEHOLDER}."

    first = restore_text(response, MAPPING)
    second = restore_text(response, MAPPING)

    assert first == f"Contact {NAME_VALUE} at {PHONE_VALUE}."
    assert second == first
    assert response == f"Contact {NAME_PLACEHOLDER} at {PHONE_PLACEHOLDER}."


def test_restore_structured_content_walks_nested_strings_and_keys():
    response: dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "content": f"Call {PHONE_PLACEHOLDER}.",
                },
            }
        ],
        NAME_PLACEHOLDER: {"value": "synthetic"},
    }
    mapping = {
        PHONE_PLACEHOLDER: PHONE_VALUE,
        NAME_PLACEHOLDER: NAME_VALUE,
    }

    restored = restore_structured_response(response, mapping)

    assert restored == {
        "choices": [
            {
                "message": {
                    "content": f"Call {PHONE_VALUE}.",
                },
            }
        ],
        NAME_VALUE: {"value": "synthetic"},
    }
    assert response["choices"][0]["message"]["content"] == (
        f"Call {PHONE_PLACEHOLDER}."
    )


@pytest.mark.parametrize(
    ("response", "error_type", "reason_code"),
    [
        (
            "Unknown <<OPENMED_PHI_NAME_AAAAAAAA_000001>>",
            UnknownPlaceholderError,
            "unknown_placeholder",
        ),
        (
            "Malformed <<OPENMED_PHI_NAME_BAD_1>>",
            MalformedPlaceholderError,
            "malformed_placeholder",
        ),
        (
            f"{NAME_PLACEHOLDER} and {NAME_PLACEHOLDER}",
            DuplicatePlaceholderError,
            "duplicate_placeholder",
        ),
    ],
)
def test_invalid_placeholders_fail_closed_without_mapping_values(
    response: str,
    error_type: type[ValueError],
    reason_code: str,
):
    with pytest.raises(error_type) as exc_info:
        restore_text(response, MAPPING)

    assert exc_info.value.reason_code == reason_code
    assert NAME_VALUE not in str(exc_info.value)
    assert PHONE_VALUE not in str(exc_info.value)


def test_duplicate_mapping_pairs_are_rejected_without_overwriting_state():
    with pytest.raises(DuplicateMappingError):
        InboundRestorationState.from_mapping(
            [(NAME_PLACEHOLDER, "first"), (NAME_PLACEHOLDER, "second")],
            request_id="request-1",
        )


def test_state_repr_and_safe_metadata_do_not_expose_mapping_values():
    state = InboundRestorationState.from_mapping(MAPPING, request_id="request-1")

    assert NAME_VALUE not in repr(state)
    assert PHONE_VALUE not in repr(state)
    assert NAME_VALUE not in str(state.to_safe_dict())
    assert state.placeholder_count == 2
    assert state.mapping_bytes > 0
    with pytest.raises(TypeError):
        state.mapping[NAME_PLACEHOLDER] = "replacement"


def test_duplicate_occurrences_can_be_allowed_explicitly():
    policy = InboundRestorationPolicy(reject_duplicates=False)

    restored = restore_text(
        f"{NAME_PLACEHOLDER} / {NAME_PLACEHOLDER}",
        {NAME_PLACEHOLDER: NAME_VALUE},
        policy=policy,
    )

    assert restored == f"{NAME_VALUE} / {NAME_VALUE}"


def test_store_is_request_scoped_bounded_and_releases_state_after_restore():
    store = InboundRestorationStore(
        max_requests=1,
        max_total_mapping_bytes=128,
    )
    store.put("request-1", {NAME_PLACEHOLDER: NAME_VALUE})

    with pytest.raises(RestorationLimitError):
        store.put("request-2", {PHONE_PLACEHOLDER: PHONE_VALUE})

    assert store.active_requests == 1
    assert store.restore("request-1", NAME_PLACEHOLDER) == NAME_VALUE
    assert store.active_requests == 0
    with pytest.raises(UnknownRequestStateError):
        store.get("request-1")


def test_response_limits_and_unsupported_content_fail_closed():
    policy = InboundRestorationPolicy(max_response_bytes=4)
    with pytest.raises(RestorationLimitError):
        restore_text("too long", MAPPING, policy=policy)

    with pytest.raises(UnsupportedResponseError):
        restore_structured_response({"content": object()}, MAPPING)
