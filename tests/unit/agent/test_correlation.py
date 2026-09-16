"""Offline tests for opaque agent correlation identifiers."""

from __future__ import annotations

import json
import re
import traceback
from typing import Any

import pytest

from openmed.agent.correlation import (
    ACTION_ID_PREFIX,
    CORRELATION_SCHEMA_VERSION,
    CORRELATION_TOKEN_BYTES,
    RUN_ID_PREFIX,
    ActionCorrelation,
    ActionId,
    CorrelationIdError,
    RunId,
)

RUN_TOKEN = bytes(range(CORRELATION_TOKEN_BYTES))
ACTION_TOKEN = bytes(range(16, 16 + CORRELATION_TOKEN_BYTES))
PARENT_TOKEN = bytes(range(32, 32 + CORRELATION_TOKEN_BYTES))


def fixed_token(token: bytes):
    """Return a deterministic token source that validates the requested width."""

    def source(size: int) -> bytes:
        assert size == CORRELATION_TOKEN_BYTES
        return token

    return source


def test_generated_ids_use_fixed_prefixes_and_128_bit_tokens() -> None:
    run_id = RunId.generate(token_source=fixed_token(RUN_TOKEN))
    action_id = ActionId.generate(token_source=fixed_token(ACTION_TOKEN))

    assert run_id.serialize() == f"{RUN_ID_PREFIX}{RUN_TOKEN.hex()}"
    assert action_id.serialize() == f"{ACTION_ID_PREFIX}{ACTION_TOKEN.hex()}"
    assert len(run_id.value.removeprefix(RUN_ID_PREFIX)) == 32
    assert len(action_id.value.removeprefix(ACTION_ID_PREFIX)) == 32
    assert RunId.parse(run_id.serialize()) == run_id
    assert ActionId.parse(action_id.serialize()) == action_id
    assert str(run_id) == run_id.value
    assert str(action_id) == action_id.value


def test_default_generation_produces_canonical_opaque_ids() -> None:
    run_id = RunId.generate()
    action_id = ActionId.generate()

    assert re.fullmatch(r"run_[0-9a-f]{32}", run_id.value)
    assert re.fullmatch(r"act_[0-9a-f]{32}", action_id.value)
    assert "opaque" in repr(run_id)
    assert "opaque" in repr(action_id)
    assert run_id.value not in repr(run_id)
    assert action_id.value not in repr(action_id)


@pytest.mark.parametrize(
    ("identifier_type", "value", "expected_code"),
    [
        (RunId, f"{RUN_ID_PREFIX}{'A' * 32}", "invalid_identifier"),
        (RunId, f"{RUN_ID_PREFIX}{'0' * 31}", "invalid_identifier"),
        (RunId, f"{RUN_ID_PREFIX}{'0' * 33}", "invalid_identifier"),
        (RunId, f"{RUN_ID_PREFIX}{'g' * 32}", "invalid_identifier"),
        (RunId, f"{ACTION_ID_PREFIX}{'0' * 32}", "wrong_identifier_kind"),
        (ActionId, f"{RUN_ID_PREFIX}{'0' * 32}", "wrong_identifier_kind"),
        (ActionId, True, "invalid_identifier"),
        (ActionId, 1, "invalid_identifier"),
    ],
)
def test_malformed_or_wrong_kind_ids_fail_closed(
    identifier_type: type[RunId] | type[ActionId],
    value: Any,
    expected_code: str,
) -> None:
    with pytest.raises(CorrelationIdError) as caught:
        identifier_type.parse(value)

    assert caught.value.code == expected_code
    assert repr(value) not in str(caught.value)


@pytest.mark.parametrize(
    "token_source",
    [
        None,
        fixed_token(b"short"),
        fixed_token(b"x" * 17),
        lambda _: "not-bytes",
    ],
)
def test_invalid_injected_token_sources_fail_without_retaining_values(
    token_source: Any,
) -> None:
    source = "not-callable" if token_source is None else token_source

    with pytest.raises(CorrelationIdError) as caught:
        RunId.generate(token_source=source)

    assert caught.value.code == "invalid_token_source"
    assert "not-bytes" not in str(caught.value)
    assert "short" not in str(caught.value)


def test_token_source_exception_is_replaced_with_value_free_error() -> None:
    sentinel = "Patient Jane Roe /private/chart"

    def failing_source(_: int) -> bytes:
        raise RuntimeError(sentinel)

    with pytest.raises(CorrelationIdError) as caught:
        ActionId.generate(token_source=failing_source)

    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert caught.value.code == "invalid_token_source"
    assert sentinel not in rendered


def test_parent_child_correlation_round_trips_deterministically() -> None:
    correlation = ActionCorrelation(
        run_id=RunId.generate(token_source=fixed_token(RUN_TOKEN)),
        action_id=ActionId.generate(token_source=fixed_token(ACTION_TOKEN)),
        parent_action_id=ActionId.generate(token_source=fixed_token(PARENT_TOKEN)),
    )

    assert correlation.is_root_action is False
    assert list(correlation.to_dict()) == [
        "schema_version",
        "run_id",
        "action_id",
        "parent_action_id",
    ]
    assert ActionCorrelation.from_dict(correlation.to_dict()) == correlation
    assert ActionCorrelation.from_json(correlation.to_json()) == correlation
    assert correlation.to_json() == json.dumps(
        correlation.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )


def test_root_action_serializes_an_explicit_null_parent() -> None:
    correlation = ActionCorrelation(
        run_id=RunId.generate(token_source=fixed_token(RUN_TOKEN)),
        action_id=ActionId.generate(token_source=fixed_token(ACTION_TOKEN)),
    )

    assert correlation.is_root_action is True
    assert correlation.to_dict()["parent_action_id"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {
            "run_id": f"{ACTION_ID_PREFIX}{ACTION_TOKEN.hex()}",
            "action_id": f"{ACTION_ID_PREFIX}{RUN_TOKEN.hex()}",
        },
        {
            "run_id": f"{RUN_ID_PREFIX}{RUN_TOKEN.hex()}",
            "action_id": f"{RUN_ID_PREFIX}{ACTION_TOKEN.hex()}",
        },
        {
            "run_id": f"{RUN_ID_PREFIX}{RUN_TOKEN.hex()}",
            "action_id": f"{ACTION_ID_PREFIX}{ACTION_TOKEN.hex()}",
            "parent_action_id": f"{RUN_ID_PREFIX}{PARENT_TOKEN.hex()}",
        },
    ],
)
def test_parent_child_fields_reject_wrong_identifier_kinds(
    payload: dict[str, str],
) -> None:
    with pytest.raises(CorrelationIdError) as caught:
        ActionCorrelation.from_dict(payload)

    assert caught.value.code == "wrong_identifier_kind"
    expected_field = next(
        field_name
        for field_name in ("run_id", "action_id", "parent_action_id")
        if field_name in payload
        and payload[field_name].startswith(
            ACTION_ID_PREFIX if field_name == "run_id" else RUN_ID_PREFIX
        )
    )
    assert caught.value.field_name == expected_field


def test_action_cannot_parent_itself() -> None:
    action_id = ActionId.generate(token_source=fixed_token(ACTION_TOKEN))

    with pytest.raises(CorrelationIdError) as caught:
        ActionCorrelation(
            run_id=RunId.generate(token_source=fixed_token(RUN_TOKEN)),
            action_id=action_id,
            parent_action_id=action_id,
        )

    assert caught.value.code == "self_parent"
    assert action_id.value not in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"run_id": f"{RUN_ID_PREFIX}{RUN_TOKEN.hex()}"},
        [RUN_ID_PREFIX, ACTION_ID_PREFIX],
        "not-a-mapping",
    ],
)
def test_missing_and_non_mapping_payloads_fail_closed(payload: Any) -> None:
    with pytest.raises(CorrelationIdError):
        ActionCorrelation.from_dict(payload)


def test_unknown_fields_and_versions_fail_without_echoing_values() -> None:
    sentinel = "Patient John Doe bearer-token"
    payload = {
        "schema_version": CORRELATION_SCHEMA_VERSION,
        "run_id": f"{RUN_ID_PREFIX}{RUN_TOKEN.hex()}",
        "action_id": f"{ACTION_ID_PREFIX}{ACTION_TOKEN.hex()}",
        "prompt": sentinel,
    }

    with pytest.raises(CorrelationIdError) as caught:
        ActionCorrelation.from_dict(payload)

    assert caught.value.code == "unknown_field"
    assert sentinel not in str(caught.value)

    payload.pop("prompt")
    payload["schema_version"] = sentinel
    with pytest.raises(CorrelationIdError) as caught:
        ActionCorrelation.from_dict(payload)
    assert caught.value.code == "invalid_schema_version"
    assert sentinel not in str(caught.value)


@pytest.mark.parametrize("payload", ["{", b"\xff", 42])
def test_malformed_json_fails_closed(payload: Any) -> None:
    with pytest.raises(CorrelationIdError) as caught:
        ActionCorrelation.from_json(payload)

    assert caught.value.code == "malformed_json"


def test_duplicate_json_fields_fail_closed() -> None:
    run_id = f"{RUN_ID_PREFIX}{RUN_TOKEN.hex()}"
    action_id = f"{ACTION_ID_PREFIX}{ACTION_TOKEN.hex()}"
    payload = f'{{"run_id":"{run_id}","run_id":"{run_id}","action_id":"{action_id}"}}'

    with pytest.raises(CorrelationIdError) as caught:
        ActionCorrelation.from_json(payload)

    assert caught.value.code == "malformed_json"


def test_correlation_contract_is_exported_from_public_agent_api() -> None:
    import openmed.agent as agent

    assert agent.RunId is RunId
    assert agent.ActionId is ActionId
    assert agent.ActionCorrelation is ActionCorrelation
    assert agent.CorrelationIdError is CorrelationIdError
    assert agent.RUN_ID_PREFIX == RUN_ID_PREFIX
    assert agent.ACTION_ID_PREFIX == ACTION_ID_PREFIX
    assert agent.CORRELATION_TOKEN_BYTES == CORRELATION_TOKEN_BYTES
    assert agent.CORRELATION_SCHEMA_VERSION == CORRELATION_SCHEMA_VERSION
