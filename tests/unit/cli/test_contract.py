"""Offline regression gate for the CLI machine-readable contract."""

from __future__ import annotations

import json
import socket

import pytest

from openmed.cli.contract import (
    CONTRACT_FIXTURES,
    OFFLINE_ERROR_CODE,
    PRIVACY_POLICY_ERROR_CODE,
    VALIDATION_ERROR_CODE,
    render_contract_fixture,
)


def test_contract_fixture_set_covers_required_outcomes() -> None:
    assert [fixture.name for fixture in CONTRACT_FIXTURES] == [
        "success",
        "validation",
        "offline",
        "privacy_policy",
    ]


@pytest.mark.parametrize(
    "fixture",
    CONTRACT_FIXTURES,
    ids=lambda fixture: fixture.name,
)
def test_contract_fixture_has_stable_envelope_and_exit_code(fixture) -> None:
    result = render_contract_fixture(fixture)

    assert result.exit_code == fixture.expected_exit_code
    assert result.payload == fixture.expected_payload
    assert set(result.payload) == {"ok", "command", "data"} or set(result.payload) == {
        "ok",
        "command",
        "error",
    }

    if fixture.expected_exit_code == 0:
        assert result.payload["ok"] is True
        assert set(result.payload["data"]) == {"status", "fixture"}
    else:
        assert result.payload["ok"] is False
        assert set(result.payload["error"]) == {"code", "message"}


def test_failure_categories_and_exit_codes_are_pinned() -> None:
    results = {
        fixture.name: render_contract_fixture(fixture) for fixture in CONTRACT_FIXTURES
    }

    assert results["validation"].exit_code == 2
    assert results["validation"].payload["error"]["code"] == VALIDATION_ERROR_CODE
    assert results["offline"].exit_code == 1
    assert results["offline"].payload["error"]["code"] == OFFLINE_ERROR_CODE
    assert results["privacy_policy"].exit_code == 1
    assert (
        results["privacy_policy"].payload["error"]["code"] == PRIVACY_POLICY_ERROR_CODE
    )


def test_contract_rendering_is_deterministic() -> None:
    for fixture in CONTRACT_FIXTURES:
        first = render_contract_fixture(fixture)
        second = render_contract_fixture(fixture)
        assert first.exit_code == second.exit_code
        assert first.json_text == second.json_text


def test_contract_payloads_are_value_free() -> None:
    raw_value = "synthetic-sensitive-value"
    forbidden_keys = {"input", "text", "raw", "path", "exception", "traceback"}

    for fixture in CONTRACT_FIXTURES:
        result = render_contract_fixture(fixture)
        serialized = json.dumps(result.payload, sort_keys=True)
        assert raw_value not in serialized

        keys = set(result.payload)
        if "data" in result.payload:
            keys.update(result.payload["data"])
        if "error" in result.payload:
            keys.update(result.payload["error"])
        assert not any(key.lower() in forbidden_keys for key in keys)


def test_contract_rendering_does_not_open_network_sockets(monkeypatch) -> None:
    def unexpected_network_call(*_args, **_kwargs):
        raise AssertionError("CLI contract fixtures must remain offline")

    monkeypatch.setattr(socket.socket, "connect", unexpected_network_call)
    monkeypatch.setattr(socket.socket, "connect_ex", unexpected_network_call)
    monkeypatch.setattr(socket, "create_connection", unexpected_network_call)

    for fixture in CONTRACT_FIXTURES:
        render_contract_fixture(fixture)
