"""Offline regression gate for the CLI machine-readable contract."""

from __future__ import annotations

import json
import socket

import pytest

from openmed.cli import main as cli_main
from openmed.cli import main_module
from openmed.cli.contract import (
    CONTRACT_FIXTURES,
    OFFLINE_ERROR_CODE,
    OFFLINE_ERROR_MESSAGE,
    PRIVACY_POLICY_ERROR_CODE,
    PRIVACY_POLICY_ERROR_MESSAGE,
    VALIDATION_ERROR_CODE,
    VALIDATION_ERROR_MESSAGE,
    render_contract_fixture,
)
from openmed.core.offline import OfflineModeError


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


def test_models_pull_offline_failure_matches_contract(monkeypatch, capsys) -> None:
    import openmed.core.hf_hub as hf_hub

    rejected_model = "synthetic-sensitive-model-id"

    def fail_prefetch(*_args, **_kwargs):
        raise OfflineModeError(f"uncached model: {rejected_model}")

    monkeypatch.setattr(hf_hub, "prefetch_model", fail_prefetch)

    exit_code = cli_main(["models", "pull", rejected_model, "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload == {
        "ok": False,
        "command": "models pull",
        "error": {
            "code": OFFLINE_ERROR_CODE,
            "message": OFFLINE_ERROR_MESSAGE,
        },
    }
    assert captured.err == ""
    assert rejected_model not in captured.out


def test_models_pull_success_emits_one_value_free_json_document(
    monkeypatch, capsys
) -> None:
    import openmed.core.hf_hub as hf_hub
    from openmed.core.hf_hub import DownloadProgress

    rejected_model = "synthetic-sensitive-model-id"
    local_path = "/synthetic/private/cache/snapshot"

    def finish_prefetch(*_args, progress_callback, **_kwargs):
        progress_callback(
            DownloadProgress(
                filename="weights.safetensors",
                bytes_done=10,
                bytes_total=10,
                files_done=1,
                files_total=1,
            )
        )
        return local_path

    monkeypatch.setattr(hf_hub, "prefetch_model", finish_prefetch)

    exit_code = cli_main(["models", "pull", rejected_model, "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {
        "ok": True,
        "command": "models pull",
        "data": {"status": "ready"},
    }
    assert captured.err == ""
    assert "weights.safetensors" not in captured.out
    assert rejected_model not in captured.out
    assert local_path not in captured.out


def test_risk_discover_validation_failure_matches_contract(
    monkeypatch, capsys, tmp_path
) -> None:
    import openmed.structured as structured
    from openmed.structured import DiscoveryConfigurationError

    rejected_value = "synthetic-sensitive-column"

    def fail_scan(*_args, **_kwargs):
        raise DiscoveryConfigurationError(rejected_value)

    monkeypatch.setattr(main_module, "_preflight_structured_paths", lambda **_: None)
    monkeypatch.setattr(structured, "scan_table", fail_scan)

    exit_code = cli_main(
        [
            "risk",
            "discover",
            str(tmp_path / "input.csv"),
            "--output",
            str(tmp_path / "report.json"),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 2
    assert payload == {
        "ok": False,
        "command": "risk discover",
        "error": {
            "code": VALIDATION_ERROR_CODE,
            "message": VALIDATION_ERROR_MESSAGE,
        },
    }
    assert captured.err == ""
    assert rejected_value not in captured.out


def test_risk_assess_policy_failure_matches_value_free_contract(
    monkeypatch, capsys, tmp_path
) -> None:
    import openmed.risk as risk
    import openmed.structured as structured

    class FailedAssessment:
        meets_policy = False

        @staticmethod
        def to_json() -> str:
            return "{}"

    rejected_path = tmp_path / "synthetic-sensitive-output.json"
    monkeypatch.setattr(
        main_module, "_validated_release_policy", lambda _args: object()
    )
    monkeypatch.setattr(main_module, "_preflight_structured_paths", lambda **_: None)
    monkeypatch.setattr(
        main_module, "_temporary_sibling_path", lambda _path: tmp_path / "staged.json"
    )
    monkeypatch.setattr(main_module, "_write_safe_text", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_module, "_publish_release_outputs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(main_module, "_unlink_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(structured, "read_table", lambda _path: [])
    monkeypatch.setattr(
        risk, "assess_release", lambda _records, _policy: FailedAssessment()
    )

    exit_code = cli_main(
        [
            "risk",
            "assess",
            str(tmp_path / "input.csv"),
            "--output",
            str(rejected_path),
            "--k",
            "2",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload == {
        "ok": False,
        "command": "risk assess",
        "error": {
            "code": PRIVACY_POLICY_ERROR_CODE,
            "message": PRIVACY_POLICY_ERROR_MESSAGE,
        },
    }
    assert captured.err == ""
    assert str(rejected_path) not in captured.out
