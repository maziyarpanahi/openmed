"""``openmed service diagnose`` and ``verify-bundle`` command contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli.main import main
from openmed.service.diagnostics import CATEGORIES, verify_diagnostic_bundle
from tests.fixtures.diagnostic_state import (
    FrozenState,
    assert_no_sentinels,
    build_frozen_state,
    plant_sentinel_environment,
)


@pytest.fixture
def state(tmp_path: Path) -> FrozenState:
    return build_frozen_state(tmp_path / "state")


def _source_args(state: FrozenState) -> list[str]:
    return [
        "--jobs-store",
        str(state.jobs_store_path),
        "--journey-store",
        str(state.journey_store_root),
        "--registry-state",
        str(state.registry_state_path),
        "--audit-chain",
        str(state.audit_chain_path),
    ]


def _diagnose(state: FrozenState, *extra: str) -> int:
    return main(
        [
            "--config-path",
            str(state.config_path),
            "service",
            "diagnose",
            *_source_args(state),
            *extra,
        ]
    )


def test_diagnose_writes_a_verifiable_bundle_without_echoing_sources(
    state: FrozenState,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plant_sentinel_environment(monkeypatch)
    target = tmp_path / "bundle.json"

    assert _diagnose(state, "--out", str(target)) == 0

    captured = capsys.readouterr()
    assert "Diagnostic bundle written: 13 entries" in captured.out
    assert "Suggested recovery: rotate_credentials" in captured.out
    assert_no_sentinels(captured.out + captured.err, str(state.root), str(target))
    payload = target.read_bytes()
    assert_no_sentinels(payload.decode("ascii"), str(state.root))
    assert verify_diagnostic_bundle(payload).valid

    assert main(["service", "verify-bundle", str(target)]) == 0
    verified = capsys.readouterr().out
    assert "Diagnostic bundle valid: 13 entries" in verified
    assert str(target) not in verified


def test_diagnose_json_summary_and_stdout_bundle(
    state: FrozenState, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _diagnose(state, "--category", "jobs", "--category", "jobs", "--json") == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["ok"] is True
    assert envelope["command"] == "service diagnose"
    data = envelope["data"]
    assert data["written"] is False
    assert data["entry_count"] == 1
    assert verify_diagnostic_bundle(data["bundle"]).valid

    assert _diagnose(state, "--category", "runtime") == 0
    document = json.loads(capsys.readouterr().out)
    assert [item["category"] for item in document["entries"]] == ["runtime"]


def test_diagnose_refuses_to_overwrite_and_validates_limits(
    state: FrozenState, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "bundle.json"
    target.write_text("existing")

    assert _diagnose(state, "--out", str(target), "--json") == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["code"] == "output_exists"
    assert str(target) not in json.dumps(envelope)
    assert target.read_text() == "existing"

    assert _diagnose(state, "--out", str(target), "--overwrite") == 0
    assert verify_diagnostic_bundle(target.read_bytes()).valid
    capsys.readouterr()

    assert _diagnose(state, "--max-files", "100000", "--json") == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == (
        "limit_above_maximum"
    )
    assert _diagnose(state, "--max-bundle-bytes", "10") == 2
    assert "limit_below_minimum" in capsys.readouterr().err


def test_diagnose_applies_stricter_limits(
    state: FrozenState, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        _diagnose(state, "--category", "jobs", "--max-file-bytes", "8", "--json") == 0
    )
    bundle = json.loads(capsys.readouterr().out)["data"]["bundle"]
    assert bundle["entries"][0]["reason_codes"] == ["file_too_large"]
    assert bundle["header"]["limits"]["max_file_bytes"] == 8


def test_verify_bundle_reports_tampering_with_stable_codes(
    state: FrozenState, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "bundle.json"
    assert _diagnose(state, "--out", str(target)) == 0
    capsys.readouterr()
    document = json.loads(target.read_text())
    document["entries"].pop()
    target.write_text(json.dumps(document))

    assert main(["service", "verify-bundle", str(target), "--json"]) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["outcome"] == "check_failed"
    assert envelope["data"]["reason_code"] == "entry_missing"
    assert envelope["data"]["valid"] is False

    target.write_text("not json")
    assert main(["service", "verify-bundle", str(target)]) == 1
    assert "bundle_not_json" in capsys.readouterr().out

    missing = tmp_path / "missing.json"
    assert main(["service", "verify-bundle", str(missing), "--json"]) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["code"] == "bundle_unreadable"
    assert str(missing) not in json.dumps(envelope)


def test_category_choices_match_the_bundle_contract() -> None:
    from openmed.cli.main import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["service", "diagnose", "--category", "filesystem"])
    assert raised.value.code == 2
    args = parser.parse_args(["service", "diagnose", "--category", CATEGORIES[0]])
    assert args.categories == [CATEGORIES[0]]
