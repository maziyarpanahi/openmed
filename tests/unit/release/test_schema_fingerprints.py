"""Bundled JSON Schema fingerprint check script."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from openmed.core.schemas import load_packaged_schema

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "check_schema_fingerprints.py"
DECISION_RESULT = "openmed/core/schemas/json/decision_result.schema.json"
SPAN = "openmed/core/schemas/json/span.schema.json"
SYNTHETIC_FIELD = "synthetic_removed_field"

spec = importlib.util.spec_from_file_location("check_schema_fingerprints", SCRIPT)
assert spec is not None
fingerprints = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = fingerprints
spec.loader.exec_module(fingerprints)

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION = re.compile(r"^(none|\d+|\d+\.\d+\.\d+|[a-z0-9_.-]+\.v\d+)$")
_FIELD = re.compile(r"^[A-Za-z0-9_$.\[\]-]+$")
_SUMMARY_KEYS = {"schemas", *fingerprints.STATUSES}


def _committed() -> dict[str, Any]:
    return fingerprints.load_snapshot(fingerprints.DEFAULT_SNAPSHOT)


def _write(tmp_path: Path, snapshot: dict[str, Any]) -> Path:
    path = tmp_path / "schema_fingerprints.json"
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    return path


def _with_removed_field(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Pretend the snapshot saw one more decision_result field than today."""

    changed = copy.deepcopy(snapshot)
    entry = changed[DECISION_RESULT]
    entry["required"] = sorted([*entry["required"], SYNTHETIC_FIELD])
    entry["properties"] = sorted([*entry["properties"], SYNTHETIC_FIELD])
    entry["fingerprint"] = "sha256:" + "0" * 64
    return changed


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = fingerprints.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_committed_snapshot_passes_the_check(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = _run(["--check"], capsys)

    assert code == fingerprints.EXIT_OK
    assert err == ""
    assert out.splitlines() == [
        f"pass schemas={len(_committed())} unchanged={len(_committed())} "
        "changed=0 breaking=0 missing_entry=0 removed=0"
    ]


def test_removed_field_without_version_bump_fails_the_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot_path = _write(tmp_path, _with_removed_field(_committed()))

    code, out, _ = _run(["--check", "--snapshot", str(snapshot_path)], capsys)

    assert code == fingerprints.EXIT_DRIFT
    lines = out.splitlines()
    assert lines[0].startswith(f"breaking {DECISION_RESULT} ")
    assert "schema_version=1.2.0 snapshot_version=1.2.0" in lines[0]
    assert lines[0].endswith(
        f"removed_required={SYNTHETIC_FIELD} removed_properties={SYNTHETIC_FIELD}"
    )
    assert lines[-1].startswith("fail ")
    assert "breaking=1" in lines[-1]


def test_report_contains_only_names_versions_fingerprints_and_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = _with_removed_field(_committed())
    del snapshot[SPAN]
    snapshot["openmed/core/schemas/json/synthetic_gone.schema.json"] = {
        "fingerprint": "sha256:" + "1" * 64,
        "properties": ["synthetic_gone_field"],
        "required": [],
        "schema_version": "1.0.0",
    }
    snapshot_path = _write(tmp_path, snapshot)

    code, out, err = _run(
        ["--check", "--verbose", "--snapshot", str(snapshot_path)], capsys
    )

    assert code == fingerprints.EXIT_DRIFT
    assert err == ""
    known = set(snapshot) | {SPAN}
    lines = out.splitlines()
    for line in lines[:-1]:
        status, schema, *tokens = line.split(" ")
        assert status in fingerprints.STATUSES
        assert schema in known
        for token in tokens:
            key, value = token.split("=", 1)
            if key in {"schema_version", "snapshot_version"}:
                assert _VERSION.fullmatch(value)
            elif key in {"fingerprint", "snapshot_fingerprint"}:
                assert value == "none" or _SHA256.fullmatch(value)
            else:
                assert key in {"removed_required", "removed_properties"}
                assert all(_FIELD.fullmatch(field) for field in value.split(","))
    result, *summary = lines[-1].split(" ")
    assert result == "fail"
    assert {token.split("=")[0] for token in summary} == _SUMMARY_KEYS
    assert load_packaged_schema(DECISION_RESULT)["title"] not in out
    assert str(tmp_path) not in out


def test_json_report_has_controlled_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot_path = _write(tmp_path, _with_removed_field(_committed()))

    code, out, _ = _run(["--check", "--json", "--snapshot", str(snapshot_path)], capsys)

    report = json.loads(out)
    assert code == fingerprints.EXIT_DRIFT
    assert set(report) == {"result", "schemas", "summary"}
    assert report["result"] == "fail"
    assert set(report["summary"]) == _SUMMARY_KEYS
    breaking = [item for item in report["schemas"] if item["status"] == "breaking"]
    assert breaking == [
        {
            "fingerprint": breaking[0]["fingerprint"],
            "removed_properties": [SYNTHETIC_FIELD],
            "removed_required": [SYNTHETIC_FIELD],
            "schema": DECISION_RESULT,
            "schema_version": "1.2.0",
            "snapshot_fingerprint": "sha256:" + "0" * 64,
            "snapshot_version": "1.2.0",
            "status": "breaking",
        }
    ]


def test_schema_without_entry_fails_the_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = _committed()
    del snapshot[SPAN]
    snapshot_path = _write(tmp_path, snapshot)

    code, out, _ = _run(["--check", "--snapshot", str(snapshot_path)], capsys)

    assert code == fingerprints.EXIT_DRIFT
    assert out.splitlines()[0].startswith(f"missing_entry {SPAN} ")
    assert "missing_entry=1" in out.splitlines()[-1]


def test_removed_schema_fails_the_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = _committed()
    gone = "openmed/core/schemas/json/synthetic_gone.schema.json"
    snapshot[gone] = {
        "fingerprint": "sha256:" + "1" * 64,
        "properties": ["synthetic_gone_field"],
        "required": ["synthetic_gone_field"],
        "schema_version": 1,
    }
    snapshot_path = _write(tmp_path, snapshot)

    code, out, _ = _run(["--check", "--snapshot", str(snapshot_path)], capsys)

    assert code == fingerprints.EXIT_DRIFT
    assert out.splitlines()[0] == (
        f"removed {gone} schema_version=none snapshot_version=1 fingerprint=none "
        f"snapshot_fingerprint=sha256:{'1' * 64} "
        "removed_required=synthetic_gone_field "
        "removed_properties=synthetic_gone_field"
    )


def test_additive_change_passes_but_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = _committed()
    entry = snapshot[DECISION_RESULT]
    entry["properties"] = entry["properties"][1:]
    entry["fingerprint"] = "sha256:" + "2" * 64
    snapshot_path = _write(tmp_path, snapshot)

    code, out, _ = _run(["--check", "--snapshot", str(snapshot_path)], capsys)

    assert code == fingerprints.EXIT_OK
    assert out.splitlines()[0].startswith(f"changed {DECISION_RESULT} ")
    assert out.splitlines()[-1].startswith("pass ")


def test_write_refuses_breaking_drift_unless_allowed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot_path = _write(tmp_path, _with_removed_field(_committed()))
    before = snapshot_path.read_bytes()

    code, _, err = _run(["--write", "--snapshot", str(snapshot_path)], capsys)

    assert code == fingerprints.EXIT_DRIFT
    assert err.strip() == "error: breaking_drift_not_written"
    assert snapshot_path.read_bytes() == before

    code, out, _ = _run(
        ["--write", "--allow-breaking", "--snapshot", str(snapshot_path)], capsys
    )

    assert code == fingerprints.EXIT_OK
    assert out.strip() == f"wrote schemas={len(_committed())}"
    assert snapshot_path.read_bytes() == fingerprints.DEFAULT_SNAPSHOT.read_bytes()


def test_write_creates_a_missing_snapshot_identical_to_the_committed_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot_path = tmp_path / "nested" / "schema_fingerprints.json"

    code, _, _ = _run(["--write", "--snapshot", str(snapshot_path)], capsys)

    assert code == fingerprints.EXIT_OK
    assert snapshot_path.read_bytes() == fingerprints.DEFAULT_SNAPSHOT.read_bytes()


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (None, "snapshot_missing"),
        ("{not json", "snapshot_unreadable"),
        ("[]", "snapshot_invalid"),
        ('{"x.schema.json": {"fingerprint": "sha256:0"}}', "snapshot_invalid"),
        (
            '{"x.schema.json": {"fingerprint": "sha256:0", "properties": [1],'
            ' "required": [], "schema_version": 1}}',
            "snapshot_invalid",
        ),
        (
            '{"x.schema.json": {"fingerprint": "sha256:0", "properties": [],'
            ' "required": [], "schema_version": true}}',
            "snapshot_invalid",
        ),
    ],
)
def test_unreadable_snapshot_exits_with_a_reason_code_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    payload: str | None,
    reason: str,
) -> None:
    snapshot_path = tmp_path / "private-dir" / "schema_fingerprints.json"
    if payload is not None:
        snapshot_path.parent.mkdir()
        snapshot_path.write_text(payload)

    code, out, err = _run(["--check", "--snapshot", str(snapshot_path)], capsys)

    assert code == fingerprints.EXIT_SNAPSHOT_ERROR
    assert out == ""
    assert err.strip() == f"error: {reason}"


def test_allow_breaking_requires_write(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = _run(["--check", "--allow-breaking"], capsys)

    assert code == fingerprints.EXIT_SNAPSHOT_ERROR
    assert err.strip() == "error: allow_breaking_requires_write"


def test_script_runs_directly_from_the_checkout() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == fingerprints.EXIT_OK, completed.stderr
    assert completed.stdout.startswith("pass schemas=")
