"""CLI tests for ``openmed migrate`` dry runs over synthetic records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from openmed.cli import migrate as migrate_cli
from openmed.cli.main import main
from openmed.clinical.journey_contracts import canonical_digest
from openmed.clinical.review_packet_migrations import migrate_review_packet
from openmed.clinical.review_transitions import ClinicalReviewPacket
from openmed.core.record_migrations import MigratorRegistry, default_registry

PACKET_ID = "reviewpacket_aaaaaaaaaaaaaaaa"
VALUE_CANARIES = (
    PACKET_ID,
    "conflict_aaaaaaaaaaaaaaaa",
    "fact_aaaaaaaaaaaaaaaa",
    "2026-09-21T10:00:00Z",
    "openmed.fact.reconciliation",
)


def _current_packet() -> dict[str, Any]:
    return ClinicalReviewPacket(
        packet_id=PACKET_ID,
        conflict_id="conflict_aaaaaaaaaaaaaaaa",
        fact_ids=("fact_aaaaaaaaaaaaaaaa", "fact_bbbbbbbbbbbbbbbb"),
        state="queued",
        priority="normal",
        created_at="2026-09-21T10:00:00Z",
        expires_at=None,
        policy_id="openmed.fact.reconciliation",
        policy_version="1.0.0",
        provenance_fingerprint=canonical_digest({"synthetic": "packet"}),
    ).to_dict()


def _legacy_packet() -> dict[str, Any]:
    packet = _current_packet()
    packet["schema_version"] = "1.0.0"
    for name in ("compatibility_policy", "extensions", "transition_ids"):
        packet.pop(name)
    return packet


def _upgrade_packet(record: dict[str, Any]) -> dict[str, Any]:
    result = migrate_review_packet(record, target_version="1.1.0")
    if not result.ok or result.value is None:
        raise ValueError(result.code)
    return result.value.packet.to_dict()


def _registry() -> MigratorRegistry:
    registry = MigratorRegistry()
    registry.register_schema(
        "clinical_review_packet",
        current_version="1.1.0",
        matcher=lambda record: "packet_id" in record and "fact_ids" in record,
    )
    registry.register_migrator(
        "clinical_review_packet",
        "1.0.0",
        "1.1.0",
        _upgrade_packet,
        lossless=True,
        fixture="clinical_review_packet/1.0.0-to-1.1.0",
    )
    return registry


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> MigratorRegistry:
    registry = _registry()
    monkeypatch.setattr(migrate_cli, "_registry", lambda: registry)
    return registry


def _write(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _snapshot(path: Path) -> tuple[str, int, int, int]:
    info = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest, info.st_size, info.st_mtime_ns, info.st_mode


def _run_json(
    capsys: pytest.CaptureFixture[str], *argv: str
) -> tuple[int, dict[str, Any], str]:
    code = main(["migrate", "check", *argv, "--json"])
    out = capsys.readouterr().out
    return code, json.loads(out), out


def _assert_value_free(text: str, tmp_path: Path) -> None:
    for canary in VALUE_CANARIES:
        assert canary not in text
    assert str(tmp_path) not in text


def test_review_packet_at_1_0_0_reports_field_paths_without_values(
    registry: MigratorRegistry,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _write(tmp_path / "packet.json", _legacy_packet())

    code, payload, out = _run_json(capsys, str(record))

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "migrate check"
    data = payload["data"]
    assert data["status"] == "migratable"
    assert data["dry_run"] is True
    assert data["written"] is False
    assert data["schema_id"] == "clinical_review_packet"
    assert (data["source_version"], data["target_version"]) == ("1.0.0", "1.1.0")
    assert data["lossy"] is False
    assert data["added"] == ["/compatibility_policy", "/extensions", "/transition_ids"]
    assert data["moved"] == []
    assert data["changed"] == ["/schema_version"]
    assert data["dropped"] == []
    assert data["counts"]["steps"] == 1
    assert data["source_digest"].startswith("sha256:")
    _assert_value_free(out, tmp_path)


def test_human_output_lists_paths_and_states_the_dry_run(
    registry: MigratorRegistry,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _write(tmp_path / "packet.json", _legacy_packet())

    assert main(["migrate", "check", str(record)]) == 0
    out = capsys.readouterr().out

    assert out.splitlines() == [
        "migrate check: migratable (clinical_review_packet 1.0.0 -> 1.1.0, lossless)",
        "  added: /compatibility_policy, /extensions, /transition_ids",
        "  moved: none",
        "  changed: /schema_version",
        "  dropped: none",
        "  step: 1.0.0 -> 1.1.0 (lossless)",
        "dry run: no file was written",
    ]
    _assert_value_free(out, tmp_path)


def test_current_record_reports_current_with_the_same_exit_code(
    registry: MigratorRegistry,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    legacy = _write(tmp_path / "legacy.json", _legacy_packet())
    current = _write(tmp_path / "current.json", _current_packet())

    legacy_code, _, _ = _run_json(capsys, str(legacy))
    code, payload, out = _run_json(capsys, str(current))

    assert code == legacy_code == 0
    assert payload["data"]["status"] == "current"
    assert payload["data"]["steps"] == []
    assert payload["data"]["source_version"] == "1.1.0"
    assert main(["migrate", "check", str(current)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "migrate check: current (clinical_review_packet 1.1.0)",
        "dry run: no file was written",
    ]
    _assert_value_free(out, tmp_path)


def test_unknown_schema_id_returns_a_coded_error_naming_only_id_and_version(
    registry: MigratorRegistry,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _write(tmp_path / "packet.json", _legacy_packet())

    code, payload, out = _run_json(
        capsys, str(record), "--schema", "unregistered_schema"
    )

    assert code == 2
    assert payload == {
        "ok": False,
        "command": "migrate check",
        "error": {
            "code": "migration_unknown_schema",
            "message": (
                "Record migration refused (unknown_schema) for "
                "unregistered_schema from 1.0.0."
            ),
        },
    }
    _assert_value_free(out, tmp_path)


def test_unidentified_record_names_only_its_version(
    registry: MigratorRegistry,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _write(
        tmp_path / "other.json",
        {"schema_version": "3.0.0", "note": "SYNTHETIC-PRIVATE-CANARY"},
    )

    code, payload, out = _run_json(capsys, str(record))

    assert code == 2
    assert payload["error"] == {
        "code": "migration_unknown_schema",
        "message": (
            "Record migration refused (unknown_schema) for unidentified schema "
            "from 3.0.0."
        ),
    }
    assert "CANARY" not in out


def test_record_with_no_migration_path_is_a_usage_error(
    registry: MigratorRegistry,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _write(tmp_path / "packet.json", _legacy_packet())

    code, payload, out = _run_json(capsys, str(record), "--to-version", "9.0.0")

    assert code == 2
    assert payload["error"] == {
        "code": "migration_no_path",
        "message": (
            "Record migration refused (no_path) for clinical_review_packet "
            "from 1.0.0 to 9.0.0."
        ),
    }
    _assert_value_free(out, tmp_path)


def test_failing_migrator_is_a_runtime_error_without_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = MigratorRegistry()
    registry.register_schema("synthetic_note", current_version="2")

    def explode(record: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(record["note"])

    registry.register_migrator(
        "synthetic_note", "1", "2", explode, lossless=True, fixture="x"
    )
    monkeypatch.setattr(migrate_cli, "_registry", lambda: registry)
    record = _write(
        tmp_path / "note.json",
        {"schema_version": "1", "note": "SYNTHETIC-PRIVATE-CANARY"},
    )

    code, payload, out = _run_json(capsys, str(record), "--schema", "synthetic_note")

    assert code == 1
    assert payload["error"]["code"] == "migration_migrator_failed"
    assert payload["error"]["message"] == (
        "Record migration refused (migrator_failed) for synthetic_note "
        "from 1 to 2 after RuntimeError."
    )
    assert "CANARY" not in out


def test_input_bytes_and_directory_are_unchanged_by_every_outcome(
    registry: MigratorRegistry,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    legacy = _write(tmp_path / "legacy.json", _legacy_packet())
    current = _write(tmp_path / "current.json", _current_packet())
    before = {path: _snapshot(path) for path in (legacy, current)}
    listing = sorted(tmp_path.iterdir())

    for argv in (
        [str(legacy)],
        [str(legacy), "--json"],
        [str(current), "--json"],
        [str(legacy), "--to-version", "9.0.0", "--json"],
        [str(legacy), "--schema", "unregistered_schema"],
    ):
        main(["migrate", "check", *argv])
        capsys.readouterr()

    assert {path: _snapshot(path) for path in (legacy, current)} == before
    assert sorted(tmp_path.iterdir()) == listing


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (b"{not json", "record_invalid"),
        (b"[1, 2, 3]", "record_invalid"),
        (b"\xff\xfe\x00", "record_invalid"),
    ],
)
def test_malformed_record_files_are_usage_errors(
    registry: MigratorRegistry,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: bytes,
    code: str,
) -> None:
    record = tmp_path / "record.json"
    record.write_bytes(content)

    exit_code, payload, out = _run_json(capsys, str(record))

    assert exit_code == 2
    assert payload["error"]["code"] == code
    assert str(tmp_path) not in out


def test_missing_directory_and_oversized_records_are_refused(
    registry: MigratorRegistry,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _write(tmp_path / "packet.json", _legacy_packet())

    missing_code, missing, missing_out = _run_json(capsys, str(tmp_path / "absent"))
    directory_code, directory, _ = _run_json(capsys, str(tmp_path))
    monkeypatch.setattr(migrate_cli, "MAX_RECORD_BYTES", 16)
    large_code, large, _ = _run_json(capsys, str(record))

    assert (missing_code, missing["error"]["code"]) == (2, "record_unreadable")
    assert (directory_code, directory["error"]["code"]) == (2, "record_unreadable")
    assert (large_code, large["error"]["code"]) == (2, "record_too_large")
    assert str(tmp_path) not in missing_out


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are POSIX-only")
def test_fifo_is_refused_without_blocking(
    registry: MigratorRegistry,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fifo = tmp_path / "record.json"
    os.mkfifo(fifo)

    code, payload, _ = _run_json(capsys, str(fifo))

    assert code == 2
    assert payload["error"]["code"] == "record_unreadable"


def test_command_uses_the_default_registry() -> None:
    assert migrate_cli._registry() is default_registry()


def test_migrate_without_a_subcommand_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["migrate"]) == 0
    assert "usage: openmed" in capsys.readouterr().out
