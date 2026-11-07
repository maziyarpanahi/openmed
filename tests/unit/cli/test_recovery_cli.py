"""CLI tests for ``openmed recovery`` with synthetic state and key files."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from openmed.cli.main import main
from tests.fixtures.service_recovery import (
    BACKUP_KEY,
    JOURNEY_KEY,
    SENTINEL,
    SyntheticServiceState,
    build_service_state,
    logical_snapshot,
)

pytest.importorskip("cryptography")


@pytest.fixture(autouse=True)
def _development_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)


@pytest.fixture
def state(tmp_path: Path) -> Iterator[SyntheticServiceState]:
    service = build_service_state(tmp_path / "state")
    yield service
    service.close()


@pytest.fixture
def keys(tmp_path: Path) -> dict[str, str]:
    directory = tmp_path / "keys"
    directory.mkdir()
    (directory / "backup.key").write_bytes(BACKUP_KEY)
    (directory / "journey.hex").write_text(JOURNEY_KEY.hex() + "\n", encoding="ascii")
    return {
        "backup": f"backup-key-1={directory / 'backup.key'}",
        "journey": f"journey-key-1={directory / 'journey.hex'}",
    }


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = main([*argv, "--json"])
    output = capsys.readouterr().out
    return code, json.loads(output)


def test_inventory_prints_categories_only(
    state: SyntheticServiceState,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, payload = _run(
        capsys, "recovery", "inventory", "--state-root", str(state.root)
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "recovery inventory"
    categories = [entry["category"] for entry in payload["data"]["entries"]]
    assert "journey_metadata" in categories and "encryption_keys" in categories
    encoded = json.dumps(payload)
    assert str(tmp_path) not in encoded
    assert SENTINEL not in encoded

    assert main(["recovery", "inventory", "--state-root", str(state.root)]) == 0
    human = capsys.readouterr().out
    assert "journey_metadata: authoritative" in human
    assert str(tmp_path) not in human


@pytest.mark.skipif(sys.platform == "win32", reason="activation is POSIX-only")
def test_backup_verify_restore_activate_and_discard(
    state: SyntheticServiceState,
    keys: dict[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    point = tmp_path / "rp"
    code, backup = _run(
        capsys,
        "recovery",
        "backup",
        str(point),
        "--state-root",
        str(state.root),
        "--key",
        keys["backup"],
        "--writers-stopped",
        "--retention-days",
        "30",
    )
    assert code == 0, backup
    assert backup["outcome"] == "completed"
    assert backup["data"]["kind"] == "full"

    code, verify = _run(
        capsys, "recovery", "verify", str(point), "--key", keys["backup"]
    )
    assert code == 0, verify
    assert verify["data"]["content_digest"] == backup["data"]["content_digest"]

    staged = tmp_path / "generation-1"
    live = tmp_path / "live"
    code, restore = _run(
        capsys,
        "recovery",
        "restore",
        str(point),
        str(staged),
        "--key",
        keys["backup"],
        "--live",
        str(live),
    )
    assert code == 0, restore

    code, activate = _run(
        capsys,
        "recovery",
        "activate",
        str(staged),
        str(live),
        "--key",
        keys["backup"],
        "--journey-key",
        keys["journey"],
    )
    assert code == 0, activate
    assert live.resolve() == staged.resolve()
    assert logical_snapshot(live, state.encryption) == logical_snapshot(
        state.root, state.encryption
    )

    code, discard = _run(
        capsys, "recovery", "discard", str(staged), "--live", str(live)
    )
    assert code == 1
    assert discard["outcome"] == "refused"
    assert discard["data"]["reason"] == "staged_is_live"

    for payload in (backup, verify, restore, activate, discard):
        encoded = json.dumps(payload)
        assert str(tmp_path) not in encoded
        assert SENTINEL not in encoded
        assert "backup-key-1" not in encoded


def test_incremental_backup_and_human_output(
    state: SyntheticServiceState,
    keys: dict[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = [
        "--state-root",
        str(state.root),
        "--key",
        keys["backup"],
        "--writers-stopped",
    ]
    assert main(["recovery", "backup", str(tmp_path / "rp-0"), *base]) == 0
    human = capsys.readouterr().out
    assert human.startswith("recovery backup completed")
    assert str(tmp_path) not in human
    state.add_job()

    code, payload = _run(
        capsys,
        "recovery",
        "backup",
        str(tmp_path / "rp-1"),
        *base,
        "--parent",
        str(tmp_path / "rp-0"),
    )

    assert code == 0, payload
    assert payload["data"]["kind"] == "incremental"
    assert payload["data"]["counts"]["objects_stored"] == 1
    code, refused = _run(
        capsys, "recovery", "verify", str(tmp_path / "rp-1"), "--key", keys["backup"]
    )
    assert code == 1
    assert refused["data"]["reason"] == "chain_incomplete"
    code, verified = _run(
        capsys,
        "recovery",
        "verify",
        str(tmp_path / "rp-1"),
        "--key",
        keys["backup"],
        "--ancestor",
        str(tmp_path / "rp-0"),
    )
    assert code == 0, verified


def test_backup_without_writer_attestation_is_refused(
    state: SyntheticServiceState,
    keys: dict[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, payload = _run(
        capsys,
        "recovery",
        "backup",
        str(tmp_path / "rp"),
        "--state-root",
        str(state.root),
        "--key",
        keys["backup"],
    )

    assert code == 1
    assert payload["outcome"] == "refused"
    assert payload["data"]["reason"] == "quiesce_required"
    assert not (tmp_path / "rp").exists()


def test_usage_errors_are_value_free(
    state: SyntheticServiceState,
    keys: dict[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_key = tmp_path / "keys" / "short.key"
    bad_key.write_bytes(b"too-short")
    cases = [
        (
            [
                "recovery",
                "backup",
                str(tmp_path / "rp"),
                "--state-root",
                str(state.root),
                "--key",
                f"backup-key-1={bad_key}",
                "--writers-stopped",
            ],
            "key_file_invalid",
        ),
        (
            [
                "recovery",
                "verify",
                str(tmp_path / "rp"),
                "--key",
                f"backup-key-1={tmp_path / 'missing.key'}",
            ],
            "key_file_unavailable",
        ),
        (
            ["recovery", "verify", str(tmp_path / "rp"), "--key", "no-separator"],
            "key_spec_invalid",
        ),
        (
            [
                "recovery",
                "inventory",
                "--state-root",
                str(state.root),
                "--jobs-store",
                str(state.root / "jobs" / "service-jobs.json"),
            ],
            "sources_conflict",
        ),
        (["recovery", "inventory"], "sources_required"),
    ]
    for argv, expected in cases:
        code, payload = _run(capsys, *argv)
        assert code == 2, (expected, payload)
        assert payload["error"]["code"] == expected
        assert str(tmp_path) not in json.dumps(payload)


def test_activate_requires_a_journey_check(
    state: SyntheticServiceState,
    keys: dict[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    point = tmp_path / "rp"
    staged = tmp_path / "generation-1"
    assert (
        main(
            [
                "recovery",
                "backup",
                str(point),
                "--state-root",
                str(state.root),
                "--key",
                keys["backup"],
                "--writers-stopped",
            ]
        )
        == 0
    )
    assert (
        main(["recovery", "restore", str(point), str(staged), "--key", keys["backup"]])
        == 0
    )
    capsys.readouterr()

    code, payload = _run(
        capsys,
        "recovery",
        "activate",
        str(staged),
        str(tmp_path / "live"),
        "--key",
        keys["backup"],
    )

    assert code == 2
    assert payload["error"]["code"] == "journey_check_required"
    assert not (tmp_path / "live").exists()
