"""CLI tests for ``openmed migrations`` with synthetic state."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from openmed.cli.main import main
from openmed.structured.store.migrations import MIGRATIONS
from tests.fixtures.service_recovery import (
    BACKUP_KEY,
    SENTINEL,
    build_service_state,
    tree_bytes,
)

pytest.importorskip("cryptography")


@pytest.fixture(autouse=True)
def _development_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_registry_prints_the_matrix(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _run(capsys, "migrations", "registry")

    assert code == 0
    assert payload["command"] == "migrations registry"
    components = {item["component"]: item for item in payload["data"]["components"]}
    assert components["journey_sqlite"]["rollback_class"] == "restore_required"
    assert components["protected_payload_store"]["presence"] == "not_present"
    assert payload["data"]["findings"] == []

    assert main(["migrations", "registry"]) == 0
    human = capsys.readouterr().out
    assert "journey_sqlite: v4, restore_required" in human
    assert "findings: none" in human


def test_preflight_is_a_value_free_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state = build_service_state(tmp_path / "state")
    state.close()
    before = tree_bytes(state.root)
    names = sorted(path.relative_to(state.root) for path in state.root.rglob("*"))

    code, payload = _run(
        capsys, "migrations", "preflight", "--state-root", str(state.root)
    )

    assert code == 0
    assert payload["data"]["admission"] == "ready"
    encoded = json.dumps(payload)
    assert str(tmp_path) not in encoded
    assert SENTINEL not in encoded
    assert tree_bytes(state.root) == before
    assert sorted(path.relative_to(state.root) for path in state.root.rglob("*")) == (
        names
    )

    assert main(["migrations", "preflight", "--state-root", str(state.root)]) == 0
    human = capsys.readouterr().out
    assert "migration preflight: ready" in human
    assert str(tmp_path) not in human


def test_preflight_refuses_a_blocked_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "state"
    database = root / "journey" / "journey.sqlite3"
    database.parent.mkdir(parents=True)
    connection = sqlite3.connect(database, isolation_level=None)
    connection.execute(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, "
        "name TEXT NOT NULL, checksum TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO schema_migrations VALUES (?, ?, ?)",
        (MIGRATIONS[-1].version + 1, "future", "f" * 64),
    )
    connection.close()

    code, payload = _run(capsys, "migrations", "preflight", "--state-root", str(root))

    assert code == 1
    assert payload["outcome"] == "refused"
    assert payload["data"]["admission"] == "blocked"
    checks = {item["name"]: item for item in payload["data"]["checks"]}
    assert checks["version_compatibility"]["code"] == "schema_newer_than_binary"


def test_preflight_rejects_a_key_without_a_recovery_point(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key = tmp_path / "backup.key"
    key.write_bytes(BACKUP_KEY)
    (tmp_path / "state").mkdir()

    code, payload = _run(
        capsys,
        "migrations",
        "preflight",
        "--state-root",
        str(tmp_path / "state"),
        "--key",
        f"backup-key-1={key}",
    )

    assert code == 2
    assert payload["error"]["code"] == "recovery_point_required"


@pytest.mark.skipif(sys.platform == "win32", reason="activation is POSIX-only")
def test_rehearse_signs_and_writes_a_partial_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key = tmp_path / "signing.key"
    key.write_text(bytes(range(32)).hex(), encoding="ascii")
    output = tmp_path / "report.json"

    code, payload = _run(
        capsys,
        "migrations",
        "rehearse",
        "--component",
        "audit_chain",
        "--signing-key",
        f"release-gate-1={key}",
        "--output",
        str(output),
    )

    assert code == 0
    data = payload["data"]
    assert data["gate"]["state"] == "partial"
    assert data["signature"]["algorithm"] == "HMAC-SHA256"
    assert json.loads(output.read_text(encoding="utf-8")) == data
    assert str(tmp_path) not in json.dumps(payload)

    again, refused = _run(capsys, "migrations", "rehearse", "--output", str(output))
    assert again == 2
    assert refused["error"]["code"] == "output_exists"
    unknown, error = _run(capsys, "migrations", "rehearse", "--component", "nope")
    assert unknown == 2
    assert error["error"]["code"] == "component_unknown"
