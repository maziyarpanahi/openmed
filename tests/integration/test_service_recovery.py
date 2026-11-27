"""Offline recovery drill for encrypted service-state backup and restore.

A synthetic service writes Journey records, job metadata, model-registry
generations, and audit entries under a managed live symlink.  The drill takes
a quiesced full backup and an incremental one, restores the chain into an
inactive generation, runs a stand-in production self-check and a
recovery-epoch hook, activates, and resumes with equivalent logical state.
A failure matrix then proves that no corrupted, incomplete, or interrupted
restore can be activated.  Network access is blocked for the whole drill.
"""

from __future__ import annotations

import errno
import os
import socket
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from openmed.service import recovery
from openmed.service.recovery import (
    JOB_STORE_PATH,
    JOURNEY_DATABASE_PATH,
    ActivationCheckResult,
    RecoveryEpochContext,
    ServiceStateSources,
    activate_restored_state,
    create_recovery_point,
    journey_store_check,
    restore_recovery_point,
)
from tests.fixtures.service_recovery import (
    BACKUP_KEY,
    JOURNEY_KEY,
    OTHER_BACKUP_KEY,
    SENTINEL,
    SENTINEL_BYTES,
    backup_provider,
    build_service_state,
    logical_snapshot,
    tree_bytes,
)

pytest.importorskip("cryptography")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32", reason="atomic symlink activation is POSIX-only"
    ),
]


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the recovery drill must not use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


class _Admission:
    """Stand-in drain coordinator and admission gate for the service."""

    def __init__(self) -> None:
        self.open = True
        self.drains = 0

    def quiesce(self) -> "_Admission":
        return self

    def __enter__(self) -> None:
        self.open = False
        self.drains += 1

    def __exit__(self, *exc: object) -> None:
        self.open = True


class _ProductionSelfCheck:
    """Stand-in for the fail-closed production self-check run before admission."""

    def __init__(self, admission: _Admission) -> None:
        self.admission = admission
        self.roots: list[Path] = []

    def __call__(self, root: Path) -> ActivationCheckResult:
        self.roots.append(root)
        ready = not self.admission.open and (root / JOURNEY_DATABASE_PATH).is_file()
        return ActivationCheckResult("production_self_check", ready, None)


class _EpochLedger:
    """Stand-in recovery-epoch store that fences pre-restore authority."""

    def __init__(self) -> None:
        self.epoch = 0
        self.contexts: list[RecoveryEpochContext] = []

    def open_recovery_epoch(self, context: RecoveryEpochContext) -> None:
        self.epoch += 1
        self.contexts.append(context)


@pytest.fixture
def drill(tmp_path: Path) -> Iterator[dict[str, Any]]:
    generations = tmp_path / "generations"
    generations.mkdir()
    initial = generations / "initial"
    service = build_service_state(initial, notes=3, jobs=2, audit_entries=2)
    live = tmp_path / "state"
    live.symlink_to(initial, target_is_directory=True)
    yield {
        "generations": generations,
        "live": live,
        "service": service,
        "backups": tmp_path / "backups",
    }
    service.close()


def _restore_chain(drill: dict[str, Any], name: str, **kwargs: Any) -> Any:
    backups = drill["backups"]
    return restore_recovery_point(
        backups / "incremental",
        drill["generations"] / name,
        key_provider=kwargs.pop("provider", None) or backup_provider(),
        ancestors=kwargs.pop("ancestors", (backups / "full",)),
        live=drill["live"],
        **kwargs,
    )


def _take_backups(drill: dict[str, Any], admission: _Admission) -> dict[str, Any]:
    service = drill["service"]
    backups = drill["backups"]
    backups.mkdir()
    sources = ServiceStateSources.from_state_root(drill["live"])
    full = create_recovery_point(
        sources,
        backups / "full",
        key_provider=backup_provider(),
        quiesce=admission.quiesce,
        retention_days=30,
        chunk_bytes=8192,
    )
    assert full.completed, full.reason
    service.add_note()
    service.add_job(status="running")
    service.add_audit_entry()
    service.write_registry(latest="OpenMed/synthetic-pii-generation-2")
    incremental = create_recovery_point(
        sources,
        backups / "incremental",
        key_provider=backup_provider(),
        parent=backups / "full",
        quiesce=admission.quiesce,
        retention_days=30,
        chunk_bytes=8192,
    )
    assert incremental.completed, incremental.reason
    return {
        "full": full,
        "incremental": incremental,
        "expected": logical_snapshot(drill["live"], service.encryption),
    }


def test_clean_recovery_drill_resumes_equivalent_state(
    drill: dict[str, Any],
) -> None:
    pytest.importorskip("pydantic")
    from openmed.service.jobs import LocalJobStore

    admission = _Admission()
    taken = _take_backups(drill, admission)
    service = drill["service"]
    previous_target = drill["live"].resolve()

    # Disaster: the live database is damaged after the recovery point.
    service.close()
    (previous_target / JOURNEY_DATABASE_PATH).write_bytes(b"damaged" * 64)

    restored = _restore_chain(drill, "restored")
    assert restored.completed, restored.reason
    assert restored.chain_length == 2
    assert drill["live"].resolve() == previous_target

    self_check = _ProductionSelfCheck(admission)
    ledger = _EpochLedger()
    with admission.quiesce():
        activation = activate_restored_state(
            drill["generations"] / "restored",
            drill["live"],
            key_provider=backup_provider(),
            checks=[journey_store_check(service.encryption), self_check],
            epoch_hooks=[ledger],
        )
        assert activation.completed, (activation.reason, activation.details)
        assert not admission.open
    assert admission.open

    assert drill["live"].resolve() == (drill["generations"] / "restored").resolve()
    assert self_check.roots == [(drill["generations"] / "restored").resolve()]
    assert ledger.epoch == 1
    assert (
        ledger.contexts[0].recovery_point_id == taken["incremental"].recovery_point_id
    )
    assert ledger.contexts[0].content_digest == taken["incremental"].content_digest
    assert logical_snapshot(drill["live"], service.encryption) == taken["expected"]
    jobs = LocalJobStore(drill["live"] / JOB_STORE_PATH)
    assert jobs.get("job-0002")["status"] == "running"

    evidence = b"".join(
        [
            tree_bytes(drill["backups"]),
            tree_bytes(drill["live"] / recovery.RECOVERY_METADATA_DIRECTORY),
            *(
                receipt.to_json().encode()
                for receipt in (
                    taken["full"],
                    taken["incremental"],
                    restored,
                    activation,
                )
            ),
        ]
    )
    assert SENTINEL_BYTES not in evidence
    assert JOURNEY_KEY not in evidence
    assert BACKUP_KEY not in evidence
    assert str(drill["generations"]).encode() not in evidence


def _corrupt(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))


def _truncate(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 40])


def _reorder(path: Path) -> None:
    data = path.read_bytes()
    header, offset, records = data[:9], 9, []
    while offset < len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        records.append(data[offset : offset + 4 + length])
        offset += 4 + length
    assert len(records) > 1
    records.reverse()
    path.write_bytes(header + b"".join(records))


@pytest.mark.parametrize(
    "fault",
    [
        "corruption",
        "truncation",
        "reordering",
        "missing_increment",
        "wrong_key",
        "wrong_schema",
        "disk_exhaustion",
        "interrupted",
    ],
)
def test_faulty_restores_can_never_be_activated(
    drill: dict[str, Any],
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taken = _take_backups(drill, _Admission())
    backups = drill["backups"]
    live_target = drill["live"].resolve()
    database_object = next(
        backups / "incremental" / "objects" / f"{item['object']}.omrc"
        for item in _manifest_objects(backups / "incremental")
        if item["path"] == JOURNEY_DATABASE_PATH
    )
    kwargs: dict[str, Any] = {}
    if fault == "corruption":
        _corrupt(database_object)
    elif fault == "truncation":
        _truncate(database_object)
    elif fault == "reordering":
        _reorder(database_object)
    elif fault == "missing_increment":
        kwargs["ancestors"] = ()
    elif fault == "wrong_key":
        kwargs["provider"] = backup_provider(OTHER_BACKUP_KEY)
    elif fault == "wrong_schema":
        monkeypatch.setattr(recovery, "LATEST_MIGRATION_VERSION", 1)
    elif fault == "disk_exhaustion":
        original = recovery._write_all
        calls = {"count": 0}

        def exhausted(descriptor: int, data: bytes) -> None:
            calls["count"] += 1
            if calls["count"] > 4:
                raise OSError(errno.ENOSPC, "No space left on device")
            original(descriptor, data)

        monkeypatch.setattr(recovery, "_write_all", exhausted)
    elif fault == "interrupted":
        original_materialize = recovery._materialize

        def interrupted(*args: Any, **kw: Any) -> Any:
            original_materialize(*args, **kw)
            raise KeyboardInterrupt

        monkeypatch.setattr(recovery, "_materialize", interrupted)

    target = drill["generations"] / "candidate"
    if fault == "interrupted":
        with pytest.raises(KeyboardInterrupt):
            _restore_chain(drill, "candidate", **kwargs)
    else:
        receipt = _restore_chain(drill, "candidate", **kwargs)
        assert not receipt.completed
        assert receipt.reason in recovery.RECOVERY_REASON_CODES
        assert SENTINEL not in receipt.to_json()
    assert not target.exists()

    activation = activate_restored_state(
        target, drill["live"], key_provider=backup_provider()
    )
    assert not activation.completed
    assert drill["live"].resolve() == live_target
    assert (
        logical_snapshot(drill["live"], drill["service"].encryption)
        == (taken["expected"])
    )
    assert os.path.islink(drill["live"])


def _manifest_objects(point: Path) -> list[dict[str, Any]]:
    import json

    from openmed.core.aead_envelope import EnvelopeCipher

    descriptor = json.loads((point / "recovery-point.json").read_text())
    cipher = EnvelopeCipher(backup_provider(), purpose=recovery.BACKUP_KEY_PURPOSE)
    manifest = json.loads(
        cipher.open(
            (point / "manifest.omae").read_bytes(),
            context={
                "point": descriptor["recovery_point_id"],
                "record": "recovery_manifest",
            },
        )
    )
    return list(manifest["objects"])
