"""Crash, disk-fault, and restart recovery for transactional model activation.

Every boundary the activation journal crosses is exercised by replaying a
recorded trace of fault-hook points: a simulated process kill or a disk-full
error is injected at each hit in turn, the root is reopened by a fresh manager,
and the recovered state is checked. Bundles are tiny synthetic files; nothing
touches the network.
"""

from __future__ import annotations

import errno
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openmed.core.model_integrity import (
    verify_artifact_manifest,
    write_local_artifact_manifest,
)
from openmed.service.model_activation import (
    ActivationOutcome,
    ActivationPolicy,
    ActivationReason,
    GenerationContext,
    ModelActivationManager,
    ModelReference,
)

pytestmark = pytest.mark.integration

SLOT = "pii-default"
MODEL_ID = "openmed-test/synthetic-activation-model"
REPRO = "sha256:" + "cd" * 32
WINDOW = 60.0


class SimulatedCrash(BaseException):
    """Stands in for a process kill: no ``except Exception`` handler runs."""


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class Source:
    def __init__(self, payload: bytes, *, corrupt: bool = False) -> None:
        self.payload = payload
        self.corrupt = corrupt

    def acquire(self, reference: ModelReference, workspace: Path) -> Path:
        bundle = workspace / "bundle"
        bundle.mkdir()
        (bundle / "config.json").write_text('{"model_type": "synthetic"}\n')
        (bundle / "model.safetensors").write_bytes(self.payload)
        write_local_artifact_manifest(
            bundle, model_id=MODEL_ID, reproducibility_hash=REPRO
        )
        if self.corrupt:
            (bundle / "model.safetensors").write_bytes(b"corrupted")
        return bundle


class Handle:
    def __init__(self, context: GenerationContext) -> None:
        self.artifact_digest = context.artifact_digest
        self.payload = (context.artifact_dir / "model.safetensors").read_bytes()


class Loader:
    def __init__(self) -> None:
        self.fail_load = False

    def load(self, context: GenerationContext) -> Handle:
        if self.fail_load:
            raise RuntimeError("synthetic load failure")
        return Handle(context)

    def warm(self, handle: Handle, context: GenerationContext) -> None:
        return None

    def unload(self, handle: Handle, context: GenerationContext) -> None:
        return None


class Hook:
    """Records boundary hits and injects one fault at the ``index``-th hit."""

    def __init__(
        self,
        index: int | None = None,
        *,
        error: BaseException | None = None,
        persistent_disk_full: bool = False,
    ) -> None:
        self.index = index
        self.error = error
        self.persistent = persistent_disk_full
        self.enabled = False
        self.points: list[str] = []
        self.tripped = False

    def __call__(self, point: str) -> None:
        if not self.enabled:
            return
        position = len(self.points)
        self.points.append(point)
        if self.persistent and self.tripped and point.endswith(".write"):
            raise OSError(errno.ENOSPC, "synthetic disk full")
        if self.index is not None and position == self.index:
            self.tripped = True
            raise self.error or SimulatedCrash(point)


def checkpoint(version: str, *parents: str) -> dict[str, Any]:
    return {
        "model_id": f"synthetic/checkpoint-{version}",
        "family": "pii",
        "version": version,
        "semver_constraint": ">=1.0.0",
        "lineage": [f"synthetic/checkpoint-{parent}" for parent in parents],
        "policy_fingerprint": "policy-v1",
        "tokenizer_ids": ["tokenizer-v1"],
        "evidence_schema_versions": ["1"],
    }


VERSIONS = {
    b"weights-v1": checkpoint("1.0.0"),
    b"weights-v2": checkpoint("1.1.0", "1.0.0"),
    b"weights-v3": checkpoint("1.2.0", "1.0.0", "1.1.0"),
}


def open_manager(
    root: Path,
    *,
    window: float,
    clock: Clock,
    hook: Hook | None = None,
    loader: Loader | None = None,
) -> tuple[ModelActivationManager, tuple[Any, ...]]:
    manager = ModelActivationManager(
        root,
        loader=loader or Loader(),
        policy=ActivationPolicy(health_window_seconds=window, max_quarantine_entries=4),
        clock=clock,
        fault_hook=hook,
    )
    return manager, manager.start()


def activate(manager: ModelActivationManager, payload: bytes, **source: Any):
    return manager.activate(
        SLOT,
        ModelReference(MODEL_ID, revision=payload.decode()),
        source=Source(payload, **source),
        compatibility=VERSIONS[payload],
    )


def settle(manager: ModelActivationManager, clock: Clock, window: float) -> None:
    if window > 0:
        clock.advance(window + 1)
        assert manager.poll_health(SLOT).outcome is ActivationOutcome.ACTIVATED


def serving_payload(manager: ModelActivationManager) -> bytes:
    with manager.lease(SLOT) as lease:
        return lease.handle.payload


def assert_recovered(manager: ModelActivationManager, root: Path) -> bytes:
    """Check the recovered root and return the payload being served."""

    status = manager.status(SLOT)
    assert status["pending_transaction"] is None
    assert status["active_generation"] == status["last_green_generation"]
    assert status["serving_generation"] == status["active_generation"]
    with manager.lease(SLOT) as lease:
        payload = lease.handle.payload
        artifact = (
            next((root / "generations").iterdir())
            / f"{lease.generation:08d}"
            / "artifact"
        )
        verified = verify_artifact_manifest(artifact / ".openmed-integrity.json")
        assert verified.actual_sha256 == lease.artifact_digest
        assert lease.handle.artifact_digest == lease.artifact_digest
    assert list((root / "staging").iterdir()) == []
    generations = [
        gen.name
        for slot_dir in (root / "generations").iterdir()
        for gen in slot_dir.iterdir()
    ]
    assert generations == [f"{status['active_generation']:08d}"]
    for journal in (root / "journal").glob("txn-*.json"):
        assert json.loads(journal.read_text())["status"] in {
            "committed",
            "rolled_back",
            "quarantined",
        }
    assert len(list((root / "quarantine").iterdir())) <= 4
    assert manager.verify_receipts()
    return payload


def trace(tmp_path: Path, window: float) -> list[str]:
    hook = Hook()
    clock = Clock()
    manager, _ = open_manager(tmp_path / "trace", window=window, clock=clock, hook=hook)
    activate(manager, b"weights-v1")
    settle(manager, clock, window)
    hook.enabled = True
    activate(manager, b"weights-v2")
    manager.close()
    return hook.points


def prepared(root: Path, window: float, hook: Hook, clock: Clock):
    manager, _ = open_manager(root, window=window, clock=clock, hook=hook)
    activate(manager, b"weights-v1")
    settle(manager, clock, window)
    hook.enabled = True
    return manager


@pytest.mark.parametrize("window", [0.0, WINDOW])
def test_process_kill_at_every_boundary_recovers(tmp_path: Path, window: float):
    points = trace(tmp_path, window)
    published = points.index("commit.published")
    assert {"acquire.before", "verify.after", "warm.after", "commit.after"} <= set(
        points
    )
    for index, point in enumerate(points):
        root = tmp_path / f"kill-{index}"
        clock = Clock()
        manager = prepared(root, window, Hook(index), clock)
        with pytest.raises(SimulatedCrash):
            activate(manager, b"weights-v2")
        manager.close()

        restarted, results = open_manager(root, window=window, clock=clock)
        expected = (
            b"weights-v2" if window == 0 and index >= published else b"weights-v1"
        )
        assert assert_recovered(restarted, root) == expected, (index, point)
        if window > 0 and index >= published:
            assert results[0].reason is ActivationReason.HEALTH_WINDOW_INTERRUPTED
        assert all(result.outcome is not ActivationOutcome.BUSY for result in results)

        follow_up = activate(restarted, b"weights-v3")
        assert follow_up.ok, (index, point, follow_up.reason)
        settle(restarted, clock, window)
        assert serving_payload(restarted) == b"weights-v3"
        restarted.close()


@pytest.mark.parametrize("window", [0.0, WINDOW])
def test_disk_full_at_every_boundary_keeps_a_usable_generation(
    tmp_path: Path, window: float
):
    points = trace(tmp_path, window)
    committed = points.index("commit.after")
    for index, point in enumerate(points):
        root = tmp_path / f"enospc-{index}"
        clock = Clock()
        hook = Hook(index, error=OSError(errno.ENOSPC, "synthetic disk full"))
        manager = prepared(root, window, hook, clock)

        result = activate(manager, b"weights-v2")

        expected = b"weights-v2" if index > committed else b"weights-v1"
        assert serving_payload(manager) == expected, (index, point)
        status = manager.status(SLOT)
        assert status["serving_generation"] == status["active_generation"]
        if index <= committed:
            assert not result.ok
            assert result.reason in {
                ActivationReason.DISK_EXHAUSTED,
                ActivationReason.ACQUIRE_FAILED,
            }, (index, point, result.reason)
        if status["pending_transaction"] is not None and window == 0:
            assert manager.poll_health(SLOT).outcome is ActivationOutcome.ACTIVATED
        manager.close()

        restarted, _ = open_manager(root, window=window, clock=clock)
        assert assert_recovered(restarted, root) == expected, (index, point)
        restarted.close()


def test_persistent_disk_full_never_serves_a_partial_generation(tmp_path: Path):
    points = trace(tmp_path, 0.0)
    for index, point in enumerate(points):
        root = tmp_path / f"full-{index}"
        clock = Clock()
        hook = Hook(
            index,
            error=OSError(errno.ENOSPC, "synthetic disk full"),
            persistent_disk_full=True,
        )
        manager = prepared(root, 0.0, hook, clock)

        activate(manager, b"weights-v2")

        status = manager.status(SLOT)
        assert status["serving_generation"] == status["active_generation"], index
        assert serving_payload(manager) in {b"weights-v1", b"weights-v2"}
        manager.close()

        restarted, _ = open_manager(root, window=0.0, clock=clock)
        assert assert_recovered(restarted, root) in {b"weights-v1", b"weights-v2"}
        restarted.close()


def test_restart_during_health_window_rolls_back(tmp_path: Path):
    clock = Clock()
    manager, _ = open_manager(tmp_path, window=WINDOW, clock=clock)
    activate(manager, b"weights-v1")
    settle(manager, clock, WINDOW)
    pending = activate(manager, b"weights-v2")
    assert pending.outcome is ActivationOutcome.PENDING_HEALTH
    manager.close()

    restarted, results = open_manager(tmp_path, window=WINDOW, clock=clock)

    assert [result.outcome for result in results] == [ActivationOutcome.ROLLED_BACK]
    assert results[0].transaction_id == pending.transaction_id
    assert assert_recovered(restarted, tmp_path) == b"weights-v1"
    assert restarted.receipts()[-1]["reason"] == "health_window_interrupted"
    restarted.close()


@pytest.mark.parametrize("fault", ["kill", "disk_full"])
def test_faults_during_health_rollback_and_promotion_recover(
    tmp_path: Path, fault: str
):
    def run(operation: str, index: int | None) -> tuple[list[str], bytes | None]:
        root = tmp_path / f"{operation}-{index}"
        clock = Clock()
        error = OSError(errno.ENOSPC, "synthetic disk full")
        hook = Hook(index, error=error if fault == "disk_full" else None)
        manager, _ = open_manager(root, window=WINDOW, clock=clock, hook=hook)
        activate(manager, b"weights-v1")
        settle(manager, clock, WINDOW)
        activate(manager, b"weights-v2")
        hook.enabled = True
        try:
            if operation == "rollback":
                manager.record_health(SLOT, healthy=False)
            else:
                clock.advance(WINDOW + 1)
                manager.poll_health(SLOT)
        except SimulatedCrash:
            assert fault == "kill"
        else:
            # Whatever failed, memory serves exactly what is durable.
            status = manager.status(SLOT)
            assert status["serving_generation"] == status["active_generation"]
            assert serving_payload(manager) in {b"weights-v1", b"weights-v2"}
        manager.close()
        if index is None:
            return hook.points, None
        restarted, _ = open_manager(root, window=WINDOW, clock=clock)
        payload = assert_recovered(restarted, root)
        restarted.close()
        return hook.points, payload

    rollback_points, _ = run("rollback", None)
    assert "rollback.published" in rollback_points
    for index in range(len(rollback_points)):
        assert run("rollback", index)[1] == b"weights-v1", index

    promote_points, _ = run("promote", None)
    assert promote_points[0] == "journal.write"
    for index in range(len(promote_points)):
        # Until the retire phase is journaled the window's pass is not durable.
        expected = b"weights-v1" if index == 0 else b"weights-v2"
        assert run("promote", index)[1] == expected, index


def test_kill_during_recovery_is_itself_recoverable(tmp_path: Path):
    clock = Clock()
    manager, _ = open_manager(tmp_path, window=WINDOW, clock=clock)
    activate(manager, b"weights-v1")
    settle(manager, clock, WINDOW)
    activate(manager, b"weights-v2")
    manager.close()

    crashing = Hook(0)
    crashing.enabled = True
    second = ModelActivationManager(
        tmp_path,
        loader=Loader(),
        policy=ActivationPolicy(health_window_seconds=WINDOW),
        clock=clock,
        fault_hook=crashing,
    )
    with pytest.raises(SimulatedCrash):
        second.start()
    assert not second.started

    restarted, _ = open_manager(tmp_path, window=WINDOW, clock=clock)
    assert assert_recovered(restarted, tmp_path) == b"weights-v1"
    restarted.close()


def test_missing_pending_journal_is_rolled_back_on_start(tmp_path: Path):
    clock = Clock()
    manager, _ = open_manager(tmp_path, window=WINDOW, clock=clock)
    activate(manager, b"weights-v1")
    settle(manager, clock, WINDOW)
    pending = activate(manager, b"weights-v2")
    manager.close()
    (tmp_path / "journal" / f"{pending.transaction_id}.json").unlink()

    restarted, results = open_manager(tmp_path, window=WINDOW, clock=clock)

    assert results[0].outcome is ActivationOutcome.ROLLED_BACK
    assert assert_recovered(restarted, tmp_path) == b"weights-v1"
    restarted.close()


OPERATIONS = (
    "good",
    "corrupt",
    "load_failure",
    "healthy",
    "unhealthy",
    "tick",
    "kill",
    "restart",
)


@settings(max_examples=25, deadline=None, derandomize=True)
@given(
    steps=st.lists(
        st.tuples(st.sampled_from(OPERATIONS), st.integers(0, 45)),
        min_size=1,
        max_size=8,
    )
)
def test_random_operation_sequences_preserve_invariants(steps):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        clock = Clock()
        loader = Loader()
        hook = Hook()
        manager, _ = open_manager(
            root, window=30.0, clock=clock, hook=hook, loader=loader
        )
        activate(manager, b"weights-v1")
        clock.advance(31)
        manager.poll_health(SLOT)
        payloads = iter([b"weights-v2", b"weights-v3"] * 8)
        known = {b"weights-v1", b"weights-v2", b"weights-v3"}

        for operation, number in steps:
            if operation in {"good", "corrupt", "load_failure", "kill"}:
                payload = next(payloads)
                loader.fail_load = operation == "load_failure"
                if operation == "kill":
                    hook.index, hook.points, hook.enabled = number, [], True
                try:
                    activate(manager, payload, corrupt=operation == "corrupt")
                except SimulatedCrash:
                    manager.close()
                    manager, _ = open_manager(
                        root, window=30.0, clock=clock, hook=hook, loader=loader
                    )
                hook.enabled = False
                loader.fail_load = False
            elif operation in {"healthy", "unhealthy"}:
                manager.record_health(SLOT, healthy=operation == "healthy")
            elif operation == "tick":
                clock.advance(31)
                manager.poll_health(SLOT)
            else:
                manager.close()
                manager, _ = open_manager(
                    root, window=30.0, clock=clock, hook=hook, loader=loader
                )

            status = manager.status(SLOT)
            if status["pending_transaction"] is None:
                assert status["active_generation"] == status["last_green_generation"]
            else:
                assert status["active_generation"] != status["last_green_generation"]
            assert status["serving_generation"] == status["active_generation"]
            assert serving_payload(manager) in known
            assert list((root / "staging").iterdir()) == []
            assert len(list((root / "quarantine").iterdir())) <= 4
            assert manager.verify_receipts()

        manager.close()
        restarted, _ = open_manager(root, window=30.0, clock=clock)
        assert assert_recovered(restarted, root) in known
        restarted.close()
