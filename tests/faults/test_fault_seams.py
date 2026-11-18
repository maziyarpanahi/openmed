"""Each injection seam has a positive control (fires) and a negative control.

The negative control runs the same call path with the injector in control
mode: the seam must count the pass (proving it sits on the call path) and
leave behaviour untouched. The positive control fires the scheduled fault.
"""

from __future__ import annotations

import asyncio
import errno
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from openmed.core.config import PROFILE_ENV_VAR
from tests.faults.guard import (
    ALREADY_ARMED_CODE,
    ARM_FORBIDDEN_CODE,
    NOT_ARMED_CODE,
    FaultInjectionForbiddenError,
)
from tests.faults.plan import FaultPlan, FaultPlanError, sha256_digest
from tests.faults.seams import (
    CLOCK_MONOTONIC,
    CLOCK_WALL,
    DISK_FSYNC,
    DISK_LINK,
    DISK_READ,
    DISK_REPLACE,
    DISK_WRITE,
    NETWORK_REQUEST,
    SQLITE_COMMIT,
    STATE_RESTART,
    FaultClock,
    FaultInjector,
    FaultSession,
    FaultTransport,
    LoopbackHTTPReceiver,
    SimulatedProcessKill,
    SQLiteCommitSeam,
    disk_faults,
)

CATEGORY_OF = {
    "enospc": "disk",
    "eio": "disk",
    "partial_write": "disk",
    "fsync_failure": "disk",
    "clock_skew": "clock",
    "clock_jump_forward": "clock",
    "clock_jump_backward": "clock",
    "monotonic_stall": "clock",
    "process_kill": "process",
    "connection_refused": "network",
    "timeout": "network",
    "connection_reset": "network",
    "cancellation": "cancellation",
    "truncated_state": "corrupted_state",
    "garbled_state": "corrupted_state",
}


def _injector(
    *,
    fault: str,
    point: str,
    mode: str = "fault",
    phase: str = "before_commit",
    at_call: int = 1,
    persistent: bool = False,
    parameters: dict[str, Any] | None = None,
) -> FaultInjector:
    case: dict[str, Any] = {
        "case_id": "unit.seam",
        "component": "unit.seam",
        "category": CATEGORY_OF[fault],
        "fault": fault,
        "injection_point": point,
        "phase": phase,
        "trigger": {"at_call": at_call, "persistent": persistent},
        "expected_outcome": "rolled_back",
    }
    if parameters:
        case["parameters"] = parameters
    plan = FaultPlan.from_mapping(
        {
            "schema_version": "1.0.0",
            "plan_id": "unit.seams",
            "seed": 11,
            "cases": [case],
        }
    )
    return FaultInjector(plan.resolved()[0], mode=mode)


def _session(tmp_path: Path, injector: FaultInjector) -> FaultSession:
    return FaultSession(injector, tmp_path / injector.mode)


def test_injector_arms_only_under_the_test_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    injector = _injector(fault="enospc", point=DISK_WRITE)
    with pytest.raises(FaultInjectionForbiddenError) as unarmed:
        injector.hit(DISK_WRITE)
    assert unarmed.value.code == NOT_ARMED_CODE
    with pytest.raises(FaultInjectionForbiddenError) as unarmed_seam:
        with disk_faults(_session(tmp_path, injector), tmp_path):
            pass
    assert unarmed_seam.value.code == NOT_ARMED_CODE

    with injector.armed():
        with pytest.raises(FaultInjectionForbiddenError) as nested:
            with injector.armed():
                pass
    assert nested.value.code == ALREADY_ARMED_CODE

    for value in (None, "prod"):
        if value is None:
            monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(PROFILE_ENV_VAR, value)
        with pytest.raises(FaultInjectionForbiddenError) as refused:
            with injector.armed():
                pass
        assert refused.value.code == ARM_FORBIDDEN_CODE
    assert not injector.is_armed


def test_disk_write_seam_controls(subtests, tmp_path: Path) -> None:
    payload = "synthetic-record-0123456789"
    for mode, fault in (
        ("control", "enospc"),
        ("fault", "enospc"),
        ("fault", "eio"),
        ("fault", "partial_write"),
    ):
        with subtests.test(mode=mode, fault=fault):
            injector = _injector(fault=fault, point=DISK_WRITE, mode=mode)
            session = _session(tmp_path / f"{mode}-{fault}", injector)
            scope = session.state_dir
            outside = tmp_path / f"{mode}-{fault}" / "outside"
            outside.mkdir()
            with injector.armed(), disk_faults(session, scope):
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=outside, delete=False
                ) as handle:
                    handle.write(payload)
                assert injector.hits() == 0
                error: OSError | None = None
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=scope, delete=False
                ) as handle:
                    name = handle.name
                    try:
                        handle.write(payload)
                    except OSError as exc:
                        error = exc
            written = Path(name).read_text(encoding="utf-8")
            assert injector.hits() == 1
            if mode == "control":
                assert error is None and written == payload
                assert injector.fired == 0
                continue
            assert error is not None and injector.fired == 1
            expected_errno = errno.EIO if fault == "eio" else errno.ENOSPC
            assert error.errno == expected_errno
            assert payload not in str(error)
            if fault == "partial_write":
                assert written == payload[: len(payload) // 2]
            else:
                assert written == ""


def test_disk_fsync_replace_link_and_read_seam_controls(
    subtests, tmp_path: Path
) -> None:
    def fsync_case(scope: Path) -> None:
        descriptor, name = tempfile.mkstemp(dir=scope)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b"synthetic")
            stream.flush()
            os.fsync(stream.fileno())

    def replace_case(scope: Path) -> None:
        source = scope / "next.json"
        source.write_text("v2", encoding="utf-8")
        os.replace(source, scope / "current.json")

    def link_case(scope: Path) -> None:
        source = scope / "staged.bin"
        source.write_bytes(b"synthetic")
        os.link(source, scope / "published.bin")

    def read_case(scope: Path) -> None:
        (scope / "current.json").read_text(encoding="utf-8")

    operations = {
        DISK_FSYNC: ("fsync_failure", fsync_case),
        DISK_REPLACE: ("eio", replace_case),
        DISK_LINK: ("eio", link_case),
        DISK_READ: ("eio", read_case),
    }
    for point, (fault, operation) in operations.items():
        for mode in ("control", "fault"):
            with subtests.test(point=point, mode=mode):
                injector = _injector(fault=fault, point=point, mode=mode)
                session = _session(tmp_path / point, injector)
                scope = session.state_dir
                (scope / "current.json").write_text("v1", encoding="utf-8")
                with injector.armed(), disk_faults(session, scope):
                    if mode == "control":
                        operation(scope)
                    else:
                        with pytest.raises(OSError) as error:
                            operation(scope)
                        assert error.value.errno == errno.EIO
                assert injector.hits() == 1
                assert injector.fired == (1 if mode == "fault" else 0)
                current = (scope / "current.json").read_text(encoding="utf-8")
                if point == DISK_REPLACE:
                    assert current == ("v2" if mode == "control" else "v1")
                if point == DISK_LINK:
                    assert (scope / "published.bin").exists() is (mode == "control")


def test_seams_restore_process_primitives(tmp_path: Path) -> None:
    originals = (
        os.replace,
        os.link,
        os.fsync,
        os.fdopen,
        tempfile.mkstemp,
        tempfile.NamedTemporaryFile,
        Path.read_text,
        Path.read_bytes,
    )
    injector = _injector(fault="eio", point=DISK_REPLACE)
    with injector.armed():
        with pytest.raises(OSError):
            with disk_faults(_session(tmp_path, injector), tmp_path):
                raise OSError(errno.EIO, "synthetic")
    assert (
        os.replace,
        os.link,
        os.fsync,
        os.fdopen,
        tempfile.mkstemp,
        tempfile.NamedTemporaryFile,
        Path.read_text,
        Path.read_bytes,
    ) == originals


def test_process_kill_seam_captures_the_durable_state_at_the_kill_instant(
    subtests, tmp_path: Path
) -> None:
    assert issubclass(SimulatedProcessKill, BaseException)
    assert not issubclass(SimulatedProcessKill, Exception)
    for mode, phase, expected in (
        ("control", "before_commit", None),
        ("fault", "before_commit", "v1"),
        ("fault", "after_commit", "v2"),
    ):
        with subtests.test(mode=mode, phase=phase):
            injector = _injector(
                fault="process_kill", point=DISK_REPLACE, mode=mode, phase=phase
            )
            session = _session(tmp_path / phase, injector)
            current = session.state_dir / "current.json"
            current.write_text("v1", encoding="utf-8")
            staged = session.state_dir / "staged.json"
            staged.write_text("v2", encoding="utf-8")
            caught_by_exception_handler = False
            with injector.armed(), disk_faults(session, session.state_dir):
                try:
                    try:
                        os.replace(staged, current)
                    except Exception:
                        caught_by_exception_handler = True
                except SimulatedProcessKill:
                    pass
            assert not caught_by_exception_handler
            assert injector.hits() == 1
            if expected is None:
                assert not session.crashed and not session.crash_dir.exists()
                continue
            assert session.crashed
            image = session.crash_dir / "current.json"
            assert image.read_text(encoding="utf-8") == expected
            assert (session.crash_dir / "staged.json").exists() is (
                phase != "after_commit"
            )


def test_clock_seam_controls(subtests, tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cases = {
        ("control", "clock_jump_forward"): [1001.0, 1002.0, 1003.0],
        ("fault", "clock_skew"): [1001.0, 4602.0, 4603.0],
        ("fault", "clock_jump_forward"): [1001.0, 4602.0, 4603.0],
        ("fault", "clock_jump_backward"): [1001.0, -2598.0, -2597.0],
        ("fault", "monotonic_stall"): [1001.0, 1002.0, 1002.0],
    }
    for (mode, fault), expected in cases.items():
        with subtests.test(mode=mode, fault=fault):
            injector = _injector(
                fault=fault, point=CLOCK_MONOTONIC, mode=mode, at_call=2
            )
            session = _session(tmp_path / fault / mode, injector)
            clock = FaultClock(session, wall_start=start, monotonic_start=1000.0)
            with injector.armed():
                readings = [clock.monotonic() for _ in range(3)]
                wall = clock.wall()
            assert readings == expected
            assert injector.hits() == 3
            assert injector.fired == (0 if mode == "control" else 1)
            assert wall.tzinfo is not None and wall > start
    with pytest.raises(FaultPlanError):
        FaultClock(
            _session(
                tmp_path / "naive", _injector(fault="clock_skew", point=CLOCK_WALL)
            ),
            wall_start=datetime(2026, 1, 1),
        )


def test_network_seam_controls(subtests, tmp_path: Path) -> None:
    cases = (
        ("control", "connection_refused", "before_commit", None, 1),
        ("fault", "connection_refused", "before_commit", httpx.ConnectError, 0),
        ("fault", "timeout", "before_commit", httpx.ConnectTimeout, 0),
        ("fault", "timeout", "during_commit", httpx.ReadTimeout, 1),
        ("fault", "connection_reset", "before_commit", httpx.ConnectError, 0),
        ("fault", "connection_reset", "during_commit", httpx.ReadError, 1),
    )
    for mode, fault, phase, error, received in cases:
        with subtests.test(mode=mode, fault=fault, phase=phase):
            injector = _injector(
                fault=fault, point=NETWORK_REQUEST, mode=mode, phase=phase
            )
            session = _session(tmp_path / f"{fault}-{phase}-{mode}", injector)
            with injector.armed():
                transport = FaultTransport(session)
                with httpx.Client(transport=transport) as client:
                    if error is None:
                        response = client.post("http://127.0.0.1/hook", content=b"x")
                        assert response.status_code == 204
                    else:
                        with pytest.raises(error):
                            client.post("http://127.0.0.1/hook", content=b"x")
            assert transport.received == received
            assert transport.body_digests == [sha256_digest(b"x")] * received
            assert injector.hits() == 1


def test_loopback_receiver_binds_loopback_only() -> None:
    with LoopbackHTTPReceiver() as receiver:
        assert receiver.url.startswith("http://127.0.0.1:")
        response = httpx.post(receiver.url, content=b"synthetic", timeout=5)
    assert response.status_code == 204
    assert receiver.body_digests == [sha256_digest(b"synthetic")]


def test_cancellation_seam_controls(subtests, tmp_path: Path) -> None:
    for mode, phase in (
        ("control", "during_commit"),
        ("fault", "during_commit"),
        ("fault", "after_commit"),
    ):
        with subtests.test(mode=mode, phase=phase):
            injector = _injector(
                fault="cancellation", point="unit.transition", mode=mode, phase=phase
            )
            session = _session(tmp_path / f"{mode}-{phase}", injector)
            effects: list[str] = []
            with injector.armed():
                if mode == "control":
                    assert (
                        session.call("unit.transition", effects.append, "ran") is None
                    )
                else:
                    with pytest.raises(asyncio.CancelledError):
                        session.call("unit.transition", effects.append, "ran")
            assert injector.hits() == 1
            assert effects == (
                [] if phase == "during_commit" and mode == "fault" else ["ran"]
            )


def test_corruption_seam_is_deterministic_and_detectable(
    subtests, tmp_path: Path
) -> None:
    original = b'{"jobs": {"synthetic": {"status": "done"}}}\n' * 4
    for fault in ("truncated_state", "garbled_state"):
        digests = []
        for run, mode in enumerate(("control", "fault", "fault")):
            with subtests.test(fault=fault, mode=mode, run=run):
                injector = _injector(fault=fault, point=STATE_RESTART, mode=mode)
                session = _session(tmp_path / fault / str(run), injector)
                target = session.state_dir / "jobs.json"
                target.write_bytes(original)
                with injector.armed():
                    fired = session.checkpoint(STATE_RESTART, target=target)
                damaged = target.read_bytes()
                assert injector.hits() == 1
                if mode == "control":
                    assert not fired and damaged == original
                    continue
                assert fired and damaged != original
                if fault == "truncated_state":
                    assert original.startswith(damaged) and len(damaged) < len(original)
                else:
                    assert len(damaged) == len(original) and damaged[0] != original[0]
                digests.append(sha256_digest(damaged))
        assert len(digests) == 2 and digests[0] == digests[1]


def test_sqlite_commit_seam_controls(subtests, tmp_path: Path) -> None:
    for mode in ("control", "fault"):
        with subtests.test(mode=mode):
            injector = _injector(
                fault="eio", point=SQLITE_COMMIT, mode=mode, phase="during_commit"
            )
            session = _session(tmp_path, injector)
            connection = sqlite3.connect(
                session.state_dir / f"{mode}.sqlite3", isolation_level=None
            )
            connection.execute("CREATE TABLE synthetic(value INTEGER)")
            seam = SQLiteCommitSeam(session, connection)
            with injector.armed():
                seam.execute("BEGIN IMMEDIATE")
                seam.execute("INSERT INTO synthetic VALUES (1)")
                if mode == "control":
                    seam.execute("COMMIT")
                else:
                    with pytest.raises(sqlite3.OperationalError):
                        seam.execute("COMMIT")
                    seam.execute("ROLLBACK")
            rows = connection.execute("SELECT COUNT(*) FROM synthetic").fetchone()[0]
            connection.close()
            assert injector.hits() == 1
            assert rows == (1 if mode == "control" else 0)
