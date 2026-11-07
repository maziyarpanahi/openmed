"""Offline tests for bounded Journey-store waits, cancellation, and reuse.

PostgreSQL behavior is exercised through a synthetic DB-API driver backed by an
in-memory SQLite database. It honors the transaction-local ``statement_timeout``
and ``lock_timeout`` the store arms, records every statement, and keeps
parameter-bearing notices so tests can prove they are dropped.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import traceback
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    EvidenceLocator,
    sha256_digest,
)
from openmed.core.budget import RequestBudget
from openmed.structured.store import (
    STORE_BUSY_CODE,
    STORE_DEADLINE_EXCEEDED_CODE,
    STORE_LOCK_WAIT_ENV_VAR,
    STORE_TIMEOUT_ENV_VAR,
    STORE_TIMEOUTS_INVALID_CODE,
    ComposedJourneyStore,
    LocalArtifactStore,
    LocalJourneyStore,
    PostgresJourneyStore,
    PostgresStoreError,
    SQLiteJourneyStore,
    StoreBusyError,
    StoreDeadline,
    StoreDeadlineExceededError,
    StoreState,
    StoreTimeouts,
    StoreTimeoutsError,
    current_store_deadline,
    store_deadline,
)
from openmed.structured.store.deadlines import ConnectionCheckout

T1 = "2026-03-04T05:06:07Z"
T2 = "2026-03-04T06:06:07Z"
SYNTHETIC_VALUE = "synthetic-parameter-value-7f3a"
# Bounds a broken implementation's wait so the suite cannot hang.
SAFETY_SECONDS = 5.0
LONG_SQLITE_QUERY = (
    "WITH RECURSIVE counter(x) AS ("
    "SELECT 1 UNION ALL SELECT x + 1 FROM counter WHERE x < {limit}"
    ") SELECT max(x) FROM counter"
)


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeDriverError(Exception):
    """Driver error whose diagnostics echo a synthetic parameter value."""

    def __init__(self, sqlstate: str) -> None:
        super().__init__(
            {
                "C": sqlstate,
                "M": "synthetic driver failure",
                "D": f"Key (artifact_id)=({SYNTHETIC_VALUE})",
            }
        )
        self.sqlstate = sqlstate


class FakeCursor:
    """DB-API cursor over the fake connection."""

    def __init__(self, connection: FakePostgresConnection) -> None:
        self.connection = connection
        self.description: Sequence[Sequence[Any]] | None = None
        self.rowcount = -1
        self._rows: list[tuple[Any, ...]] = []

    def execute(self, operation: str, parameters: Sequence[Any] = ()) -> None:
        self.connection.run(self, operation, tuple(parameters))

    def executemany(
        self,
        operation: str,
        parameters: Sequence[Sequence[Any]],
    ) -> None:
        for row in parameters:
            self.connection.run(self, operation, tuple(row))

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows.pop(0) if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows, self._rows = self._rows, []
        return rows

    def close(self) -> None:
        self._rows = []

    def set_result(self, names: Sequence[str], rows: list[tuple[Any, ...]]) -> None:
        self.description = tuple(
            (name, None, None, None, None, None, None) for name in names
        )
        self._rows = rows


class FakePostgresConnection:
    """SQLite-backed DB-API connection that enforces PostgreSQL wait settings."""

    def __init__(self) -> None:
        self.db = sqlite3.connect(
            ":memory:",
            isolation_level=None,
            check_same_thread=False,
        )
        self.autocommit = False
        self.closed = False
        self.statements: list[str] = []
        self.local: dict[str, str] = {}
        self.session: dict[str, str] = {}
        self.armed: list[dict[str, str]] = []
        self.notices: deque[str] = deque()
        self.block: Callable[[str], str | None] = lambda sql: None
        self.fail_on: Callable[[str], bool] = lambda sql: False
        self.release = threading.Event()
        self.blocking = threading.Event()
        self.fail_commit: str | None = None
        self.fail_reset = False
        self.unbounded_waits = 0

    @property
    def in_transaction(self) -> bool:
        return self.db.in_transaction

    def cursor(self) -> FakeCursor:
        self._require_open()
        return FakeCursor(self)

    def commit(self) -> None:
        self._require_open()
        if self.fail_commit == "lost_ack":
            self._end("COMMIT")
            raise FakeDriverError("08006")
        if self.fail_commit == "refused":
            raise FakeDriverError("08006")
        self._end("COMMIT")

    def rollback(self) -> None:
        self._require_open()
        self._end("ROLLBACK")

    def close(self) -> None:
        self.closed = True

    def run(
        self,
        cursor: FakeCursor,
        operation: str,
        parameters: tuple[Any, ...],
    ) -> None:
        self._require_open()
        sql = " ".join(operation.split())
        self.statements.append(sql)
        if parameters:
            self.notices.append(f"NOTICE: values {parameters!r}")
        if self.fail_on(sql):
            raise FakeDriverError("XX000")
        self._wait_if_blocked(sql)
        upper = sql.upper()
        if upper.startswith("BEGIN"):
            if self.db.in_transaction:
                self.notices.append("WARNING: there is already a transaction")
            else:
                self.db.execute("BEGIN")
            return
        if upper.startswith(("LOCK TABLE", "CREATE SCHEMA", "SET LOCAL SEARCH_PATH")):
            return
        if upper.startswith("SET SEARCH_PATH"):
            self.session["search_path"] = sql.rsplit(" ", 1)[-1]
            return
        if upper == "RESET ALL":
            if self.fail_reset:
                raise FakeDriverError("08006")
            self.session.clear()
            return
        if "SET_CONFIG(" in upper:
            names = re.findall(r"set_config\('(\w+)'", sql)
            values = dict(zip(names, (str(value) for value in parameters), strict=True))
            if self.db.in_transaction:
                self.local.update(values)
            self.armed.append(values)
            cursor.set_result(names, [tuple(values[name] for name in names)])
            return
        if "PG_ADVISORY_XACT_LOCK" in upper:
            cursor.set_result(("pg_advisory_xact_lock",), [(None,)])
            return
        try:
            result = self.db.execute(operation.replace("%s", "?"), parameters)
            rows = result.fetchall() if result.description else []
        except sqlite3.IntegrityError:
            raise FakeDriverError("23505") from None
        except sqlite3.Error:
            raise FakeDriverError("42601") from None
        cursor.description = result.description
        cursor.rowcount = result.rowcount
        cursor._rows = rows

    def _wait_if_blocked(self, sql: str) -> None:
        kind = self.block(sql)
        if kind is None:
            return
        bounds = (
            [(int(self.local["statement_timeout"]), "57014")]
            if ("statement_timeout" in self.local)
            else []
        )
        if kind == "lock" and "lock_timeout" in self.local:
            bounds.append((int(self.local["lock_timeout"]), "55P03"))
        self.blocking.set()
        if not bounds:
            self.unbounded_waits += 1
            self.release.wait(SAFETY_SECONDS)
            return
        wait_ms, sqlstate = min(bounds)
        if self.release.wait(wait_ms / 1000):
            return
        raise FakeDriverError(sqlstate)

    def _end(self, statement: str) -> None:
        if self.db.in_transaction:
            self.db.execute(statement)
        self.local.clear()

    def _require_open(self) -> None:
        if self.closed:
            raise FakeDriverError("08003")


def _artifact(number: int = 1) -> tuple[ClinicalArtifact, bytes]:
    content = f"synthetic bounded journey artifact {number}".encode()
    suffix = f"{number:016d}"
    artifact = ClinicalArtifact(
        artifact_id=f"artifact_{suffix}",
        artifact_type="clinical_note",
        media_type="text/plain",
        content_hash=sha256_digest(content),
        byte_size=len(content),
        source_id=f"source_{suffix}",
        recorded_at=T1,
        subject_id=f"subject_{suffix}",
    )
    return artifact, content


def _evidence(artifact: ClinicalArtifact) -> EvidenceLocator:
    return EvidenceLocator(
        locator_id=f"evidence_{artifact.artifact_id.removeprefix('artifact_')}",
        artifact_id=artifact.artifact_id,
        location_type="text_span",
        location={"start": 0, "end": 9},
    )


@pytest.fixture
def fake() -> Iterator[FakePostgresConnection]:
    connection = FakePostgresConnection()
    yield connection
    connection.release.set()


def _postgres(
    connection: FakePostgresConnection,
    **timeouts: Any,
) -> PostgresJourneyStore:
    return PostgresJourneyStore(connection, timeouts=StoreTimeouts(**timeouts))


def _statements_after(connection: FakePostgresConnection, start: int) -> list[str]:
    return connection.statements[start:]


class _StalledHolder:
    """Hold a store transaction open on another thread until released."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.entered = threading.Event()
        self.release = threading.Event()
        self.thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self) -> None:
        with self.store.transaction(committed_at=T1):
            self.entered.set()
            self.release.wait(SAFETY_SECONDS)

    def __enter__(self) -> _StalledHolder:
        self.thread.start()
        assert self.entered.wait(SAFETY_SECONDS)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release.set()
        self.thread.join(SAFETY_SECONDS)
        assert not self.thread.is_alive()


# --- Deadline primitives -------------------------------------------------


def test_store_timeouts_validate_caps_without_echoing_values(subtests) -> None:
    defaults = StoreTimeouts()
    assert (defaults.operation_seconds, defaults.lock_wait_seconds) == (30.0, 5.0)
    assert StoreTimeouts.from_env({}) == defaults
    assert StoreTimeouts.from_env(
        {STORE_TIMEOUT_ENV_VAR: " 12.5 ", STORE_LOCK_WAIT_ENV_VAR: "0.25"}
    ) == StoreTimeouts(operation_seconds=12.5, lock_wait_seconds=0.25)

    for raw in ("not-a-number", "0", "-1", "nan", "inf", "3601"):
        with subtests.test(raw=raw):
            with pytest.raises(StoreTimeoutsError) as caught:
                StoreTimeouts.from_env({STORE_LOCK_WAIT_ENV_VAR: raw})
            assert str(caught.value) == (
                f"{STORE_LOCK_WAIT_ENV_VAR} must be a number of seconds in (0, 3600]"
            )
            assert caught.value.code == STORE_TIMEOUTS_INVALID_CODE
    for value in (True, 0, -0.5, float("nan"), 3_601, "5"):
        with subtests.test(value=value):
            with pytest.raises(StoreTimeoutsError):
                StoreTimeouts(operation_seconds=value)  # type: ignore[arg-type]


def test_store_deadline_scopes_only_shorten_and_reuse_request_budgets() -> None:
    clock = FakeClock()
    timeouts = StoreTimeouts(operation_seconds=20, lock_wait_seconds=2, clock=clock)

    assert current_store_deadline() is None
    assert timeouts.operation_deadline().remaining() == 20
    explicit = StoreDeadline.after(3, clock=clock)
    assert timeouts.operation_deadline(explicit) is explicit
    assert timeouts.operation_deadline(60).remaining() == 20

    with store_deadline(StoreDeadline.after(5, clock=clock)) as outer:
        assert outer is current_store_deadline()
        with store_deadline(StoreDeadline.after(9, clock=clock)) as inner:
            assert inner is outer
        with store_deadline(None) as unchanged:
            assert unchanged is outer
        assert timeouts.operation_deadline().remaining() == 5
        assert timeouts.operation_deadline(explicit).remaining() == 3
    assert current_store_deadline() is None

    budget = RequestBudget(max_wall_time=0.5).start()
    derived = StoreDeadline.from_budget(budget, clock=clock)
    assert derived is not None and 0 < derived.remaining() <= 0.5
    assert StoreDeadline.from_budget(RequestBudget().start()) is None
    assert timeouts.operation_deadline(budget).remaining() <= 0.5

    expired = StoreDeadline.after(0, clock=clock)
    assert expired.expired
    with pytest.raises(StoreDeadlineExceededError):
        expired.milliseconds(1)
    assert StoreDeadline.after(0.0001, clock=clock).milliseconds(5) == 1
    for invalid in ("5", True, object()):
        with pytest.raises(TypeError):
            timeouts.operation_deadline(invalid)  # type: ignore[arg-type]


def test_connection_checkout_is_reentrant_and_never_blocks_forever() -> None:
    checkout = ConnectionCheckout(StoreTimeouts(lock_wait_seconds=0.05))
    checkout.acquire()
    checkout.acquire()
    assert checkout.held
    outcome: list[BaseException] = []

    def contend() -> None:
        try:
            checkout.acquire()
        except BaseException as error:  # noqa: BLE001 - recorded for the assertion
            outcome.append(error)
        try:
            checkout.release()
        except BaseException as error:  # noqa: BLE001 - recorded for the assertion
            outcome.append(error)

    worker = threading.Thread(target=contend)
    worker.start()
    worker.join(SAFETY_SECONDS)
    assert [type(error) for error in outcome] == [StoreBusyError, RuntimeError]
    assert str(outcome[0]) == "journey store is busy"
    checkout.release()
    checkout.release()
    assert not checkout.held
    with pytest.raises(StoreDeadlineExceededError):
        checkout.acquire(StoreDeadline.after(0))


# --- PostgreSQL ------------------------------------------------------------


def test_postgres_arms_transaction_local_timeouts_from_the_earliest_deadline(
    fake: FakePostgresConnection,
) -> None:
    clock = FakeClock()
    store = _postgres(fake, operation_seconds=20, lock_wait_seconds=2, clock=clock)
    artifact, _ = _artifact()
    start = len(fake.statements)

    written = store.put_artifact(artifact, committed_at=T1)

    assert written.ok and written.created
    issued = _statements_after(fake, start)
    assert issued[0] == "BEGIN ISOLATION LEVEL SERIALIZABLE"
    assert "set_config('statement_timeout', %s, true)" in issued[1]
    assert fake.armed[-1] == {"statement_timeout": "20000", "lock_timeout": "2000"}
    assert fake.local == {} and not fake.in_transaction
    assert "RESET ALL" not in issued

    with store_deadline(StoreDeadline.after(0.75, clock=clock)):
        start = len(fake.statements)
        read = store.get_artifact(artifact.artifact_id)
    assert read.ok and read.value == artifact
    assert _statements_after(fake, start)[0] == "BEGIN READ ONLY"
    assert fake.armed[-1] == {"statement_timeout": "750", "lock_timeout": "750"}

    second, _ = _artifact(2)
    with store.transaction(
        committed_at=T2,
        deadline=StoreDeadline.after(1.5, clock=clock),
    ) as transaction:
        assert fake.armed[-1] == {"statement_timeout": "1500", "lock_timeout": "1500"}
        assert transaction.put_artifact(second).ok
        start = len(fake.statements)
        nested = store.get_artifact(second.artifact_id)
        assert nested.ok
        assert "BEGIN READ ONLY" not in _statements_after(fake, start)
    assert store.latest_revision == 2
    assert fake.local == {} and not fake.in_transaction


def test_postgres_stalled_checkout_returns_store_busy_and_keeps_the_connection(
    fake: FakePostgresConnection,
) -> None:
    clock = FakeClock()
    store = _postgres(fake, clock=clock)
    artifact, _ = _artifact()

    with _StalledHolder(store) as holder:
        start = len(fake.statements)
        with store_deadline(StoreDeadline.after(0.05, clock=clock)):
            busy_write = store.put_artifact(artifact, committed_at=T2)
            busy_read = store.get_artifact(artifact.artifact_id)
        assert holder.thread.is_alive()
        assert _statements_after(fake, start) == []

    for busy in (busy_write, busy_read):
        assert busy.state is StoreState.FAILURE
        assert busy.code == STORE_BUSY_CODE
        assert busy.value is None
        assert artifact.artifact_id not in repr(busy)
    assert not fake.closed and not fake.in_transaction

    retried = store.put_artifact(artifact, committed_at=T2)
    assert retried.ok and retried.created and retried.revision == 1
    store.close()
    assert fake.closed


def test_postgres_lock_wait_cap_bounds_checkout_without_a_caller_deadline(
    fake: FakePostgresConnection,
) -> None:
    store = _postgres(fake, lock_wait_seconds=0.05)
    artifact, _ = _artifact()

    with _StalledHolder(store):
        busy = store.put_artifact(artifact, committed_at=T2)

    assert busy.code == STORE_BUSY_CODE
    assert store.put_artifact(artifact, committed_at=T2).ok


def test_postgres_blocked_query_is_cancelled_and_the_session_is_reset(
    fake: FakePostgresConnection,
) -> None:
    clock = FakeClock()
    store = _postgres(fake, clock=clock)
    artifact, _ = _artifact()
    fake.block = lambda sql: (
        "statement" if sql.startswith("INSERT INTO artifacts") else None
    )
    # A context leaked by an earlier request must not survive the reset.
    fake.session["app.request_policy"] = "synthetic-earlier-request"
    start = len(fake.statements)

    with store_deadline(StoreDeadline.after(0.1, clock=clock)):
        timed_out = store.put_artifact(artifact, committed_at=T1)

    assert timed_out.state is StoreState.FAILURE
    assert timed_out.code == STORE_DEADLINE_EXCEEDED_CODE
    assert fake.unbounded_waits == 0
    assert fake.armed[-1] == {"statement_timeout": "100", "lock_timeout": "100"}
    issued = _statements_after(fake, start)
    blocked_at = next(
        index
        for index, sql in enumerate(issued)
        if sql.startswith("INSERT INTO artifacts")
    )
    assert issued[blocked_at + 1 :] == [
        "RESET ALL",
        "SET search_path TO openmed_journey",
    ]
    assert fake.session == {"search_path": "openmed_journey"}
    assert fake.local == {} and not fake.in_transaction
    assert not fake.notices and not fake.closed

    fake.block = lambda sql: None
    missing = store.get_artifact(artifact.artifact_id)
    assert missing.code == "artifact_not_found"
    assert store.latest_revision is None
    written = store.put_artifact(artifact, committed_at=T1)
    assert written.ok and written.created and written.revision == 1


def test_postgres_lock_timeout_reports_store_busy(
    fake: FakePostgresConnection,
) -> None:
    store = _postgres(fake, lock_wait_seconds=0.05, clock=FakeClock())
    artifact, _ = _artifact()
    fake.block = lambda sql: (
        "lock" if sql.startswith("LOCK TABLE store_revisions") else None
    )

    busy = store.put_artifact(artifact, committed_at=T1)

    assert busy.code == STORE_BUSY_CODE
    assert fake.armed[-1] == {"statement_timeout": "30000", "lock_timeout": "50"}
    assert fake.unbounded_waits == 0
    assert "RESET ALL" in fake.statements
    fake.block = lambda sql: None
    assert store.put_artifact(artifact, committed_at=T1).ok


def test_postgres_never_commits_after_the_deadline(
    fake: FakePostgresConnection,
) -> None:
    clock = FakeClock()
    store = _postgres(fake, clock=clock)
    artifact, _ = _artifact()

    with pytest.raises(StoreDeadlineExceededError):
        with store.transaction(
            committed_at=T1,
            deadline=StoreDeadline.after(5, clock=clock),
        ) as transaction:
            assert transaction.put_artifact(artifact).ok
            clock.advance(6)
            start = len(fake.statements)
            with pytest.raises(StoreDeadlineExceededError):
                transaction.put_evidence(_evidence(artifact))
            assert _statements_after(fake, start) == []

    assert "RESET ALL" in fake.statements
    assert not fake.in_transaction
    assert store.get_artifact(artifact.artifact_id).code == "artifact_not_found"
    assert store.latest_revision is None


def test_postgres_rearms_statement_timeout_as_the_deadline_approaches(
    fake: FakePostgresConnection,
) -> None:
    clock = FakeClock()
    store = _postgres(fake, clock=clock)
    artifact, _ = _artifact()

    with store.transaction(
        committed_at=T1,
        deadline=StoreDeadline.after(1, clock=clock),
    ) as transaction:
        assert fake.local["statement_timeout"] == "1000"
        clock.advance(0.01)
        assert transaction.put_artifact(artifact).ok
        assert fake.local["statement_timeout"] == "1000"
        clock.advance(0.49)
        assert transaction.put_evidence(_evidence(artifact)).ok
        assert fake.local["statement_timeout"] == "500"
        assert fake.armed[-1] == {"statement_timeout": "500"}

    assert store.get_evidence(_evidence(artifact).locator_id).ok


def test_postgres_ambiguous_commit_is_reported_and_reconciled_by_replay(
    fake: FakePostgresConnection,
) -> None:
    store = _postgres(fake)
    landed, _ = _artifact(1)
    refused, _ = _artifact(2)

    fake.fail_commit = "lost_ack"
    unknown_landed = store.put_artifact(landed, committed_at=T1)
    fake.fail_commit = "refused"
    unknown_refused = store.put_artifact(refused, committed_at=T1)
    fake.fail_commit = None

    for unknown in (unknown_landed, unknown_refused):
        assert unknown.state is StoreState.UNKNOWN
        assert unknown.code == "commit_status_unknown"
    assert fake.statements.count("RESET ALL") == 2
    assert not fake.closed and not fake.in_transaction

    replay_landed = store.put_artifact(landed, committed_at=T2)
    replay_refused = store.put_artifact(refused, committed_at=T2)
    assert replay_landed.ok and not replay_landed.created
    assert replay_refused.ok and replay_refused.created


def test_postgres_connection_that_cannot_be_reset_is_closed_not_reused(
    fake: FakePostgresConnection,
) -> None:
    clock = FakeClock()
    store = _postgres(fake, clock=clock)
    artifact, _ = _artifact()
    fake.block = lambda sql: (
        "statement" if sql.startswith("INSERT INTO artifacts") else None
    )
    fake.fail_reset = True

    with store_deadline(StoreDeadline.after(0.05, clock=clock)):
        timed_out = store.put_artifact(artifact, committed_at=T1)

    assert timed_out.code == STORE_DEADLINE_EXCEEDED_CODE
    assert fake.closed
    issued = len(fake.statements)
    after_close = store.put_artifact(artifact, committed_at=T1)
    assert after_close.state is StoreState.FAILURE
    assert len(fake.statements) == issued


def test_postgres_errors_and_connection_retain_no_parameter_diagnostics(
    fake: FakePostgresConnection,
) -> None:
    store = _postgres(fake)
    artifact, _ = _artifact()
    fake.fail_on = lambda sql: sql.startswith("INSERT INTO artifacts")

    with pytest.raises(PostgresStoreError) as caught:
        with store.transaction(committed_at=T1) as transaction:
            transaction.put_artifact(artifact)

    error = caught.value
    assert error.__cause__ is None and error.__context__ is None
    rendered = "".join(traceback.format_exception(error))
    assert SYNTHETIC_VALUE not in rendered
    assert artifact.artifact_id not in rendered
    assert not fake.notices
    assert fake.session == {"search_path": "openmed_journey"}
    fake.fail_on = lambda sql: False
    assert store.put_artifact(artifact, committed_at=T1).ok
    assert not fake.notices


def test_postgres_connect_rejects_invalid_timeouts_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_import(name: str) -> None:
        raise AssertionError("driver import attempted")

    monkeypatch.setattr(
        "openmed.structured.store.postgres.import_module",
        forbidden_import,
    )
    monkeypatch.setenv(STORE_TIMEOUT_ENV_VAR, "synthetic-invalid")

    result = PostgresJourneyStore.connect("postgresql://openmed:secret@db/openmed")
    opened = PostgresJourneyStore.open(FakePostgresConnection())

    for outcome in (result, opened):
        assert outcome.state is StoreState.FAILURE
        assert outcome.code == STORE_TIMEOUTS_INVALID_CODE
        assert "synthetic-invalid" not in repr(outcome)
        assert "secret" not in repr(outcome)


def test_composed_ingestion_discards_new_bytes_when_metadata_is_busy(
    fake: FakePostgresConnection,
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    artifacts = LocalArtifactStore(tmp_path / "objects")
    store = ComposedJourneyStore(artifacts, _postgres(fake, clock=clock))
    artifact, content = _artifact()

    with _StalledHolder(store.metadata):
        with store_deadline(StoreDeadline.after(0.05, clock=clock)):
            busy = store.ingest_graph(artifact, content, committed_at=T2)

    assert busy.code == STORE_BUSY_CODE
    assert artifacts.get_bytes(artifact.content_hash).code == "artifact_not_found"
    assert store.ingest_graph(artifact, content, committed_at=T2).ok


# --- SQLite ----------------------------------------------------------------


def test_sqlite_stalled_checkout_returns_store_busy_and_discards_new_bytes(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    store = LocalJourneyStore(
        tmp_path / "journey",
        timeouts=StoreTimeouts(clock=clock),
    )
    artifact, content = _artifact()

    with _StalledHolder(store.metadata) as holder:
        with store_deadline(StoreDeadline.after(0.05, clock=clock)):
            busy_put = store.metadata.put_artifact(artifact, committed_at=T2)
            busy_graph = store.ingest_graph(artifact, content, committed_at=T2)
        assert holder.thread.is_alive()

    for busy in (busy_put, busy_graph):
        assert busy.state is StoreState.FAILURE
        assert busy.code == STORE_BUSY_CODE
    assert store.artifacts.get_bytes(artifact.content_hash).code == (
        "artifact_not_found"
    )
    graph = store.ingest_graph(artifact, content, committed_at=T2)
    assert graph.ok and graph.revision == 1
    store.close()


def test_sqlite_write_lock_wait_is_bounded_and_busy_timeout_restored(
    tmp_path: Path,
) -> None:
    path = tmp_path / "journey.sqlite3"
    store = SQLiteJourneyStore(path, timeouts=StoreTimeouts(lock_wait_seconds=0.05))
    blocker = sqlite3.connect(path, isolation_level=None)
    artifact, _ = _artifact()
    try:
        blocker.execute("BEGIN IMMEDIATE")
        busy = store.put_artifact(artifact, committed_at=T1)
        assert busy.code == STORE_BUSY_CODE
        assert store._connection.execute("PRAGMA busy_timeout").fetchone()[0] == 50
        assert not store._connection.in_transaction
        blocker.execute("ROLLBACK")
        assert store.put_artifact(artifact, committed_at=T1).ok
    finally:
        blocker.close()
        store.close()


def test_sqlite_statement_past_the_deadline_is_interrupted_and_rolled_back(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    store = SQLiteJourneyStore(
        tmp_path / "journey.sqlite3",
        timeouts=StoreTimeouts(operation_seconds=10, clock=clock),
    )
    artifact, _ = _artifact()

    with pytest.raises(StoreDeadlineExceededError):
        with store.transaction(committed_at=T1) as transaction:
            assert transaction.put_artifact(artifact).ok
            clock.advance(11)
            with pytest.raises(sqlite3.OperationalError, match="interrupted"):
                store._connection.execute(
                    LONG_SQLITE_QUERY.format(limit=10_000_000)
                ).fetchone()

    assert store.get_artifact(artifact.artifact_id).code == "artifact_not_found"
    assert store.latest_revision is None
    # The deadline handler is removed, so later statements run to completion.
    finished = store._connection.execute(LONG_SQLITE_QUERY.format(limit=20_000))
    assert finished.fetchone()[0] == 20_000
    assert store._connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5_000
    store.close()


def test_sqlite_explicit_deadline_and_budget_bound_transactions(tmp_path: Path) -> None:
    store = SQLiteJourneyStore(tmp_path / "journey.sqlite3")
    artifact, _ = _artifact()

    with pytest.raises(StoreDeadlineExceededError):
        with store.transaction(committed_at=T1, deadline=0):
            pytest.fail("an expired deadline must not open a transaction")

    clock = FakeClock()
    with pytest.raises(StoreDeadlineExceededError):
        with store.transaction(
            committed_at=T1,
            deadline=StoreDeadline.after(1, clock=clock),
        ) as transaction:
            assert transaction.put_artifact(artifact).ok
            clock.advance(2)
    assert store.latest_revision is None

    budget = RequestBudget(max_wall_time=30).start()
    with store.transaction(committed_at=T1, deadline=budget) as transaction:
        assert transaction.put_artifact(artifact).ok
    assert store.latest_revision == 1
    store.close()


class _CommitFailingConnection:
    """Proxy that fails COMMIT, optionally after applying it."""

    def __init__(self, inner: sqlite3.Connection, *, apply_commit: bool) -> None:
        self._inner = inner
        self._apply_commit = apply_commit

    def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
        if sql == "COMMIT":
            if self._apply_commit:
                self._inner.execute("COMMIT")
            raise sqlite3.OperationalError("synthetic disk I/O error")
        return self._inner.execute(sql, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_sqlite_ambiguous_commit_is_unknown_and_provable_failure_rolls_back(
    tmp_path: Path,
) -> None:
    store = LocalJourneyStore(tmp_path / "journey")
    metadata = store.metadata
    inner = metadata._connection
    landed, landed_bytes = _artifact(1)
    refused, _ = _artifact(2)
    try:
        metadata._connection = _CommitFailingConnection(inner, apply_commit=True)
        unknown = store.ingest_graph(landed, landed_bytes, committed_at=T1)
        metadata._connection = _CommitFailingConnection(inner, apply_commit=False)
        failed = metadata.put_artifact(refused, committed_at=T1)
    finally:
        metadata._connection = inner

    assert unknown.state is StoreState.UNKNOWN
    assert unknown.code == "commit_status_unknown"
    # An unacknowledged commit may reference the bytes, so they are kept.
    assert store.artifacts.get_bytes(landed.content_hash).ok
    assert failed.state is StoreState.FAILURE and failed.code == "transaction_failed"
    assert metadata.get_artifact(refused.artifact_id).code == "artifact_not_found"
    replay = metadata.put_artifact(landed, committed_at=T2)
    assert replay.ok and not replay.created
    store.close()


def test_sqlite_invalid_timeout_environment_fails_before_creating_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(STORE_LOCK_WAIT_ENV_VAR, "synthetic-invalid")
    path = tmp_path / "journey" / "journey.sqlite3"

    opened = SQLiteJourneyStore.open(path)

    assert opened.state is StoreState.FAILURE
    assert opened.code == STORE_TIMEOUTS_INVALID_CODE
    assert "synthetic-invalid" not in repr(opened)
    assert not path.parent.exists()
    monkeypatch.setenv(STORE_LOCK_WAIT_ENV_VAR, "0.5")
    configured = SQLiteJourneyStore(path)
    assert configured.timeouts.lock_wait_seconds == 0.5
    assert configured._connection.execute("PRAGMA busy_timeout").fetchone()[0] == 500
    configured.close()
