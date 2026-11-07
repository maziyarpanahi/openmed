"""Offline rehearsal of forward and rollback-safe migrations for durable schemas.

:func:`rehearse_service_migrations` exercises every present component in
:mod:`openmed.service.schema_registry` through its *native* engine, against
synthetic fixtures in a throwaway workspace, for every supported source
version:

``forward_upgrade``
    Upgrade from the source version (``0`` is a clean install); the result
    is ready at the current version and aggregate counts are preserved.
``repeat``
    Run the migration again; nothing changes.
``interrupted``
    Kill the migrating process mid-migration (a child process dies inside
    the SQLite transaction or at the atomic publish), drop the PostgreSQL
    connection mid-transaction, or inject a fault mid-write; the schema stays
    at the source version, readiness stays false, and a retry completes.
``drift``
    Tamper with the recorded migration history; the engine refuses with its
    controlled code before mutating anything.
``unknown_version``
    Present state written by a newer binary (old binary against new schema);
    the engine refuses with its controlled code before mutating anything.
``previous_binary``
    The previous release's engine against the current schema behaves as the
    registry declares.  It is executed where the previous engine is
    reproducible from this codebase (the SQLite engine with the released
    migration chain, in a child process) and reported as ``declared``
    otherwise.
``restore_rollback`` and ``retry_after_recovery``
    Capture a recovery point of the source state with
    :func:`~openmed.service.recovery.create_recovery_point`, upgrade, restore
    the point into a new target, keep readiness false until activation,
    activate it, prove the previous state (and, where reproducible, the
    previous binary) is back, and then re-run the forward migration.

PostgreSQL scenarios run the real :class:`PostgresJourneyStore` engine and
the deployment readiness rule from :mod:`openmed.deploy.journey`.  Offline,
the engine talks to :class:`SQLitePostgresDouble`, a SQLite-backed DB-API
double that emulates the few PostgreSQL control statements the engine
issues; pass ``postgres_connect`` to rehearse against a real server instead.

The :class:`RehearsalReport` is a value-free compatibility matrix (versions,
categories, counts, schema-metadata digests, states, reason codes) with a
release gate verdict and an optional HMAC-SHA256 signature.  Fixtures are
synthetic, keys are ephemeral and never persisted, and nothing outside the
workspace (and, when supplied, the rehearsal's own PostgreSQL schemas) is
touched.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import openmed
from openmed.__about__ import __version__
from openmed.core.audit import AuditSignature
from openmed.core.registry_slots import registry_slot_key
from openmed.service._rehearsal_worker import KILL_EXIT_CODE
from openmed.service.recovery import (
    AUDIT_CHAIN_DIRECTORY,
    BACKUP_KEY_PURPOSE,
    JOB_STORE_PATH,
    REGISTRY_STATE_FILE,
    ServiceStateSources,
    activate_restored_state,
    create_recovery_point,
    journey_store_check,
    restore_recovery_point,
    verify_recovery_point,
)
from openmed.service.schema_registry import (
    AUDIT_CHAIN,
    JOB_STORE,
    JOURNEY_ARTIFACT_MARKER,
    JOURNEY_POSTGRES,
    JOURNEY_SQLITE,
    MODEL_REGISTRY,
    PREVIOUS_RELEASE,
    RECOVERY_MANIFEST,
    RELEASED_CHECKSUMS,
    ComponentObservation,
    DurableSchema,
    ObservedState,
    Presence,
    PreviousBinary,
    RecoveryPointKind,
    RollbackClass,
    durable_schemas,
    get_schema,
    observe_artifact_marker,
    observe_audit_chains,
    observe_job_store,
    observe_journey_postgres,
    observe_journey_sqlite,
    observe_model_registry,
    observe_recovery_point,
    plan_service_migrations,
    registry_digest,
    registry_findings,
)
from openmed.structured.store.migrations import MIGRATIONS
from openmed.structured.store.postgres import POSTGRES_MIGRATIONS

__all__ = [
    "REHEARSAL_SCHEMA",
    "SIGNATURE_ALGORITHM",
    "ComponentRehearsal",
    "RehearsalReport",
    "Scenario",
    "ScenarioOutcome",
    "ScenarioResult",
    "SQLitePostgresDouble",
    "rehearse_service_migrations",
]

REHEARSAL_SCHEMA = "openmed.service.migration-rehearsal.v1"
SIGNATURE_ALGORITHM = "HMAC-SHA256"

_COMMITTED_AT = "2026-01-01T00:00:00Z"
_SEEDED_REVISIONS = 3
_WORKER_TIMEOUT_SECONDS = 300
_KEY_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_PG_HISTORY_DDL = """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version BIGINT PRIMARY KEY,
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
_SQLITE_HISTORY_DDL = """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                checksum TEXT NOT NULL
            )
            """


class Scenario(str, Enum):
    """Rehearsal scenarios, in report order."""

    FORWARD_UPGRADE = "forward_upgrade"
    REPEAT = "repeat"
    INTERRUPTED = "interrupted"
    DRIFT = "drift"
    UNKNOWN_VERSION = "unknown_version"
    PREVIOUS_BINARY = "previous_binary"
    RESTORE_ROLLBACK = "restore_rollback"
    RETRY_AFTER_RECOVERY = "retry_after_recovery"


_SCENARIO_ORDER = {scenario: index for index, scenario in enumerate(Scenario)}


class ScenarioOutcome(str, Enum):
    """Outcome of one scenario.

    ``declared`` records a registry declaration that cannot be executed
    offline; ``external`` records a rollback path owned by the operator (for
    example a PostgreSQL backup).  Neither is a failure.
    """

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    DECLARED = "declared"
    EXTERNAL = "external"


@dataclass(frozen=True)
class ScenarioResult:
    """Value-free result of one scenario for one source version.

    Attributes:
        component: Component identifier.
        scenario: Scenario name.
        source_version: Source schema version (``0`` is no prior state).
        outcome: Scenario outcome.
        code: Controlled reason code: the refusal code observed, or the
            first failing check.
        evidence: How the result was produced, for example
            ``native_engine``, ``killed_process``, ``connection_lost``,
            ``fault_injected``, ``previous_release_chain``, ``declared``.
        observed_version: Schema version observed at the end.
        checks: Named boolean assertions the outcome is derived from.
        counts: Aggregate counters.
    """

    component: str
    scenario: Scenario
    source_version: int
    outcome: ScenarioOutcome
    code: str | None = None
    evidence: str = "native_engine"
    observed_version: int | None = None
    checks: Mapping[str, bool] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free mapping."""

        return {
            "checks": dict(sorted(self.checks.items())),
            "code": self.code,
            "component": self.component,
            "counts": dict(sorted(self.counts.items())),
            "evidence": self.evidence,
            "observed_version": self.observed_version,
            "outcome": self.outcome.value,
            "scenario": self.scenario.value,
            "source_version": self.source_version,
        }


@dataclass(frozen=True)
class ComponentRehearsal:
    """Rehearsal results for one registered component.

    Attributes:
        component: Component identifier.
        presence: ``present`` or ``not_present``.
        current_version: Version this binary writes.
        rollback_class: Registry rollback class of this release's transition.
        engine: Engine the scenarios ran against.
        scenarios: Scenario results.
        tracking_issue: Issue that adds a ``not_present`` component.
    """

    component: str
    presence: str
    current_version: int | None
    rollback_class: str | None
    engine: str
    scenarios: tuple[ScenarioResult, ...] = ()
    tracking_issue: int | None = None

    @property
    def failed(self) -> tuple[ScenarioResult, ...]:
        """Return the failed scenarios."""

        return tuple(
            item for item in self.scenarios if item.outcome is ScenarioOutcome.FAILED
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free mapping."""

        return {
            "component": self.component,
            "current_version": self.current_version,
            "engine": self.engine,
            "presence": self.presence,
            "rollback_class": self.rollback_class,
            "scenarios": [item.to_dict() for item in self.scenarios],
            "tracking_issue": self.tracking_issue,
        }


@dataclass(frozen=True)
class RehearsalReport:
    """Signed-able, value-free compatibility matrix and release gate verdict.

    Attributes:
        components: Per-component results in registry order.
        gate_state: ``passed``, ``blocked``, or ``partial`` (a subset ran).
        gate_codes: Controlled codes of every blocking finding.
        registry_digest: Digest of the canonical registry document.
        generated_at: UTC timestamp.
        release: Version of this binary.
        previous_release: Release a rollback returns to.
        schema: Report schema identifier.
        packet_digest: Digest over every unsigned field, set by :meth:`sign`.
        signature: HMAC-SHA256 signature, set by :meth:`sign`.
    """

    components: tuple[ComponentRehearsal, ...]
    gate_state: str
    gate_codes: tuple[str, ...]
    registry_digest: str
    generated_at: str
    release: str = str(__version__)
    previous_release: str = PREVIOUS_RELEASE
    schema: str = REHEARSAL_SCHEMA
    packet_digest: str = ""
    signature: AuditSignature | None = None

    @property
    def passed(self) -> bool:
        """Return whether the release gate passed."""

        return self.gate_state == "passed"

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "components": [item.to_dict() for item in self.components],
            "gate": {"codes": list(self.gate_codes), "state": self.gate_state},
            "generated_at": self.generated_at,
            "previous_release": self.previous_release,
            "registry_digest": self.registry_digest,
            "release": self.release,
            "schema": self.schema,
        }

    def recompute_packet_digest(self) -> str:
        """Recompute the packet digest without trusting the stored value."""

        return _digest(_canonical_json(self._unsigned_payload()))

    def sign(self, key: bytes, *, key_id: str) -> "RehearsalReport":
        """Return a copy signed with HMAC-SHA256 over every field.

        Args:
            key: Caller-owned signing key of at least 32 bytes.
            key_id: Operator label for the key (letters, digits, ``._-``).

        Returns:
            The signed report.

        Raises:
            ValueError: If the key or key identifier is invalid.
        """

        if not isinstance(key, bytes) or len(key) < 32:
            raise ValueError("signing key must contain at least 32 bytes")
        if not isinstance(key_id, str) or _KEY_ID_RE.fullmatch(key_id) is None:
            raise ValueError("signing key id must be a short controlled label")
        packet_digest = self.recompute_packet_digest()
        value = hmac.new(
            key,
            _canonical_json(
                {**self._unsigned_payload(), "packet_digest": packet_digest}
            ),
            hashlib.sha256,
        ).hexdigest()
        return replace(
            self,
            packet_digest=packet_digest,
            signature=AuditSignature(
                key_id=key_id, algorithm=SIGNATURE_ALGORITHM, value=value
            ),
        )

    def verify(self, key: bytes) -> bool:
        """Return whether the packet digest and signature are intact.

        Args:
            key: The signing key.

        Returns:
            ``True`` only for an intact, signed report.
        """

        if self.signature is None or self.signature.algorithm != SIGNATURE_ALGORITHM:
            return False
        if self.recompute_packet_digest() != self.packet_digest:
            return False
        expected = hmac.new(
            key,
            _canonical_json(
                {**self._unsigned_payload(), "packet_digest": self.packet_digest}
            ),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, self.signature.value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free mapping."""

        return {
            **self._unsigned_payload(),
            "packet_digest": self.packet_digest or self.recompute_packet_digest(),
            "signature": None if self.signature is None else self.signature.to_dict(),
        }

    def to_json(self) -> str:
        """Return the report as deterministic JSON text."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def summary(self) -> str:
        """Return a short human summary of the matrix."""

        present = [item for item in self.components if item.presence == "present"]
        absent = [item for item in self.components if item.presence != "present"]
        lines = [
            f"migration rehearsal {self.gate_state}: release {self.release}, "
            f"previous release {self.previous_release}, {len(present)} present "
            f"component(s), {len(absent)} not present"
        ]
        for item in present:
            cells = []
            for scenario in Scenario:
                rows = [row for row in item.scenarios if row.scenario is scenario]
                if not rows:
                    continue
                cells.append(f"{scenario.value} {_cell(rows)}")
            lines.append(
                f"  {item.component} v{item.current_version} "
                f"{item.rollback_class} ({item.engine}): " + ", ".join(cells)
            )
        for item in absent:
            lines.append(
                f"  {item.component}: not_present (issue #{item.tracking_issue})"
            )
        for code in self.gate_codes:
            lines.append(f"  blocking: {code}")
        return "\n".join(lines)


def _cell(rows: Sequence[ScenarioResult]) -> str:
    executed = [
        row
        for row in rows
        if row.outcome in {ScenarioOutcome.PASSED, ScenarioOutcome.FAILED}
    ]
    if executed:
        passed = sum(1 for row in executed if row.outcome is ScenarioOutcome.PASSED)
        return f"{passed}/{len(executed)}"
    return rows[0].outcome.value


# ---------------------------------------------------------------------------
# PostgreSQL DB-API double and fault injection
# ---------------------------------------------------------------------------


class _DoubleError(Exception):
    """Emulated database error; carries no SQL or values."""


_PG_QUALIFIED = re.compile(
    r"\b(FROM|INTO|UPDATE|TABLE|JOIN)\s+[a-z][a-z0-9_]{0,62}\.", re.IGNORECASE
)
_PG_TABLES = re.compile(
    r"^SELECT table_name FROM information_schema\.tables", re.IGNORECASE
)


class _DoubleCursor:
    def __init__(self, connection: "_DoubleConnection") -> None:
        self._owner = connection
        self._rows: list[tuple[Any, ...]] = []
        self.description: list[tuple[Any, ...]] | None = None
        self.rowcount = -1

    def execute(self, operation: str, parameters: Sequence[Any] = ()) -> None:
        self._owner._require_open()
        sql = operation.strip()
        head = " ".join(sql.split()[:3]).upper()
        self.description = None
        self._rows = []
        if head.startswith(("CREATE SCHEMA", "SET ", "LOCK TABLE", "DROP SCHEMA")):
            return
        if head.startswith("SELECT PG_ADVISORY_XACT_LOCK"):
            self.description = [("pg_advisory_xact_lock",)]
            self._rows = [(None,)]
            return
        if head.startswith("BEGIN"):
            self._owner._sqlite.execute("BEGIN IMMEDIATE")
            return
        if head in {"COMMIT", "ROLLBACK"}:
            if self._owner._sqlite.in_transaction:
                self._owner._sqlite.execute(head)
            return
        if _PG_TABLES.match(sql):
            sql = "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            parameters = ()
        sql = _PG_QUALIFIED.sub(lambda match: f"{match.group(1)} ", sql)
        try:
            cursor = self._owner._sqlite.execute(
                sql.replace("%s", "?"), tuple(parameters)
            )
        except sqlite3.IntegrityError:
            raise _DoubleError({"C": "23505"}) from None
        except sqlite3.Error:
            raise _DoubleError({"C": "XX000"}) from None
        self.description = (
            None
            if cursor.description is None
            else [tuple(column) for column in cursor.description]
        )
        self._rows = list(cursor.fetchall()) if cursor.description else []
        self.rowcount = cursor.rowcount

    def executemany(self, operation: str, parameters: Sequence[Sequence[Any]]) -> None:
        for item in parameters:
            self.execute(operation, item)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows.pop(0) if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows, self._rows = self._rows, []
        return rows

    def close(self) -> None:
        self._rows = []


class _DoubleConnection:
    def __init__(self, path: Path) -> None:
        self._sqlite = sqlite3.connect(
            path, isolation_level=None, check_same_thread=False, timeout=5.0
        )
        self._sqlite.execute("PRAGMA foreign_keys = ON")
        self._closed = False
        self.autocommit = False

    def _require_open(self) -> None:
        if self._closed:
            raise _DoubleError({"C": "08003"})

    def cursor(self) -> _DoubleCursor:
        self._require_open()
        return _DoubleCursor(self)

    def commit(self) -> None:
        self._require_open()
        if self._sqlite.in_transaction:
            self._sqlite.execute("COMMIT")

    def rollback(self) -> None:
        self._require_open()
        if self._sqlite.in_transaction:
            self._sqlite.execute("ROLLBACK")

    def close(self) -> None:
        if not self._closed:
            # Like a server dropping a session: an open transaction aborts.
            self._sqlite.close()
            self._closed = True


class SQLitePostgresDouble:
    """SQLite-backed DB-API double for offline PostgreSQL migration rehearsal.

    Each double is one emulated PostgreSQL schema stored in a SQLite file.
    It emulates the control statements the Journey PostgreSQL engine issues
    (``CREATE SCHEMA``, ``SET [LOCAL] search_path``, ``pg_advisory_xact_lock``,
    ``LOCK TABLE``, explicit ``BEGIN``/``COMMIT``/``ROLLBACK``) and the
    ``information_schema.tables`` listing, ignores schema qualification, and
    runs the engine's DDL and DML on SQLite.  Closing a connection mid
    transaction aborts it, as a server does for a dropped session.  It proves
    the engine's control flow (history checks, drift and version refusal,
    transactional rollback, idempotence) offline; it is not a substitute for
    the opt-in server rehearsal.

    Args:
        path: SQLite file backing the emulated schema.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> Any:
        """Return a new DB-API connection to the emulated schema."""

        return _DoubleConnection(self.path)


class _ConnectionLost(Exception):
    """Injected mid-migration connection loss."""


class _FaultingCursor:
    def __init__(self, owner: "_FaultingConnection", cursor: Any) -> None:
        self._owner = owner
        self._cursor = cursor

    @property
    def description(self) -> Any:
        return self._cursor.description

    @property
    def rowcount(self) -> int:
        return int(getattr(self._cursor, "rowcount", -1))

    def execute(self, operation: str, parameters: Sequence[Any] = ()) -> Any:
        self._owner._before(operation)
        return self._cursor.execute(operation, parameters)

    def executemany(self, operation: str, parameters: Sequence[Sequence[Any]]) -> Any:
        self._owner._before(operation)
        return self._cursor.executemany(operation, parameters)

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    def fetchall(self) -> Any:
        return self._cursor.fetchall()

    def close(self) -> None:
        try:
            self._cursor.close()
        except Exception:
            pass


class _FaultingConnection:
    """Drop the session when the Nth migration-history insert starts."""

    def __init__(self, connection: Any, *, kill_after: int) -> None:
        self._connection = connection
        self._kill_after = kill_after
        self._seen = 0
        self.lost = False

    @property
    def autocommit(self) -> bool:
        return bool(self._connection.autocommit)

    @autocommit.setter
    def autocommit(self, value: bool) -> None:
        self._connection.autocommit = value

    def _before(self, operation: str) -> None:
        if self.lost:
            raise _ConnectionLost
        normalized = " ".join(operation.split()).upper()
        if normalized.startswith("INSERT INTO SCHEMA_MIGRATIONS"):
            self._seen += 1
            if self._seen >= self._kill_after:
                self.lost = True
                try:
                    self._connection.close()
                except Exception:
                    pass
                raise _ConnectionLost

    def cursor(self) -> _FaultingCursor:
        if self.lost:
            raise _ConnectionLost
        return _FaultingCursor(self, self._connection.cursor())

    def commit(self) -> None:
        if self.lost:
            raise _ConnectionLost
        self._connection.commit()

    def rollback(self) -> None:
        if self.lost:
            raise _ConnectionLost
        self._connection.rollback()

    def close(self) -> None:
        if not self.lost:
            self._connection.close()


class _DoubleBackend:
    engine = "native_engine_sqlite_dbapi_double"

    def __init__(self, root: Path) -> None:
        self._root = root

    def new_schema(self) -> str:
        return f"rehearsal_{secrets.token_hex(6)}"

    def connect(self, schema: str) -> Any:
        return SQLitePostgresDouble(self._root / f"{schema}.sqlite3").connect()

    def cleanup(self, schema: str) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            (self._root / f"{schema}.sqlite3{suffix}").unlink(missing_ok=True)


class _ServerBackend:
    engine = "native_engine_postgresql_server"

    def __init__(self, connect: Callable[[], Any]) -> None:
        self._connect = connect

    def new_schema(self) -> str:
        return f"openmed_rehearsal_{secrets.token_hex(6)}"

    def connect(self, schema: str) -> Any:
        del schema
        return self._connect()

    def cleanup(self, schema: str) -> None:
        connection = self._connect()
        try:
            connection.autocommit = True
            _pg_execute(connection, f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        finally:
            connection.close()


def _pg_execute(connection: Any, sql: str, parameters: Sequence[Any] = ()) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute(sql, tuple(parameters))
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# Component adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Probe:
    """Native-engine outcome: ready, refusal code, and version."""

    ready: bool
    code: str | None = None
    version: int | None = None
    evidence: str = "native_engine"
    interrupted: bool = False


@dataclass(frozen=True)
class _Previous:
    outcome: ScenarioOutcome
    code: str | None
    evidence: str
    ready: bool = False


class _Runner:
    """Per-run shared state: workspace, ephemeral keys, and worker launcher."""

    def __init__(self, workspace: Path, postgres: Any) -> None:
        from openmed.core.key_provider import InMemoryKeyProvider
        from openmed.structured.store import (
            JOURNEY_KEY_PURPOSE,
            JourneyStoreEncryption,
        )

        self.workspace = workspace
        self.postgres = postgres
        # Ephemeral keys: synthetic stores only, never persisted or printed.
        self.journey_encryption = JourneyStoreEncryption.encrypted(
            InMemoryKeyProvider.generate(JOURNEY_KEY_PURPOSE)
        )
        self.backup_provider = InMemoryKeyProvider.generate(BACKUP_KEY_PURPOSE)
        package_root = Path(openmed.__file__).resolve().parent.parent
        existing = os.environ.get("PYTHONPATH")
        self._env = {
            **os.environ,
            "PYTHONPATH": (
                str(package_root)
                if not existing
                else os.pathsep.join((str(package_root), existing))
            ),
        }

    def worker(self, *arguments: str) -> tuple[int, dict[str, Any]]:
        """Run one operation in a throwaway child process."""

        command = [
            sys.executable,
            "-P",
            "-c",
            "import sys\n"
            "from openmed.service._rehearsal_worker import main\n"
            "raise SystemExit(main(sys.argv[1:]))",
            *arguments,
        ]
        try:
            completed = subprocess.run(  # noqa: S603 - fixed interpreter and code
                command,
                cwd=self.workspace,
                env=self._env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=_WORKER_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return -1, {"state": "failed", "code": "worker_unavailable"}
        payload: dict[str, Any] = {}
        lines = completed.stdout.decode("utf-8", "replace").strip().splitlines()
        if lines:
            try:
                parsed = json.loads(lines[-1])
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                payload = parsed
        return completed.returncode, payload


class _Adapter:
    """Scenario primitives for one component, implemented natively."""

    component = ""
    engine = "native_engine"
    interruptible: frozenset[int] = frozenset()
    observer_detects_drift = True

    def __init__(self, runner: _Runner, schema: DurableSchema) -> None:
        self.runner = runner
        self.schema = schema

    @property
    def current(self) -> int:
        return int(self.schema.current_version or 0)

    def seed(self, root: Path, version: int) -> None:
        raise NotImplementedError

    def migrate(self, root: Path) -> _Probe:
        raise NotImplementedError

    def observe(self, root: Path) -> ComponentObservation:
        raise NotImplementedError

    def fingerprint(self, root: Path) -> str:
        raise NotImplementedError

    def tamper(self, root: Path, version: int) -> bool:
        return False

    def future(self, root: Path, version: int) -> None:
        raise NotImplementedError

    def interrupt(self, root: Path, version: int) -> _Probe:
        raise NotImplementedError

    def previous_binary(self, root: Path) -> _Previous:
        previous = self.schema.previous_binary
        if previous is PreviousBinary.IGNORES:
            return _Previous(
                ScenarioOutcome.NOT_APPLICABLE, "introduced_in_release", "declared"
            )
        return _Previous(
            ScenarioOutcome.DECLARED,
            self.schema.previous_binary_code
            or ("previous_release_reads" if previous is PreviousBinary.READS else None),
            "declared",
        )

    def activation_checks(self) -> list[Any]:
        return []

    def cleanup(self, root: Path) -> None:
        del root

    def count_strays(self, root: Path) -> int:
        del root
        return 0


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)


def _write_private(path: Path, data: bytes) -> None:
    _private_dir(path.parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _file_digest(path: Path) -> str:
    if not path.exists():
        return "absent"
    return _digest(path.read_bytes())


def _sqlite_dump_digest(database: Path) -> str:
    if not database.exists():
        return "absent"
    connection = sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=rw", uri=True, timeout=5.0
    )
    try:
        return _digest("\n".join(connection.iterdump()).encode("utf-8"))
    finally:
        connection.close()


def _seed_sqlite_chain(database: Path, chain: Sequence[Any]) -> None:
    """Write the store a release with exactly ``chain`` left behind."""

    _private_dir(database.parent)
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        os.chmod(database, 0o600)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(_SQLITE_HISTORY_DDL)
        for migration in chain:
            for statement in migration.statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, name, checksum) "
                "VALUES (?, ?, ?)",
                (migration.version, migration.name, migration.checksum),
            )
        if chain:
            for _ in range(_SEEDED_REVISIONS):
                connection.execute(
                    "INSERT INTO store_revisions(committed_at) VALUES (?)",
                    (_COMMITTED_AT,),
                )
        connection.execute("COMMIT")
    finally:
        connection.close()


class _JourneySQLite(_Adapter):
    component = JOURNEY_SQLITE

    def __init__(self, runner: _Runner, schema: DurableSchema) -> None:
        super().__init__(runner, schema)
        self.interruptible = frozenset(range(0, self.current))

    def _database(self, root: Path) -> Path:
        return root / "journey" / "journey.sqlite3"

    def seed(self, root: Path, version: int) -> None:
        if version == 0:
            return
        _seed_sqlite_chain(self._database(root), MIGRATIONS[:version])
        if version == self.current:
            # A current store has been opened by this binary at least once.
            self.migrate(root)

    def migrate(self, root: Path) -> _Probe:
        from openmed.structured.store import SQLiteJourneyStore

        result = SQLiteJourneyStore.open(
            self._database(root), encryption=self.runner.journey_encryption
        )
        if not result.ok or result.value is None:
            return _Probe(False, result.code or "store_open_failed")
        store = result.value
        try:
            version = store.schema_version
            integrity = store.integrity_check()
        finally:
            store.close()
        ready = version == self.current and integrity.ok
        return _Probe(
            ready, None if ready else integrity.code or "schema_pending", version
        )

    def observe(self, root: Path) -> ComponentObservation:
        return observe_journey_sqlite(self._database(root))

    def fingerprint(self, root: Path) -> str:
        return _sqlite_dump_digest(self._database(root))

    def tamper(self, root: Path, version: int) -> bool:
        if version < 1:
            return False
        connection = sqlite3.connect(self._database(root), isolation_level=None)
        try:
            connection.execute(
                "UPDATE schema_migrations SET checksum = ? WHERE version = 1",
                ("0" * 64,),
            )
        finally:
            connection.close()
        return True

    def future(self, root: Path, version: int) -> None:
        database = self._database(root)
        _private_dir(database.parent)
        connection = sqlite3.connect(database, isolation_level=None)
        try:
            connection.execute(_SQLITE_HISTORY_DDL)
            connection.execute(
                "INSERT INTO schema_migrations(version, name, checksum) "
                "VALUES (?, ?, ?)",
                (self.current + 1, "future_release", "f" * 64),
            )
        finally:
            connection.close()

    def interrupt(self, root: Path, version: int) -> _Probe:
        database = self._database(root)
        _private_dir(database.parent)
        status, payload = self.runner.worker(
            "sqlite-kill", str(database), str(self.current - version)
        )
        if status == KILL_EXIT_CODE:
            return _Probe(
                False, "migration_killed", evidence="killed_process", interrupted=True
            )
        return _Probe(
            False,
            _controlled(payload.get("code")) or "fault_not_reached",
            evidence="killed_process",
        )

    def previous_binary(self, root: Path) -> _Previous:
        status, payload = self.runner.worker(
            "sqlite-previous",
            str(self._database(root)),
            str(self.schema.previous_release_version or 0),
        )
        state = payload.get("state")
        code = _controlled(payload.get("code"))
        if status != 0 or state not in {"ready", "refused"}:
            return _Previous(
                ScenarioOutcome.FAILED,
                code or "worker_failed",
                "previous_release_chain",
            )
        declared = self.schema.previous_binary
        matches = (
            state == "refused"
            and declared is PreviousBinary.REFUSES
            and code == self.schema.previous_binary_code
        ) or (state == "ready" and declared is PreviousBinary.READS)
        return _Previous(
            ScenarioOutcome.PASSED if matches else ScenarioOutcome.FAILED,
            code if state == "refused" else "previous_release_reads",
            "previous_release_chain",
            ready=state == "ready",
        )

    def previous_reads(self, root: Path) -> bool:
        status, payload = self.runner.worker(
            "sqlite-previous",
            str(self._database(root)),
            str(self.schema.previous_release_version or 0),
        )
        return status == 0 and payload.get("state") == "ready"

    def activation_checks(self) -> list[Any]:
        return [journey_store_check(self.runner.journey_encryption)]


class _JourneyPostgres(_Adapter):
    component = JOURNEY_POSTGRES

    def __init__(self, runner: _Runner, schema: DurableSchema) -> None:
        super().__init__(runner, schema)
        self.backend = runner.postgres
        self.engine = self.backend.engine
        self.interruptible = frozenset(range(0, self.current))
        self._schemas: dict[Path, str] = {}
        self._lock = threading.Lock()

    def _schema(self, root: Path) -> str:
        with self._lock:
            if root not in self._schemas:
                self._schemas[root] = self.backend.new_schema()
            return self._schemas[root]

    @contextmanager
    def _connection(self, root: Path) -> Iterator[Any]:
        connection = self.backend.connect(self._schema(root))
        try:
            connection.autocommit = True
            yield connection
        finally:
            try:
                connection.close()
            except Exception:
                pass

    def seed(self, root: Path, version: int) -> None:
        schema = self._schema(root)
        if version == 0:
            return
        with self._connection(root) as connection:
            _pg_execute(connection, "BEGIN")
            _pg_execute(connection, f"CREATE SCHEMA IF NOT EXISTS {schema}")
            _pg_execute(connection, f"SET LOCAL search_path TO {schema}")
            _pg_execute(connection, _PG_HISTORY_DDL)
            for migration in POSTGRES_MIGRATIONS[:version]:
                for statement in migration.statements:
                    _pg_execute(connection, statement)
                _pg_execute(
                    connection,
                    "INSERT INTO schema_migrations(version, name, checksum) "
                    "VALUES (%s, %s, %s)",
                    (migration.version, migration.name, migration.checksum),
                )
            for revision in range(1, _SEEDED_REVISIONS + 1):
                _pg_execute(
                    connection,
                    "INSERT INTO store_revisions(revision, committed_at) "
                    "VALUES (%s, %s)",
                    (revision, _COMMITTED_AT),
                )
            _pg_execute(connection, "COMMIT")

    def _open(self, connection: Any, root: Path) -> _Probe:
        from openmed.deploy.journey import migration_health
        from openmed.structured.store import PostgresJourneyStore

        health = migration_health(
            PostgresJourneyStore.open(connection, schema=self._schema(root))
        )
        migration = health.components["migration"]
        return _Probe(
            health.ready,
            None if health.ready else migration.code,
            migration.schema_version,
        )

    def migrate(self, root: Path) -> _Probe:
        connection = self.backend.connect(self._schema(root))
        return self._open(connection, root)

    def observe(self, root: Path) -> ComponentObservation:
        with self._connection(root) as connection:
            return observe_journey_postgres(connection, self._schema(root))

    def fingerprint(self, root: Path) -> str:
        observation = self.observe(root)
        return _digest(
            _canonical_json(
                {
                    "counts": dict(observation.counts),
                    "digest": observation.digest,
                    "version": observation.version,
                }
            )
        )

    def tamper(self, root: Path, version: int) -> bool:
        if version < 1:
            return False
        schema = self._schema(root)
        with self._connection(root) as connection:
            _pg_execute(
                connection,
                f"UPDATE {schema}.schema_migrations SET checksum = %s "
                "WHERE version = %s",
                ("0" * 64, 1),
            )
        return True

    def future(self, root: Path, version: int) -> None:
        schema = self._schema(root)
        with self._connection(root) as connection:
            if version == 0:
                _pg_execute(connection, f"CREATE SCHEMA IF NOT EXISTS {schema}")
                _pg_execute(connection, f"SET search_path TO {schema}")
                _pg_execute(connection, _PG_HISTORY_DDL)
            _pg_execute(
                connection,
                f"INSERT INTO {schema}.schema_migrations(version, name, checksum) "
                "VALUES (%s, %s, %s)",
                (self.current + 1, "future_release", "f" * 64),
            )

    def interrupt(self, root: Path, version: int) -> _Probe:
        faulting = _FaultingConnection(
            self.backend.connect(self._schema(root)),
            kill_after=self.current - version,
        )
        probe = self._open(faulting, root)
        return replace(
            probe,
            evidence="connection_lost",
            interrupted=faulting.lost,
        )

    def previous_binary(self, root: Path) -> _Previous:
        previous = self.schema.previous_release_version or 0
        released = RELEASED_CHECKSUMS.get(self.component, {})
        chain = tuple(item.checksum for item in POSTGRES_MIGRATIONS[:previous])
        if previous != self.current or chain != tuple(
            released.get(index) for index in range(1, previous + 1)
        ):
            return _Previous(
                ScenarioOutcome.DECLARED, self.schema.previous_binary_code, "declared"
            )
        # The released chain is this chain, so this engine is the previous
        # release's engine for schema purposes.
        probe = self.migrate(root)
        return _Previous(
            ScenarioOutcome.PASSED if probe.ready else ScenarioOutcome.FAILED,
            "previous_release_reads" if probe.ready else probe.code,
            "previous_release_chain",
            ready=probe.ready,
        )

    def cleanup(self, root: Path) -> None:
        with self._lock:
            schema = self._schemas.pop(root, None)
        if schema is not None:
            try:
                self.backend.cleanup(schema)
            except Exception:
                pass


class _ArtifactMarker(_Adapter):
    component = JOURNEY_ARTIFACT_MARKER
    interruptible = frozenset({0})

    def _artifacts(self, root: Path) -> Path:
        return root / "journey" / "artifacts"

    def _database(self, root: Path) -> Path:
        return root / "journey" / "journey.sqlite3"

    def seed(self, root: Path, version: int) -> None:
        if version == 0:
            # The previous release's layout: its metadata chain, no marker.
            previous = get_schema(JOURNEY_SQLITE).previous_release_version or 0
            _seed_sqlite_chain(self._database(root), MIGRATIONS[:previous])
            _private_dir(self._artifacts(root) / "blobs" / "sha256")
            return
        self.migrate(root)

    def migrate(self, root: Path) -> _Probe:
        from openmed.core.key_provider import EncryptionError
        from openmed.structured.store import LocalJourneyStore, LocalStoreError

        try:
            # The artifact store (and its marker) opens before the metadata.
            LocalJourneyStore(
                root / "journey", encryption=self.runner.journey_encryption
            ).close()
        except EncryptionError as exc:
            return _Probe(False, exc.code)
        except LocalStoreError:
            return _Probe(False, "store_open_failed")
        return _Probe(True, None, self.observe(root).version)

    def observe(self, root: Path) -> ComponentObservation:
        return observe_artifact_marker(self._artifacts(root))

    def fingerprint(self, root: Path) -> str:
        return _digest(
            _canonical_json(
                [
                    _file_digest(self._artifacts(root) / "encryption.json"),
                    _sqlite_dump_digest(self._database(root)),
                ]
            )
        )

    def tamper(self, root: Path, version: int) -> bool:
        marker = self._artifacts(root) / "encryption.json"
        if version < 1 or not marker.exists():
            return False
        document = json.loads(marker.read_text(encoding="utf-8"))
        document["shadow_field"] = {}
        _write_private(marker, json.dumps(document, sort_keys=True).encode("ascii"))
        return True

    def future(self, root: Path, version: int) -> None:
        document = {
            "envelope_version": 1,
            "key_purpose": "journey-store",
            "key_verifiers": {},
            "schema_version": self.current + 1,
            "state": "encrypted",
        }
        _write_private(
            self._artifacts(root) / "encryption.json",
            json.dumps(document, sort_keys=True).encode("ascii"),
        )

    def interrupt(self, root: Path, version: int) -> _Probe:
        status, payload = self.runner.worker(
            "replace-kill", "marker", str(self._artifacts(root))
        )
        if status == KILL_EXIT_CODE:
            return _Probe(
                False, "migration_killed", evidence="killed_process", interrupted=True
            )
        return _Probe(
            False,
            _controlled(payload.get("code")) or "fault_not_reached",
            evidence="killed_process",
        )

    def activation_checks(self) -> list[Any]:
        return [journey_store_check(self.runner.journey_encryption)]

    def count_strays(self, root: Path) -> int:
        artifacts = self._artifacts(root)
        if not artifacts.is_dir():
            return 0
        return sum(1 for item in artifacts.iterdir() if item.name.startswith("."))


def _synthetic_jobs() -> dict[str, dict[str, Any]]:
    return {
        f"job-{index:04d}": {
            "created_at": _COMMITTED_AT,
            "document_count": 1,
            "expires_at": "2099-01-01T00:00:00Z",
            "id": f"job-{index:04d}",
            "status": "done",
            "updated_at": _COMMITTED_AT,
        }
        for index in range(2)
    }


class _JobStore(_Adapter):
    component = JOB_STORE

    def _path(self, root: Path) -> Path:
        return root / JOB_STORE_PATH

    def seed(self, root: Path, version: int) -> None:
        if version == 0:
            return
        # The previous release's layout: no ``schema_version`` field.
        _write_private(
            self._path(root),
            json.dumps({"jobs": _synthetic_jobs()}, indent=2, sort_keys=True).encode(
                "ascii"
            ),
        )

    def migrate(self, root: Path) -> _Probe:
        try:
            from openmed.service.jobs import JobStoreSchemaError, LocalJobStore
        except ImportError:
            return _Probe(False, "engine_unavailable")
        path = self._path(root)
        try:
            store = LocalJobStore(
                path, clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
            )
            if not path.exists():
                # A clean install creates the store on its first write.
                store.create(next(iter(_synthetic_jobs().values())))
        except JobStoreSchemaError as exc:
            return _Probe(False, exc.code)
        except ValueError:
            return _Probe(False, "job_store_invalid")
        return _Probe(True, None, self.observe(root).version)

    def observe(self, root: Path) -> ComponentObservation:
        return observe_job_store(self._path(root))

    def fingerprint(self, root: Path) -> str:
        return _file_digest(self._path(root))

    def tamper(self, root: Path, version: int) -> bool:
        path = self._path(root)
        if version < 1 or not path.exists():
            return False
        document = json.loads(path.read_text(encoding="utf-8"))
        document["shadow_index"] = {}
        _write_private(path, json.dumps(document, sort_keys=True).encode("ascii"))
        return True

    def future(self, root: Path, version: int) -> None:
        jobs = _synthetic_jobs() if version else {}
        _write_private(
            self._path(root),
            json.dumps(
                {"jobs": jobs, "schema_version": self.current + 1}, sort_keys=True
            ).encode("ascii"),
        )


_REGISTRY_TARGET = "OpenMed/OpenMed-Synthetic-Rehearsal-v1"
_REGISTRY_SLOT = registry_slot_key("synthetic", "small", "pytorch")


class _ModelRegistry(_Adapter):
    component = MODEL_REGISTRY
    interruptible = frozenset({1})

    def _state(self, root: Path) -> Path:
        return root / REGISTRY_STATE_FILE

    def _inputs(self, root: Path) -> tuple[Path, Path]:
        inputs = root.parent / "registry-inputs"
        return inputs / "models.jsonl", inputs / "baseline.json"

    def seed(self, root: Path, version: int) -> None:
        from openmed.core.baseline import write_baseline_store

        manifest, baseline = self._inputs(root)
        row = {
            "family": "synthetic",
            "formats": ["pytorch"],
            "repo_id": _REGISTRY_TARGET,
            "tier": "small",
        }
        _write_private(manifest, (json.dumps(row, sort_keys=True) + "\n").encode())
        write_baseline_store(
            {
                "entries": {
                    _REGISTRY_SLOT: {
                        "family": "synthetic",
                        "format": "pytorch",
                        "key": _REGISTRY_SLOT,
                        "metrics": {},
                        "repo_id": _REGISTRY_TARGET,
                        "reproducibility_hash": "sha256:" + "ab" * 32,
                        "tier": "small",
                    }
                },
                "schema_version": 1,
            },
            baseline,
        )
        pointers = {"canary": None, "last_green": _REGISTRY_TARGET}
        pointers["latest"] = _REGISTRY_TARGET
        if version == 1:
            document: dict[str, Any] = {
                "families": {
                    "synthetic": {
                        "lineage": [],
                        "pointers": pointers,
                        "versions": {_REGISTRY_TARGET: "1.0.0"},
                    }
                },
                "schema_version": 1,
            }
        else:
            document = {
                "schema_version": self.current,
                "slots": {
                    _REGISTRY_SLOT: {
                        "checkpoints": {_REGISTRY_TARGET: "1.0.0"},
                        "lineage": [],
                        "pointers": pointers,
                    }
                },
            }
        _write_private(
            self._state(root), json.dumps(document, sort_keys=True).encode("ascii")
        )

    def migrate(self, root: Path) -> _Probe:
        from openmed.core.registry_errors import RegistryError
        from openmed.core.registry_slots import (
            load_registry_state,
            migrate_registry_state_file,
        )

        observation = self.observe(root)
        if observation.state is ObservedState.EXPLICIT_MIGRATION_REQUIRED:
            manifest, baseline = self._inputs(root)
            try:
                migrate_registry_state_file(
                    state_path=self._state(root),
                    manifest_path=manifest,
                    baseline_path=baseline,
                )
            except RegistryError:
                return _Probe(False, "registry_migration_failed", 1)
        try:
            load_registry_state(self._state(root))
        except RegistryError:
            observation = self.observe(root)
            return _Probe(
                False, observation.code or "registry_state_invalid", observation.version
            )
        return _Probe(True, None, self.observe(root).version)

    def observe(self, root: Path) -> ComponentObservation:
        return observe_model_registry(self._state(root))

    def fingerprint(self, root: Path) -> str:
        return _file_digest(self._state(root))

    def tamper(self, root: Path, version: int) -> bool:
        if version < 2:
            return False
        path = self._state(root)
        document = json.loads(path.read_text(encoding="utf-8"))
        entry = document["slots"].pop(_REGISTRY_SLOT)
        document["slots"] = {"Synthetic::Small::PyTorch": entry}
        _write_private(path, json.dumps(document, sort_keys=True).encode("ascii"))
        return True

    def future(self, root: Path, version: int) -> None:
        _write_private(
            self._state(root),
            json.dumps(
                {"schema_version": self.current + 1, "slots": {}}, sort_keys=True
            ).encode("ascii"),
        )

    def interrupt(self, root: Path, version: int) -> _Probe:
        manifest, baseline = self._inputs(root)
        status, payload = self.runner.worker(
            "replace-kill",
            "registry",
            str(self._state(root)),
            str(manifest),
            str(baseline),
        )
        if status == KILL_EXIT_CODE:
            return _Probe(
                False, "migration_killed", evidence="killed_process", interrupted=True
            )
        return _Probe(
            False,
            _controlled(payload.get("code")) or "fault_not_reached",
            evidence="killed_process",
        )

    def count_strays(self, root: Path) -> int:
        directory = self._state(root).parent
        if not directory.is_dir():
            return 0
        return sum(
            1 for item in directory.iterdir() if item.name != "registry_state.json"
        )


def _synthetic_audit_report(index: int) -> Any:
    from openmed.core.audit import AuditReport, AuditSpan, DetectorInfo, hash_text

    original = f"synthetic rehearsal record {index:04d} token"
    start = original.index("token")
    end = start + len("token")
    return AuditReport(
        policy="hipaa_safe_harbor",
        resolved_profile={"method": "mask", "language": "en"},
        detectors=[
            DetectorInfo(source="ml", model_id="synthetic-model", model_format="x")
        ],
        safety_sweep={"spans_added": 0},
        spans=[
            AuditSpan(
                start=start,
                end=end,
                label="NAME",
                canonical_label="PERSON",
                sources=["ml"],
                confidence=0.99,
                threshold=0.5,
                action="mask",
                surrogate="[PERSON]",
                text_hash=hash_text(original[start:end]),
            )
        ],
        thresholds={"PERSON": 0.5},
        residual_risk={},
        openmed_version="0.0.0",
        manifest_hash=hash_text("synthetic manifest"),
        document_length=len(original),
        input_hash=hash_text(original),
        deidentified_text_hash=hash_text("synthetic rehearsal record [PERSON]"),
    )


class _AuditChain(_Adapter):
    component = AUDIT_CHAIN

    def _path(self, root: Path) -> Path:
        return root / AUDIT_CHAIN_DIRECTORY / "chain-0000.json"

    def _write(self, root: Path, entries: int) -> None:
        from openmed.core.audit_chain import AuditChain

        chain = AuditChain()
        for index in range(entries):
            chain.append(_synthetic_audit_report(index))
        _private_dir(self._path(root).parent)
        chain.write(self._path(root))

    def seed(self, root: Path, version: int) -> None:
        if version >= 1:
            self._write(root, 2)

    def migrate(self, root: Path) -> _Probe:
        from openmed.core.audit_chain import AuditChain

        if not self._path(root).exists():
            # A clean install writes the chain on its first audited run.
            self._write(root, 1)
        try:
            chain = AuditChain.load(self._path(root))
        except (ValueError, TypeError):
            # The reader refuses without a code; the observer names the cause.
            observation = self.observe(root)
            return _Probe(False, observation.code or "audit_chain_invalid")
        if not chain.verify().valid:
            return _Probe(False, "audit_chain_invalid")
        return _Probe(True, None, self.observe(root).version)

    def observe(self, root: Path) -> ComponentObservation:
        path = self._path(root)
        return observe_audit_chains((path,) if path.exists() else ())

    def fingerprint(self, root: Path) -> str:
        return _file_digest(self._path(root))

    def tamper(self, root: Path, version: int) -> bool:
        path = self._path(root)
        if version < 1 or not path.exists():
            return False
        document = json.loads(path.read_text(encoding="utf-8"))
        document["head_hash"] = "sha256:" + "0" * 64
        _write_private(path, json.dumps(document, sort_keys=True).encode("ascii"))
        return True

    def future(self, root: Path, version: int) -> None:
        if version >= 1:
            document = json.loads(self._path(root).read_text(encoding="utf-8"))
        else:
            document = {"entries": [], "format": "openmed.audit-chain"}
        document["version"] = self.current + 1
        _write_private(
            self._path(root), json.dumps(document, sort_keys=True).encode("ascii")
        )


class _FaultingKeyProvider:
    """Backup key provider that fails on its first key resolution."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.faulted = False

    def active_key(self, purpose: str) -> Any:
        return self._inner.active_key(purpose)

    def resolve_key(self, key_id: str, *, purpose: str) -> Any:
        from openmed.core.key_provider import KeyProviderError

        if not self.faulted:
            self.faulted = True
            raise KeyProviderError("key_provider_failed")
        return self._inner.resolve_key(key_id, purpose=purpose)


class _RecoveryManifest(_Adapter):
    component = RECOVERY_MANIFEST
    interruptible = frozenset({0})
    # The manifest is sealed; only a keyed verification detects tampering.
    observer_detects_drift = False

    def _point(self, root: Path) -> Path:
        return root / "point"

    def _sources(self, root: Path) -> ServiceStateSources:
        source = root / "source"
        path = source / JOB_STORE_PATH
        if not path.exists():
            _write_private(
                path,
                json.dumps({"jobs": _synthetic_jobs()}, sort_keys=True).encode("ascii"),
            )
        return ServiceStateSources.from_state_root(source)

    def _create(self, root: Path, provider: Any) -> Any:
        _private_dir(root)
        return create_recovery_point(
            self._sources(root),
            self._point(root),
            key_provider=provider,
            writers_stopped=True,
        )

    def seed(self, root: Path, version: int) -> None:
        if version >= 1:
            self._create(root, self.runner.backup_provider).require_completed()

    def migrate(self, root: Path) -> _Probe:
        if not self._point(root).exists():
            created = self._create(root, self.runner.backup_provider)
            if not created.completed:
                return _Probe(False, created.reason)
        receipt = verify_recovery_point(
            self._point(root), key_provider=self.runner.backup_provider
        )
        return _Probe(receipt.completed, receipt.reason, self.observe(root).version)

    def observe(self, root: Path) -> ComponentObservation:
        return observe_recovery_point(self._point(root))

    def fingerprint(self, root: Path) -> str:
        return _tree_digest(self._point(root))

    def tamper(self, root: Path, version: int) -> bool:
        manifest = self._point(root) / "manifest.omae"
        if version < 1 or not manifest.exists():
            return False
        data = bytearray(manifest.read_bytes())
        data[-1] ^= 0x01
        _write_private(manifest, bytes(data))
        return True

    def future(self, root: Path, version: int) -> None:
        if not self._point(root).exists():
            self.seed(root, 1)
        descriptor = self._point(root) / "recovery-point.json"
        document = json.loads(descriptor.read_text(encoding="utf-8"))
        document["format_version"] = self.current + 1
        _write_private(descriptor, json.dumps(document, sort_keys=True).encode())

    def interrupt(self, root: Path, version: int) -> _Probe:
        provider = _FaultingKeyProvider(self.runner.backup_provider)
        receipt = self._create(root, provider)
        return _Probe(
            receipt.completed,
            receipt.reason,
            evidence="fault_injected",
            interrupted=provider.faulted and not receipt.completed,
        )

    def count_strays(self, root: Path) -> int:
        if not root.is_dir():
            return 0
        return sum(
            1 for item in root.iterdir() if item.name.startswith(".recovery-partial-")
        )


_ADAPTERS: Mapping[str, type[_Adapter]] = {
    JOURNEY_SQLITE: _JourneySQLite,
    JOURNEY_POSTGRES: _JourneyPostgres,
    JOURNEY_ARTIFACT_MARKER: _ArtifactMarker,
    JOB_STORE: _JobStore,
    MODEL_REGISTRY: _ModelRegistry,
    AUDIT_CHAIN: _AuditChain,
    RECOVERY_MANIFEST: _RecoveryManifest,
}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _result(
    adapter: _Adapter,
    scenario: Scenario,
    source: int,
    checks: Mapping[str, bool],
    *,
    code: str | None = None,
    evidence: str = "native_engine",
    observed: int | None = None,
    counts: Mapping[str, int] | None = None,
) -> ScenarioResult:
    passed = all(checks.values())
    failing = next((name for name, value in checks.items() if not value), None)
    return ScenarioResult(
        component=adapter.component,
        scenario=scenario,
        source_version=source,
        outcome=ScenarioOutcome.PASSED if passed else ScenarioOutcome.FAILED,
        code=code if passed else (_controlled(code) or failing),
        evidence=evidence,
        observed_version=observed,
        checks=dict(checks),
        counts=dict(counts or {}),
    )


def _skip(
    adapter: _Adapter,
    scenario: Scenario,
    source: int,
    code: str,
    outcome: ScenarioOutcome = ScenarioOutcome.NOT_APPLICABLE,
    evidence: str = "none",
) -> ScenarioResult:
    return ScenarioResult(
        component=adapter.component,
        scenario=scenario,
        source_version=source,
        outcome=outcome,
        code=code,
        evidence=evidence,
    )


def _preserved(before: ComponentObservation, after: ComponentObservation) -> bool:
    return all(
        after.counts.get(name, 0) >= value for name, value in before.counts.items()
    )


def _forward(adapter: _Adapter, base: Path, source: int) -> ScenarioResult:
    root = base / "state"
    _private_dir(root)
    adapter.seed(root, source)
    before = adapter.observe(root)
    checks: dict[str, bool] = {}
    if before.state is ObservedState.EXPLICIT_MIGRATION_REQUIRED:
        checks["runtime_refused_before_explicit_migration"] = (
            before.code == adapter.schema.refusal_codes.get("explicit")
        )
    probe = adapter.migrate(root)
    after = adapter.observe(root)
    checks["ready"] = probe.ready
    checks["current_version"] = after.version == adapter.current
    checks["counts_preserved"] = _preserved(before, after)
    return _result(
        adapter,
        Scenario.FORWARD_UPGRADE,
        source,
        checks,
        code=probe.code,
        observed=after.version,
        counts=after.counts,
    )


def _repeat(adapter: _Adapter, base: Path, source: int) -> ScenarioResult:
    root = base / "state"
    _private_dir(root)
    adapter.seed(root, source)
    first = adapter.migrate(root)
    observed = adapter.observe(root)
    fingerprint = adapter.fingerprint(root)
    second = adapter.migrate(root)
    again = adapter.observe(root)
    checks = {
        "first_ready": first.ready,
        "second_ready": second.ready,
        "schema_unchanged": again.to_dict() == observed.to_dict(),
        "state_unchanged": adapter.fingerprint(root) == fingerprint,
    }
    return _result(
        adapter,
        Scenario.REPEAT,
        source,
        checks,
        code=second.code,
        observed=again.version,
    )


def _interrupted(adapter: _Adapter, base: Path, source: int) -> ScenarioResult:
    root = base / "state"
    _private_dir(root)
    adapter.seed(root, source)
    before = adapter.observe(root)
    fingerprint = adapter.fingerprint(root)
    probe = adapter.interrupt(root, source)
    middle = adapter.observe(root)
    sources = ServiceStateSources.from_state_root(root)
    plan = (
        None
        if adapter.schema.recovery_category is None
        else plan_service_migrations(sources, state_root=root)
    )
    checks = {
        "interrupted": probe.interrupted,
        "not_ready_after_interruption": not probe.ready,
        "no_half_upgraded_schema": (
            middle.version == before.version and middle.digest == before.digest
        ),
    }
    if before.state is not ObservedState.ABSENT:
        # The whole migration rolled back, data included.
        checks["state_unchanged"] = adapter.fingerprint(root) == fingerprint
    if plan is not None:
        checks["plan_not_ready"] = not plan.ready
    strays = adapter.count_strays(root)
    retry = adapter.migrate(root)
    after = adapter.observe(root)
    checks["retry_ready"] = retry.ready
    checks["retry_current_version"] = after.version == adapter.current
    return _result(
        adapter,
        Scenario.INTERRUPTED,
        source,
        checks,
        code=probe.code,
        evidence=probe.evidence,
        observed=after.version,
        counts={"stray_temporary_files": strays},
    )


def _refusal(
    adapter: _Adapter, base: Path, source: int, scenario: Scenario
) -> ScenarioResult:
    root = base / "state"
    _private_dir(root)
    adapter.seed(root, source)
    if scenario is Scenario.DRIFT:
        if not adapter.tamper(root, source):
            return _skip(adapter, scenario, source, "no_recorded_history")
        expected = adapter.schema.refusal_codes.get("drift")
        refusing = {ObservedState.DRIFTED, ObservedState.INVALID}
    else:
        adapter.future(root, source)
        expected = adapter.schema.refusal_codes.get("unknown_version")
        refusing = {ObservedState.NEWER_THAN_BINARY}
    fingerprint = adapter.fingerprint(root)
    observed = adapter.observe(root)
    probe = adapter.migrate(root)
    checks = {
        "refused": not probe.ready,
        "controlled_code": probe.code == expected,
        "refused_before_mutation": adapter.fingerprint(root) == fingerprint,
    }
    if scenario is Scenario.UNKNOWN_VERSION or adapter.observer_detects_drift:
        checks["observed_refusing_state"] = observed.state in refusing
    return _result(
        adapter,
        scenario,
        source,
        checks,
        code=probe.code,
        observed=adapter.observe(root).version,
    )


def _previous(adapter: _Adapter, base: Path) -> ScenarioResult:
    root = base / "state"
    _private_dir(root)
    adapter.seed(root, adapter.current)
    result = adapter.previous_binary(root)
    return ScenarioResult(
        component=adapter.component,
        scenario=Scenario.PREVIOUS_BINARY,
        source_version=adapter.current,
        outcome=result.outcome,
        code=result.code,
        evidence=result.evidence,
        observed_version=adapter.observe(root).version,
        checks=(
            {}
            if result.outcome is not ScenarioOutcome.PASSED
            else {"matches_declaration": True}
        ),
    )


def _restore(
    adapter: _Adapter, base: Path, source: int
) -> tuple[ScenarioResult, ScenarioResult]:
    generations = base / "generations"
    initial = generations / "g0"
    _private_dir(initial)
    adapter.seed(initial, source)
    live = base / "live"
    os.symlink(initial, live, target_is_directory=True)
    before = adapter.observe(live)
    fingerprint = adapter.fingerprint(live)
    provider = adapter.runner.backup_provider
    point = base / "point"
    backup = create_recovery_point(
        ServiceStateSources.from_state_root(live),
        point,
        key_provider=provider,
        writers_stopped=True,
    )
    checks: dict[str, bool] = {"backup_completed": backup.completed}
    code = backup.reason
    upgrade = adapter.migrate(live)
    checks["upgraded"] = upgrade.ready
    staged = generations / "restored"
    restore = restore_recovery_point(point, staged, key_provider=provider, live=live)
    checks["restore_completed"] = restore.completed
    code = code or restore.reason
    if restore.completed:
        staged_plan = plan_service_migrations(
            ServiceStateSources.from_state_root(staged), state_root=staged
        )
        checks["not_ready_before_activation"] = not staged_plan.ready
    activation = activate_restored_state(
        staged, live, key_provider=provider, checks=adapter.activation_checks()
    )
    checks["activation_completed"] = activation.completed
    code = code or activation.reason
    restored = adapter.observe(live)
    checks["restored_version"] = restored.version == before.version
    checks["restored_state_identical"] = adapter.fingerprint(live) == fingerprint
    if (
        isinstance(adapter, _JourneySQLite)
        and source == adapter.schema.previous_release_version
        and activation.completed
    ):
        checks["previous_binary_reads_restored"] = adapter.previous_reads(live)
    restored_result = _result(
        adapter,
        Scenario.RESTORE_ROLLBACK,
        source,
        checks,
        code=code,
        evidence="service_recovery_point",
        observed=restored.version,
    )
    if not activation.completed:
        retry_result = _skip(
            adapter,
            Scenario.RETRY_AFTER_RECOVERY,
            source,
            "restore_not_activated",
            ScenarioOutcome.FAILED,
        )
        return restored_result, retry_result
    retry = adapter.migrate(live)
    after = adapter.observe(live)
    plan = plan_service_migrations(
        ServiceStateSources.from_state_root(live), state_root=live
    )
    retry_result = _result(
        adapter,
        Scenario.RETRY_AFTER_RECOVERY,
        source,
        {
            "retry_ready": retry.ready,
            "retry_current_version": after.version == adapter.current,
            "plan_ready": plan.ready,
        },
        code=retry.code,
        observed=after.version,
    )
    return restored_result, retry_result


def _unit(adapter: _Adapter, base: Path, source: int) -> list[ScenarioResult]:
    """Run every applicable scenario for one component and source version."""

    results: list[ScenarioResult] = []
    roots: list[Path] = []

    def run(scenario: Scenario, action: Callable[[Path], Any]) -> None:
        directory = base / scenario.value
        roots.append(directory / "state")
        roots.append(directory / "live")
        try:
            outcome = action(directory)
        except Exception as exc:
            # Only the exception type is kept: messages may carry paths.
            name = re.sub(r"[^a-z0-9_]", "", type(exc).__name__.lower())[:48]
            outcome = _skip(
                adapter,
                scenario,
                source,
                f"raised_{name}" if name else "scenario_raised",
                ScenarioOutcome.FAILED,
            )
        if isinstance(outcome, tuple):
            results.extend(outcome)
        else:
            results.append(outcome)

    run(Scenario.FORWARD_UPGRADE, lambda path: _forward(adapter, path, source))
    run(Scenario.REPEAT, lambda path: _repeat(adapter, path, source))
    if source in adapter.interruptible:
        run(Scenario.INTERRUPTED, lambda path: _interrupted(adapter, path, source))
    else:
        results.append(
            _skip(adapter, Scenario.INTERRUPTED, source, "no_pending_migration")
        )
    run(Scenario.DRIFT, lambda path: _refusal(adapter, path, source, Scenario.DRIFT))
    run(
        Scenario.UNKNOWN_VERSION,
        lambda path: _refusal(adapter, path, source, Scenario.UNKNOWN_VERSION),
    )
    schema = adapter.schema
    if source in schema.restorable_versions and schema.recovery_category is not None:
        run(Scenario.RESTORE_ROLLBACK, lambda path: _restore(adapter, path, source))
    else:
        outcome = ScenarioOutcome.NOT_APPLICABLE
        evidence = "none"
        if schema.recovery_point is RecoveryPointKind.OPERATOR_DATABASE_BACKUP:
            code, outcome, evidence = (
                "operator_database_backup",
                ScenarioOutcome.EXTERNAL,
                "external",
            )
        elif schema.recovery_point is RecoveryPointKind.NOT_REQUIRED:
            code = "rollback_not_required"
        elif source == 0:
            code = "no_prior_state"
        else:
            code = "source_not_restorable"
        for scenario in (Scenario.RESTORE_ROLLBACK, Scenario.RETRY_AFTER_RECOVERY):
            results.append(_skip(adapter, scenario, source, code, outcome, evidence))
    if source == adapter.current:
        run(Scenario.PREVIOUS_BINARY, lambda path: _previous(adapter, path))
    for root in roots:
        adapter.cleanup(root)
    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def rehearse_service_migrations(
    workspace: str | Path | None = None,
    *,
    components: Sequence[str] | None = None,
    postgres_connect: Callable[[], Any] | None = None,
    clock: Callable[[], datetime] | None = None,
    max_workers: int | None = None,
) -> RehearsalReport:
    """Rehearse every registered migration path offline and gate the release.

    Args:
        workspace: Empty or missing directory for synthetic fixtures; a
            temporary directory is created and removed when omitted.
        components: Optional subset of components; the gate is then
            ``partial`` at best.
        postgres_connect: Optional factory of dedicated DB-API connections
            to a disposable PostgreSQL server.  The rehearsal creates and
            drops its own randomly named schemas there.  Without it the
            PostgreSQL engine runs against :class:`SQLitePostgresDouble`.
        clock: Optional UTC clock for the report timestamp.
        max_workers: Optional bound on parallel scenario units.

    Returns:
        The value-free rehearsal report.

    Raises:
        ValueError: If ``workspace`` is not empty or a component is unknown.
    """

    schemas = durable_schemas()
    known = {schema.component for schema in schemas}
    selected = set(known if components is None else components)
    if selected - known:
        raise ValueError("unknown component requested")
    if workspace is None:
        with tempfile.TemporaryDirectory(prefix="openmed-rehearsal-") as directory:
            return _rehearse(
                Path(directory), schemas, selected, postgres_connect, clock, max_workers
            )
    target = Path(workspace)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("workspace must be a new or empty directory")
    _private_dir(target)
    return _rehearse(target, schemas, selected, postgres_connect, clock, max_workers)


def _rehearse(
    workspace: Path,
    schemas: Sequence[DurableSchema],
    selected: set[str],
    postgres_connect: Callable[[], Any] | None,
    clock: Callable[[], datetime] | None,
    max_workers: int | None,
) -> RehearsalReport:
    postgres_root = workspace / "postgres"
    _private_dir(postgres_root)
    backend: Any = (
        _DoubleBackend(postgres_root)
        if postgres_connect is None
        else _ServerBackend(postgres_connect)
    )
    runner = _Runner(workspace, backend)
    adapters = {
        schema.component: _ADAPTERS[schema.component](runner, schema)
        for schema in schemas
        if schema.presence is Presence.PRESENT
        and schema.component in selected
        and schema.component in _ADAPTERS
    }
    units = [
        (component, source)
        for component, adapter in adapters.items()
        for source in adapter.schema.source_versions
    ]
    workers = max_workers or min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            unit: pool.submit(
                _unit,
                adapters[unit[0]],
                workspace / unit[0] / f"v{unit[1]}",
                unit[1],
            )
            for unit in units
        }
        outcomes = {unit: future.result() for unit, future in futures.items()}
    components: list[ComponentRehearsal] = []
    for schema in schemas:
        if schema.component not in selected:
            continue
        rollback = schema.rollback_class
        if schema.component not in adapters:
            components.append(
                ComponentRehearsal(
                    component=schema.component,
                    presence=schema.presence.value,
                    current_version=schema.current_version,
                    rollback_class=None if rollback is None else rollback.value,
                    engine="none",
                    tracking_issue=schema.tracking_issue,
                )
            )
            continue
        rows = [
            row
            for (component, _source), result in outcomes.items()
            if component == schema.component
            for row in result
        ]
        rows.sort(key=lambda row: (_SCENARIO_ORDER[row.scenario], row.source_version))
        components.append(
            ComponentRehearsal(
                component=schema.component,
                presence=schema.presence.value,
                current_version=schema.current_version,
                rollback_class=None if rollback is None else rollback.value,
                engine=adapters[schema.component].engine,
                scenarios=tuple(rows),
                tracking_issue=schema.tracking_issue,
            )
        )
    codes = list(registry_findings(schemas))
    for item in components:
        if item.presence != Presence.PRESENT.value:
            continue
        if item.rollback_class == RollbackClass.RELEASE_BLOCKING.value:
            codes.append(f"{item.component}:release_blocking")
        for row in item.failed:
            codes.append(
                f"{item.component}:{row.scenario.value}:v{row.source_version}:"
                f"{row.code or 'failed'}"
            )
    if codes:
        state = "blocked"
    elif selected != {schema.component for schema in schemas}:
        state = "partial"
    else:
        state = "passed"
    moment = (clock or (lambda: datetime.now(timezone.utc)))()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return RehearsalReport(
        components=tuple(components),
        gate_state=state,
        gate_codes=tuple(sorted(set(codes))),
        registry_digest=registry_digest(),
        generated_at=moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _controlled(value: Any) -> str | None:
    if isinstance(value, str) and _CODE_RE.fullmatch(value):
        return value
    return None


def _tree_digest(root: Path) -> str:
    if not root.exists():
        return "absent"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()
