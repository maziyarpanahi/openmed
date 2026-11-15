"""Throwaway child process for crash and previous-release migration rehearsals.

Only :mod:`openmed.service.migration_rehearsal` starts this worker, against
synthetic fixtures it created.  Each operation patches *this* process alone,
then runs a component's native engine:

* ``sqlite-kill`` kills the process (``os._exit``) when SQLite starts the
  Nth ``INSERT INTO schema_migrations`` statement, mid-transaction, exactly
  like a crash or ``SIGKILL`` during the migration.
* ``sqlite-previous`` runs the native SQLite migration engine with the
  released migration chain of the previous release, which is what that
  release's binary does when it opens a store.
* ``replace-kill`` kills the process at the atomic ``os.replace`` that would
  publish a migrated or first document (the artifact encryption marker, the
  model registry state, a sealed job payload, or an audit continuity
  manifest).
* ``activation-kill`` kills the process at the first model-activation
  pointer-state write, after the candidate generation was staged and
  verified and its journal recorded.

The worker prints one JSON line of controlled fields and never prints paths,
values, or exception text.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

KILL_EXIT_CODE = 86
"""Exit status of a worker killed mid-migration."""


def _emit(payload: dict[str, Any]) -> int:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()
    return 0


def _die(*_args: Any, **_kwargs: Any) -> None:
    os._exit(KILL_EXIT_CODE)


def _ephemeral_journey_encryption() -> Any:
    from openmed.core.key_provider import InMemoryKeyProvider
    from openmed.structured.store import JOURNEY_KEY_PURPOSE, JourneyStoreEncryption

    return JourneyStoreEncryption.encrypted(
        InMemoryKeyProvider.generate(JOURNEY_KEY_PURPOSE)
    )


def _sqlite_kill(database: Path, inserts: int) -> int:
    original = sqlite3.connect
    seen = {"count": 0}

    def trace(statement: str) -> None:
        if statement.lstrip().upper().startswith("INSERT INTO SCHEMA_MIGRATIONS"):
            seen["count"] += 1
            if seen["count"] >= inserts:
                os._exit(KILL_EXIT_CODE)

    def traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = original(*args, **kwargs)
        connection.set_trace_callback(trace)
        return connection

    sqlite3.connect = traced_connect  # type: ignore[assignment]
    from openmed.structured.store import SQLiteJourneyStore

    result = SQLiteJourneyStore.open(
        database, encryption=_ephemeral_journey_encryption()
    )
    if result.ok and result.value is not None:
        result.value.close()
    return _emit({"state": "not_killed", "code": result.code})


def _sqlite_previous(database: Path, released: int) -> int:
    from openmed.structured.store import local
    from openmed.structured.store.migrations import MIGRATIONS

    # The previous release opened a store with exactly these pragmas and then
    # ran this (unchanged) engine over its own, shorter migration chain.
    local.MIGRATIONS = MIGRATIONS[:released]  # type: ignore[misc]
    local.LATEST_MIGRATION_VERSION = released  # type: ignore[misc]
    store = object.__new__(local.SQLiteJourneyStore)
    connection = sqlite3.connect(database, isolation_level=None)
    connection.row_factory = sqlite3.Row
    store._connection = connection
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 5000")
        store._apply_migrations()
        version = connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()[0]
    except local.StoreCompatibilityError:
        return _emit({"state": "refused", "code": "schema_unsupported"})
    except (local.LocalStoreError, sqlite3.Error):
        return _emit({"state": "failed", "code": "store_open_failed"})
    finally:
        connection.close()
    return _emit({"state": "ready", "code": None, "version": int(version)})


def _replace_kill(component: str, arguments: list[str]) -> int:
    os.replace = _die  # type: ignore[assignment]
    if component == "marker":
        from openmed.structured.store import LocalArtifactStore

        LocalArtifactStore(
            Path(arguments[0]), encryption=_ephemeral_journey_encryption()
        )
        return _emit({"state": "not_killed", "code": None})
    if component == "registry":
        from openmed.core.registry_slots import migrate_registry_state_file

        migrate_registry_state_file(
            state_path=Path(arguments[0]),
            manifest_path=Path(arguments[1]),
            baseline_path=Path(arguments[2]),
        )
        return _emit({"state": "not_killed", "code": None})
    if component == "payload":
        from openmed.core.key_provider import InMemoryKeyProvider
        from openmed.service.job_payloads import (
            JOB_PAYLOAD_KEY_PURPOSE,
            PAYLOAD_KIND_INPUT,
            EncryptedJobPayloadStore,
        )

        # The remnant stays sealed under a key that dies with this process.
        EncryptedJobPayloadStore(
            Path(arguments[0]),
            key_provider=InMemoryKeyProvider.generate(JOB_PAYLOAD_KEY_PURPOSE),
        ).create(
            PAYLOAD_KIND_INPUT, "job-0000", {"synthetic": True}, retention_seconds=60
        )
        return _emit({"state": "not_killed", "code": None})
    if component == "continuity":
        import secrets

        from openmed.core.audit_continuity import AuditContinuityStore

        AuditContinuityStore(
            Path(arguments[0]), key_provider={arguments[1]: secrets.token_bytes(32)}
        ).initialize(key_id=arguments[1], chain_id=arguments[2])
        return _emit({"state": "not_killed", "code": None})
    return _emit({"state": "failed", "code": "worker_operation_invalid"})


class _NullLoader:
    def load(self, context: Any) -> Any:
        return context.generation

    def warm(self, handle: Any, context: Any) -> None:
        del handle, context

    def unload(self, handle: Any, context: Any) -> None:
        del handle, context


def _activation_kill(root: Path, bundle: Path, slot: str, model_id: str) -> int:
    from openmed.service.model_activation import (
        LocalBundleSource,
        ModelActivationManager,
        ModelReference,
    )

    def crash(point: str) -> None:
        # The first pointer-state write allocates the verified candidate's
        # generation; nothing has been published yet.
        if point == "state.write":
            os._exit(KILL_EXIT_CODE)

    manager = ModelActivationManager(root, loader=_NullLoader(), fault_hook=crash)
    manager.start()
    manager.activate(slot, ModelReference(model_id), source=LocalBundleSource(bundle))
    manager.close()
    return _emit({"state": "not_killed", "code": None})


def main(argv: list[str]) -> int:
    """Run one rehearsal operation in this throwaway process.

    Args:
        argv: Operation name followed by its arguments.

    Returns:
        Process exit status (``KILL_EXIT_CODE`` never returns; the process
        dies).
    """

    try:
        operation = argv[0]
        if operation == "sqlite-kill":
            return _sqlite_kill(Path(argv[1]), int(argv[2]))
        if operation == "sqlite-previous":
            return _sqlite_previous(Path(argv[1]), int(argv[2]))
        if operation == "replace-kill":
            return _replace_kill(argv[1], argv[2:])
        if operation == "activation-kill":
            return _activation_kill(Path(argv[1]), Path(argv[2]), argv[3], argv[4])
    except Exception:
        return _emit({"state": "failed", "code": "worker_failed"})
    return _emit({"state": "failed", "code": "worker_operation_invalid"})


if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    raise SystemExit(main(sys.argv[1:]))
