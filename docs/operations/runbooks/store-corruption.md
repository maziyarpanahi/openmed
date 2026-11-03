# Job and Journey store corruption runbook

Use this runbook when the async job store or a Journey store cannot be opened,
fails an integrity check, or returns damaged records. Read the
[shared rules](service-runbooks.md#rules-for-tickets-logs-and-chat) first. If
the damage came from a full disk, fix capacity first with the
[disk exhaustion runbook](disk-exhaustion.md).

In every case, keep the damaged data. Move it into quarantine and record its
SHA-256 digest and size. Do not edit it in place, and do not open it in an
editor or pager. A bounded diagnostic bundle is pending in #3600, and an
operator recovery CLI is pending in #3703. Until they ship, use the manual
steps below.

## Async job store

`openmed.service.jobs.LocalJobStore` keeps job metadata in one JSON file at
`OPENMED_SERVICE_JOBS_STORE_PATH`. The default is
`~/.cache/openmed/service-jobs.json`. The service loads it at startup and
refuses to start if the file is damaged. It never repairs or replaces the
file on its own. The local store is for one service process. Two processes
sharing one path can overwrite each other's changes.

### Detection signals

The service stops during startup with one of these fixed messages. The
messages contain no file content:

- `Job metadata store contains invalid JSON`
- `Job metadata store must contain a JSON object`
- `Job metadata store must contain a jobs object`
- `Job metadata store contains an invalid job record`

An unreadable file, for example one with wrong ownership, stops startup with
an operating system permission error instead. The process restarts in a loop
and never reports ready.

You can check the configured store without starting the service. Run this in
the service environment, or set `OPENMED_SERVICE_JOBS_STORE_PATH` to a
quarantined copy. It prints one of three fixed lines and never prints the
file's contents or path:

```bash
python - <<'PY'
import os

from openmed.service.jobs import (
    SERVICE_JOBS_STORE_ENV_VAR,
    LocalJobStore,
    parse_jobs_store_path,
)

path = parse_jobs_store_path(os.getenv(SERVICE_JOBS_STORE_ENV_VAR))
if not path.exists():
    print("job store: absent")
else:
    try:
        LocalJobStore(path)
    except (OSError, ValueError):
        print("job store: unreadable or invalid")
    else:
        print("job store: loadable")
PY
```

### Containment

1. Stop every service process that uses this store path.
2. Move the file into quarantine. Do not copy it to a shared location:

    ```bash
    umask 077
    mkdir -p <quarantine-dir>
    mv <store-path> <quarantine-dir>/service-jobs.quarantined.json
    shasum -a 256 <quarantine-dir>/service-jobs.quarantined.json
    ```

    Record only the digest and the file size in the incident.

### Recovery

1. Start the service. With no file at the store path, it starts with an empty
   store and writes a new file on the first job.
2. Tell clients that job IDs issued before the incident now return `404` from
   `GET /jobs/{job_id}`. They must submit unfinished work again. Jobs that were
   queued or running at the failure were lost in any case, because their text
   lived only in memory. Durable jobs with replay are pending in #3574.
3. If you have a known-good copy of the store from your own backups, you can
   restore it instead. Run the check above against the copy first, then move
   it into place while the service is stopped. Restore tooling for service
   state is pending in #3577.

### Verification

1. The check above prints `job store: loadable` for the live path once the
   first job has been saved.
2. `GET /readyz` returns `200`. A synthetic job submitted with `POST /jobs`
   reaches `done` in `GET /jobs/{job_id}`.

### Keep out of tickets and logs

- The quarantined file and any excerpt from it. It holds job IDs, document
  IDs, span offsets, labels, text hashes and webhook delivery results.
- The store path, if it shows a user name or tenant. Record the digest, the
  size and the startup message.

## Journey PostgreSQL store

The Compose profile and the Helm chart keep Journey metadata in PostgreSQL
(`OPENMED_JOURNEY_POSTGRES_DSN`, `OPENMED_JOURNEY_SCHEMA`) and artifacts under
`OPENMED_JOURNEY_ARTIFACT_ROOT`. `python -m openmed.deploy.journey` (also
installed as `openmed-journey`) runs `migrate`, `probe`, `worker` and `golden`.
Its output contains only controlled states and codes.

Opening the store with these commands, with the worker, or with
`openmed.structured.store.PostgresJourneyStore` applies any pending
migrations. No read-only inspection command exists yet. One is pending in #3703.
Take a database backup before you run any of them against a store you think is
damaged.

### Detection signals

- The probe or the worker readiness output shows one of these codes:
  `postgres_connect_failed` (the database is unreachable or the credential was
  rejected), `store_open_failed` (open or migration failed), `schema_unsupported`
  (the schema is newer than this release, or an applied migration checksum
  differs; the probe cannot tell these apart), `store_probe_failed`, or
  `dsn_not_configured`.
- Reads return a `failure` result with `stored_payload_invalid`.
- An integrity check returns `foreign_key_integrity_failed` or
  `payload_hash_mismatch`.

### Containment

1. Stop the processes that write Journey data. Keep the database running so
   that you can back it up.
2. Take a database-level backup or volume snapshot with your PostgreSQL
   tooling. Store it encrypted with restricted access.
3. If the code is `schema_unsupported` right after an image rollback, the
   schema is probably newer than the running release. Roll forward to the
   release that wrote it. Do not run `migrate` again with the older image.

### Recovery

1. Restore your most recent good backup into a new database or schema. Do not
   restore over the damaged one.
2. Point `OPENMED_JOURNEY_POSTGRES_DSN` and `OPENMED_JOURNEY_SCHEMA` at the
   restored copy in a maintenance shell, then run the integrity check. The DSN
   is read from the environment and is never printed:

    ```bash
    python - <<'PY'
    import os

    from openmed.structured.store import PostgresJourneyStore

    opened = PostgresJourneyStore.connect(
        os.environ["OPENMED_JOURNEY_POSTGRES_DSN"],
        schema=os.getenv("OPENMED_JOURNEY_SCHEMA", "openmed_journey"),
    )
    if not opened.ok:
        print(f"open: {opened.state.value} {opened.code}")
    else:
        result = opened.value.integrity_check()
        opened.value.close()
        print(f"integrity: {result.state.value} {result.code or ''}".strip())
        if result.ok:
            print(dict(result.value))
    PY
    ```

    On success it prints `integrity: success` and row counts for each table.
3. Switch the service and worker to the restored database, then run
   `python -m openmed.deploy.journey migrate` once.
4. Data written after the backup is not in the restored copy. Re-run the
   ingestion that produced it from its original sources. OpenMed backup and
   restore for service state is pending in #3577.

### Verification

```bash
docker compose -f deploy/openmed-compose.yaml exec openmed \
  python3 -m openmed.deploy.journey probe \
  --worker-url http://journey-worker:8091/readyz \
  --model-url http://127.0.0.1:8080/readyz
```

The migration component must show `migration_current`, the store component
`store_ready`, and the artifact component `artifact_store_ready`. Under Helm,
run the same probe command inside the service pod.

### Keep out of tickets and logs

- The DSN, the database password, and database dumps or row exports.
- Query results from the Journey tables. Record the integrity code and the row
  counts only.

## Local SQLite Journey store and artifacts

Library users can persist Journey data with
`openmed.structured.store.SQLiteJourneyStore` and
`openmed.structured.store.LocalArtifactStore`, or both together through
`openmed.structured.store.LocalJourneyStore`. SQLite uses write-ahead logging,
so the database file, its `-wal` file and its `-shm` file belong together.

### Detection signals

- `SQLiteJourneyStore.open` returns `store_open_failed` or
  `schema_unsupported`.
- `integrity_check()` returns `sqlite_integrity_failed`,
  `foreign_key_integrity_failed`, `stored_payload_invalid` or
  `payload_hash_mismatch`.
- Artifact reads return `artifact_integrity_failed` (the bytes do not match
  their digest), `artifact_read_failed`, or `artifact_path_unsafe` (the blob
  path is a symbolic link). Every read verifies its digest. There is no bulk
  artifact sweep; that is pending in #3600.
- A metadata commit returns `commit_status_unknown`. Read the record back
  before you retry.

### Containment

1. Stop the process that writes the store. Do not delete the `-wal` file.
2. Check a copy, never the original. Opening a store applies pending
   migrations. This snippet copies the database through the SQLite backup API,
   which includes committed write-ahead log content, and checks the copy. It
   prints only states, codes and row counts:

```bash
JOURNEY_SQLITE_PATH=<store-file> JOURNEY_CHECK_COPY=<quarantine-dir>/check.sqlite3 \
  python - <<'PY'
import os
import sqlite3
from pathlib import Path

from openmed.structured.store import SQLiteJourneyStore

os.umask(0o077)
source = Path(os.environ["JOURNEY_SQLITE_PATH"]).resolve()
copy = Path(os.environ["JOURNEY_CHECK_COPY"])
try:
    reader = sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)
    writer = sqlite3.connect(copy)
    reader.backup(writer)
    reader.close()
    writer.close()
except sqlite3.Error:
    print("copy: failed")
else:
    opened = SQLiteJourneyStore.open(copy)
    if not opened.ok:
        print(f"open: {opened.state.value} {opened.code}")
    else:
        result = opened.value.integrity_check()
        opened.value.close()
        print(f"integrity: {result.state.value} {result.code or ''}".strip())
        if result.ok:
            print(dict(result.value))
PY
```

### Recovery

1. Restore the database from a backup that was taken after `close()`, or with
   the SQLite backup API. Restore it together with its write-ahead log state.
   A file copied from a live database without its `-wal` file can lose
   committed revisions.
2. Artifacts are content-addressed. To repair a blob that fails
   `artifact_integrity_failed`, write the original bytes again through
   `put_bytes` with the same artifact record. The store accepts the bytes only
   if they match the recorded digest and size. If the source is gone, record
   the artifact as unrecoverable. Do not substitute other bytes.
3. Move the damaged database and blobs into quarantine with mode `0600` and
   record their digests.

### Verification

1. The snippet above prints `integrity: success` for the restored store.
2. Read each repaired artifact back with `get_bytes`. It must return a
   successful result.

### Keep out of tickets and logs

- Database files, `-wal` files, artifact blobs, and record payloads.
- Content digests next to patient or document identifiers. Record counts,
  codes, and the digests of quarantined files only.
