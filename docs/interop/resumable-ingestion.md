# Resumable ingestion control plane

OpenMed provides a local-first control plane for replay-safe Journey ingestion.
It records source identity, job state, exclusive leases, checkpoints, retries,
cancellations, and quarantined parser output without putting source values in
diagnostics. The same logical ledger runs on SQLite for a single local process
or PostgreSQL for coordinated workers.

This layer coordinates ingestion. It does not make a parser correct, turn a
partial parse into trusted clinical data, or make an external side effect
transactional with the ledger.

## Versioned records

The public records use schema version `1.0.0`:

- `SourceManifest` binds the source identifier, artifact digests, policy
  digest, and pipeline digest into a content-derived replay identity.
- `IngestionJob` records the append-only state and checkpoint sequence.
- `Lease` grants one worker time-bounded ownership of one job.
- `Checkpoint` acknowledges one `(job, step, input digest)` boundary.
- `Retry` contains a stable classification and controlled reason code.
- `Cancellation` records an explicit actor digest and controlled reason.
- `QuarantineResult` isolates malformed, partial, ambiguous, or unsafe output.

Bundled JSON Schemas can be loaded without network access:

```python
from openmed.interop.ingest import load_all_ingestion_schemas

schemas = load_all_ingestion_schemas()
assert schemas["ingestion_job"]["schema_version"] == 1
```

Unknown fields, duplicate JSON keys, non-finite numbers, invalid identifiers,
and unsupported record versions fail closed. Errors contain field or category
names, never the rejected value.

## Register a source once

Use `SQLiteIngestionStore` for the default offline path:

```python
from openmed.clinical.journey_contracts import canonical_digest
from openmed.interop.ingest import (
    IngestionCoordinator,
    SQLiteIngestionStore,
    SourceManifest,
)

store = SQLiteIngestionStore("./journey-data/journey.sqlite3")
coordinator = IngestionCoordinator(store)
manifest = SourceManifest(
    manifest_id="manifest_0123456789abcdef",
    source_id="source_0123456789abcdef",
    artifact_digests=(
        "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    ),
    policy_digest=canonical_digest({"policy": "local"}),
    pipeline_digest=canonical_digest({"pipeline": "extract-v1"}),
    created_at="2026-01-02T03:04:05Z",
)
registration = coordinator.register(
    manifest,
    recorded_at="2026-01-02T03:04:05Z",
)
assert registration.ok
```

Manifest identity is derived from source, artifact, policy, pipeline, and
schema content. It intentionally excludes the manifest record identifier and
creation time. Registering unchanged content therefore returns the original
job and appends a `ReplayAudit` with action `noop`; it does not create a second
job or duplicate facts. Reusing one manifest identifier for different content
returns a conflict.

## Execute idempotent steps

`IngestionCoordinator.execute_step()` performs this sequence:

1. Read an existing checkpoint for the exact job, step, and input digest.
2. Acquire or confirm the caller's live lease.
3. Read the checkpoint again after lease acquisition to close the race.
4. Run the caller operation.
5. Persist either a trusted checkpoint, a value-free retry, or quarantine.

A trusted operation returns only output identity and an optional committed
Journey revision:

```python
from openmed.interop.ingest import StepOutput

result = coordinator.execute_step(
    job_id=registration.value.job.job_id,
    worker_id="worker_0123456789abcdef",
    step="extract",
    input_digest=manifest.artifact_digests[0],
    acquired_at="2026-01-02T03:05:05Z",
    lease_seconds=300,
    attempt=1,
    operation=lambda: StepOutput(
        output_digest=canonical_digest({"result": "synthetic"}),
    ),
)
assert result.ok
```

The operation must be idempotent under its own stable record identifiers. A
worker can terminate after committing a fact but before acknowledging the
checkpoint. On restart, the lease expires, the operation runs again, and the
Journey store's immutable identifier and payload checks prevent a duplicate
fact. The checkpoint is written only after the operation returns successfully.

Ordinary exceptions become `Retry` records with stable categories such as
`transient`, `dependency`, `resource`, or `policy_denied`. Exception messages
are not inspected or persisted. Process-ending exceptions are not caught; the
lease is allowed to expire so another worker can replay the absent checkpoint.

## Lease and state rules

A job moves from `queued` to `running` when its first lease is acquired. One
live job lease can belong to one worker. The same worker may reacquire its live
lease idempotently; another worker receives `lease_held`. A worker can take over
only after expiry, which also records release of the stale lease.

Successful checkpoints advance the sequence without gaps. A job can then be
completed under its live lease. Cancellation is an explicit terminal action.
Completed, cancelled, and failed jobs cannot be leased again. State history is
append-only even though reads return the latest version.

SQLite uses `BEGIN IMMEDIATE` to serialize ledger writes. PostgreSQL uses
serializable transactions and a transaction-scoped advisory lock. The
PostgreSQL lock is deliberately correctness-first and serializes ingestion
ledger mutations in this version; applications requiring higher write
parallelism should shard ledgers by deployment rather than weaken lease checks.

## Quarantine is not success

Malformed and partial parser results return a `QuarantineRequest`. The ledger
stores only classification, controlled reason, counts, and optional output
digest; it marks the job `quarantined` and does not create a trusted checkpoint
or clinical fact. New leases are refused until an explicit
`QuarantinePromotion` records reviewer and evidence digests.

Promotion returns the job to `queued`. It is an authorization to resume
processing, not evidence that the original candidate output was correct.
Applications must keep untrusted payloads in a separately controlled location
and write trusted records only after review.

## PostgreSQL workers

Construct `PostgresIngestionStore` with the same dedicated connection and
schema controls used by `PostgresJourneyStore`:

```python
import psycopg

from openmed.interop.ingest import PostgresIngestionStore

store = PostgresIngestionStore(
    psycopg.connect("postgresql://application@localhost/journey"),
    schema="openmed_journey",
)
```

TLS, credentials, database routing, retention, and availability are deployment
responsibilities. A lost PostgreSQL commit acknowledgement returns
`commit_status_unknown`; reconcile by reading the manifest or checkpoint before
retrying. Do not convert an ambiguous commit into success.

## Recovery and integrity checks

After a restart:

1. Open the same SQLite path or PostgreSQL schema and require healthy migrations.
2. Run `ingestion_integrity_check()` and require an `ok` result.
3. Let stale leases expire; do not delete lease rows manually.
4. Replay the same manifest and resume each step with the same input digest and
   deterministic record identifiers.
5. Investigate `unknown`, `partial`, or `commit_status_unknown` outcomes before
   advancing the job.

The integrity check validates stored payload hashes and record parsability. The
underlying Journey integrity check validates graph constraints. Neither check
proves clinical correctness or verifies bytes held by a separate object store;
run the artifact digest checks described in the storage guide as well.

The synthetic local suite covers termination at every checkpoint, duplicate
fact prevention, replay audits, migration from the original Journey schema,
lease expiry, retry classification, cancellation, quarantine promotion, policy
denials, and tamper detection:

```bash
pytest tests/unit/interop/ingest tests/integration/test_ingestion_recovery.py -q
```

The optional PostgreSQL race test requires a disposable database. It creates
and removes only a randomly named `openmed_test_*` schema:

```bash
OPENMED_TEST_POSTGRES_DSN='postgresql://...' \
  pytest tests/integration/test_ingestion_recovery.py -q
```
