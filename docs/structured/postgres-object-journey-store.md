# PostgreSQL and object-backed Journey storage

OpenMed can run the same Journey metadata contracts on PostgreSQL and keep
content-addressed artifact bytes in an explicitly bounded `fsspec` namespace.
Local filesystem and SQLite storage remain the default. Nothing in this path
enables a network backend, downloads a driver, or sends telemetry unless the
caller supplies and invokes those components.

These stores persist caller-created records. They do not diagnose, recommend
treatment, or turn an ambiguous clinical state into an apparent success.

## Backend-neutral composition

`ComposedJourneyStore` joins any compensating artifact store to any
`TransactionalJourneyStore`. The common graph-ingestion path therefore has the
same ordering, typed outcomes, idempotency, and metadata rollback behavior for
local and PostgreSQL deployments.

```python
import psycopg  # Caller-installed driver.

from openmed.structured.store import (
    ComposedJourneyStore,
    FsspecArtifactStore,
    ObjectStoreNamespace,
    PostgresJourneyStore,
)

namespace = ObjectStoreNamespace(
    "file:///srv/openmed/journey-objects",
)
artifacts = FsspecArtifactStore(namespace)
metadata = PostgresJourneyStore(
    psycopg.connect("postgresql://application@localhost/journey"),
    schema="openmed_journey",
)
store = ComposedJourneyStore(artifacts, metadata)
```

The supplied database connection is dedicated to the store, changed to
autocommit mode for bounded reads and explicit transactions, and closed by
`store.close()`. Install and configure the PostgreSQL driver in the application;
OpenMed does not add a mandatory database client.

## PostgreSQL transaction parity

`PostgresJourneyStore` implements the same public metadata protocols as
`SQLiteJourneyStore`. It uses:

- a controlled, caller-selected PostgreSQL schema (default
  `openmed_journey`);
- serializable write transactions and an exact committed revision;
- append-only facts, conflicts, resolutions, canonical versions, datasets, and
  job states;
- deterministic `?`-style internal parameters translated to DB-API parameters;
- point-in-time reads bounded by `StorePoint`;
- controlled errors and typed `partial`, `unknown`, `conflict`, `unsupported`,
  `denied`, and `failure` outcomes.

The connection or DSN must use TLS, authentication, routing, and access controls
appropriate to the deployment. OpenMed neither weakens those controls nor puts
connection details into result representations. `PostgresJourneyStore.connect()`
is an optional convenience that lazily imports the BSD-licensed `pg8000` driver
from the `journey` extra. It accepts a PostgreSQL URL with an optional
`sslmode=verify-full` query (using the system trust store); other URL options
fail closed. Direct construction accepts a compatible dedicated DB-API
connection when deployment-specific TLS or routing is required.

## Migration state and recovery

PostgreSQL migrations are ordered, checksummed, and applied under one advisory
transaction lock. PostgreSQL transactional DDL means a failed or interrupted
upgrade rolls back the migration row and its schema changes together. A
half-applied migration is never reported as healthy.

After a successful open, inspect `migration_report`:

```python
from openmed.structured.store import MigrationHealth

assert metadata.migration_report.state is MigrationHealth.HEALTHY
```

Opening fails closed when an applied checksum differs or the database has a
newer migration version. `open()` returns `unsupported` for compatibility
failures and `failure` for connection or migration failures. Do not rewrite a
checksum or delete a migration row to bypass either state. Restore a known-good
backup or deploy code that understands the schema.

## Allowlisted object namespaces

`ObjectStoreNamespace` defaults to the `file` protocol. Any other protocol must
be named explicitly:

```python
namespace = ObjectStoreNamespace(
    "s3://example-controlled-bucket/openmed/journey",
    allowed_protocols=frozenset({"s3"}),
    storage_options={"profile": "application-owned-profile"},
)
artifacts = FsspecArtifactStore(namespace)
```

This example only configures an adapter; it does not validate the service,
residency, encryption, retention, or credential policy. Install the matching
`fsspec` backend separately. Storage options are excluded from representations
and are never persisted by OpenMed.

The namespace rejects:

- a protocol not present in its explicit allowlist;
- credentials embedded in the URI;
- query strings, fragments, and parent traversal;
- an empty or unbounded resolved object root.

Object keys cannot be supplied by a record. They are derived exclusively from a
validated SHA-256 digest under
`blobs/sha256/<prefix>/<digest>`, then checked to remain below the resolved
namespace. Reads recompute the digest and fail closed on corruption.

## Atomicity boundary

PostgreSQL transactions cannot atomically commit an independent object store.
`ComposedJourneyStore` first verifies and writes immutable bytes, then commits
the complete metadata graph in one database transaction. If metadata fails and
this call created the object, the artifact backend performs a bounded
compensating delete.

If the database driver loses the commit acknowledgement, the facade returns
`unknown` with code `commit_status_unknown` and retains the object. It never
turns an ambiguous commit into success and never deletes bytes that a completed
but unacknowledged transaction may reference. Reconcile by reading the artifact
metadata by identifier or digest before retrying.

Use a single ingestion owner per digest namespace or add deployment-level
coordination when concurrent writers can ingest the same new digest. Periodic
orphan reconciliation should compare digest objects against committed artifact
metadata. Never delete an object merely because one transient reader cannot see
its metadata.

## Backup and restore

Treat database and object retention as one recovery set. Back up PostgreSQL
first at a named consistency point, record the highest committed revision, then
capture or version the object namespace. A database snapshot alone does not
contain artifact bytes.

Example operator workflow, with credentials supplied through the operator's
normal secret mechanism:

```bash
pg_dump --format=custom --schema=openmed_journey \
  --file=openmed-journey.dump "$OPENMED_POSTGRES_DSN"

pg_restore --clean --if-exists --schema=openmed_journey \
  --dbname="$OPENMED_POSTGRES_DSN" openmed-journey.dump
```

Restore into an isolated database first. Open the store, require a healthy
migration report, run `integrity_check()`, and sample artifact digest reads
before directing application traffic to it. `integrity_check()` reports only
table counts and controlled failures; it validates canonical payload hashes and
that PostgreSQL constraints are validated.

Do not put database dumps, object copies, credentials, raw clinical content, or
restricted datasets in source control. Encryption, object versioning, deletion
holds, disaster-recovery testing, and retention remain operator responsibilities.

## Conformance testing

The integration suite exercises the same synthetic graph against SQLite and a
caller-provided ephemeral PostgreSQL database, including idempotency, rollback,
historical canonical reads, restart, integrity verification, and interrupted
migration recovery:

```bash
OPENMED_TEST_POSTGRES_DSN='postgresql://...' \
  pytest tests/integration/test_journey_postgres_conformance.py -q
```

Use a disposable database role and database. The test creates a randomly named
`openmed_test_*` schema and removes only that schema during cleanup.
