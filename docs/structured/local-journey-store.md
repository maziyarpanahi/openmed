# Durable local Journey storage

OpenMed v3 provides an offline-first persistence boundary for the immutable
Journey contracts. It separates source bytes from metadata:

- `LocalArtifactStore` writes verified bytes to content-addressed SHA-256 paths.
- `SQLiteJourneyStore` appends artifact metadata, evidence, facts, conflicts,
  resolutions, canonical pointers, dataset snapshots, and PHI-free job metadata.
- `LocalJourneyStore` joins both stores for atomic source-to-fact metadata
  ingestion.

These stores do not perform clinical interpretation. They persist caller-created
records and preserve explicit uncertainty or conflict states for review.

## Typed outcomes

Every expected read or mutation outcome is a `StoreResult`. Check `result.ok`
or its `StoreState`; do not interpret a missing value as success.

| State | Meaning |
| --- | --- |
| `success` | The requested record is available or was committed. |
| `partial` | A required parent, evidence item, or source is absent. |
| `unknown` | The requested identifier has no visible record. |
| `conflict` | An existing identifier or relationship disagrees. |
| `unsupported` | The persisted schema is newer or its migration checksum drifted. |
| `denied` | The configured storage policy rejected the operation. |
| `failure` | Integrity, parsing, I/O, or transaction verification failed. |

Non-success results carry controlled error codes only. Clinical values and
source bytes are excluded from result representations and exceptions.

## Content-addressed artifacts

Artifact writes validate both `byte_size` and `content_hash` before writing.
Files are created with owner-only permissions, written through a temporary
file, flushed, and linked into an immutable digest path. Reads recompute the
digest and fail closed on corruption.

```python
from pathlib import Path

from openmed.clinical import ClinicalArtifact, sha256_digest
from openmed.structured.store import LocalArtifactStore

content = b"synthetic clinical fixture"
artifact = ClinicalArtifact(
    artifact_id="artifact_aaaaaaaaaaaaaaaa",
    artifact_type="clinical_note",
    media_type="text/plain",
    content_hash=sha256_digest(content),
    byte_size=len(content),
    source_id="source_aaaaaaaaaaaaaaaa",
    recorded_at="2026-01-02T03:04:05Z",
)

store = LocalArtifactStore(Path("./journey-data/artifacts"))
result = store.put_bytes(artifact, content)
assert result.ok
```

The digest path reveals equality between identical byte sequences. Use a
caller-controlled encrypted volume and access controls when local artifacts
contain sensitive data. OpenMed does not claim application-level at-rest
encryption in this store.

## Atomic graph transactions

`SQLiteJourneyStore.transaction()` assigns an exact revision and commits all
mutations together. A raised exception or any typed non-success result from a
transaction mutation rolls back the revision and every earlier mutation in that
transaction.

```python
from openmed.structured.store import SQLiteJourneyStore

metadata = SQLiteJourneyStore("./journey-data/journey.sqlite3")

with metadata.transaction(committed_at="2026-01-02T03:04:05Z") as transaction:
    artifact_result = transaction.put_artifact(artifact)
    if not artifact_result.ok:
        raise RuntimeError("artifact metadata was not accepted")
    # Add evidence before facts, and facts before conflicts or canonical records.
```

For the common ingestion path, `LocalJourneyStore.ingest_graph()` verifies the
blob and atomically writes artifact metadata, evidence, facts, and conflicts.
Replaying identical records is idempotent and creates neither duplicate rows nor
an empty revision.

## Point-in-time history

Facts are immutable rows. A correction is a new `ClinicalFact` whose
`parent_fact_ids` reference earlier facts. `CanonicalRecord` appends the selected
fact for one logical record; it never updates an earlier row.

```python
from openmed.structured.store import StorePoint

earlier = metadata.get_canonical(
    "canonical_aaaaaaaaaaaaaaaa",
    as_of=StorePoint(revision=12),
)
current = metadata.get_canonical("canonical_aaaaaaaaaaaaaaaa")
```

`list_canonical_versions()` and `list_facts()` return deterministic append-only
history through an optional revision. Later corrections therefore do not change
the result of an earlier point-in-time read.

## Schema and migration compatibility

The local persisted schema is version `1.0.0`; the SQLite migration level is
`LATEST_MIGRATION_VERSION`. Each ordered migration has a SHA-256 checksum stored
in `schema_migrations`.

Opening a database fails with typed state `unsupported` when it contains a newer
migration or when an already-applied migration checksum differs. OpenMed never
guesses how to reinterpret an unknown persisted schema. Migrations execute in a
single immediate transaction and are safe to check again after restart.

`CanonicalRecord` and `JobMetadata` include their own explicit schema versions.
The store also verifies that its supported major version matches the Journey
contract major version before opening.

## Privacy boundary

- No network client, hosted check, or telemetry path is part of these stores.
- Committed fixtures are synthetic.
- SQLite rows use opaque identifiers and canonical record payloads.
- Job metadata rejects common raw-text, secret, credential, token, vault, and
  PHI-bearing field names recursively.
- Store results, errors, and integrity reports expose controlled codes and
  counts, not clinical content.
- Restricted datasets, terminology, and credentials remain caller supplied.

The storage policy interface can deny read, write, or verification operations.
A denial returns `StoreState.DENIED`; it is not represented as an empty success.
Metadata key rejection is defense in depth, not PHI detection. Callers remain
responsible for supplying counts, controlled codes, and opaque identifiers
rather than sensitive values.

## Recovery and verification

SQLite uses foreign keys, full synchronous writes, and write-ahead logging.
`close()` checkpoints the write-ahead log. On restart, use
`integrity_check()` to verify SQLite structure and every persisted payload hash.
Artifact reads independently verify their content digest.

Backup and retention remain operator responsibilities. Do not copy a live
database and omit its WAL files; close or checkpoint it first, or use SQLite's
backup API.
