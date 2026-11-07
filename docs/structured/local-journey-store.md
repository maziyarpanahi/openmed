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

Every local store requires an explicit encryption choice: a caller-supplied
key provider, or the development-only plaintext opt-in. See
[Application-layer encryption](#application-layer-encryption).

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
from openmed.core.key_provider import InMemoryKeyProvider, KeyState
from openmed.structured.store import (
    JOURNEY_KEY_PURPOSE,
    JourneyStoreEncryption,
    LocalArtifactStore,
)

# Load key material from your own custody system; never hard-code it.
provider = InMemoryKeyProvider()
provider.add_key(
    "journey-2026-01",
    load_journey_key(),  # caller-supplied, 32 bytes
    purpose=JOURNEY_KEY_PURPOSE,
    state=KeyState.ACTIVE,
)
encryption = JourneyStoreEncryption.encrypted(provider)

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

store = LocalArtifactStore(Path("./journey-data/artifacts"), encryption=encryption)
result = store.put_bytes(artifact, content)
assert result.ok
```

With an encrypted configuration the blob file, and the temporary file it is
written through, contain only an AEAD envelope; reads decrypt and then
recompute the digest. The digest path still reveals equality between identical
byte sequences and lets anyone holding a candidate document confirm it is
stored. Keep the store on an access-controlled volume.

## Atomic graph transactions

`SQLiteJourneyStore.transaction()` assigns an exact revision and commits all
mutations together. A raised exception or any typed non-success result from a
transaction mutation rolls back the revision and every earlier mutation in that
transaction.

```python
from openmed.structured.store import SQLiteJourneyStore

metadata = SQLiteJourneyStore(
    "./journey-data/journey.sqlite3",
    encryption=encryption,
)

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

## Application-layer encryption

Local Journey storage encrypts clinical content before it reaches the
filesystem, independently of any whole-disk encryption. `LocalArtifactStore`,
`SQLiteJourneyStore`, `LocalJourneyStore`, and `SQLiteIngestionStore` all take
an `encryption=` argument, and none of them selects a mode on its own:

| Configuration | Behavior |
| --- | --- |
| `JourneyStoreEncryption.encrypted(provider)` | Seals artifact bytes and every Journey `payload_json` value with the provider's active key. |
| `JourneyStoreEncryption.development_plaintext()` | Stores readable payloads. For development and tests only. |
| no argument | Refused with `JourneyEncryptionError` code `encryption_required`. |

The production profile (`OPENMED_PROFILE=prod`, which the bundled container
images, Compose files, and Helm chart set) refuses the plaintext opt-in with
`plaintext_refused_in_production`. Code cannot weaken the profile: passing
`profile="dev"` does not override a production environment, while
`profile="prod"` enforces production rules without the environment variable.
Encryption needs the optional `cryptography` dependency (`openmed[integrity]`);
without it, encrypted operations fail closed with
`encryption_dependency_unavailable`.

### Key providers

Callers own the keys. A key provider implements
`openmed.core.key_provider.KeyProvider`:

- `active_key(purpose)` returns the single key used for new ciphertext.
- `resolve_key(key_id, purpose=...)` returns the key named in an envelope
  header.

Keys are 32 bytes and carry an opaque, non-secret identifier and a lifecycle
state: `active`, `decrypt_only` (superseded by a rotation, still readable),
`retired`, or `revoked`. OpenMed never persists, logs, pickles, or represents
key material. `InMemoryKeyProvider` holds caller-supplied material in process
memory. It suits tests, synthetic runs, and adapters that load material from a
keychain, hardware module, or custody bridge. Bundled cloud KMS clients are out
of scope. Back up keys separately from data; losing every key for a purpose
makes its data unrecoverable by design.

### Envelope format

`openmed.core.aead_envelope` defines version 1 of the envelope: magic
`OMAE`, version, algorithm, key identifier, a 96-bit random nonce, and an
AES-256-GCM ciphertext with its tag. The cipher key is derived from the
provider key per purpose. The header and a record context are authenticated as
associated data. For SQLite rows the context is the table and primary key (for
example `canonical_record_versions` with `canonical_id` and `version`). For
blobs it is the content digest. A ciphertext copied onto another record, or
relabelled with another key identifier, fails with
`envelope_authentication_failed`. SQLite stores the envelope as ASCII text
with an `omae:` prefix.

Each store also persists a verifier per key identifier: an envelope of a fixed
constant. If a provider returns different material under a known key
identifier, opening the store fails with `key_check_failed` before anything is
written.

### What stays in cleartext

Payload-bearing values are sealed. The following stay readable so the stores
can enforce identity, ordering, idempotency, and integrity without keys:

- opaque identifiers (`artifact_id`, `subject_id`, `fact_id`, and so on),
  revisions, version numbers, and commit timestamps;
- the controlled `fact_type` and job `state` codes;
- `content_hash` and `payload_hash`, which are unkeyed SHA-256 digests of the
  plaintext and therefore reveal equality and allow confirming a guessed value;
- the encryption state, key purpose, envelope version, key identifiers, and
  key verifiers;
- the ingestion ledger tables of `SQLiteIngestionStore`, whose contracts carry
  only digests, opaque identifiers, reason codes, and timestamps.

### SQLite journals and temporary files

Payloads are sealed before SQLite sees them, so the database, its write-ahead
log (`-wal`), and its shared-memory index (`-shm`) receive only ciphertext for
payload columns, plus the cleartext fields above. Stores open with
`secure_delete` on, so freed cells are zeroed. They also set
`temp_store = MEMORY`, so sorts and transient indexes stay off disk. Blob
writes go through an owner-only temporary file that already holds the sealed
envelope. Previously written plaintext is the exception: until a migration
completes, the database and its write-ahead log may still hold plaintext
pages. Completing a migration vacuums the database and truncates the
write-ahead log. Application-layer encryption cannot scrub blocks the
filesystem already released, so migrate legacy stores on an encrypted volume
when that matters.

### Key rotation

1. Add the new key to the provider as `active`; the previous key becomes
   `decrypt_only`. New writes use the new key immediately.
2. Run `store.reencrypt()`. It re-seals rows and blobs that are still sealed
   under an older key, in atomic batches. Identifiers, payload hashes,
   revisions, history, and point-in-time reads do not change.
3. Repeat until the report's `complete` is `True`, then retire the old key.

Re-encryption is restart-safe. Each SQLite batch commits atomically, each blob
is replaced atomically, and progress is derived from the envelope headers, so
an interrupted run loses at most the batch in flight and resumes when called
again. `batch_size` and `max_batches` (or `max_files`) bound the work per call.
A typed failure carries a `ReencryptionReport` of the progress committed
before it.

### Migrating a plaintext store

Stores created before encryption, or with the plaintext opt-in, refuse an
encrypted configuration with `store_migration_required`. Migrate them
explicitly:

```python
from openmed.structured.store import LocalJourneyStore

result = LocalJourneyStore.migrate_to_encrypted(
    "./journey-data",
    encryption=encryption,
)
assert result.ok and result.value.complete
```

1. Close the store and back it up, including its `-wal` file, or use SQLite's
   backup API.
2. Run the migration. The store is marked `migrating` first. From then on it
   opens only with an encrypted configuration, and reads accept both forms.
3. If it is interrupted, run it again. It resumes from the remaining plaintext;
   the batch in flight is rolled back.
4. On completion the store is marked `encrypted`, stale temporary blob files
   are removed, and the database is vacuumed with its write-ahead log
   truncated.
5. Verify reads and `integrity_check()`, then destroy or separately protect
   the plaintext backup. The migration cannot reach copies outside the store.

`SQLiteJourneyStore.migrate_to_encrypted()` and
`LocalArtifactStore.migrate_to_encrypted()` migrate one half each. An encrypted
store never reverts to plaintext.

### Failure codes

Key and envelope failures never return partial plaintext and never change the
stored bytes. They surface as `StoreState.FAILURE` with one of these codes, or
as `JourneyEncryptionError` from a constructor:

| Code | Meaning |
| --- | --- |
| `encryption_required` | No encryption configuration was supplied. |
| `plaintext_refused_in_production` | The plaintext opt-in was used under the production profile. |
| `store_migration_required` | A plaintext store was opened with an encrypted configuration. |
| `store_encrypted` | An encrypted or migrating store was opened as plaintext. |
| `key_purpose_mismatch` | The store was encrypted for another key purpose. |
| `key_check_failed` | The provider's material does not match a known key identifier. |
| `key_unavailable` | The key is unknown, destroyed, or the provider failed. |
| `key_retired` / `key_revoked` | The key was retired or revoked. |
| `envelope_authentication_failed` | Wrong key material, tampering, or a moved ciphertext. |
| `envelope_malformed` / `envelope_version_unsupported` | The stored envelope cannot be parsed. |
| `payload_not_encrypted` | Plaintext was found where ciphertext is required. |
| `encryption_dependency_unavailable` | `cryptography` is not installed. |

Error messages are fixed strings. They never include key identifiers, key
material, record identifiers, or content.

### Scope

Encrypted backup and restore that carries the encryption metadata without
bundling keys is tracked separately. `PostgresJourneyStore` does not seal
payload columns; rely on PostgreSQL access controls and server-side storage
encryption, and note that its `migrate_to_encrypted()` returns `unsupported`.
`FsspecArtifactStore` and the identity-resolution store are also unchanged.

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
`integrity_check()` to verify SQLite structure and every persisted payload hash;
for an encrypted store this decrypts every payload, so it also proves the
provider can still read the whole store. Artifact reads independently verify
their content digest after decryption.

Backup and retention remain operator responsibilities. Do not copy a live
database and omit its WAL files; close or checkpoint it first, or use SQLite's
backup API.
