# Encrypted Service Backup and Restore

`openmed recovery` creates and restores encrypted, integrity-checked recovery
points for the durable local state an OpenMed service needs to resume safely.
Restore always writes to a new, inactive directory. The live state changes
only at an explicit activation step, which re-verifies the restored files and
then atomically repoints one symbolic link.

The same operations are available from Python in `openmed.service.recovery`.

## What a recovery point contains

| Category | Source | Notes |
| --- | --- | --- |
| `journey_metadata` | local Journey `journey.sqlite3` | captured with the SQLite online backup API, in memory |
| `journey_artifacts` | `artifacts/blobs/sha256/**` | content-addressed blobs, already sealed by the Journey key |
| `journey_encryption` | `artifacts/encryption.json` | key identifiers and verifiers only, never key material |
| `job_metadata` | async job store JSON | metadata only; raw job payloads never leave memory |
| `model_registry` | registry slot, pointer, and lineage state | schema version checked on restore |
| `audit_chain` | audit-chain JSON files | each chain must verify before backup and after restore |

These categories are classified but never copied:

| Category | Disposition |
| --- | --- |
| `encryption_keys` | excluded secret: resolved through caller-owned key providers |
| `service_credentials` | excluded secret: API keys, JWKS, TLS keys, DSNs |
| `model_artifacts` | rebuildable: re-acquire through verified downloads or an air-gapped kit |
| `result_cache` | rebuildable |
| `sqlite_journal_files` | transient: `-wal`/`-shm` content is folded in by the online backup |
| `staging_files` | transient |
| `job_payloads` | transient: in memory only |

PostgreSQL-backed Journey deployments keep using `pg_dump` and storage
snapshots, as described in the Compose and Helm guides. This tool covers the
local SQLite and filesystem state.

## State layout and the live link

Restored state always uses one canonical layout under a state root:

```text
<state-root>/
  journey/journey.sqlite3
  journey/artifacts/encryption.json
  journey/artifacts/blobs/sha256/<2>/<64>
  jobs/service-jobs.json
  registry/registry_state.json
  audit/chain-0000.json
```

Run the service against a symbolic link, for example `/var/lib/openmed/state`,
that points at the current generation directory. Set
`OPENMED_SERVICE_JOBS_STORE_PATH=/var/lib/openmed/state/jobs/service-jobs.json`
and open any local Journey store at `/var/lib/openmed/state/journey`.
Activation replaces that link with one atomic `rename`. A live path that is a
real directory is
refused (`live_target_unmanaged`); move the directory into a generation and
create the link once while the service is stopped. Activation is not
supported on Windows (`atomic_swap_unsupported`).

## Keys

Backups are sealed with AES-256-GCM envelopes under the caller's
`service-backup` key. Keys never enter a backup. The CLI reads keys from
caller-owned files given as `KEY_ID=FILE`; each file holds exactly 32 raw
bytes or 64 hexadecimal characters. The first `--key` is the active key. Add
older keys after it, so recovery points sealed before a rotation still open.

```bash
openmed recovery inventory --state-root /var/lib/openmed/state
```

The inventory prints categories, object counts, sizes, and schema versions
only.

## Back up

Stop admission and every writer first, then assert it with
`--writers-stopped`. The Python API also accepts a `quiesce` context-manager
factory that a drain coordinator can provide instead.

```bash
openmed recovery backup /backups/rp-0 \
  --state-root /var/lib/openmed/state \
  --key backup-2026=/run/secrets/openmed-backup.key \
  --writers-stopped --retention-days 35

openmed recovery backup /backups/rp-1 --parent /backups/rp-0 \
  --state-root /var/lib/openmed/state \
  --key backup-2026=/run/secrets/openmed-backup.key --writers-stopped
```

A backup fingerprints every source before and after capture. If anything
changed in between, the backup is refused (`snapshot_boundary_moved`). The
point is written to a hidden sibling directory, verified by decrypting every
stored object, and renamed into place only when complete.

An incremental point stores only objects whose digest changed since its
parent and references the rest. The manifest records the parent manifest
digest, the chain sequence, and the snapshot bounds (Journey revision, job
count, audit entry counts and heads, and schema versions).

## Verify, restore, and activate

```bash
openmed recovery verify /backups/rp-1 --ancestor /backups/rp-0 \
  --key backup-2026=/run/secrets/openmed-backup.key --max-age-hours 48

openmed recovery restore /backups/rp-1 /var/lib/openmed/generations/g7 \
  --ancestor /backups/rp-0 --live /var/lib/openmed/state \
  --key backup-2026=/run/secrets/openmed-backup.key

openmed recovery activate /var/lib/openmed/generations/g7 /var/lib/openmed/state \
  --key backup-2026=/run/secrets/openmed-backup.key \
  --journey-key journey-2026=/run/secrets/openmed-journey.key
```

Restore refuses, and removes its partial target, unless every check passes:

- the manifest and every chunk authenticate under the backup key, so
  corruption, truncation, reordering, and objects moved between points fail;
- every object digest and size matches the manifest;
- the chain is complete, its parent digests and sequence numbers match, and
  its sequence bounds never regress;
- the point has not passed its retention or the `--max-age-hours` limit;
- component schema versions are supported and the Journey migrations match
  known checksums;
- every Journey artifact row has its blob, and the encryption metadata agrees
  with the database;
- the restored content reproduces the manifest's snapshot boundary.

Only then is a sealed staging record written. `activate` authenticates that
record, re-hashes every staged file, opens the restored Journey store with the
Journey key (an integrity check plus reading every artifact), runs any further
checks, and calls any recovery-epoch hooks. Only after all of that does it
repoint the link. Without `--journey-key`, activation of a state that has a
Journey store is refused. Keep traffic stopped until activation completes and
the service's own readiness checks pass.

From Python, pass the production self-check through `checks=` and components
that must fence pre-restore authority (for example approval nonce stores)
through `epoch_hooks=`. Each hook's `open_recovery_epoch(context)` receives
the recovery-point id, manifest and content digests, and snapshot bounds
before the swap; raising refuses activation.

The previous generation is kept. To roll back, restore again into a new
directory: a generation that has served traffic no longer matches its
staging record. `openmed recovery discard <dir> --live <link>` removes an
inactive restored directory and never the live target.

## Receipts and reason codes

Every command prints a versioned receipt
(`openmed.service.recovery-receipt.v1`) holding the opaque recovery-point id,
component categories, digests, sizes, schema versions, snapshot bounds,
counts, timestamps, and one controlled reason code. Receipts, manifests, and
errors never contain paths, file names, key identifiers, key material, or
state content. Refusals exit with status 1; usage errors exit with status 2.

| Reason | Meaning |
| --- | --- |
| `quiesce_required` | no drain hook and no `--writers-stopped` assertion |
| `snapshot_boundary_moved` | a source changed during capture |
| `manifest_authentication_failed` | wrong key or tampered manifest |
| `object_authentication_failed`, `object_corrupt` | tampered, truncated, or reordered object |
| `chain_incomplete`, `chain_invalid` | missing or substituted increment |
| `schema_incompatible` | a component schema this release cannot read |
| `cross_reference_missing_artifact`, `snapshot_boundary_mismatch` | restored components disagree |
| `recovery_point_expired`, `recovery_point_stale` | retention or maximum age exceeded |
| `restore_incomplete`, `staged_state_modified` | the staged directory cannot be activated |
| `activation_check_failed` | a pre-activation check failed; `details` names it |

## Limits

- The Journey database is snapshotted into memory, so backup memory grows with
  the database size.
- The CLI cannot drain a running service; stop writers before `backup`.
- PostgreSQL Journey stores, remote artifact stores, and cloud snapshots are
  out of scope.
