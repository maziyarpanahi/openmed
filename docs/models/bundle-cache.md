# Crash-safe bundle generation caches

Contract version 1 (OpenMed v4.0). OpenMedKit for Android, OpenMedKit for
Swift, and the `openmed` web package store [signed portable
bundles](portable-bundle.md) in a generation cache with one lifecycle. The
cache stages bytes in isolation, resumes only from verified partial state,
verifies every declared file against the signed manifest, and switches an
activation pointer only after a complete, atomic commit. Quota eviction never
removes a generation that is active, last green, kept for rollback, pinned, or
leased.

The cache never opens a network connection. Bytes arrive through a
caller-supplied *range source*; a local directory source is built in, so a
prepared bundle can be imported and opened fully offline. Network acquisition
stays explicit: an app plugs its own downloader in as a range source.

| Kit | Entry point | Storage |
|---|---|---|
| Android | `com.openmed.openmedkit.cache.BundleGenerationCache` (also `ModelCache.bundleGenerations()`) | app-private directory, `FileChannel` locks |
| Swift | `OpenMedBundleCache` (also `OpenMedModelStore.bundleCache()`) | `FileManager` directory, `flock(2)` locks |
| Web | `BundleGenerationCache` with `openBundleCacheStorage()` | OPFS, IndexedDB fallback, Web Locks |

## Identities

Cache metadata holds opaque identities and declared digests only:

- **Bundle key**: `b-` followed by the first 32 hex digits of the SHA-256 of
  the manifest's `bundle.id`. Every bundle has its own directory, lock, and
  activation pointer.
- **Generation**: `g-` followed by the first 32 hex digits of the manifest
  content identity (`sha256:` of the manifest bytes). The same signed manifest
  always maps to the same generation, so concurrent or repeated acquisitions
  converge on one copy. A changed manifest is a different generation and can
  never reuse another generation's bytes.
- **Source digest**: `sha256:` of the range source's opaque identity (for
  example a URL plus an ETag). The identity itself is never stored.
- **Trust digest**: `sha256:` of the RFC 8785 encoding of
  `{"keys":[{"kid","not_after","not_before","revoked","x"}],"revoked_content_ids":[...],"threshold":n}`,
  with keys sorted by `kid`, revoked content ids sorted, and `x` as unpadded
  base64url.

No record, error, log line, or report contains a filesystem path, URL,
source identity, credential, or file content.

## Generation states

Each generation has a persisted record state (`staging`, `verifying`, `ready`,
or `quarantined`) plus an activation pointer. The reported state is derived
from them:

| State | Meaning |
|---|---|
| `absent` | Nothing is stored for the generation. |
| `staging` | An owner holding the bundle lock is writing bytes. |
| `resumable` | Staged bytes with an intact resume record and no live owner (after a crash, cancellation, or failed transfer). |
| `verifying` | All bytes are staged and the owner is checking every declared file. |
| `ready` | Verified and committed, never activated, and protected (for example pinned). |
| `active` | The activation pointer names it. |
| `superseded` | Verified and previously active, and still protected (rollback target, last green, pinned, or leased). |
| `evictable` | Verified, inactive, and unprotected. Only evictable generations are quota candidates. |
| `quarantined` | Failed verification or detected corruption. Unreachable until recovery purges it. |
| `evicted` | Removed by quota eviction. Afterwards the generation is `absent`. |

`staging` and `verifying` are visible only while an operation runs. A snapshot
taken between operations reports staged bytes as `resumable`.

Transitions:

```text
absent ──acquire──▶ staging ──all bytes──▶ verifying ──verified──▶ ready/evictable
   ▲                  │  ▲                   │                       │
   │       interrupt, │  │ resume            │ mismatch              │ activate
   │   cancel, crash  ▼  │                   ▼                       ▼
   │               resumable            quarantined ◀── corrupt ── active
   │                  │                      │                       │ next activate
   │  expired/orphan  │                      │ recover               ▼
   └──────────────────┴──────────────────────┴─── evict ◀─ evictable ◀─ superseded
```

## Protections and the activation pointer

The pointer record holds `active`, `last_green`, and `rollback`:

- `activate(g, confirm)`: the previous active generation becomes `rollback`,
  `g` becomes `active`, and with `confirm` (the default) `g` also becomes
  `last_green`. With `confirm: false` the activation is on probation: the old
  `last_green` stays protected until `confirm()`.
- `confirm()`: sets `last_green` to the active generation.
- `rollback(quarantine)`: reactivates the `rollback` generation without any
  transfer and clears `rollback`. If `last_green` named the abandoned
  generation, it moves to the restored one. With `quarantine: true` the
  abandoned generation is quarantined with reason `rolled_back`. Without a
  rollback target the call fails with `rollback_unavailable`.

A generation is protected when it is `active`, `last_green`, `rollback`,
`pinned`, or `leased`. Snapshots list protections in that order.

**Leases** mark generations in use. `lease()` and `open()` write a lease record
with an owner id and an expiry. A lease held by the current cache instance
protects its generation until it is released. A lease left by another owner
(another process, or a process that died) protects it until its expiry; after
that, recovery removes it with `lease_expired`.

## Acquisition

`acquire` runs these steps under the bundle lock, in order:

1. Verify the manifest and envelope with the portable-bundle verifier, without
   files. A refusal returns `bundle_rejected` with the verifier reason; nothing
   is written.
2. If the generation is already ready, return it (`from_cache`) and activate
   it when asked. A ready record whose files no longer match is quarantined
   (`payload_mismatch`) and the pointer is repaired before restaging.
3. Discard staging of every other generation of the same bundle, and any
   quarantine entry of this generation.
4. Check resume state. Staged bytes are reused only when the content identity,
   source digest, trust digest, and chunk size all match the resume record.
   Otherwise everything is discarded (`source_changed`,
   `trust_policy_changed`, `chunk_size_changed`). Each kept file is re-hashed
   chunk by chunk against the recorded chunk digests; the file is truncated at
   the first missing or mismatching chunk (`partial_digest_mismatch`).
5. Preflight: fail with `storage_pressure` when free storage is below the
   bytes still needed, then with `quota_exceeded` when evicting every
   evictable generation would not make room. Planned evictions run only after
   both checks pass.
6. Write the resume record, then request each incomplete file from its
   verified offset. Bytes are buffered and appended one chunk at a time
   (4 MiB by default); after every chunk the resume record gains that chunk's
   SHA-256. A source that answers from offset 0 instead of the requested
   offset restarts that file (`range_not_honored`); any other offset fails with
   `range_mismatch`. A source failure or early end of file returns
   `transfer_interrupted`, and a cancellation request (checked after every
   chunk) returns `cancelled`. Both keep the staged bytes resumable.
7. Record the `verifying` phase and verify every declared file in manifest
   order (presence, size, full SHA-256), then refuse undeclared files. A
   source that sends more bytes than declared fails the same way. Failures
   quarantine the staging with the verifier reason and return
   `verification_failed`.
8. Commit: write the ready record and move the staging into the generation
   area in one atomic step (a directory rename on Android and Swift, a single
   record write on the web).
9. Activate when asked, by an atomic pointer replacement.
10. Trim to the quota, never evicting the generation just committed.

No ready path or handle exists before step 8 completes.

## Quota and eviction

Usage is the declared byte total of every ready generation across all bundles
in the cache. Eviction candidates are evictable generations only, ordered by
last access time and then by generation id, so the same state always produces
the same plan. A generation is evicted by first making it unreachable (a rename
into the trash area, or deleting its ready record) and then deleting its bytes.
`evict(additional_bytes)` returns a quota decision with the evicted
generations, the remaining usage, and whether the quota is satisfied. When
only protected generations remain, the decision is unsatisfied and nothing
protected is touched.

## Locks

Every mutating operation holds the bundle lock: an in-process mutex plus an
OS advisory lock on the bundle's lock file (`FileChannel.lock` on Android,
`flock` on Swift, the Web Locks API in browsers). Waiting is bounded by
`lock_timeout_ms`; a timeout fails with `lock_timeout` and changes nothing.
The holder also writes an owner record. A record that exists when the lock is
acquired belonged to a holder that died without releasing it; recovery reports
it as `stale_lock_cleared`. Because the OS releases advisory locks when a
process dies, a stale record never blocks another process. Browsers without
Web Locks fall back to an in-page mutex, which does not exclude other tabs.

## Crash recovery

Call `recover()` at startup. It is idempotent and bounded, and processes each
bundle in bundle-key order:

| Order | Condition | Action |
|---|---|---|
| 1 | Lock owner record left by a dead holder | `stale_lock_cleared` |
| 2 | Trash entry (an interrupted eviction or cleanup) | `trash_purged` |
| 3 | Quarantine entry | `quarantine_purged` |
| 4 | Staging without a readable resume record | `staging_orphaned` (removed) |
| 4 | Staging older than `staging_max_age_ms` (default 7 days) since its last checkpoint | `staging_expired` (removed) |
| 4 | Other staging, including an interrupted `verifying` phase | `staging_resumable` (kept; re-verified before any commit) |
| 5 | Ready generation with an unreadable record or a file whose size changed | `generation_quarantined` (reason `payload_mismatch`) |
| 6 | Pointer naming a generation that is no longer ready | `pointer_repaired`: `active` falls back to `last_green`, then `rollback`, then none |
| 7 | Lease of another owner past its expiry | `lease_expired` |

Within a step, entries are processed in generation-id order. Staging is never
promoted by recovery. A generation quarantined in step 5 stays visible until
the next recovery purges it. Recovery reads sizes only; `open(verification:
full)` re-hashes every file, and on a mismatch quarantines the generation,
repairs the pointer the same way, and fails with `generation_corrupt`.

Named crash checkpoints (used by the shared vectors) are
`staging_record_written`, `chunk_committed`, `verifying_record_written`,
`generation_committed`, `pointer_committed`, and `eviction_committed`. A crash
at any of them leaves the previous active and last-green generations intact.

## Reason codes

Operation failures: `bundle_rejected`, `lock_timeout`, `quota_exceeded`,
`storage_pressure`, `transfer_interrupted`, `cancelled`, `range_mismatch`,
`verification_failed`, `storage_io`, `generation_not_ready`,
`generation_corrupt`, `no_active_generation`, `rollback_unavailable`,
`lease_unknown`. `bundle_rejected`, `verification_failed`, and
`generation_corrupt` also carry the portable-bundle verifier reason.

Resume reasons: `source_changed`, `trust_policy_changed`, `chunk_size_changed`,
`partial_digest_mismatch`, `range_not_honored`.

Quarantine reasons: `file_missing`, `file_size_mismatch`,
`file_digest_mismatch`, `file_undeclared`, `payload_mismatch`, `rolled_back`.

## Public result

Every kit returns the same value-free fields:

- acquisition: `ok`, `reason`, `bundle_reason`, `bundle_key`, `generation`,
  `state`, `from_cache`, `resumed`, `resume_reason`, `bytes_transferred`,
  `verified_bytes`, `files_verified`, `evicted`;
- snapshot (`openmed.bundle_cache.snapshot`, version 1): the pointer, each
  generation's `state`, `total_bytes`, `verified_bytes`, and `protections`,
  the quarantined generations with their reasons, and `used_bytes`;
- quota decision: `quota_bytes`, `used_bytes`, `additional_bytes`, `evicted`,
  `satisfied`;
- recovery report: the ordered `actions`, each with its generation or `null`.

A lease or open result exposes the payload location to the loader, but its
string form omits it.

## Storage layout

Android and Swift use one directory per bundle:

```text
<root>/<bundle key>/
  lock, lock.json          bundle lock and owner record
  pointer.json             activation pointer (replaced atomically)
  staging/<generation>/    staging.json + payload/
  generations/<generation>/ generation.json + payload/
  quarantine/<generation>/ quarantine.json + the moved content
  trash/                   entries being deleted
  leases/<generation>/<lease>.json
```

The web adapter keeps the same records as keys in OPFS files or an IndexedDB
object store, and stores payload bytes as one entry per verified chunk, so a
crash can never leave a torn chunk and resuming never rewrites earlier bytes.
Every write is atomic in both: an OPFS writable stream replaces its file on
close, and an IndexedDB write is a single transaction. A lease reads a file
back as bytes or as a `Blob` assembled from the chunk entries.

## Shared vectors

`tests/fixtures/bundle_cache/vectors.json` holds synthetic signed bundles and
32 scripted scenarios: resume, changed source or trust policy, tampered
partial bytes, range handling, crashes at every checkpoint, restarts,
verification failures, signature and schema refusals, cancellation, storage
pressure, quota, LRU order, pins, leases, rollback, probation, corruption, and
offline opening. Android, Swift, and web tests all replay them with a fake
range source, a fake clock, and crash checkpoints. Regenerate or check them
with:

```bash
python tests/fixtures/bundle_cache/generate_vectors.py
python tests/fixtures/bundle_cache/generate_vectors.py --check
```

## Relationship to other caches

- `openmed.core.model_cache_policy` (Python) applies the same deterministic
  rule, least recently used first with a stable tie breaker and pinned entries
  excluded, to artifacts an application registers.
- `ModelCache` and the Android format-aware acquisition keep serving Hugging
  Face snapshots. A downloader with HTTP Range support can serve a bundle
  generation cache by implementing the range-source interface.
- `OpenMedModelStore` (Swift) keeps downloading MLX snapshots; its errors carry
  reason codes and status codes only.
