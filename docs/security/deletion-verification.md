# Deletion verification

`openmed.core.deletion_verify` provides a local, deterministic guard for
removing sensitive artifacts such as a redacted source file or a temporary
mapping. It does not make network requests and it does not certify that a
clinical or regulatory obligation has been met.

## Contract

Callers must provide an explicit path and the expected SHA-256 fingerprint for
each regular file:

```python
from openmed.core.deletion_verify import (
    DeletionArtifact,
    delete_verified_artifacts,
    fingerprint_file,
)

artifact = "redacted-source.bin"
fingerprint = fingerprint_file(artifact)
result = delete_verified_artifacts(
    ".",
    [DeletionArtifact(artifact, fingerprint)],
    evidence_path="deletion-evidence.json",
)
```

The helper accepts a bare 64-character SHA-256 digest or the canonical
`sha256:<digest>` form. Fingerprints are checked while the file is open, and
the file identity is checked again immediately before it is staged.

All requested files are verified before any file is moved. The operation then
uses a private same-filesystem quarantine, an independent recovery copy, and a
retained read-only recovery descriptor. An error during staging, deletion,
recovery cleanup, or evidence publication restores and re-verifies every
original path. A request is limited to 128 files to bound open descriptors,
and the filesystem must have enough temporary capacity for one recovery copy
of each artifact.

Input paths must be inside `root`, must not contain `.` or `..` aliases, and
must not be symlinks, directories, or hard links. These restrictions
intentionally fail closed when the requested object is ambiguous. Callers
must also prevent concurrent writers from renaming or mutating the governed
directory tree during the transaction; the identity and fingerprint checks
detect observed changes but are not a substitute for exclusive ownership.

## Evidence and privacy

`DeletionEvidence.to_dict()` and an optional `evidence_path` contain only:

- the schema version;
- requested, verified, deleted, and rolled-back counts; and
- a stable operation status.

They contain no file names, paths, fingerprints, content, timestamps, or
free-form errors. Public exceptions expose only a stable failure code. Keep
the returned `DeletionArtifact` records in memory and do not log them.

The evidence file is written atomically. A failed preflight writes a
`rejected` count record when an evidence path was supplied. A transaction that
has to restore staged files writes `rolled_back` evidence. Completed evidence
is published only after all artifact and recovery paths have been removed. A
successful empty request is a deterministic completed no-op.

Rollback covers failures observed by the running process. It is not a
journaled filesystem transaction and cannot guarantee recovery after process,
kernel, storage-device, or power failure. Deletion also removes directory
entries; it is not a claim of physical secure erasure on flash or copy-on-write
storage.

### Windows recovery

Windows uses a verified in-memory recovery copy because an open backup handle
prevents rename and deletion. The combined recovery payload is limited to 128 MiB
per request. Requests that exceed this limit roll back staged files before any
payload is deleted. Recovery references are released after completion or rollback.
All platforms fingerprint and copy file contents in binary mode.
