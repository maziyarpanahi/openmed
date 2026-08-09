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
uses a private same-filesystem quarantine and recovery link so an error during
staging or deletion restores the verified paths. The input paths must be
inside `root`, must not contain `.` or `..` aliases, and must not be symlinks,
directories, or hard links. These restrictions intentionally fail closed when
the requested object is ambiguous.

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
has to restore staged files writes `rolled_back` evidence. A successful empty
request is a deterministic completed no-op.
