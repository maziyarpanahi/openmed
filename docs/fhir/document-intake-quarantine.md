# Document-intake quarantine policy

`openmed.interop.document_quarantine` provides a local, deterministic intake
gate for attachments that must be classified before a parser sees them. It
does not make a clinical decision, certify compliance, or contact a remote
validator.

## Classify an attachment

Pass the bytes, the MIME value supplied by the transport boundary, and the
filename used for the extension check:

```python
from openmed.interop.document_quarantine import classify_document

result = classify_document(
    b"%PDF-1.7\nsynthetic fixture",
    declared_mime="application/pdf",
    filename="synthetic-report.pdf",
)

safe_report = result.to_dict()
```

The default policy accepts common local document, image, text, JSON, XML, and
archive types up to 25,000,000 bytes. A top-level archive has depth one; one
archive layer is allowed, while nested archives beyond the configured limit
are rejected. ZIP and tar members are checked for path traversal, symlinks,
entry count, declared uncompressed size, and recursively inspectable nested
archives.

Use `DocumentQuarantinePolicy` to change the limits or allowlists for a local
deployment:

```python
from openmed.interop.document_quarantine import (
    DocumentQuarantinePolicy,
    classify_document,
)

policy = DocumentQuarantinePolicy(
    max_size_bytes=10_000_000,
    max_archive_depth=1,
)
result = classify_document(
    payload,
    declared_mime="application/pdf",
    filename="synthetic-report.pdf",
    policy=policy,
)
```

## Dispositions and reason codes

An attachment is `accepted` only when its declared MIME, extension, and local
content sniff agree, the MIME and extension are allowlisted, and archive
limits pass. Missing or conflicting signals are `quarantined` for an explicit
local review. Empty or oversized payloads, unsupported types, malformed
archives, unsafe archive members, and archive-depth violations are
`rejected`.

`QuarantineResult` exposes only:

- `disposition`: `accepted`, `quarantined`, or `rejected`;
- `reason_codes`: stable machine-readable values such as
  `declared_mime_sniff_mismatch` or `archive_depth_exceeded`; and
- `sha256`: a digest of the supplied bytes.

`to_dict()` is safe to place in a log, report, or audit record. It does not
include the filename, declared MIME, sniffed type, archive member names,
payload bytes, or exception details. Callers should discard or separately
protect the original payload after routing it according to the disposition.

The classifier uses only Python's local standard-library inspection helpers.
It performs no mandatory network call and has no telemetry or remote fallback.
