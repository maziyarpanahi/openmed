# Bounded asset digests

Use `digest_asset` to compute a SHA-256 digest without loading a PDF, image,
DICOM object, audio file, or other binary asset into memory at once.

```python
from openmed.multimodal.digest import digest_asset

with open("synthetic-scan.dcm", "rb") as stream:
    result = digest_asset(stream, max_bytes=2 * 1024**3)

print(result.sha256)
print(result.byte_count)
```

Streams are read from their current position in chunks no larger than 1 MiB.
Seekable streams are restored to that position on success or failure, and the
helper never closes a caller-owned stream. Non-seekable streams are consumed.
Passing `bytes` hashes the existing in-memory value directly.

`max_bytes` is an optional hard limit. The helper reads at most one byte beyond
the limit to detect overflow, then raises `DigestLimitExceededError`. Its error
contains only the `digest_size_limit` category, `maximum_bytes`, and
`bytes_read`; it never includes paths, filenames, or asset content. Other
stream failures raise a value-free `DigestStreamError`.

The result contains the lowercase 64-character SHA-256 hex digest and the exact
number of bytes hashed. This helper does not open paths, validate media, scan
for malware, sign content, or provide content-addressed storage.
