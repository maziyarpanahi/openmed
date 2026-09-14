# Bounded media-type detection

`openmed.multimodal.media_type` provides a dependency-free preflight check for
uploaded or in-memory clinical media. It inspects at most the first 132 bytes
and does not use filenames, paths, or extensions.

Supported signatures:

- PDF
- PNG and JPEG
- little-endian and big-endian TIFF
- DICOM Part 10 with the `DICM` marker at byte offset 128
- WAV with `RIFF` and `WAVE` markers

```python
from openmed.multimodal.media_type import (
    MediaTypeStatus,
    detect_media_type,
    validate_media_type,
)

prefix = b"%PDF-1.7\n"

assert detect_media_type(prefix) == "application/pdf"
assert validate_media_type(prefix, "application/pdf") is MediaTypeStatus.MATCH
assert validate_media_type(prefix, "image/png") is MediaTypeStatus.MISMATCH
```

Truncated, ambiguous, and unsupported prefixes return `None` from
`detect_media_type()` and `MediaTypeStatus.UNKNOWN` from `validate_media_type()`.
Invalid declared types fail closed with a value-free error.

This preflight check does not fully validate a file, scan for malware,
decompress content, or replace format-specific parsers. Callers should pass
only the bounded prefix and should not log source bytes.
