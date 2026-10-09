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
- HEIF/HEIC and M4A with a complete bounded first `ftyp` box (including
  extended sizes); image sequence brands are distinguished

M4A maps to `audio/mp4`. HEIF maps to `image/heif` or `image/heic`, with
`-sequence` for sequence brands. Generic MP4, ambiguous image/audio brands,
AVIF and file-type boxes extending beyond the prefix are unknown. See
[HEIF/M4A header preflight](iso-bmff-preflight.md) for declared metadata and limits.

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
