# Bounded media-type detection

`openmed.multimodal.media_type` provides a dependency-free preflight check for
uploaded or in-memory clinical media. It inspects at most the first 132 bytes
and does not use filenames, paths, or extensions.

Supported signatures:

- PDF
- PNG and JPEG
- little-endian and big-endian TIFF
- GIF (GIF87a and GIF89a)
- WebP with `RIFF` and `WEBP` markers
- BMP with a complete 14-byte file header and zero reserved fields
- DICOM Part 10 with the `DICM` marker at byte offset 128
- WAV with `RIFF` and `WAVE` markers
- HEIF/HEIC and M4A with a complete bounded first `ftyp` box (including
  extended sizes), with distinct types for image sequences
- Generic MP4 video with a recognized major brand

M4A maps to `audio/mp4`. HEIF maps to `image/heif` or `image/heic`, with
`-sequence` for sequence brands. Ambiguous image/audio brands and AVIF are
unknown. The generic MP4 video signature can be recognized from a partial box.
See
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

Prefixes without a recognized signature return `None` from
`detect_media_type()` and `MediaTypeStatus.UNKNOWN` from `validate_media_type()`.
Invalid declared types fail closed with a value-free error.

The DICOM Part 10 marker takes precedence over signatures in its arbitrary
128-byte preamble, including GIF and BMP look-alikes.

This preflight check does not fully validate a file, scan for malware,
decompress content, or replace format-specific parsers. Callers should pass
only the bounded prefix and should not log source bytes.
