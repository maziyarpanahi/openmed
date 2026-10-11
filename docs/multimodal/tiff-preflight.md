# TIFF Header Preflight

`read_tiff_metadata` reads declared classic TIFF geometry and sample layout
without an imaging dependency. It is an explicit helper, not an automatically
registered decoder or a pipeline gate.

```python
from openmed.multimodal.tiff_metadata import read_tiff_metadata

with open("synthetic.tif", "rb") as stream:
    metadata = read_tiff_metadata(stream, max_pixels=20_000_000)
    assert stream.tell() == 0
print(metadata.width, metadata.height, metadata.bits_per_sample)
```

## Supported boundary

Both byte orders are accepted: `II` (little-endian) and `MM` (big-endian),
with magic 42. Only the first IFD is read. A nonzero next-IFD offset is
recorded by the file but never followed, so multipage enumeration, EXIF
sub-IFDs and BigTIFF are out of scope, as is decoding.

Seven tags are allowlisted: `ImageWidth` (256), `ImageLength` (257),
`BitsPerSample` (258), `Compression` (259), `PhotometricInterpretation` (262),
`SamplesPerPixel` (277) and `PlanarConfiguration` (284). Any other tag is
skipped without its type, count, or offset being interpreted, so a hostile
tag cannot steer a read. `ImageWidth` and `ImageLength` accept `SHORT` or
`LONG`; `BYTE`, `SHORT` and `LONG` are the only value types read at all.

`ImageWidth`, `ImageLength` and `PhotometricInterpretation` are required.
`SamplesPerPixel`, `Compression` and `PlanarConfiguration` default to `1`, and
`BitsPerSample` defaults to one bit per sample, matching the TIFF defaults.
`BitsPerSample` must carry exactly `SamplesPerPixel` entries.

## Offsets, limits and stream ownership

TIFF resolves values through file offsets, so the helper loads a bounded
prefix once — up to `max_header_bytes`, default 65 536 — and answers every
offset inside it. That is the one behavioral difference from the sequential
BMP/GIF/PNG readers: a nonseekable stream is consumed up to that bound or to
end of file rather than only through a fixed header. Seekable streams are
restored after success and failure, and the caller retains ownership; the
helper never closes a stream. Short reads are supported.

Every out-of-line value must lie at or after byte 8, must not point at the
IFD itself, and must fit inside the loaded prefix. An offset that has already
been followed is rejected as a cycle rather than read twice. `count * size` is
overflow-checked before any slice, the IFD entry count is bounded by
`max_ifd_entries` (default 512) before entries are read, and the pixel budget
is checked with division, so no allocation is proportional to declared
dimensions.

## Failures

`TiffMetadataError` is a `ValueError` with a stable `.category`; its string is
the same category. Offsets, tag values and paths are never echoed. Categories
are `tiff_byte_order_invalid`, `tiff_magic_invalid`, `tiff_header_truncated`,
`tiff_ifd_offset_invalid`, `tiff_ifd_truncated`, `tiff_ifd_empty`,
`tiff_ifd_entry_limit_exceeded`, `tiff_tag_duplicate`,
`tiff_value_type_unsupported`, `tiff_value_count_invalid`,
`tiff_value_length_overflow`, `tiff_value_offset_invalid`,
`tiff_value_offset_cycle`, `tiff_value_truncated`, `tiff_dimensions_missing`,
`tiff_dimensions_invalid`, `tiff_pixel_limit_exceeded`,
`tiff_photometric_missing`, `tiff_samples_per_pixel_invalid`,
`tiff_bits_per_sample_count_invalid` and `tiff_bits_per_sample_invalid`.
Stream failures use `tiff_stream_contract_error`, `tiff_stream_read_error`,
`tiff_stream_position_error` or `tiff_stream_restore_error` without retaining
an underlying I/O message. Invalid API argument types are reported separately
as `TypeError`/`ValueError` with constant messages.

An accepted header is not a clinical or security approval, and it does not
prove that strip or tile payloads are present or decodable.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_tiff_metadata.py -q
```

Fixtures are synthetic. Tests cover both byte orders, inline and out-of-line
values, grayscale and RGB layouts, TIFF defaults, every truncation prefix,
invalid magic, offsets, types, counts, overflow and cycles, short and
nonseekable streams, and compare Pillow-written uncompressed, LZW and Deflate
files. Pillow is an existing development dependency, not a runtime import.

Layout reference: Adobe [TIFF 6.0](https://web.archive.org/web/20210108172930/https://www.adobe.io/open/standards/TIFF.html).

## Slide-level de-identification

`openmed.multimodal.slide_deidentification.deidentify_tiff_slide` is a separate,
local, dependency-free operation. Unlike `read_tiff_metadata`, it traverses a
bounded directory chain in classic TIFF **and BigTIFF**, then rebuilds a compact
copy. It does not implement pyramid geometry preflight, tiling or provenance.

```python
from pathlib import Path
from openmed.multimodal.slide_deidentification import (
    SlideDeidLimits, deidentify_tiff_slide,
)

# Synthetic/caller-owned protected bytes only; enforce intake size before loading.
source = Path("synthetic-slide.svs")
budget = SlideDeidLimits(max_file_bytes=32 * 1024 * 1024)
if source.stat().st_size > budget.max_file_bytes:
    raise ValueError("slide_file_limit")
result = deidentify_tiff_slide(source.read_bytes(), limits=budget)
Path("synthetic-clean.svs").write_bytes(result.data)
print(result.to_audit_report())  # roles, counts, SHA-256 digests and outcome only
```

### Layouts and retained content

The supported conservative subset is:

- [Generic tiled TIFF pyramids](https://openslide.org/formats/generic-tiff/):
  first directory has `NewSubfileType=0`; later tiled levels must have subtype 1
  (the reduced-resolution bit) and decreasing dimensions. Untiled directories
  in a generic pyramid are ambiguous and refused.
- [Aperio conventions](https://openslide.org/formats/aperio/): a tiled first
  directory whose description starts with `Aperio`; a stripped second directory
  with subtype 0 is a thumbnail; stripped subtype 1 and 9 directories at the end
  are respectively label and macro. There can be at most one of each associated
  role. Later tiled levels retain the baseline compression and tile dimensions.
- Both byte orders, TIFF magic 42 and
  [BigTIFF magic 43 with 8-byte offsets](https://libtiff.gitlab.io/libtiff/specification/bigtiff.html).
  Only the linked IFD chain is supported. SubIFD, EXIF and GPS directory links
  are refused. Other vendor layouts are not inferred from filenames or text.

All associated directories, **including thumbnails**, are removed. Only level
payloads are copied, byte for byte, without decoding or re-encoding; the report
contains each original block's SHA-256 digest. The output contains no removed
payload, original description, private/text tag or orphan source bytes.

Descriptions are replaced with a controlled prefix and only `MPP` and `AppMag`
key/value pairs having a positive decimal value at most 100,000 (at most six
integer and six fractional digits). Duplicate retained keys refuse the copy.
Names, paths, dates, free text and all other description fields are dropped.
Numeric baseline imaging tags (including TIFF resolution) are retained.
The supported sample layouts are grayscale, RGB and YCbCr with 1–64-bit samples;
palette/color-map and other unsupported layouts refuse rather than losing
required decoding metadata. ICC,
XMP and vendor metadata are omitted; color-profile metadata is not preserved.
JPEG tables may contain only quantization/Huffman table segments and delimiters;
comment and application metadata segments are refused. Supported declared
compression codes are 1, 5, 7, 8, 32946, 33003 and 33005. Decoder qualification
and compressed tile contents are outside this structural operation.

A slide containing only levels returns `no_associated_images`; its description
and other metadata are still scrubbed. This is not a byte-for-byte file pass-through.
Successful removal returns `deidentified`. Calling the helper again on its output
retains the same bytes and level digests.

### Resource and refusal contract

Defaults are 256 MiB input/output bytes, 64 directories, 128 entries per directory,
1 MiB per interpreted value and 1,000,000 total image blocks. Bounds are positive
integers. Applications enforce their intake budget before loading bytes and may
choose tighter device-specific limits. Files larger than the chosen budget are
refused; this API does not claim streaming support for arbitrarily large slides.

No allocation is proportional to declared dimensions. Interpreted ranges must fit
in the input and may not alias directories, metadata or image payloads; cumulative
claimed bytes are bounded to prevent alias-driven allocation amplification.
Unknown layouts, extra associated images, repeated directories, duplicate tags,
unsupported types/compression, excessive chains/values/blocks and incomplete
ranges fail closed. A refusal raises `TiffMetadataError` with a controlled
`slide_*` category; no source value, private path or partial copy is exposed.
Caller-owned input bytes are never modified, logged or written to temporary files.

### DICOM whole-slide instances

The existing Python `deidentify_dicom_headers`, `redact_dicom_pixels` and
`redact_document` routes recognize the VL Whole Slide Microscopy SOP Class.
[Image Type value 3](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.12.4.html)
`LABEL` and `OVERVIEW` refuse by default with
`dicom_slide_associated_image_refused`, even when `BurnedInAnnotation=NO`.
Missing/unknown WSI roles and a `VOLUME`/`THUMBNAIL` declaring a specimen label
refuse with `dicom_slide_image_type_unknown`. Non-WSI behavior is unchanged.

A caller may explicitly set `keep_slide_associated_images=True` and
`reviewer_confirmed=True` with a caller-supplied **offline** OCR engine/PII
provider. Pixel redaction must run with residual verification and fail-on-residual
both enabled. Header-only retention additionally requires the actual
`DicomPixelRedactionResult` via `slide_pixel_redaction=...`: processed frame counts
must match the verified residual report and its pixel digest must match the
current bytes. Changed pixels or unverified results cannot authorize retention.
The dispatcher carries this result through automatically. No clinical decision
or cloud fallback is added.

### OpenMedKit and review limits

`SlideDeidentification.deidentifyTIFF(Data, limits:)` implements the same local
traversal, rewrite, bounds, outcome and audit fields. Shared synthetic vectors
exercise classic/BigTIFF and both byte orders. Native applications without a
DICOM decoder use `validateDICOMInstance` before writing their caller-owned
instance; retained associated images require actual local redaction evidence,
matching pixel digest and explicit reviewer confirmation. This guard is not a
DICOM parser or an invented provider qualification.

**Non-diagnostic:** structural removal does not detect identifiers burned into
pyramid tiles or prove those pixels are safe. Output image bytes remain protected.
Before consequential release, applications must perform pixel privacy checking
and obtain explicit human review; never auto-trigger clinical decisions. Reports
are deterministic roles/counts/digests only and contain no source description,
file path, OCR text or patient identifiers. No models, datasets, network clients
or new dependencies are bundled.

### Synthetic validation

```bash
.venv/bin/python -m pytest tests/unit/multimodal/test_slide_deidentification.py tests/unit/multimodal/test_dicom_slides.py tests/integration/multimodal/test_slide_deidentification.py -q
cd swift/OpenMedKit && swift test
```

Fixtures are hand-built synthetic files with identifier sentinels, not clinical
samples. Checks cover stripped associated payloads, description/private/orphan
metadata, immutable source files, exact tile digests, known pixel values through
Pillow's supported variants, shared native rewrite digests, every truncated
prefix, cycles, aliases, hidden IFDs, bounds and unexpected associated roles.
Pillow 12 does not recognize big-endian BigTIFF; native parity vectors cover that
variant and libtiff can independently inspect/decode the synthetic output.
