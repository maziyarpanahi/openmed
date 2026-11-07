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

## Opt-in BigTIFF pyramid preflight

`openmed.multimodal.bigtiff_metadata.read_bigtiff_metadata` is a separate,
dependency-free metadata adapter. The classic helper above remains unchanged.
The caller must declare a supported profile and a verified complete source
byte length; the adapter does not open paths or inspect identifying metadata.

```python
from openmed.multimodal.bigtiff_metadata import (
    BigTiffLimits, BigTiffProfile, read_bigtiff_metadata,
)

# stream is caller-owned and positioned at the beginning of a local slide.
# source_size is its verified byte length from that position.
pyramid = read_bigtiff_metadata(
    stream,
    profile=BigTiffProfile.IFD_CHAIN,
    source_size=source_size,
    limits=BigTiffLimits(max_pixels=10_000_000_000),
)
geometry = [(level.width, level.height, level.tile_count) for level in pyramid.levels]
```

The two profiles are `bigtiff-ifd-chain-v1` (a next-IFD chain) and
`bigtiff-subifds-v1` (a root with direct, ordered SubIFDs). Both support either
byte order, BigTIFF magic 43, eight-byte offsets and zero reserved header bits.
Directories and out-of-line tables must be eight-byte aligned. The first level
has `NewSubfileType=0` (or the default); every later level must declare reduced
image type `1`. Levels must decrease in at least one dimension, never increase
in either dimension, and remain in declared order. A single base level is valid.
SubIFD children cannot have further children or next-IFD links.

Only dimensions, level order, tile dimensions, grid rows/columns, tile counts
and compression categories enter the immutable manifest. Compression categories
are `uncompressed` (1), `lzw` (5), `jpeg` (7) and `deflate` (8 or 32946).
These are header declarations, **not** decoder availability or qualification.
Geometry/table values accept unsigned SHORT, LONG or LONG8; SubIFDs additionally
accept IFD8. Tile tables must have exactly
`ceil(width / tile_width) * ceil(height / tile_height)` entries. Only chunky
planar configuration is supported, with 1–64 samples per pixel. Strip layouts,
volumes, private vendor tags, unknown compression, associated-image flags,
nested SubIFDs and mixed chain/SubIFD layouts fail explicitly. This is a narrow
declared layout contract, not an automatic vendor detector or general WSI reader.
Other standard tags, including descriptions, are skipped without following their
offsets. Text-based vendor hints are not interpreted or certified.

The reader reuses classic TIFF bounded loading, short reads, stream restoration
and value-free errors. `BigTiffLimits` defaults to 65,536 header bytes,
100,000,000 pixels per level, 512 entries per directory, 32 directories,
32 levels, 1,000,000 total tiles and a maximum exclusive range end of
`2**63 - 1`. All ceilings are configurable positive integers. Metadata must fit
entirely within the bounded prefix. Distant metadata is rejected rather than
seeked to; increasing the prefix bound is an explicit caller choice. Offset
addition and count multiplication are checked before allocation. Tile byte
counts must be positive, ranges must fit the declared source length and offset
ceiling, and metadata and tile ranges must be disjoint without aliases.
The full byte length must match when passing Python `bytes`; streams use the
caller-verified size and are not scanned for length.

Malformed inputs raise `TiffMetadataError` with a controlled `bigtiff_*`
category: `header_limit_or_truncated`, `byte_order_invalid`,
`header_unsupported`, `directory_offset_invalid`, `directory_cycle`,
`directory_limit_exceeded`, `level_limit_exceeded`, `entry_limit_exceeded`,
`offset_overflow`, `offset_limit_exceeded`, `range_out_of_source`,
`value_offset_invalid`, `value_type_unsupported`, `value_count_invalid`,
`value_count_limit_exceeded`, `tag_duplicate`, `geometry_invalid`,
`pixel_limit_exceeded`, `pyramid_geometry_inconsistent`, `tile_limit_exceeded`,
`tile_table_inconsistent`, `tile_range_invalid`, `range_overlap`,
`compression_unsupported` or `layout_unsupported`. API declaration errors use
constant `bigtiff_limit_invalid`, `bigtiff_limits_invalid`,
`bigtiff_profile_unsupported` or `bigtiff_source_size_invalid` messages.
Stream failures retain the classic `tiff_stream_*` codes. Paths and tag strings
never enter manifests, errors or logs; no raw temporary artifacts are written.

### OpenMedKit parity

`BigTiffPreflight.read(_:profile:limits:)` accepts caller-owned `Data`.
For a gigapixel slide, use
`BigTiffPreflight.read(prefix:sourceSize:profile:limits:)` with only a bounded
local prefix and the verified full byte length. No whole-file materialization
is needed. The caller owns any file handle and local loading. `BigTiffLimits`
has the same defaults and ceilings, and `BigTiffPyramid` encodes the same JSON
geometry fields as Python. `BigTiffPreflightError` exposes only the categorical
code. Shared synthetic fixtures cover both byte orders and both profiles.

Preflight success proves only consistency of supported metadata declarations.
It does not prove tile payloads are present, correctly encoded or safe to decode.
Pixel decoding, tissue detection, tile inference, transforms and clinical
interpretation remain outside this API. The future tiler (#2794) can consume
the validated level geometry; this change does not invent its planner contract.
No diagnostic or consequential output is produced, and there is no cloud
fallback or automatic clinical action. Any later consequential interpretation
requires its own non-diagnostic notice and explicit reviewer confirmation.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/test_bigtiff_metadata.py tests/integration/multimodal/test_bigtiff_preflight.py -q
cd swift/OpenMedKit && swift test
```

Format reference: [LibTIFF BigTIFF design](https://libtiff.gitlab.io/libtiff/specification/bigtiff.html).
