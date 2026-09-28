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
