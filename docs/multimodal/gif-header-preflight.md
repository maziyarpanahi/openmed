# GIF header preflight

`read_gif_dimensions` returns declared logical-screen geometry and global
color-table size from GIF87a or GIF89a. It does not decode an image or inspect
animation frames and is not automatically wired into an image pipeline.

```python
from openmed.multimodal.gif_dimensions import read_gif_dimensions

with open("synthetic.gif", "rb") as stream:
    geometry = read_gif_dimensions(stream)
    assert stream.tell() == 0
print(geometry.width, geometry.height, geometry.version)
print(geometry.global_color_table_entries, geometry.global_color_table_bytes)
```

## Read boundary

The first 13 bytes contain the signature and logical-screen descriptor. If a
Global Color Table is declared, exactly its bounded 3-bytes-per-entry extent is
read and discarded, so a truncated table is rejected. No palette values enter
the returned immutable record. Without a global table, no table bytes are read.
The helper stops before image descriptors, local tables, extension blocks,
comments or compressed pixels. The returned canvas is not necessarily the size
of an individual frame. Later content is not checked for presence or validity.

Both `bytes` and caller-owned binary streams are supported. Streams begin at their
current offset. Short reads are handled, nonseekable reads stop at the boundary,
and seekable streams are restored on both success and failure. No stream is closed.

## Limits and failures

Defaults are `max_header_bytes=781` (13 plus the maximum 768-byte global table) and
`max_pixels=100_000_000`. Both must be positive integers; booleans are rejected.
Zero dimensions, an over-limit canvas, and an out-of-range background palette
index when a global table exists are rejected. Pixel area is checked by division,
not fixed-width multiplication. No allocation scales with canvas dimensions.

`GifDimensionsError` extends `ValueError`; `.category` and the exception string
are the same value-free category: `gif_signature_invalid`,
`gif_dimensions_invalid`, `gif_background_index_invalid`,
`gif_pixel_limit_exceeded`, `gif_header_limit_exceeded`, `gif_header_truncated`,
or `gif_stream_contract_error`, `gif_stream_read_error`,
`gif_stream_position_error`, `gif_stream_restore_error`.
Underlying I/O messages and filenames are not retained in these exceptions.
Invalid API argument types are separate constant-message `TypeError`/`ValueError`s.

An accepted header does not certify a complete GIF, remove metadata, establish
clinical fitness, or replace validation by the selected decoder.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_gif_dimensions.py -q
```

Synthetic tests cover both versions, all global-table sizes, absent tables,
truncations and limits. File-level tests use Pillow-generated images and a
complete local-table-only image as independent format checks. Pillow is used
only in tests and is already a development dependency.

Layout reference: W3C's [GIF89a specification](https://www.w3.org/Graphics/GIF/spec-gif89a.txt).
