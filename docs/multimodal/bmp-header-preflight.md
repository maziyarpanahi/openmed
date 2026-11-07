# BMP header preflight

`read_bmp_dimensions` reads declared BMP geometry without decoding pixels.
It is an explicit helper, not an automatically registered decoder or pipeline gate.

```python
from openmed.multimodal.bmp_dimensions import read_bmp_dimensions

with open("synthetic.bmp", "rb") as stream:
    geometry = read_bmp_dimensions(stream, max_pixels=20_000_000)
    assert stream.tell() == 0
print(geometry.width, geometry.height, geometry.bit_depth, geometry.top_down)
```

## Supported boundary

The helper accepts `bytes` or a caller-owned binary stream at its current position.
It supports the 12-byte Windows CORE DIB and the uncompressed 40-byte INFO DIB
(`BI_RGB`). CORE depths are 1, 4, 8 and 24; INFO also supports 16 and 32.
INFO negative height denotes top-down rows; the returned height is positive.
The plane count must be one. Dimensions must be nonzero and width positive.
Other DIB layouts and compression modes are explicitly unsupported, not guessed.

Only 26 bytes (CORE) or 54 bytes (INFO) are read. Palette counts and the declared
pixel offset/file extent are checked arithmetically, without reading the palette.
Rows use four-byte alignment. INFO image size may be zero or the computed size.
This does not prove that pixel payload bytes are present or decodable. Trailing
content, color management, metadata, compression, OCR and sanitization are outside
scope. An accepted header is not a clinical or security approval.

## Limits and stream ownership

The defaults are `max_header_bytes=54` and `max_pixels=100_000_000`. Both must be
positive integers (not booleans). Lower limits can be supplied per call. The pixel
limit is checked using division; declared file extents must fit their 32-bit fields.
No allocation proportional to image dimensions or declared offsets is performed.

Short reads are supported. A read never exceeds the remaining header budget.
Seekable streams are restored after success and failure, including a nonzero
starting offset. Nonseekable streams are consumed only through the required
header. The caller retains ownership; the helper never closes a stream.

## Failures

`BmpDimensionsError` is a `ValueError` with a stable `.category`; its string is the
same category. Input values, paths, resolutions and palette entries are not echoed.
Categories include `bmp_signature_invalid`, `bmp_reserved_fields_invalid`,
`bmp_dib_header_unsupported`, `bmp_compression_unsupported`, `bmp_planes_invalid`,
`bmp_bit_depth_unsupported`, `bmp_dimensions_invalid`, `bmp_color_table_size_invalid`,
`bmp_pixel_offset_invalid`, `bmp_file_size_invalid`, `bmp_image_size_invalid`,
`bmp_pixel_limit_exceeded`, `bmp_header_limit_exceeded` and `bmp_header_truncated`.
Stream failures use `bmp_stream_contract_error`, `bmp_stream_read_error`,
`bmp_stream_position_error` or `bmp_stream_restore_error` without retaining an
underlying I/O message. Unsupported formats should be handled explicitly by the
caller rather than retried with larger limits. Invalid API argument types are
reported separately as `TypeError`/`ValueError` with constant messages.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_bmp_dimensions.py -q
```

Fixtures are synthetic. Tests compare complete files with Pillow, include CORE
and top-down examples, exercise short/nonseekable streams and assert the precise
read boundary. Pillow is an existing development dependency, not a runtime import.

Layout references: Microsoft's [BITMAPCOREHEADER](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-bitmapcoreheader)
and [BITMAPINFOHEADER](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-bitmapinfoheader).
