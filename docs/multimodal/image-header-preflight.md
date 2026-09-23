# PNG and JPEG Header Preflight

`read_image_header` reads declared image geometry before Pillow or a vision
backend is imported. It is an explicit helper, not an automatically registered
decoder or a pipeline gate.

```python
from openmed.multimodal.image_header import read_image_header

with open("synthetic.png", "rb") as stream:
    header = read_image_header(stream, max_pixels=20_000_000)
    assert stream.tell() == 0
print(header.image_format, header.width, header.height, header.bit_depth)
```

`read_png_header` and `read_jpeg_header` do the same work for one format and
reject the other, which is useful when a media type is already known.

## Supported boundary

The helper accepts `bytes` or a caller-owned binary stream at its current
position. `read_image_header` selects the grammar from the first two bytes.

For PNG, the eight-byte signature, the IHDR chunk length, the IHDR CRC-32, and
the declared compression, filter and interlace methods are checked. Color
types 0, 2, 3, 4 and 6 are accepted with the bit depths PNG allows for each,
and `component_count` is derived from the color type. Adam7 interlace is
reported as `progressive`. No chunk after IHDR is read, so a PLTE, IDAT or
text chunk is never parsed.

For JPEG, the marker scan starts at SOI and skips segments without retaining
them until it reaches the first frame header. SOF0, SOF1, SOF3, SOF5, SOF7,
SOF9, SOF11 and SOF15 are sequential; SOF2, SOF6, SOF10 and SOF14 report
`progressive=True`. Sample precision, height, width and the frame component
count are read, and the segment length must match `8 + 3 * components`.
Reaching SOS or EOI without a frame header fails rather than guessing. The
scan stops at the frame header and never touches entropy-coded data.

BigTIFF, TIFF, EXIF orientation, ICC profiles, animation, metadata scrubbing
and clinical image-quality assessment are outside scope. An accepted header is
not a clinical or security approval, and it does not prove that pixel data is
present or decodable.

## Limits and stream ownership

The defaults are `max_header_bytes=65_536`, `max_pixels=100_000_000` and
`max_jpeg_markers=256`. All three must be positive integers, not booleans, and
lower limits can be supplied per call. Every read, including a skipped JPEG
segment, is charged to the header budget, so a padded APP segment cannot make
the reader run past the limit. The pixel limit is checked with division, and
no allocation proportional to declared dimensions is performed.

Short reads are supported and a read never exceeds the remaining budget.
Seekable streams are restored after success and failure, including from a
nonzero starting offset; nonseekable streams are consumed only through the
required header. The caller retains ownership and the helper never closes a
stream.

## Failures

`ImageHeaderError` is a `ValueError` with a stable `.category`; its string is
the same category. Dimensions, paths and byte values are never echoed.
Shared categories are `image_signature_unsupported`, `image_pixel_limit_exceeded`,
`image_header_limit_exceeded` and `image_header_truncated`. PNG adds
`png_signature_invalid`, `png_ihdr_missing`, `png_ihdr_length_invalid`,
`png_ihdr_crc_mismatch`, `png_dimensions_invalid`, `png_color_type_unsupported`,
`png_bit_depth_unsupported`, `png_method_unsupported` and
`png_interlace_unsupported`. JPEG adds `jpeg_signature_invalid`,
`jpeg_marker_invalid`, `jpeg_marker_unexpected`, `jpeg_marker_limit_exceeded`,
`jpeg_marker_fill_limit_exceeded`, `jpeg_segment_length_invalid`,
`jpeg_frame_header_missing`, `jpeg_frame_length_invalid`,
`jpeg_component_count_invalid`, `jpeg_precision_unsupported` and
`jpeg_dimensions_invalid`.

Stream failures use `image_stream_contract_error`, `image_stream_read_error`,
`image_stream_position_error` or `image_stream_restore_error` without retaining
an underlying I/O message. Unsupported formats should be handled explicitly by
the caller rather than retried with larger limits. Invalid API argument types
are reported separately as `TypeError`/`ValueError` with constant messages.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_image_header.py -q
```

Fixtures are synthetic. Tests pin every truncation boundary, zero-sized and
overflowing dimensions, invalid lengths and CRCs, marker loops and fill runs,
the exact read boundary, short and nonseekable streams, and compare complete
Pillow-written PNG and baseline/progressive JPEG files. Pillow is an existing
development dependency, not a runtime import.

Layout references: the W3C [PNG specification](https://www.w3.org/TR/png/) and
ITU-T [T.81](https://www.w3.org/Graphics/JPEG/itu-t81.pdf) for JPEG markers.
