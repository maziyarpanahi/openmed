# WebP header preflight

`read_webp_dimensions` reads declared geometry from a RIFF/WEBP envelope whose
first chunk is VP8, VP8L or VP8X. The helper uses the standard library only and
is not automatically registered as a decoder or pipeline validator.

```python
from openmed.multimodal.webp_dimensions import read_webp_dimensions

with open("synthetic.webp", "rb") as stream:
    geometry = read_webp_dimensions(stream, max_pixels=20_000_000)
    assert stream.tell() == 0
print(geometry.width, geometry.height, geometry.chunk_type)
print(geometry.has_alpha, geometry.is_animated)
```

## Declared metadata, not decoded content

VP8 reads the key-frame header's 14-bit width/height; scaling bits are masked.
VP8L reads the lossless signature, 14-bit dimensions plus one, the alpha hint
and the zero version field. VP8X reads 24-bit canvas dimensions plus one and
alpha/animation flags while rejecting nonzero reserved bits.
`has_alpha` reports a header declaration/hint, not inspection of actual pixels.
For animated VP8X, geometry is the declared canvas, not an inspection of frames.

The RIFF size must be even and at most `2**32 - 10`. The declared first chunk,
including its odd-byte padding, must fit inside the RIFF envelope. VP8X's chunk
size must be exactly ten. Canvas area cannot exceed `2**32 - 1` even when the
caller supplies a larger pixel budget. Unknown first-chunk layouts are refused;
the parser never scans through arbitrary metadata looking for another chunk.

Only 25 bytes for VP8L or 30 for VP8/VP8X are read. Pixel payloads, later chunks,
actual padding contents, trailing bytes and full-file length are not inspected.
The caller must not treat a valid header as proof that payloads exist, animation
is well formed, metadata is safe, or the file will decode. No pixel decoding,
animation traversal, EXIF/XMP extraction, OCR or clinical interpretation occurs.

## Ownership, limits and errors

Sources are `bytes` or binary streams at their current offset. Short reads work;
seekable streams are restored on success/failure; caller streams are never closed.
Defaults are `max_header_bytes=30` and `max_pixels=100_000_000`; overrides must be
positive integers excluding booleans. No allocation scales with a declared chunk
size or canvas size.

`WebpDimensionsError` extends `ValueError`, with identical value-free string and
`.category`. Categories are `webp_signature_invalid`, `webp_riff_size_invalid`,
`webp_chunk_size_invalid`, `webp_layout_unsupported`, `webp_vp8_header_invalid`,
`webp_vp8_version_unsupported`, `webp_vp8l_header_invalid`,
`webp_vp8l_version_unsupported`, `webp_vp8x_reserved_bits_invalid`,
`webp_dimensions_invalid`, `webp_pixel_limit_exceeded`,
`webp_header_limit_exceeded`, `webp_header_truncated`, and
`webp_stream_contract_error`, `webp_stream_read_error`,
`webp_stream_position_error`, `webp_stream_restore_error`.
Invalid API argument types raise separate constant-message errors.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_webp_dimensions.py -q
```

Pillow-generated lossy, lossless, alpha and animated files provide real-file
oracles; handcrafted headers exercise malformed envelopes and exact read
boundaries. Tests require Pillow with WebP support, already present in the
project's development dependency. Runtime parsing never imports Pillow.

References: [WebP RIFF container](https://developers.google.com/speed/webp/docs/riff_container),
[lossless bitstream](https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification),
and [VP8 frame header](https://www.rfc-editor.org/rfc/rfc6386.html#section-9.1).
