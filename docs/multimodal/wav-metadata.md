# WAV metadata preflight and RF64 refusal

The existing `read_wav_metadata` helper reads bounded RIFF/WAVE metadata for
PCM or IEEE-float data. This contribution adds regression fixtures; it does
**not** add RF64 support or change the runtime parser.

```python
from openmed.multimodal.wav_metadata import WavMetadataError, read_wav_metadata

synthetic_rf64 = b"RF64\xff\xff\xff\xffWAVE"
try:
    read_wav_metadata(synthetic_rf64)
except WavMetadataError as error:
    assert error.category == "wav_signature_invalid"
```

## Unsupported RF64 boundary

An RF64 envelope is rejected after reading its 12-byte header, before any `ds64`
chunk header or content is consumed. The current category is
`wav_signature_invalid`. Here that means RF64 is outside the accepted RIFF/WAVE
signature set; it does not assert that every RF64 file is malformed. Keep this
observable category stable unless a deliberate API change is made later.

`tests/fixtures/multimodal/rf64.py` defines five deterministic, tiny cases:
missing `ds64`, truncated `ds64`, valid-looking size metadata, duplicate `ds64`,
and an oversized declared `ds64` chunk. Each is at most 128 bytes, even where a
length field describes gigabytes. They contain structural numeric headers only,
no audio samples, descriptive metadata, patient information or real source names.
These are refusal fixtures, not an RF64 conformance suite or decoding examples.

A source shorter than the required envelope produces `wav_header_truncated`.
A header budget too small for the envelope produces `wav_header_limit_exceeded`
before signature inspection. Tests pin those existing precedence rules as well.
Short-read streams are supported; successful restoration preserves nonzero
initial positions on seekable streams. Caller-owned streams are never closed.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_wav_metadata.py -q
```

The added cases exercise byte input, real temporary files, short-read/nonseekable
streams, read boundaries and seek restoration. An empty RIFF/WAVE file written
by Python's standard-library `wave` writer remains a successful control. Existing
RIFF/WAVE tests are retained, unmodified except for added imports and new tests.
No external decoders, model weights, credentials or network calls are needed.
