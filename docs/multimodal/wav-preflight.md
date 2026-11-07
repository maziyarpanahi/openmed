# WAV metadata preflight

`openmed.multimodal.wav_metadata` reads privacy-safe metadata from PCM and
IEEE-float RIFF/WAVE headers without importing an audio decoder.

```python
from openmed.multimodal.wav_metadata import read_wav_metadata

with open("synthetic-audio.wav", "rb") as stream:
    metadata = read_wav_metadata(stream)

print(metadata.channels)
print(metadata.sample_rate_hz)
print(metadata.duration_seconds)
```

The result contains only the numeric format code, channel count, sample rate,
bit depth, data-byte count, frame count, and duration. The parser stops at the
`data` chunk header, so it neither reads nor returns sample bytes. It accepts
uncompressed PCM (format code `1`) and IEEE float (format code `3`); compressed
codecs and big-endian `RIFX` files fail closed.

Parsing is limited to 64 KiB of header data by default. Set
`max_header_bytes` to a smaller positive integer when a tighter bound is
required. RIFF and chunk sizes, odd-byte padding, format consistency, and frame
alignment are validated before metadata is returned.

Binary streams are read from their current position. Seekable streams are
restored on success or failure, non-seekable streams remain consumed through
the `data` header, and caller-owned streams are never closed. Errors contain
only stable categories and do not include paths, source bytes, or stream error
details.

This preflight does not decode samples, validate that the declared sample bytes
are present, assess audio quality, resample content, or replace a full WAV
decoder.
