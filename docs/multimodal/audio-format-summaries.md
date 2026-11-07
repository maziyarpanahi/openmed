# Audio format summaries

`summarize_audio_formats` aggregates validated per-asset audio descriptors
into deployment-planning counts without retaining any content: channel,
sample-rate, bit-depth, duration-bucket, and container-format categories
with small-cell suppression and deterministic sorting. It never carries raw
samples, filenames, paths, identifiers, transcripts, or free-form metadata.

```python
from openmed.multimodal.audio_format_summary import (
    AudioFormat,
    AudioFormatRecord,
    summarize_audio_formats,
)

summary = summarize_audio_formats(
    [
        AudioFormatRecord(
            format=AudioFormat.WAV,
            channels=1,
            sample_rate_hz=16_000,
            bit_depth=16,
            duration_seconds=5.0,
        ),
    ],
    min_cell_count=1,
)
print(summary.to_json())
```

## Supported boundary

Every input is an `AudioFormatRecord` with exactly five closed or bounded
fields: a container format from the closed set `wav`, `flac`, `mp3`, `m4a`,
and `ogg`; channels (1 to 64); sample rate in hertz (1 to 2^32 - 1); bit
depth (1 to 64); and a finite non-negative duration in seconds. Any other
field would be free-form metadata, so the record type has none. All records
are validated before aggregation: one invalid record raises and no partial
summary is returned.

Durations are bucketed by inclusive lower bounds into `under_1s`,
`1s_to_10s`, `10s_to_1m`, `1m_to_10m`, and `10m_or_more`.

## Small-cell suppression and determinism

`min_cell_count` (default 3, inclusive: equal passes, below suppresses)
hides any category with fewer assets in a single trailing `suppressed`
entry per breakdown, whose count is the sum of the suppressed cells. Each
breakdown is sorted by category, so `to_dict()`/`to_json()` are
byte-identical for reordered inputs. `total_assets` is reported alongside
the breakdowns; the suppressed pool's count is the sum of suppressed cells
by design.

## Failures

`AudioFormatSummaryError` is a `ValueError` with a stable `.category`; its
string is the same category, and no input value is ever echoed. Categories
are `audio_format_record_type` for non-record entries and
`audio_format_format_invalid`, `audio_format_channels_invalid`,
`audio_format_sample_rate_invalid`, `audio_format_bit_depth_invalid`, and
`audio_format_duration_invalid` for field-level rejections. Invalid
`min_cell_count` API arguments are reported separately as plain
`ValueError` with a constant message.

Audio decoding, ASR, quality scoring, speaker analysis, and transcription
are out of scope; a summary is a planning artifact, not a content analysis.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_audio_format_summary.py -q
```

Tests use synthetic records only. They cover empty, single-record, mixed,
suppressed, boundary, and reordered inputs against golden JSON, inclusive
thresholds, every rejection category, and deterministic serialization,
offline.
