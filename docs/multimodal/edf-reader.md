# Bounded EDF and EDF+ reader

`openmed.multimodal.edf.read_edf` and OpenMedKit's `EDFReader.read` decode
caller-supplied EDF, continuous EDF+C and discontinuous EDF+D recordings locally.
There are no provider calls, model assets, network access, file writes or clinical
interpretations. This reader follows the [EDF specification](https://www.edfplus.info/specs/edf.html)
and [EDF+ specification](https://www.edfplus.info/specs/edfplus.html).

## Windowed signals and gaps

```python
from openmed.multimodal.edf import EdfLimits, read_edf

# recording_bytes is supplied by the caller; no path is recorded in the result.
result = read_edf(recording_bytes, start_seconds=5, end_seconds=6,
                  limits=EdfLimits(max_output_samples=100_000))
for record in result.records:
    for window in record.signals:
        integer_samples = window.digital_samples
        scaled_samples = window.physical_samples
report = result.report()
# Only an explicit human decision may authorize consequential handoff.
reviewed = result.reviewed(confirmed=True)
```

```swift
import OpenMedKit

let result = try EDFReader.read(recordingData, startSeconds: 5, endSeconds: 6)
let report = result.report()
let reviewed = try result.reviewed(confirmed: true)
```

Offsets are seconds from the withheld header start second, including the first
EDF+ record's fractional-second offset. Windows are half-open `[start, end)`;
a sample exactly at the end is excluded. Each record retains its onset, duration
and each signal's first sample index. Sample offset within a record is
`first_sample_index * record_duration / samples_per_record`. Signals may have
different sample rates. No resampling, lead normalization or clock alignment
is performed. ADC conversion is:

```text
physical_min + (integer_sample - digital_min)
             * (physical_max - physical_min) / (digital_max - digital_min)
```

Negative physical gain is supported. Integer samples must fall in their declared
digital range when decoded. Samples outside the requested window are not decoded
or range-checked. Header validity, record completeness, all EDF+ annotation
syntax and all record timings are checked across the complete bounded recording.

`record_onsets_seconds` / `recordOnsetsSeconds` describes every data record.
`gaps` describes every missing acquisition interval, even outside the window.
For synthetic one-second records at offsets `0` and `5`, the explicit gap is
`[1, 5)`. A `[5, 6)` window returns the second record only. Synthetic integer
samples `(2, 0, -1, -2)` with digital range `[-2, 2]` and physical range
`[-1, 1]` become `(1, 0, -0.5, -1)`. These are engineering controls, not clinical
validation.

EDF+D supports zero-duration point records with one sample per ordinary signal;
their sampling rate is absent. Annotation-only EDF+ records are also supported.
For zero-duration files the default window ends one second after the final point.
Otherwise, omitting the end requests the recording's end, subject to the window
budget. An explicit window may extend beyond the recording and return no samples
when it covers only a gap or lies beyond acquisition.

## Withheld content and review

Patient and recording identification, start date/time, transducer, prefilter and
reserved fields are never returned. Identification exposes only presence and
`absent`, `placeholder_only` or `not_verified` status. Only exact EDF+ placeholders
`X X X X` and `Startdate X X X X` receive `placeholder_only`; extra subfields do
not. Placeholders can mean unknown information and **do not prove anonymization**.

Signal indices, physical/digital ranges, samples per record and sampling rates
are available. Exact controlled labels (`ECG`, `EEG`, `EMG`, `EOG`, `Temp rectal`,
`Body temp`, `SaO2`, `SpO2`, `ECG I/II/III/aVR/aVL/aVF/V1–V6`, `EEG Fpz-Cz`,
`EEG Pz-Oz`) are preserved without normalization. Arbitrary labels become
`withheld`. Physical dimensions are limited to `V`, `mV`, `uV`, `nV`, `degreeC`,
`%`, `Ohm` and `mmHg`; other dimensions become `withheld`. This prevents accidental
free-text header disclosures without inferring patient or lead identity.

Annotation-channel sample bytes are never exposed. Each nonempty annotation list
(TAL) contributes only its record/signal index, onset, optional duration and count
of nonempty annotations. Empty timekeeping annotations are excluded from counts.
Lists are selected by onset within the window, or positive duration overlapping
it, even when their containing record is outside the window. Negative event
onsets are allowed. Annotation text must be valid UTF-8 and is discarded after
syntax validation. Timekeeping in the first annotation channel is mandatory for
every EDF+ record; missing timing, overlaps, backward timing and discontinuities
in EDF+C fail closed.

Every result and report carries a non-diagnostic notice. `reviewed(confirmed=True)`
(or Swift's `reviewed(confirmed: true)`) binds explicit reviewer confirmation;
false confirmation fails with `edf_review_required`. Downstream consequential
consumers must enforce that state. This reader makes no clinical decision.
Waveform samples remain sensitive, even though these headers are withheld.
Reports contain controlled counts/statuses and the notice, omitting samples,
labels, absolute dates and annotation text. Error messages contain only reason
codes; underlying stream messages and Python exception chains are withheld.

## Bounds and streams

Default budgets are 64 MiB input, 64 signals, 100,000 records, 1 MiB per record,
86,400 seconds maximum acquisition offset/end, 3,600 seconds per window,
1,000,000 retained samples and 100,000 annotation lists (including timekeepers).
Callers can supply positive budgets; unknown record count `-1` is determined
from complete bounded records at EOF. Timing fields are bounded to 28 ASCII
characters; header exponents are bounded to ±12. EDF+ onset/duration fields use
decimal syntax without exponents or spaces. Rational arithmetic in Python and
bounded decimal arithmetic in Swift validate continuity before returning
floating-point offsets. Long precision fields fail with `edf_numeric_invalid`.

Python accepts bytes and caller-owned binary streams. Seekable streams are
restored on success/failure; nonseekable streams are consumed. Swift accepts
`Data` and caller-opened `InputStream`, consuming the stream without closing it.
Reads never exceed 64 KiB; an extra one-byte probe validates EOF at the exact
input budget. At most one record and bounded annotation lists are parsed at a
time, plus the selected output and content-free timing/gap index.

Stable failures include `edf_header_invalid`, `edf_header_size_invalid`,
`edf_numeric_invalid`, `edf_range_invalid`, `edf_samples_invalid`,
`edf_sample_range_invalid`, `edf_record_count_invalid`, `edf_truncated`,
`edf_annotation_header_invalid`, `edf_annotation_channel_missing`,
`edf_annotation_invalid`, `edf_timekeeping_invalid`, `edf_record_timing_invalid`,
`edf_duration_invalid`, `edf_window_invalid`, `edf_limits_invalid`,
`edf_stream_read_error` and Python's `edf_stream_restore_error`. Budget failures
use `edf_byte_limit`, `edf_signal_limit`, `edf_record_limit`,
`edf_record_byte_limit`, `edf_duration_limit`, `edf_output_sample_limit` and
`edf_annotation_limit`.

The reader owns only parsing. Header/annotation redaction artifacts, ECG contract
integration, lead normalization and provenance clock alignment remain separate
capabilities. No release, model qualification or clinical-validation claim is
made by successful decoding.
