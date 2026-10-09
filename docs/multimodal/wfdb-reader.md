# Bounded WFDB ECG reader

`openmed.multimodal.wfdb.read_wfdb_record` reads single-segment WFDB headers and
formats **16** (little-endian signed 16-bit), **212** (packed signed 12-bit), and
**80** (unsigned 8-bit minus 128). All examples and bundled fixtures are synthetic.
It uses the standard library, performs no network calls, opens no paths, and adds
no model assets or dependencies.

```python
from openmed.multimodal.wfdb import read_wfdb_record

header = b"synthetic 1 250 3\nsignal.dat 16 200(0)/mV 16 0 1 6 0 II\n"
signal = b"\x01\x00\x02\x00\x03\x00"
record = read_wfdb_record(header, [signal], start_sample=1, sample_count=2)
assert record.signals[0].samples == (2, 3)
assert record.signals[0].gain == 200
assert record.start_seconds == 0.004
assert record.duration_seconds == 0.008
assert record.signals[0].checksum_verified
print(record.notice)
# Before any consequential downstream use, obtain an explicit human review:
record.require_reviewer_confirmation(confirmed=True)
```

## OpenMedKit surface

OpenMedKit provides the matching `WFDBReader`, `WFDBRecord`, `WFDBLimits`,
`WFDBSignal` and `WFDBAnnotations` values with the same controlled reason codes,
metadata allow lists, default budgets, checksum behavior and confirmation gate.
Pass bounded header `Data` (or a header `WFDBByteSource`) and `WFDBDataSource`
signal/annotation sources, or `WFDBFileSource` wrapping caller-owned local `FileHandle` instances. File sources
restore their initial offsets on each read and remain open; they must not be used
concurrently. An injected `WFDBByteSource` can provide bounded byte-range access,
including short reads. Swift sources declare their byte count up front; the reader
checks that budget before decoding. Header bytes are bounded before parsing.

```swift
let header = Data("synthetic 1 250 3\nsignal.dat 16 200(0)/mV 16 0 1 6 0 II\n".utf8)
let signal = WFDBDataSource(Data([1, 0, 2, 0, 3, 0]))
let record = try WFDBReader.read(
    header: header, signals: [signal], startSample: 1, sampleCount: 2)
assert(record.signals[0].samples == [2, 3])
try record.requireReviewerConfirmation(confirmed: true)
```

Both surfaces implement only parsing. The pending canonical waveform contract
and downstream lead/quality/annotation-redaction work are independent consumers.
There is no model invocation or Apple Foundation Models cloud fallback.

## Input and output contract

Supply one bytes object or binary stream per distinct signal file, in the order
of first appearance in the header. Consecutive signals sharing a file are
interleaved in declaration order. File-name fields are used only to group signals;
they are never resolved, opened, returned, or interpolated in diagnostics.
Caller-owned streams remain open. Reads begin at their current positions;
seekable streams are restored on success and failure. Forward-only streams are
consumed. Short reads are supported.

Returned signals contain integer samples for the window, declared gain, baseline,
unit and lead label, format code, and checksum-verification status. Sampling
frequency, total sample count, window offset and relative timing are available
on the immutable record. No physical-unit conversion, lead normalization,
missing-value substitution, quality gate or diagnostic interpretation is applied.
The WFDB minimum integer sentinel is preserved as an integer for downstream
missingness handling.

Header comments, record names, file names/paths, wall-clock dates/times and
annotation auxiliary text are withheld. Presence flags indicate comments, record
names, file-name fields and timing fields. Free-text signal descriptions and unit
fields may also contain identifiers. The reader therefore preserves only these
exact declared strings, without normalization:

- Lead labels: `I`, `II`, `III`, `aVR`, `aVL`, `aVF`, `AVR`, `AVL`, `AVF`,
  `MLI`, `MLII`, `MLIII`, `MCL1`, `MCL6`, `ECG`, and `V1` through `V9`.
- Voltage units: `mV`, `uV`, `µV`, `μV`, and `V`.

Other descriptions/units become `None`, with `descriptions_withheld` or
`units_withheld` set. Missing descriptions remain `None`; missing gain defaults
to 200 and missing units to mV, as specified by WFDB. Missing baseline uses ADC
zero (default 0). Explicit zero or non-finite gain is rejected. A positive, finite
sampling frequency and a positive declared sample count are required. Counter
frequency syntax is accepted but not returned. Relative timing uses only the
sampling frequency.

An optional **MIT** annotation source produces total event count and absolute
sample positions inside the requested half-open window. The reader discards
event types, NUM/SUB/CHN metadata, and AUX payloads, handling their even-byte
padding. SKIP uses signed PDP-11 intervals. Annotation positions must be inside
the declared record. EOF must be explicit. No annotations are inferred when the
source is absent. AHA and reserved annotation codes are unsupported; no automatic
format detection or annotation redaction is performed.

`record.report()` contains only controlled counts, offsets, presence flags and
the mandatory non-diagnostic notice. It omits amplitudes, lead/unit strings and
annotation positions. The record itself still contains potentially sensitive
physiological data and is **not** a de-identification result. Consequential use
must call the explicit reviewer-confirmation gate; the library never triggers
clinical actions or qualifies clinical use.

## Bounds and integrity

`WfdbLimits` defaults to 64 KiB headers, 32 signals, 20 million frames per signal,
100,000 returned frames per signal, 256 MiB per signal file, 4 MiB annotations,
and 100,000 annotation events. Budgets must be positive integers. Source lengths
and declared sizes are checked before allocating output; forward-only inputs
are bounded as they are read, including trailing bytes.

The reader scans **all declared samples** in chunks of at most 8192 bytes,
retaining only the requested window. It verifies each declared checksum modulo
65536, including corruption outside that window, and fails on truncated signals.
Absent checksums remain explicitly unverified. This trades whole-record read time
for integrity; it does not promise I/O proportional to the window. Memory scales
with the window plus bounded header/annotation metadata and fixed decode scratch,
not record duration. A packed 212 odd sample count requires only the final two
bytes for the last sample; optional padding is ignored.

Only one sample per frame, zero skew, and zero block size are supported. Byte
offsets and multiple signal files are supported. Unsupported modifiers fail
instead of producing misleading timing. Multi-segment records fail before
reading signal payloads.

## Stable failures

`WfdbError.reason_code` and the exception message contain only a controlled code:

| Code | Meaning |
| --- | --- |
| `wfdb_signal_truncated` | Missing declared signal or preamble bytes |
| `wfdb_gain_invalid` | Zero or non-finite calibration |
| `wfdb_checksum_mismatch` | Full declared signal checksum differs |
| `wfdb_signal_limit_exceeded` / `wfdb_sample_limit_exceeded` | Declared counts exceed budgets or are non-positive |
| `wfdb_window_invalid` / `wfdb_window_limit_exceeded` | Invalid or excessive window |
| `wfdb_file_limit_exceeded` | Header, signal or annotation byte budget exceeded |
| `wfdb_multisegment_unsupported` | Multi-segment header |
| `wfdb_format_unsupported` / `wfdb_layout_unsupported` | Unsupported encoding or layout |
| `wfdb_header_invalid` / `wfdb_rate_invalid` / `wfdb_sample_count_required` | Invalid or ambiguous structural metadata |
| `wfdb_signal_group_invalid` / `wfdb_source_count_invalid` | Inconsistent file groups or supplied sources |
| `wfdb_annotation_truncated` / `wfdb_annotation_invalid` | Malformed MIT annotation data |
| `wfdb_annotation_format_unsupported` / `wfdb_annotation_position_invalid` / `wfdb_annotation_limit_exceeded` | Unsupported code, out-of-range position or too many events |
| `wfdb_stream_contract_error` / `wfdb_stream_read_error` / `wfdb_stream_restore_error` | Transport failure; original exception is withheld |
| `wfdb_limits_invalid` | Invalid budgets |
| `wfdb_reviewer_confirmation_required` | Explicit confirmation missing |

## Validation and format references

```bash
.venv/bin/python -m pytest tests/unit/multimodal/test_wfdb.py tests/integration/test_wfdb_reader.py -q
```

The tests include hand-checked encodings, extrema, interleaving, odd packed
samples, offsets, checksum corruption outside a window, short reads, restored
streams, annotation padding/SKIP, multilingual synthetic identifier leakage
controls, budget refusals and measured peak allocations for generated long
streams. No public ECG dataset, clinical benchmark or provider qualification is
claimed.

The binary layouts follow the primary WFDB specifications:
[headers](https://physionet.org/physiotools/wag/header-5.htm),
[signals](https://physionet.org/physiotools/wag/signal-5.htm), and
[MIT annotations](https://physionet.org/physiotools/wag/annot-5.htm).
