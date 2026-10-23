# DICOM ECG waveform reader

`openmed.multimodal.dicom_waveform.read_dicom_ecg_waveform` decodes a caller-supplied
pydicom `Dataset` locally. Install the optional `openmed[multimodal]` extra for
pydicom. It reads no files, makes no network calls, stores no source dataset and
writes no temporary copies. File parsing and transport belong to the caller.
This Python slice owns DICOM decoding; lead normalization, header de-identification
and annotation redaction remain separate capabilities. It does not require the
pending ECG normalization contract or add a Swift DICOM parser.

Every record carries a **non-diagnostic notice** and `review_required=True`.
Call `record.confirm_for_consequential_use(reviewer_confirmed=True)` only after
qualified human review. Other values fail with `review_required`. Confirmation
does not establish clinical validity; never auto-trigger clinical decisions.
Decoded amplitudes are protected signal data, not a claim of anonymization.

## Public behavior

```python
from openmed.multimodal.dicom_waveform import read_dicom_ecg_waveform

# dataset is a caller-supplied pydicom Dataset, with file_meta.TransferSyntaxUID.
record = read_dicom_ecg_waveform(dataset, start_sample=0, sample_count=2500)
summary = record.to_report()  # counts and controlled codes only
channel = record.groups[0].channels[0]  # protected physical samples, original unit
```

The immutable reader-local record keeps multiplex groups separate, retaining
zero-based group/channel indices, source codes validated against pydicom's CID 3001
vocabulary, UCUM voltage units (`V`, `mV`, `uV`), sampling frequency and calibration.
It never copies source code meanings or labels and does not rename or align leads.
Unknown codes and source modifiers fail closed rather than silently dropping their
semantics. This is a supported decoding subset, not full DICOM IOD conformance
validation or device qualification.

Physical amplitude is `raw * sensitivity * correction_factor + baseline`; the
baseline is already in the declared physical unit. Units are preserved, not
converted to millivolts. For the synthetic single-lead fixture, raw samples
`(-2, 0, 2, 4)`, sensitivity `1 uV`, correction `0.5`, baseline `10 uV` decode
to **`(9, 10, 11, 12) uV`**.

Timing is relative only. The first returned sample offset is multiplex group
offset (milliseconds converted to seconds), plus channel time skew (or fractional
sample skew divided by frequency), channel offset, and `start_sample / frequency`.
If both skew forms exist they must agree. Absolute dates/times and synchronization
identifiers are never retained. A waveform padding value maps to `None`, not an
amplitude. Returned integer samples must respect signed bits stored, including
sign extension. No resampling or clinical interpretation occurs.

## Supported input and bounds

Supported ECG SOP classes are twelve-lead, general, ambulatory and general 32-bit
ECG Storage (`1.2.840.10008.5.1.4.1.1.9.1.1` through `.4`). Signed 16-bit `SS`
is supported for all four; signed 32-bit `SL` is also supported for general 32-bit.
Other interpretations fail closed. Transfer syntaxes are implicit VR little
endian, explicit VR little endian and explicit VR big endian. Encapsulated,
compressed, deflated and unknown transfer syntaxes are rejected before decoding.

Each group must have one definition per channel, complete finite calibration,
positive sampling frequency, supported source/unit codes and at least one skew
form. Data length must exactly match channel count, sample count and allocated
width. No extra/truncated bytes are accepted. Window requests apply independently
to every group; short final windows are clipped, but a start beyond any group's
end rejects the entire read.

Bounds are 16 groups, 64 channels per group, 128 MiB total encoded waveform bytes,
1,000,000 total returned channel samples, 10,000 annotations, and 1–10,000 samples
per requested window (default 2,500). Only the selected windows are decoded.
The caller must independently bound memory/file parsing before constructing the
Dataset; these bounds apply to waveform reading after parsing.

## Annotation and privacy policy

All annotation payloads are withheld, including numeric values, coded findings,
code meanings, text and references. Only three category counts are exposed:

- `measurement_count`: numeric annotation items without categorical values/text.
- `device_interpretation_count`: coded or otherwise nonnumeric observations,
  conservatively counted as possible device statements, without asserting origin.
- `free_text_count`: items containing unformatted text or a text value.

`device_interpretation_present` is conservative: it is true for either nonnumeric
observations or free text, since uncoded text may also contain device interpretation.
It is not an authorship classifier. Mixed items with text count as free text;
mixed numeric/categorical items count as possible statements. No source statement
is emitted as an OpenMed finding. The synthetic fixture produces:

```json
{
  "measurement_count": 1,
  "device_interpretation_count": 1,
  "free_text_count": 1,
  "device_interpretation_present": true,
  "values_withheld": true
}
```

Patient attributes, private tags, UIDs, dates, labels, annotation text, measurements
and device statements do not enter records or reports. Reports omit amplitudes,
calibration and time/frequency values; record representations also omit sample
arrays. The reader suppresses pydicom diagnostics during decoding in the calling
thread and replaces unexpected parser exceptions without exposing their text or
exception chains. Caller-owned DICOM parsing must have its own privacy controls.

`DicomWaveformError.reason_code` and its message contain only one of:
`invalid_dataset`, `unsupported_sop_class`, `unsupported_transfer_syntax`,
`unsupported_multiplex_layout`, `missing_channel_definitions`,
`unsupported_channel_source`, `unsupported_units`, `invalid_calibration`,
`invalid_timing`, `unsupported_sample_interpretation`, `inconsistent_sample_count`,
`invalid_sample_value`, `invalid_window`, `resource_limit`, `review_required`.
Missing pydicom raises the existing `MissingDependencyError` with a static install
hint. No patient values or source payloads are used in either error surface.

## Offline validation

Synthetic Python builders produce small one-lead and twelve-lead datasets with
hand-checked physical values per channel. In-memory integration tests serialize
and parse all three supported transfer syntaxes before decoding. Negative controls
cover malformed metadata, data lengths, interpretations, windows, bounds,
review confirmation and text/identifier withholding. No patient data, models,
restricted assets, diagnostic benchmarks or clinical qualification are bundled.

References: [DICOM waveform module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.10.9.html),
[waveform annotation module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.10.10.html),
[CID 3001 ECG Lead](https://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_3001.html).
