# Non-diagnostic ECG waveform input contract

The Python contract in `openmed.multimodal.ecg.waveform` normalizes caller-supplied
ECG samples locally and deterministically. It performs no file reads, network
calls, model inference, disease interpretation, or clinical decision. Its schema
is `openmed.multimodal.ecg.waveform.v1`. This issue's implementation is the Python
input contract; it adds no Swift provider or clinical workflow.

Every record, report, and validation error carries this boundary:

> Non-diagnostic ECG input normalization only. No disease interpretation or
> clinical validation. Explicit reviewer confirmation is required before
> consequential use; never automatically trigger clinical decisions.

## Canonical representation

- `EcgLead` contains immutable tuples of millivolt samples and boolean validity
  masks (`True` means observed). Missing samples are exactly `None` with a false
  mask. An omitted conversion mask is derived from `None`. NaN, infinity,
  non-numbers, and conflicting masks are rejected; values are never interpolated.
- Lead identifiers normalize whitespace and case to `I`, `II`, `III`, `aVR`,
  `aVL`, `aVF`, and `V1` through `V6`. Records sort by this order and reject
  duplicate aliases. Unknown/custom leads must be mapped explicitly by the
  caller; no lead is inferred or derived from other leads.
- `EcgLead.from_samples` requires explicit units: `V`, `mV`, `uV`, `µV`, `μV`,
  or `adc`. Voltage converts to millivolts. ADC conversion is
  `(sample - baseline_counts) / gain_counts_per_mv`; both calibration values
  must be supplied, finite, and gain must be positive. Physical voltage rejects
  ADC calibration to prevent double scaling. A source byte digest binds the
  caller's original header/calibration; the report does not copy that header.
- `EcgWaveform` requires one to twelve unique aligned leads with equal sample
  counts and an explicit finite rate in `(0, 100000]` Hz. There are at most
  1,000,000 samples across all leads. Each lead has at least one sample; a wholly
  missing lead is representable and is not a quality qualification.
- Relative sample seconds are `start_offset_seconds + index / sample_rate_hz`.
  Start defaults to zero. Optional explicit offsets must match that regular
  grid within `1e-9` of a sample interval. Duration (`count / rate`) and final
  offset cannot exceed 86,400 seconds. Absolute timestamps, irregular timing,
  per-lead rates, gaps without masks, resampling, and asynchronous alignment
  are unsupported. Split or prepare these inputs explicitly before admission.
- `EcgProvenance` accepts only a supported source format (`array`, `dicom`, `edf`,
  `wfdb`, `hl7-aecg`) and a lowercase source SHA-256 digest. Format declarations
  do not imply a reader, provider qualification, or clinical validation.

## Synthetic example

```python
import hashlib
from openmed.multimodal.ecg.waveform import EcgLead, EcgProvenance, EcgWaveform

record = EcgWaveform(
    leads=(EcgLead.from_samples(" ii ", (0, 1000, None), unit="uV"),),
    sample_rate_hz=250,
    provenance=EcgProvenance("array", hashlib.sha256(b"synthetic ECG v1").hexdigest()),
)
assert record.leads[0].samples_mv == (0.0, 1.0, None)
assert record.sample_offsets_seconds == (0.0, 0.004, 0.008)
assert record.to_report()["leads"] == [{"lead_id": "II", "missing_count": 1}]
```

## Privacy and review boundary

Samples remain sensitive in memory. Normalization is **not de-identification**.
`repr`, `to_report`, and deterministic `to_json` omit amplitudes, individual
sample offsets, masks, source paths, header text, and patient metadata. Reports
contain only schema/notice, review requirement, source format/digest, canonical
units/leads, rate, sample count, and missing counts. Digest linkage can still be
sensitive; callers must apply their access and retention policies. Do not log
raw attributes or serialize the record with generic dataclass serializers.

Validation raises `WaveformValidationError` with static reason codes and
actionable instructions, without reflecting supplied values or native exception
context. Reasons cover unsupported leads/units/formats, invalid digest/rate,
calibration/missingness/timing ambiguity, alignment, and resource limits.

`record.require_reviewer_confirmation()` fails closed. A caller must supply
`confirmed=True` following explicit review before consequential use. The method
does not approve or release an output, persist consent, or certify clinical
safety. Downstream outputs must retain the notice and their own explicit review
gate; the report always keeps `review_required=True`. There is no automatic
clinical action or cloud fallback.

All tests use synthetic values and digest literals. No dataset, credentials,
model assets, or additional dependencies are bundled.
