# Non-ECG physiological waveform acquisition

`openmed.multimodal.physiological_waveforms` and OpenMedKit's
`PhysiologicalWaveformChannel` share the same offline acquisition contract for
PPG, respiration, invasive pressure, non-invasive pressure and capnography.
Unknown kinds, including ECG, fail closed. ECG lead contracts, device adapters,
clock alignment and synthetic waveform generation APIs belong to separate work.

**Acquisition quality only; not vital-sign measurements, diagnosis or alarms.
Consequential use requires explicit reviewer confirmation.** A `pass` state
means only that these acquisition checks found no issue. Neither passing nor
review confirmation establishes clinical suitability or provider qualification.
There are no models, transports, cloud fallbacks, clinical actions or alarms.

## Version-one encodings

| Kind | Exact unit | Encoding envelope | Rate (Hz, inclusive) | Motion step fraction |
| --- | --- | --- | --- | --- |
| `ppg` | `normalized` | 0 to 1 | 10 to 2000 | 0.4 |
| `respiration` | `normalized` | -1 to 1 | 1 to 200 | 0.6 |
| `invasive_pressure` | `mmHg` | -50 to 400 | 10 to 2000 | 0.3 |
| `non_invasive_pressure` | `mmHg` | 0 to 400 | 1 to 1000 | 0.3 |
| `capnography` | `mmHg` | 0 to 150 | 1 to 500 | 0.5 |

These are supported **encoding envelopes and engineering checks**, not normal
physiological ranges, clinically validated thresholds or universal device
limits. Other units and rates require explicit caller-side conversion and a
compatible encoding; the boundary never guesses calibration or units. PPG and
respiration must arrive in caller-normalized units. Non-invasive pressure means
a regularly sampled waveform, not sporadic cuff readings or derived blood
pressure values. A capnography channel contains caller-supplied waveform
samples; the contract does not calculate end-tidal values.

Every channel declares its actual acquisition minimum and maximum within the
encoding envelope. Samples must lie within those rails; an out-of-range sample
fails validation, while samples exactly on a rail are counted as saturation.
Use actual acquisition rails rather than patient-specific bounds. The contract
cannot verify caller calibration or provenance authenticity.

Supply at least two finite samples and equal-length nonnegative relative time
offsets. Adjacent offsets must increase by one sampling period with at most 1%
period error. NaN, infinity, boolean numbers (Python), repeated/backward offsets,
and unexplained time gaps are rejected with controlled codes. Use `None` in
Python or `nil` in Swift for missing samples on the regular clock; never remove
missing slots or interpolate them inside this boundary. Channels may have
different rates, durations and starting offsets. They remain independent;
validation does not assert cross-channel clock alignment.

There are at most 64 channels and one million samples per recording, including
missing slots. Provenance requires a caller-supplied lowercase source SHA-256.
It contains no free-text device identity, path, patient identifier or header.

## Deterministic quality states

Each channel returns counts in input order:

- `dropout_count`: missing slots.
- `saturation_count`: samples exactly on either declared acquisition rail.
- `flatline_count`: transitions in contiguous runs with steps at most one
  millionth of the declared rail span, lasting at least one second
  (`ceil(rate)` transitions). Dropout breaks runs.
- `motion_proxy_count`: adjacent present samples whose absolute step exceeds
  the per-kind fraction of the declared rail span. This crude acquisition proxy
  does not prove motion or distinguish patient physiology from artifact.

`pass` has no positive counts. Any positive count produces `limited_use`.
`review` overrides it for any qualifying flatline run, dropout or saturation
in at least 10% of sample slots, or motion proxies in at least 10% of adjacent
slots. Counts can overlap. Brief constant runs and artifacts not caught by these
heuristics can still be unsafe. No repair, filtering, segmentation, rate,
interval, oxygen saturation, pressure summary or other vital sign is derived.

Every report includes `non_diagnostic` and `reviewer_confirmation_required`.
The channel and recording expose the full notice separately for application
presentation. Call `report.require_reviewer_confirmation(confirmed=True)` or
`try report.requireReviewerConfirmation(confirmed: true)` only after explicit
review. Applications must present the notice with consequential outputs; the
library cannot enforce arbitrary downstream UI or authorize clinical actions.

## Python example

```python
from openmed.multimodal.physiological_waveforms import (
    ChannelKind, WaveformChannel, WaveformProvenance, evaluate_recording,
)

# Synthetic acquisition input; digest syntax is checked, not source authenticity.
channel = WaveformChannel(
    kind=ChannelKind.PPG,
    unit="normalized",
    sample_rate_hz=10,
    acquisition_minimum=0,
    acquisition_maximum=1,
    samples=(0.45, 0.55, None),
    offsets_seconds=(0, 0.1, 0.2),
    provenance=WaveformProvenance("a" * 64),
)
report = evaluate_recording((channel,))
print(report.to_json())  # Only kinds, counts and controlled codes.
```

The channel report is:

```json
{"kind":"ppg","state":"review","sample_count":3,"dropout_count":1,"saturation_count":0,"flatline_count":0,"motion_proxy_count":0,"codes":["dropout"]}
```

Swift accepts the same fields through `PhysiologicalWaveformChannel` with raw
kind strings, typed numeric inputs and `sourceSHA256`. Use
`PhysiologicalRecordingQuality.evaluate([channel])` and `reportJSON()` for the
same report shape and codes. JSON key order differs, but parsed reports agree.
The checked-in synthetic fixture matrix is exercised on both surfaces.

## Privacy and validation limits

Protected channels retain sensitive amplitudes and timing. Validation does not
de-identify them. Do not log, cache or serialize raw inputs; the boundary
performs no I/O and provides no source-data serializer. Input descriptions omit
samples, timing and provenance. Reports omit units, rails, amplitudes, offsets,
digests, paths and source headers. Rejections use only controlled codes, never
submitted payload text. Reports contain kinds, counts and codes only.

The synthetic matrix tests all five kinds with passing, flatline, clipped,
dropout, limited-dropout, limited-clipping and abrupt-step fixtures. Negative
controls cover units, rails, numeric values, rates, timing, unknown kinds,
provenance, resource limits, metadata leakage and mandatory confirmation.
This is software-contract evidence, not clinical validation, sensor
qualification or measured diagnostic performance.
