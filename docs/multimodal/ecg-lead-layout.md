# ECG lead identity, order and layout

`openmed.multimodal.ecg.lead_layout` compares observed ECG labels, display positions and acquisition timing with an independent layout declaration. It reports structural anomalies for technician or clinician review. It never diagnoses their cause, analyzes waveform morphology, verifies physical electrode placement or triggers a clinical action.

Use it before canonical channel reordering. `EcgWaveform` already normalizes lead names and orders its validated channels; that normalized object cannot recover an original channel-order mistake. Supply original positions to `LeadObservation` while retaining source bytes separately, then normalize accepted input through the waveform contract. Image-derived input requires explicit label confirmation and known timing before it can have no review findings.

## Digital and image-derived input

```python
from openmed.multimodal.ecg.lead_layout import (
    AcquisitionKind,
    LeadLayout,
    LeadObservation,
    assess_lead_layout,
)

layout = LeadLayout.digital(("I", "II"))
observations = (
    LeadObservation(label="I", row=0, start_seconds=0, duration_seconds=2,
                    sample_rate_hz=500, sample_count=1000),
    LeadObservation(label="II", row=1, start_seconds=0, duration_seconds=2,
                    sample_rate_hz=500, sample_count=1000),
)
report = assess_lead_layout(
    observations,
    layout,
    source_digest="a" * 64,  # synthetic fixture digest
    acquisition=AcquisitionKind.DIGITAL,
)
assert not report.requires_review
```

The default digital profile expects all twelve standard leads in canonical order. Declare a smaller expected set explicitly for a single-lead or limb-lead acquisition. Known case and surrounding-whitespace variants normalize to standard names; other labels produce `unknown_lead` without appearing in the report.

`LeadLayout.three_by_four()` declares the common row-major arrangement `I/aVR/V1/V4`, `II/aVL/V2/V5`, `III/aVF/V3/V6`. Each column has its own timing group: the three leads within a column must align, while separate columns may show different source windows. A custom `LeadLayout` declares other arrangements, timing groups or simultaneous display windows. Do not infer an expected layout from the observations being checked.

Custom slots use zero-based rows and columns. Primary slots must cover the declared expected set exactly once. An intentionally repeated rhythm strip needs `SlotRole.RHYTHM` and its own declared slot; it is not counted as an accidental duplicate primary lead. Place it in a separate timing group if it shows a different window.

## Review findings

| Condition | Finding |
| --- | --- |
| Unrecognized label | `unknown_lead` |
| Missing, extra or repeated position | `missing_slot`, `undeclared_slot`, `duplicate_slot` |
| Label disagrees with its declared position | `lead_order_mismatch` |
| A canonical lead repeats in primary positions | `duplicate_primary_lead` |
| Timing is absent or differs within a group | `timing_unverified`, `timing_misaligned` |
| Digital sampling is absent or inconsistent | `sampling_unverified`, `sampling_misaligned` |
| Image labels lack explicit reviewer confirmation | `image_labels_unconfirmed` |

Durations use the half-open acquisition convention `sample_count / sample_rate_hz`. Sample rates and counts must agree within a timing group. An observed sampling duration must match its declared window. Relative source offsets and durations are bounded to one day; no wall-clock dates enter the contract. Timing comparisons allow a fixed one-nanosecond numerical tolerance, which is not a clinically validated tolerance or signal-quality threshold.

Undigitized images may omit sampling counts and rates, but their declared time windows must still be known. Passing `image_labels_confirmed=True` records the caller's assertion that label review occurred; the application must authenticate and retain the actual approval. It does not clear missing leads, timing defects or other structural findings. A correctly labeled but physically swapped electrode cannot be detected from labels alone.

## Privacy and boundaries

Raw observed labels are protected in-memory input and are omitted from representations and reports. Reports include a source digest, canonical expected lead names, controlled codes, channel indices and slot coordinates. They contain no waveform samples, patient labels, paths or arbitrary diagnostic strings. Digests remain linkable metadata and need the caller's access and retention controls.

Malformed declarations fail with a value-free `LeadLayoutError`. Up to 64 observations and slots are accepted; expected primary sets contain one to twelve unique standard leads. Validation is deterministic, standard-library only and performs no network calls or logging. The report version is `openmed.ecg.lead-layout.v1`. This Python slice does not add an ECG acquisition engine or clinical qualification.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/ecg/test_lead_layout.py -q
```

Synthetic controls exercise swapped and duplicated labels, missing and repeated slots, expected subsets, sequential image windows, rhythm strips, unknown timing, sampling mismatches, bounded inputs and value-free output.
