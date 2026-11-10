# ECG interval candidates for human review

`openmed.multimodal.ecg.interval_candidates` derives candidate rate, PR, QRS and uncorrected QT timing from actual normalized ECG samples. It first reruns the [acquisition quality gates](ecg-signal-quality.md) against the original waveform and bound acquisition evidence, requires explicit permission to measure, and uses only the independently selected required leads. It returns protected per-lead fiducials, measurement envelopes and controlled abstentions. It never supplies disease labels, treatment recommendations, corrected QT values or automatic clinical actions.

This Python implementation is a conservative baseline-excursion detector, not a clinically qualified general-purpose ECG delineator. No model, dataset or third-party algorithm is downloaded or bundled. Its synthetic tests establish deterministic software behavior and refusal paths; they do not establish clinical sensitivity, boundary accuracy or suitability for a particular device or population. The host must independently evaluate the acquisition policy, detector assumptions and every threshold before using real recordings. Swift measurement support is outside this Python issue's scope.

## Actual extraction

The detector requires separated excursions that return to a stable record baseline, a dominant sufficiently steep ventricular complex, and adequate observed context before and after each complex. `IntervalPolicy` has no clinical default thresholds. All sample-domain settings are bound to an exact sampling rate, and the complete policy plus its declared evaluation digest is included in algorithm provenance.

1. Reject any missing sample in a required lead, even if the acquisition policy permits a small gap. Do not interpolate, filter, resample or impute a missing beat.
2. Subtract the record median in memory and require the declared fraction of samples to lie within the low baseline threshold. This centering does not change stored source samples or indices. It is not a baseline-wander correction.
3. Find absolute excursions above the low threshold and join only internal gaps within `join_samples`. Require baseline guards on both sides. High-amplitude excursions must also meet the ventricular slope and width requirements; an ambiguous high-amplitude excursion holds the lead instead of being silently dropped.
4. Derive onset and offset brackets from the low and high threshold crossings. Retain the unique dominant absolute peak or a contiguous peak plateau. Separated equal maxima, excessive boundary sensitivity, truncated excursions and out-of-policy peak spacing hold the lead. A dominant negative complex is supported; the anchor is called a ventricular peak candidate, without assuming that the largest absolute deflection is an anatomical R wave.
5. Search before and after each ventricular complex for a single complete observed P/T excursion. Search distances are constraints, never substituted fiducials. Missing, competing, truncated, too-small, too-large or poorly resolved excursions abstain on the dependent interval. An additional excursion inside a search window prevents selection even if it is weaker than the main candidate. An excursion assigned to multiple wave identities, such as T for one beat and P for the next, withholds every competing assignment.
6. Require the configured number of leads, identical beat counts and bounded displacement between corresponding ventricular peaks. Every required lead participates; the implementation never chooses a favorable subset. A lead disagreement holds all four aggregate measurements. Otherwise each interval is also checked across every beat and lead, without averaging away an outlier.

The [PhysioNet QT Database annotation description](https://physionet.org/files/qtdb/1.0.0/doc/node4.html) distinguishes waveform onset/end annotations from peak and beat annotations. The [NeuroKit ECG documentation](https://neuropsychology.github.io/NeuroKit/functions/ecg.html) likewise treats peak detection and delineation as distinct steps. This implementation therefore does not substitute ventricular peaks for P onset, QRS onset/offset or T offset. It does not claim equivalence to either project's algorithms, and no data or source code from them is included.

## Measurements, confidence and abstention

| Candidate | Calculation from observed fiducial brackets | Unit |
| --- | --- | --- |
| Rate | `60 * sample_rate_hz / consecutive_peak_spacing` | beats/minute |
| PR | QRS onset minus P onset | milliseconds |
| QRS | QRS offset minus QRS onset | milliseconds |
| QT | T offset minus QRS onset | milliseconds |

Every accepted value remains `review-candidate`. Its value is the median of per-observation bracket midpoints. `lower` and `upper` cover all per-beat/per-lead brackets, with reciprocal bounds for rate. The interval envelope must fit `interval_spread_ms`; the rate envelope must fit `rate_spread_bpm`. Equality is allowed. `agreement_score` is `1 - envelope_width / allowed_spread`, between zero and one. It measures consistency under this policy, not the probability of being correct. The brackets capture sample/threshold sensitivity and observed disagreement; they are **not clinical error bounds** and do not capture missed waves, shared systematic error or all noise effects.

An `abstained` candidate has no value, bounds or confidence. It contains controlled reasons such as `p-boundaries-unresolved`, `lead-beat-disagreement` or `measurement-disagreement`. Unresolved P waves withhold PR; unresolved T waves withhold QT. Rate and QRS may remain review candidates if their own requirements pass. Beat count, ventricular delineation, required-lead or source-quality failures cannot be bypassed by accepting another interval.

The narrow morphology assumptions can fail on many real recordings, including overlapping waves, varying baseline, low-amplitude or non-dominant ventricular complexes, tall T waves, pacing artifacts, highly variable spacing and indistinct terminal boundaries. A quality pass does not prove these assumptions. Agreement between leads cannot prove correct identification, and threshold changes must not be used to manufacture agreement after observing a failure. The host must retain human review and independent qualification rather than interpreting abstention as a diagnosis or an instruction to reacquire.

## Source binding and protected data

`IntervalCandidates` retains the original source digest, full normalized waveform fingerprint, freshly computed quality report, detector policy digest and algorithm version `openmed.ecg.excursion-intervals.v1`. `digest` binds the complete result, including all protected measurements and fiducials. Hashes detect accidental stale evidence; they do not authenticate a caller or certify an acquisition. `extract_interval_candidates` accepts no prior pass report or caller-provided landmarks.

Fiducials use inclusive original-grid sample brackets. `window_start` and `window_end` define a half-open source window containing an excursion and its quiet guards. Sample `i` maps to source-relative seconds as `start_offset_seconds + i / sample_rate_hz`; it is not a wall-clock timestamp. No normalization, centering or lead selection resets that offset. Original lead IDs and sample count are retained. The result contains no waveform samples.

Measurement values, timing metadata, fiducials and the quality object are excluded from result `repr`. Treat their in-memory fields as protected clinical data. `to_report_json()` emits only algorithm and provenance hashes, controlled states/reasons, lead references and observation counts, with the non-diagnostic notice. It omits rate, interval values, confidence values, sample positions, sample rate, source offset and raw quality metrics. Do not log `dataclasses.asdict(result)` or persist the protected result as an audit report. The function performs no network call, file write or logging; host code remains responsible for its own storage and access controls.

Resources are bounded by the waveform contract, 4,096 excursions and 256 ventricular candidates per lead. Exceeding a detector bound produces abstention, not truncated measurement evidence. Inputs with numeric amplitudes beyond the detector's arithmetic resource limit also abstain. These are implementation limits, not physiological thresholds.

## Runnable synthetic example

These triangle pulses and thresholds are software fixtures, not clinical recommendations. The example constructs complete original acquisition evidence and an independently declared quality policy, then checks that the known synthetic timing falls within the returned envelope. It prints no signal or measurement values.

```python
from openmed.multimodal.ecg.interval_candidates import (
    CandidateState, IntervalPolicy, SampleRange, TimingKind,
    extract_interval_candidates,
)
from openmed.multimodal.ecg.lead_layout import (
    AcquisitionKind, LeadLayout, LeadObservation,
)
from openmed.multimodal.ecg.signal_quality import (
    AcquisitionEvidence, CalibrationSegment, LeadCalibration, QualityLimits,
    SignalQualityPolicy, waveform_fingerprint,
)
from openmed.multimodal.ecg.waveform import EcgLead, EcgProvenance, EcgWaveform

values = [0.0] * 450
for anchor in (100, 200, 300):
    for start, peak, end, amplitude in (
        (anchor - 30, anchor - 25, anchor - 20, .3),
        (anchor - 4, anchor, anchor + 4, 1),
        (anchor + 20, anchor + 30, anchor + 40, .4),
    ):
        for i in range(start, end + 1):
            fraction = ((i - start) / (peak - start) if i <= peak
                        else (end - i) / (end - peak))
            values[i] += amplitude * fraction

ids = ("I", "II")
wave = EcgWaveform(
    tuple(EcgLead.from_samples(lead, values, unit="mV") for lead in ids),
    100, EcgProvenance("array", "a" * 64), start_offset_seconds=17.25,
)
quality = SignalQualityPolicy(
    version="synthetic-v1", evidence_digest="b" * 64, required_leads=ids,
    sample_rate_hz=100, minimum_duration_seconds=1, block_samples=50,
    minimum_block_valid_fraction=.9, flatline_tolerance_mv=.001,
    rail_tolerance_mv=.01, missing_fraction=QualityLimits(.01, .05),
    gap_seconds=QualityLimits(.02, .1), clipped_fraction=QualityLimits(0, .01),
    flatline_seconds=QualityLimits(2, 3), baseline_span_mv=QualityLimits(.2, .4),
    roughness_mv=QualityLimits(.3, .6), gain_relative_error=QualityLimits(.001, .01),
)
acquisition = AcquisitionEvidence(
    source_digest=wave.provenance.source_digest,
    waveform_digest=waveform_fingerprint(wave), layout=LeadLayout.digital(ids),
    acquisition=AcquisitionKind.DIGITAL,
    observations=tuple(LeadObservation(
        label=lead, row=i, start_seconds=17.25, duration_seconds=4.5,
        sample_rate_hz=100, sample_count=450,
    ) for i, lead in enumerate(ids)),
    calibrations=tuple(LeadCalibration(
        lead=lead, expected_gain_counts_per_mv=200,
        segments=(CalibrationSegment(
            start_sample=0, end_sample=450, gain_counts_per_mv=200,
            lower_rail_mv=-5, upper_rail_mv=5,
        ),),
    ) for lead in ids),
)
policy = IntervalPolicy(
    version="synthetic-v1", evidence_digest="d" * 64, sample_rate_hz=100,
    boundary_low_mv=.01, boundary_high_mv=.025, secondary_peak_mv=.1,
    qrs_peak_mv=.8, qrs_slope_mv_per_sample=.2, quiet_samples=2, join_samples=1,
    baseline_fraction=.5, qrs_width=SampleRange(3, 15),
    secondary_width=SampleRange(5, 30), rr_spacing=SampleRange(60, 150),
    p_search=SampleRange(5, 35), t_search=SampleRange(5, 60),
    boundary_uncertainty_samples=3, alignment_samples=3, minimum_beats=3,
    minimum_leads=2, rate_spread_bpm=10, interval_spread_ms=40,
)
result = extract_interval_candidates(
    wave, quality, acquisition, policy, reviewer_confirmed=True,
)
qt = next(item for item in result.measurements if item.kind is TimingKind.QT)
assert qt.state is CandidateState.REVIEW
assert qt.lower <= 440 <= qt.upper
assert result.leads[0].beats[0].qrs.peak.lower == 100
assert result.waveform_digest == waveform_fingerprint(wave)
audit_json = result.to_report_json()  # Controlled metadata only; no printing needed.
```

Run the focused synthetic checks with `.venv/bin/python -m pytest tests/unit/multimodal/ecg/test_interval_candidates.py -q`. The controls cover source-relative timing, rate conversion, inverted/biphasic excursions, missing and ambiguous waves, per-lead and per-beat disagreements, quality/consent binding, resource bounds, and value-free audit output.
