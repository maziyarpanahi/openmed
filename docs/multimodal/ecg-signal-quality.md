# ECG signal quality and acquisition gates

`openmed.multimodal.ecg.signal_quality` measures acquisition-quality indicators for each normalized ECG lead and gates an explicit downstream measurement callback. Every threshold is supplied in a versioned policy. This Python slice performs no diagnostic interpretation, filtering, interpolation, model inference, network access or automatic clinical action. Its synthetic controls establish software behavior, not clinical qualification or measurement accuracy.

Use the [waveform contract](ecg-waveform-contract.md) to normalize samples, and retain [original lead/layout observations](ecg-lead-layout.md) before canonical reordering. Normalized millivolt samples do not retain the original ADC gain or hardware clipping rails. Supply those independently through `AcquisitionEvidence`; unknown calibration remains a review finding. A complete waveform with an unknown acquisition history cannot receive a quality pass merely because its amplitudes look plausible.

## States and enforcement

| State | Meaning and permitted behavior |
| --- | --- |
| `pass` | All required leads and structural gates meet this particular caller policy. The guarded callback still requires explicit reviewer confirmation. This is not a clinical safety or accuracy claim. |
| `limited-use` | A measured metric exceeds its first limit. Retain for review or visualization; the guarded measurement callback is blocked. |
| `review` | Acquisition evidence, metric coverage, policy applicability or input binding is unverified. The callback is blocked. |
| `reacquire` | A metric exceeds its second limit, or a required lead has no observed samples. The callback is blocked; a person decides whether to reacquire. |

Each metric uses `QualityLimits(limited_above, reacquire_above)`. Comparisons are strict: equality at the first limit remains `pass`, equality at the second remains `limited-use`. Equal limits are allowed to skip the intermediate state. Severity aggregates as `pass < limited-use < review < reacquire`; no good metric compensates for a failing required metric. All-missing input cannot pass, even with permissive fraction limits.

The overall state includes every structural finding and the worst required-lead state. Optional leads retain their own findings, but are excluded from the callback input. Original layout inconsistencies block the entire acquisition, including problems involving an optional channel. The caller must select required leads independently of observed quality, rather than weakening the set after a failure.

`assess_signal_quality` produces evidence only. `run_gated_measurement` freshly reassesses the actual immutable waveform, policy and acquisition evidence on every invocation. It accepts no previous report or override. It never calls the supplied callback on blocked quality or absent confirmation. A successful callback receives a new waveform containing only required leads. `GatedMeasurement.value` holds its protected in-memory result, excluded from `repr`, alongside the quality report. Callback failures become a fixed, value-free `SignalQualityError` without exception chaining. Callbacks remain application code, not a sandbox: the application must prevent their own logging, network access and clinical side effects.

## Measured indicators

| Metric | Deterministic calculation and limits |
| --- | --- |
| `missing_fraction` | Number of explicit missing samples divided by total samples. Missing values are never filled. |
| `gap_seconds` | Longest contiguous missing run divided by the actual sample rate. |
| `clipped_fraction` | Fraction of observed samples at, outside, or within the policy margin of declared calibrated acquisition rails. Rails may vary by calibration segment. Observed minima/maxima are never used as invented rails. |
| `flatline_seconds` | Longest contiguous observed run whose entire maximum-minus-minimum range is within the policy tolerance, divided by sample rate. Missing samples break a run. Small adjacent changes cannot chain an arbitrarily large drift into a flatline. |
| `baseline_span_mv` | Maximum-minus-minimum of medians from nonoverlapping sample blocks. Requires at least two evaluable blocks. This is a baseline-change proxy, not a separation of baseline wander from physiology. |
| `roughness_mv` | Maximum across blocks of median absolute second differences, `abs(x[i+2] - 2*x[i+1] + x[i])`. Only three consecutive observed samples contribute. A gap is never bridged. Taking the maximum block score avoids diluting a sustained local burst across a long clean record. |
| `gain_relative_error` | Maximum `abs(actual_gain / independently_expected_gain - 1)` across all calibration segments. No gain is inferred from signal amplitude. |

The two block proxies require at least three observed samples per block, the policy's minimum observed fraction, and at least one consecutive observed triple for roughness. A short or unmeasurable final block causes review; it is never discarded to hide a corrupted tail. A single block cannot establish baseline span. These conservative coverage requirements are independent of missing-fraction limits. Thresholds must be evaluated at the exact policy sample rate, block size and preprocessing configuration. A sample-rate mismatch blocks processing; reported durations still use the actual waveform rate.

Second-difference magnitude can include physiological sharp changes and can miss sparse, smooth or aliased artifacts. Block medians can miss within-block drift and can reflect physiological changes. Neither proxy identifies a particular artifact source or establishes clean ECG morphology. Flatline and clipping indicators also describe acquisition concerns without interpreting their cause. The [PhysioNet noise stress test resource](https://physionet.org/content/nstdb/1.0.0/) distinguishes baseline wander, muscle artifact and electrode-motion artifact; its discussion of electrode-motion artifacts illustrates why a simple amplitude or noise score cannot guarantee reliable ECG interpretation. No external dataset or threshold from that resource is bundled or claimed as validation here.

## Acquisition and replay binding

`AcquisitionEvidence` contains the original source digest and a `waveform_fingerprint` captured when associating calibration with the normalized record. The fingerprint binds canonical samples, missingness, lead IDs, sample count, rate, relative offset and source provenance. Reusing evidence after modifying normalized samples or timing produces `acquisition_binding_mismatch`, even when the original file digest is unchanged. It detects accidental stale evidence; it does not authenticate a caller who fabricates matching hashes and metadata.

The gate reassesses original label/position observations against an independently supplied `LeadLayout`. Expected primary leads must match the actual waveform set. Each primary observation must match the waveform's count, rate, relative start and half-open duration (`count / rate`). Missing timing/sampling evidence or unconfirmed image-derived labels blocks measurements. A sequential image layout must be normalized into aligned windows with corresponding observations before this gate can pass; arbitrary windows cannot be treated as simultaneous leads.

`LeadCalibration` has one independent expected gain and ordered `CalibrationSegment` entries covering the whole record exactly once. Gaps, overlaps, reversed segments, out-of-range endpoints or incomplete coverage require review. Missing actual gain, expected gain or rails also requires review. A rail tolerance covering the entire declared rail range is rejected as unverified. Metadata assertions cannot prove physical electrode placement, correct device calibration, undisclosed filtering or dropped samples that an upstream importer silently removed. Importers must retain missingness and original timing/count evidence.

The report binds the complete acquisition declaration and policy by SHA-256, together with algorithm version `openmed.ecg.signal-quality.v1`. `policy.version` is a short opaque identifier; `policy.evidence_digest` identifies the caller's evaluation evidence. Neither a supplied digest nor reviewer confirmation certifies that evaluation was adequate.

## Synthetic example

These numbers exercise the API using a small generated signal. They are not recommended thresholds for patient data. Real use requires an independently evaluated acquisition/task policy, appropriate measurement validation, and human review.

```python
from openmed.multimodal.ecg.lead_layout import (
    AcquisitionKind, LeadLayout, LeadObservation,
)
from openmed.multimodal.ecg.signal_quality import (
    AcquisitionEvidence, CalibrationSegment, LeadCalibration, QualityLimits,
    SignalQualityPolicy, run_gated_measurement, waveform_fingerprint,
)
from openmed.multimodal.ecg.waveform import EcgLead, EcgProvenance, EcgWaveform

waveform = EcgWaveform(
    leads=(EcgLead.from_samples(
        "I", [0, .1, .2, .1, 0, -.1, -.2, -.1] * 25, unit="mV",
    ),),
    sample_rate_hz=100,
    provenance=EcgProvenance("array", "a" * 64),
)
policy = SignalQualityPolicy(
    version="synthetic-v1", evidence_digest="b" * 64, required_leads=("I",),
    sample_rate_hz=100, minimum_duration_seconds=1, block_samples=20,
    minimum_block_valid_fraction=.5, flatline_tolerance_mv=.001,
    rail_tolerance_mv=.01, missing_fraction=QualityLimits(.01, .05),
    gap_seconds=QualityLimits(.02, .1), clipped_fraction=QualityLimits(0, .01),
    flatline_seconds=QualityLimits(.05, .1), baseline_span_mv=QualityLimits(.2, .4),
    roughness_mv=QualityLimits(.3, .6), gain_relative_error=QualityLimits(.001, .01),
)
acquisition = AcquisitionEvidence(
    source_digest=waveform.provenance.source_digest,
    waveform_digest=waveform_fingerprint(waveform),
    layout=LeadLayout.digital(("I",)), acquisition=AcquisitionKind.DIGITAL,
    observations=(LeadObservation(
        label="I", row=0, start_seconds=0, duration_seconds=2,
        sample_rate_hz=100, sample_count=200,
    ),),
    calibrations=(LeadCalibration(
        lead="I", expected_gain_counts_per_mv=200,
        segments=(CalibrationSegment(
            start_sample=0, end_sample=200, gain_counts_per_mv=200,
            lower_rail_mv=-5, upper_rail_mv=5,
        ),),
    ),),
)
# A simple callback demonstrates gating, not interval extraction or diagnosis.
result = run_gated_measurement(
    waveform, policy, acquisition,
    lambda accepted: len(accepted.leads[0].samples_mv),
    reviewer_confirmed=True,
)
assert result.value == 200
assert result.quality.measurement_allowed
```

## Privacy, bounds and verification

Raw samples, observed labels, relative time arrays, calibration values and callback results remain protected in memory. Reports contain aggregate metrics, canonical lead names, controlled findings, states and digests; they omit raw samples, labels, headers, paths and calibration values. Do not put identifiers in policy versions. Aggregate metrics and digests remain linkable, sensitive metadata; applications must control access and retention. No logging or storage is performed by the gate.

The existing waveform bound is one million total samples across at most twelve leads. Evidence allows at most 64 original observations, twelve lead calibrations and 4096 segments per lead. The flatline scan is linear; proxy sorting is bounded by the sample count. Calibration and policy numbers are bounded to magnitude `1e12`, with positive gain at least `1e-12`; samples outside the arithmetic bound produce review without computing overflow-prone metrics. These bounds protect computation and are not physiological limits. Invalid declarations fail with a fixed error that omits supplied values.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/ecg/test_signal_quality.py -q
```

Synthetic controls cover noise bursts, clipping, drift, flatlines, missing runs, all-missing input, complete calibration coverage, gain changes, threshold equality, stale evidence, optional-lead exclusion, blocked callback non-invocation and value-free failures. This issue is scoped to the Python acquisition gate; it introduces no Swift acquisition pipeline or clinical model qualification.
