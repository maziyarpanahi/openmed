"""Synthetic acquisition controls; thresholds are not clinically qualified."""

from dataclasses import replace
from itertools import product
from unittest.mock import Mock

import pytest

from openmed.multimodal.ecg.lead_layout import (
    AcquisitionKind,
    LeadLayout,
    LeadObservation,
)
from openmed.multimodal.ecg.signal_quality import (
    AcquisitionEvidence,
    CalibrationSegment,
    LeadCalibration,
    QualityLimits,
    QualityState,
    SignalQualityError,
    SignalQualityPolicy,
    assess_signal_quality,
    run_gated_measurement,
    waveform_fingerprint,
)
from openmed.multimodal.ecg.waveform import EcgLead, EcgProvenance, EcgWaveform


def policy(**changes):
    return replace(
        SignalQualityPolicy(
            version="synthetic-v1",
            evidence_digest="b" * 64,
            required_leads=("I",),
            sample_rate_hz=100,
            minimum_duration_seconds=1,
            block_samples=20,
            minimum_block_valid_fraction=0.5,
            flatline_tolerance_mv=0.001,
            rail_tolerance_mv=0.01,
            missing_fraction=QualityLimits(0.01, 0.05),
            gap_seconds=QualityLimits(0.02, 0.1),
            clipped_fraction=QualityLimits(0, 0.01),
            flatline_seconds=QualityLimits(0.05, 0.1),
            baseline_span_mv=QualityLimits(0.2, 0.4),
            roughness_mv=QualityLimits(0.3, 0.6),
            gain_relative_error=QualityLimits(0.001, 0.01),
        ),
        **changes,
    )


def samples():
    return [0, 0.1, 0.2, 0.1, 0, -0.1, -0.2, -0.1] * 25


def waveform(values=None, *, leads=("I",), rate=100):
    return EcgWaveform(
        tuple(
            EcgLead.from_samples(
                lead, samples() if values is None else values, unit="mV"
            )
            for lead in leads
        ),
        rate,
        EcgProvenance("array", "a" * 64),
    )


def evidence(wave):
    count = len(wave.leads[0].samples_mv)
    ids = tuple(lead.lead_id for lead in wave.leads)
    return AcquisitionEvidence(
        source_digest=wave.provenance.source_digest,
        waveform_digest=waveform_fingerprint(wave),
        layout=LeadLayout.digital(ids),
        acquisition=AcquisitionKind.DIGITAL,
        observations=tuple(
            LeadObservation(
                label=lead,
                row=i,
                start_seconds=wave.start_offset_seconds,
                duration_seconds=count / wave.sample_rate_hz,
                sample_rate_hz=wave.sample_rate_hz,
                sample_count=count,
            )
            for i, lead in enumerate(ids)
        ),
        calibrations=tuple(
            LeadCalibration(
                lead=lead,
                expected_gain_counts_per_mv=200,
                segments=(
                    CalibrationSegment(
                        start_sample=0,
                        end_sample=count,
                        gain_counts_per_mv=200,
                        lower_rail_mv=-5,
                        upper_rail_mv=5,
                    ),
                ),
            )
            for lead in ids
        ),
    )


def assess(values=None, **changes):
    wave = waveform(values)
    return assess_signal_quality(wave, policy(**changes), evidence(wave))


def test_synthetic_pass_is_deterministic_and_bound():
    wave = waveform()
    report = assess_signal_quality(wave, policy(), evidence(wave))
    assert report.state is QualityState.PASS
    assert report.measurement_allowed
    assert report.to_json() == assess().to_json()
    assert report.waveform_digest == waveform_fingerprint(wave)
    assert report.policy_digest == policy().digest
    assert "never automatically trigger clinical decisions" in report.to_json()


@pytest.mark.parametrize(
    ("value", "state"),
    [
        (0, QualityState.PASS),
        (0.1, QualityState.PASS),
        (0.10001, QualityState.LIMITED_USE),
        (0.2, QualityState.LIMITED_USE),
        (0.20001, QualityState.REACQUIRE),
    ],
)
def test_threshold_equality_and_states(value, state):
    assert QualityLimits(0.1, 0.2).classify(value) is state


@pytest.mark.parametrize(
    ("kind", "metric"),
    [
        ("noise", "roughness_mv"),
        ("drift", "baseline_span_mv"),
        ("clipping", "clipped_fraction"),
        ("gap", "gap_seconds"),
        ("flatline", "flatline_seconds"),
    ],
)
def test_injected_defects_are_measured_and_block_callback(kind, metric):
    values = samples()
    if kind == "noise":
        values[80:100] = [
            2,
            -2,
        ] * 10  # Short burst must not be diluted by clean blocks.
    elif kind == "drift":
        values = [value + i / 100 for i, value in enumerate(values)]
    elif kind == "clipping":
        values[80:100] = [5, -5] * 10
    elif kind == "gap":
        values[80:100] = [None] * 20
    else:
        values[80:100] = [0] * 20
    wave = waveform(values)
    report = assess_signal_quality(wave, policy(), evidence(wave))
    assert report.state is QualityState.REACQUIRE
    assert metric in [finding.code for finding in report.leads[0].findings]
    callback = Mock()
    with pytest.raises(SignalQualityError, match="blocked"):
        run_gated_measurement(
            wave, policy(), evidence(wave), callback, reviewer_confirmed=True
        )
    callback.assert_not_called()


def test_small_missing_burden_is_limited_use_not_measurement_permission():
    values = samples()
    values[50:53] = [None] * 3
    wave = waveform(values)
    report = assess_signal_quality(wave, policy(), evidence(wave))
    assert report.state is QualityState.LIMITED_USE
    assert report.leads[0].missing_fraction == 3 / 200
    assert report.leads[0].gap_seconds == 0.03
    callback = Mock()
    with pytest.raises(SignalQualityError, match="blocked"):
        run_gated_measurement(
            wave, policy(), evidence(wave), callback, reviewer_confirmed=True
        )
    callback.assert_not_called()


def test_all_missing_never_passes_permissive_policy():
    report = assess(
        [None] * 200,
        missing_fraction=QualityLimits(1, 1),
        gap_seconds=QualityLimits(100, 100),
    )
    assert report.state is QualityState.REACQUIRE
    assert "no_observed_samples" in [item.code for item in report.leads[0].findings]


def test_gap_breaks_derivatives_and_flatline_runs():
    # Any derivative across the gap would create a huge false noise value.
    values = [0] * 9 + [None] + [4] * 10
    report = assess(values * 10)
    assert report.leads[0].roughness_mv == 0
    assert report.leads[0].flatline_seconds == 0.1


def test_slow_drift_does_not_chain_into_a_flatline():
    report = assess([i * 0.001 for i in range(200)], flatline_tolerance_mv=0.01)
    assert report.leads[0].flatline_seconds <= 0.11
    assert report.leads[0].baseline_span_mv > 0.17


def test_flatline_sliding_window_matches_exhaustive_synthetic_runs():
    # Independent quadratic reference, including overlapping candidates and gaps.
    for prefix in product((None, 0.0, 0.1), repeat=4):
        values = list(prefix) + [0.3, -0.3] * 18
        best = 0
        for start in range(len(values)):
            observed = []
            for end in range(start, len(values)):
                value = values[end]
                if value is None:
                    break
                observed.append(value)
                if max(observed) - min(observed) <= 0.15:
                    best = max(best, len(observed))
        result = assess(values, flatline_tolerance_mv=0.15)
        assert result.leads[0].flatline_seconds == best / 100


@pytest.mark.parametrize("missing", ["gain", "rails", "expected_gain", "calibration"])
def test_unknown_acquisition_metadata_requires_review(missing):
    wave = waveform()
    acquisition = evidence(wave)
    calibration = acquisition.calibrations[0]
    segment = calibration.segments[0]
    if missing == "gain":
        calibration = replace(
            calibration, segments=(replace(segment, gain_counts_per_mv=None),)
        )
    elif missing == "rails":
        calibration = replace(
            calibration,
            segments=(replace(segment, lower_rail_mv=None, upper_rail_mv=None),),
        )
    elif missing == "expected_gain":
        calibration = replace(calibration, expected_gain_counts_per_mv=None)
    acquisition = replace(
        acquisition, calibrations=() if missing == "calibration" else (calibration,)
    )
    report = assess_signal_quality(wave, policy(), acquisition)
    assert report.state is QualityState.REVIEW
    callback = Mock()
    with pytest.raises(SignalQualityError, match="blocked"):
        run_gated_measurement(
            wave, policy(), acquisition, callback, reviewer_confirmed=True
        )
    callback.assert_not_called()


@pytest.mark.parametrize(("start", "end"), [(1, 200), (0, 199), (0, 201)])
def test_calibration_must_cover_whole_record(start, end):
    wave = waveform()
    acquisition = evidence(wave)
    calibration = acquisition.calibrations[0]
    calibration = replace(
        calibration,
        segments=(
            replace(calibration.segments[0], start_sample=start, end_sample=end),
        ),
    )
    report = assess_signal_quality(
        wave, policy(), replace(acquisition, calibrations=(calibration,))
    )
    assert report.state is QualityState.REVIEW
    assert report.leads[0].clipped_fraction is None


@pytest.mark.parametrize("defect", ["overlap", "reversed", "gain_change", "late_rails"])
def test_calibration_segments_are_ordered_and_each_is_checked(defect):
    wave = waveform()
    acquisition = evidence(wave)
    calibration = acquisition.calibrations[0]
    first = replace(calibration.segments[0], end_sample=100)
    second = replace(first, start_sample=100, end_sample=200)
    if defect == "overlap":
        second = replace(second, start_sample=99)
    elif defect == "gain_change":
        second = replace(second, gain_counts_per_mv=400)
    elif defect == "late_rails":
        second = replace(second, lower_rail_mv=-0.15, upper_rail_mv=0.15)
    segments = (second, first) if defect == "reversed" else (first, second)
    acquisition = replace(
        acquisition, calibrations=(replace(calibration, segments=segments),)
    )
    report = assess_signal_quality(wave, policy(), acquisition)
    assert report.state is (
        QualityState.REVIEW
        if defect in ("overlap", "reversed")
        else QualityState.REACQUIRE
    )
    if defect == "gain_change":
        assert report.leads[0].gain_relative_error == 1


def test_rail_margin_cannot_cover_the_entire_signal_range():
    report = assess(rail_tolerance_mv=5)
    assert report.state is QualityState.REVIEW
    assert report.leads[0].clipped_fraction is None


def test_signal_extrema_are_not_invented_clipping_rails():
    report = assess()
    assert report.leads[0].clipped_fraction == 0


@pytest.mark.parametrize(
    "change", ["samples", "source", "rate", "offset", "missingness", "lead"]
)
def test_stale_evidence_cannot_authorize_modified_waveform(change):
    wave = waveform()
    acquisition = evidence(wave)
    if change == "samples":
        wave = waveform([value + 0.01 for value in samples()])
    elif change == "source":
        wave = replace(wave, provenance=EcgProvenance("array", "c" * 64))
    elif change == "rate":
        wave = waveform(rate=200)
    elif change == "offset":
        wave = replace(wave, start_offset_seconds=1, sample_offsets_seconds=None)
    elif change == "missingness":
        wave = waveform([None] + samples()[1:])
    else:
        wave = waveform(leads=("II",))
    report = assess_signal_quality(wave, policy(), acquisition)
    assert "acquisition_binding_mismatch" in [item.code for item in report.findings]
    assert not report.measurement_allowed


@pytest.mark.parametrize(
    "change", ["label", "rate", "count", "offset", "image_unconfirmed"]
)
def test_original_acquisition_observations_are_checked_against_waveform(change):
    wave = waveform()
    acquisition = evidence(wave)
    observation = acquisition.observations[0]
    if change == "label":
        observation = replace(observation, label="protected-unrecognized-label")
    elif change == "rate":
        observation = replace(observation, sample_rate_hz=50, duration_seconds=4)
    elif change == "count":
        observation = replace(observation, sample_count=100, duration_seconds=1)
    elif change == "offset":
        observation = replace(observation, start_seconds=1)
    else:
        acquisition = replace(acquisition, acquisition=AcquisitionKind.IMAGE_DERIVED)
    acquisition = replace(acquisition, observations=(observation,))
    report = assess_signal_quality(wave, policy(), acquisition)
    assert report.state is QualityState.REVIEW
    assert "protected-unrecognized-label" not in report.to_json()


def test_policy_and_acquisition_changes_have_distinct_fingerprints():
    wave = waveform()
    acquisition = evidence(wave)
    original = assess_signal_quality(wave, policy(), acquisition)
    changed = assess_signal_quality(wave, policy(version="synthetic-v2"), acquisition)
    assert original.policy_digest != changed.policy_digest
    changed = assess_signal_quality(wave, policy(rail_tolerance_mv=0.02), acquisition)
    assert original.policy_digest != changed.policy_digest
    changed = assess_signal_quality(
        wave, policy(), replace(acquisition, image_labels_confirmed=True)
    )
    assert original.acquisition_digest != changed.acquisition_digest


def test_missing_required_lead_blocks_measurement():
    report = assess(required_leads=("I", "II"))
    assert report.state is QualityState.REVIEW
    assert "required_lead_missing" in [item.code for item in report.findings]


def test_invalid_optional_lead_never_reaches_measurement_callback():
    wave = waveform(leads=("I", "II"))
    wave = replace(
        wave, leads=(wave.leads[0], EcgLead.from_samples("II", [None] * 200, unit="mV"))
    )
    report = assess_signal_quality(wave, policy(), evidence(wave))
    assert report.state is QualityState.PASS
    assert report.leads[1].state is QualityState.REACQUIRE
    callback = Mock(return_value="protected-result")
    result = run_gated_measurement(
        wave, policy(), evidence(wave), callback, reviewer_confirmed=True
    )
    assert tuple(lead.lead_id for lead in callback.call_args.args[0].leads) == ("I",)
    assert result.value == "protected-result"
    assert "protected-result" not in repr(result)


def test_positive_gate_requires_explicit_confirmation_and_sanitizes_callback_errors():
    wave = waveform()
    callback = Mock(side_effect=RuntimeError("protected-callback-detail"))
    with pytest.raises(SignalQualityError, match="confirmation"):
        run_gated_measurement(wave, policy(), evidence(wave), callback)
    callback.assert_not_called()
    with pytest.raises(SignalQualityError) as raised:
        run_gated_measurement(
            wave, policy(), evidence(wave), callback, reviewer_confirmed=True
        )
    assert "protected-callback-detail" not in str(raised.value)
    assert raised.value.__context__ is None
    callback.assert_called_once()


def test_numeric_extremes_fail_closed_without_sample_values():
    report = assess([9.876543210123e200] * 200)
    assert report.state is QualityState.REVIEW
    assert "numeric_range_unverified" in [
        item.code for item in report.leads[0].findings
    ]
    assert "9.876543210123" not in report.to_json()
    assert "NaN" not in report.to_json()


@pytest.mark.parametrize("values", [[0], [0, 0.1] * 9 + [None] * 2, [None, 0.1] * 100])
def test_insufficient_proxy_evidence_never_passes(values):
    report = assess(
        values,
        minimum_duration_seconds=0.001,
        missing_fraction=QualityLimits(1, 1),
        gap_seconds=QualityLimits(100, 100),
    )
    assert not report.measurement_allowed
    assert report.leads[0].baseline_span_mv is None


def test_wrong_policy_rate_blocks_but_durations_use_actual_rate():
    wave = waveform([None] * 10 + samples()[10:], rate=200)
    report = assess_signal_quality(wave, policy(), evidence(wave))
    assert "policy_rate_mismatch" in [item.code for item in report.findings]
    assert report.leads[0].gap_seconds == 0.05


@pytest.mark.parametrize(
    "changes",
    [
        {"version": "protected/patient-label"},
        {"evidence_digest": "bad"},
        {"required_leads": ("patient",)},
        {"required_leads": ("I", "I")},
        {"sample_rate_hz": float("inf")},
        {"minimum_duration_seconds": 0},
        {"block_samples": True},
        {"block_samples": 1_000_001},
        {"minimum_block_valid_fraction": 0},
        {"minimum_block_valid_fraction": 1.1},
        {"flatline_tolerance_mv": float("nan")},
        {"rail_tolerance_mv": -1},
        {"roughness_mv": "unknown"},
        {"clipped_fraction": QualityLimits(2, 2)},
    ],
)
def test_malformed_policy_is_rejected_without_values(changes):
    with pytest.raises(SignalQualityError) as raised:
        policy(**changes)
    assert str(raised.value) == "invalid ECG quality declaration"


def test_sensitive_input_metadata_is_omitted_from_representations_and_reports():
    wave = waveform()
    acquisition = evidence(wave)
    protected = "protected-patient-header"
    acquisition = replace(
        acquisition,
        observations=(replace(acquisition.observations[0], label=protected),),
    )
    report = assess_signal_quality(wave, policy(), acquisition)
    assert protected not in repr(acquisition) + report.to_json() + repr(report)
    assert "samples_mv" not in report.to_json()
    assert "lower_rail_mv" not in report.to_json()


def test_calibration_resource_and_type_guards():
    segment = CalibrationSegment(
        start_sample=0,
        end_sample=200,
        gain_counts_per_mv=200,
        lower_rail_mv=-5,
        upper_rail_mv=5,
    )
    with pytest.raises(SignalQualityError):
        LeadCalibration(
            lead="I", expected_gain_counts_per_mv=200, segments=(segment,) * 4097
        )
    with pytest.raises(SignalQualityError):
        replace(segment, gain_counts_per_mv=True)
    with pytest.raises(SignalQualityError):
        replace(segment, upper_rail_mv=None)
    with pytest.raises(SignalQualityError):
        replace(segment, lower_rail_mv=5)
