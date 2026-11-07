"""Synthetic sample-domain truth and refusal controls, not clinical validation."""

import json
import socket
from dataclasses import replace

import pytest

from openmed.multimodal.ecg.interval_candidates import (
    ALGORITHM_VERSION,
    Abstention,
    CandidateState,
    IntervalCandidateError,
    IntervalPolicy,
    SampleRange,
    TimingKind,
    extract_interval_candidates,
)
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
    SignalQualityError,
    SignalQualityPolicy,
    waveform_fingerprint,
)
from openmed.multimodal.ecg.waveform import EcgLead, EcgProvenance, EcgWaveform


def triangle(values, start, peak, end, amplitude):
    for index in range(start, end + 1):
        values[index] += amplitude * (
            (index - start) / (peak - start)
            if index <= peak
            else (end - index) / (end - peak)
        )


def samples(*, anchors=(100, 200, 300), include_p=True, include_t=True, qrs_half=4):
    values = [0.0] * 450
    for peak in anchors:
        triangle(values, peak - qrs_half, peak, peak + qrs_half, 1)
        if include_p:
            triangle(values, peak - 30, peak - 25, peak - 20, 0.3)
        if include_t:
            triangle(values, peak + 20, peak + 30, peak + 40, 0.4)
    return values


def policy(**changes):
    return replace(
        IntervalPolicy(
            version="synthetic-v1",
            evidence_digest="d" * 64,
            sample_rate_hz=100,
            boundary_low_mv=0.01,
            boundary_high_mv=0.025,
            secondary_peak_mv=0.1,
            qrs_peak_mv=0.8,
            qrs_slope_mv_per_sample=0.2,
            quiet_samples=2,
            join_samples=1,
            baseline_fraction=0.5,
            qrs_width=SampleRange(3, 15),
            secondary_width=SampleRange(5, 30),
            rr_spacing=SampleRange(60, 150),
            p_search=SampleRange(5, 35),
            t_search=SampleRange(5, 60),
            boundary_uncertainty_samples=3,
            alignment_samples=3,
            minimum_beats=3,
            minimum_leads=2,
            rate_spread_bpm=10,
            interval_spread_ms=40,
        ),
        **changes,
    )


def waveform(first=None, second=None, *, offset=17.25):
    return EcgWaveform(
        leads=(
            EcgLead.from_samples("I", samples() if first is None else first, unit="mV"),
            EcgLead.from_samples(
                "II", samples() if second is None else second, unit="mV"
            ),
        ),
        sample_rate_hz=100,
        provenance=EcgProvenance("array", "a" * 64),
        start_offset_seconds=offset,
    )


def quality(wave, **changes):
    return replace(
        SignalQualityPolicy(
            version="synthetic-v1",
            evidence_digest="b" * 64,
            required_leads=tuple(lead.lead_id for lead in wave.leads),
            sample_rate_hz=100,
            minimum_duration_seconds=1,
            block_samples=50,
            minimum_block_valid_fraction=0.9,
            flatline_tolerance_mv=0.001,
            rail_tolerance_mv=0.01,
            missing_fraction=QualityLimits(0.01, 0.05),
            gap_seconds=QualityLimits(0.02, 0.1),
            clipped_fraction=QualityLimits(0, 0.01),
            flatline_seconds=QualityLimits(2, 3),
            baseline_span_mv=QualityLimits(0.2, 0.4),
            roughness_mv=QualityLimits(0.3, 0.6),
            gain_relative_error=QualityLimits(0.001, 0.01),
        ),
        **changes,
    )


def acquisition(wave):
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
                row=index,
                start_seconds=wave.start_offset_seconds,
                duration_seconds=count / wave.sample_rate_hz,
                sample_rate_hz=wave.sample_rate_hz,
                sample_count=count,
            )
            for index, lead in enumerate(ids)
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


def extract(wave=None, interval=None, **kwargs):
    wave = waveform() if wave is None else wave
    return extract_interval_candidates(
        wave,
        quality(wave),
        acquisition(wave),
        interval or policy(),
        reviewer_confirmed=True,
        **kwargs,
    )


def candidates(result):
    return {item.kind: item for item in result.measurements}


def test_actual_samples_yield_bounded_timings_and_original_fiducials():
    result = extract()
    expected = {
        TimingKind.RATE: 60,
        TimingKind.PR: 260,
        TimingKind.QRS: 80,
        TimingKind.QT: 440,
    }
    for item in result.measurements:
        assert item.state is CandidateState.REVIEW
        assert item.lower <= expected[item.kind] <= item.upper
        assert item.lower <= item.value <= item.upper
        assert 0 <= item.agreement_score <= 1
        assert item.leads == ("I", "II")
    first = result.leads[0].beats[0]
    assert first.qrs.peak.lower == first.qrs.peak.upper == 100
    assert first.qrs.onset.lower == 96
    assert first.p.onset.lower == 70
    assert first.t.offset.upper == 140
    assert (
        result.start_offset_seconds + first.qrs.peak.lower / result.sample_rate_hz
        == 18.25
    )
    assert all(
        0 <= beat.qrs.window_start < beat.qrs.window_end <= result.sample_count
        for lead in result.leads
        for beat in lead.beats
    )


def test_no_network_and_deterministic_provenance(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network access")

    monkeypatch.setattr(socket, "socket", forbidden)
    wave = waveform()
    first, second = extract(wave), extract(wave)
    assert first == second
    assert first.to_report_json() == second.to_report_json()
    assert first.waveform_digest == waveform_fingerprint(wave)
    assert first.policy_digest == policy().digest
    assert json.loads(first.to_report_json())["algorithm"] == ALGORITHM_VERSION
    assert extract(wave, policy(version="synthetic-v2")).digest != first.digest
    assert extract(waveform(offset=18)).digest != first.digest


def test_signed_lead_and_baseline_offset_preserve_timings():
    result = extract(
        waveform([v + 0.5 for v in samples()], [-v + 0.2 for v in samples()])
    )
    assert all(item.state is CandidateState.REVIEW for item in result.measurements)
    assert candidates(result)[TimingKind.RATE].value == 60


@pytest.mark.parametrize(
    ("missing", "kind", "reason"),
    [
        ("include_p", TimingKind.PR, Abstention.P),
        ("include_t", TimingKind.QT, Abstention.T),
    ],
)
def test_absent_wave_abstains_only_dependent_measurement(missing, kind, reason):
    result = candidates(extract(waveform(second=samples(**{missing: False}))))
    assert result[kind].reasons == (reason,)
    assert result[kind].value is None
    assert result[kind].agreement_score is None
    assert all(
        item.state is CandidateState.REVIEW
        for key, item in result.items()
        if key is not kind
    )


def test_multiple_p_alternatives_are_not_resolved_by_strength():
    values = samples(include_p=False)
    for peak in (100, 200, 300):
        triangle(values, peak - 31, peak - 28, peak - 25, 0.3)
        triangle(values, peak - 21, peak - 18, peak - 15, 0.2)
    result = candidates(extract(waveform(second=values)))
    assert result[TimingKind.PR].reasons == (Abstention.P,)
    assert result[TimingKind.RATE].state is CandidateState.REVIEW


def test_noise_excursion_inside_search_window_prevents_p_selection():
    values = samples()
    values[85] = 0.02  # Above boundary floor, below candidate peak threshold.
    result = candidates(extract(waveform(second=values)))
    assert result[TimingKind.PR].state is CandidateState.ABSTAINED


@pytest.mark.parametrize(
    "second", [samples(anchors=(110, 210, 310)), samples(anchors=(100, 300))]
)
def test_lead_alignment_and_missing_beats_abstain_all(second):
    result = extract(waveform(second=second))
    assert all(item.state is CandidateState.ABSTAINED for item in result.measurements)
    assert all(Abstention.ALIGNMENT in item.reasons for item in result.measurements)


def test_cross_lead_interval_disagreement_not_averaged_away():
    values = samples(include_t=False)
    for peak in (100, 200, 300):
        triangle(values, peak + 20, peak + 35, peak + 50, 0.4)
    result = candidates(extract(waveform(second=values)))
    assert result[TimingKind.QT].reasons == (Abstention.SPREAD,)
    assert result[TimingKind.QT].value is None
    assert result[TimingKind.PR].state is CandidateState.REVIEW


def test_variable_spacing_within_search_range_abstains_on_rate_spread():
    values = samples(anchors=(100, 180, 300))
    result = candidates(extract(waveform(values, values)))
    assert result[TimingKind.RATE].reasons == (Abstention.SPREAD,)
    assert result[TimingKind.QRS].state is CandidateState.REVIEW


def test_spacing_outside_policy_never_fills_missing_beats():
    values = samples(anchors=(70, 140, 320))
    result = extract(waveform(values, values))
    assert all(item.reasons == (Abstention.SPACING,) for item in result.measurements)


def test_one_required_lead_does_not_claim_agreement_with_optional_lead():
    wave = waveform()
    result = extract_interval_candidates(
        wave,
        quality(wave, required_leads=("I",)),
        acquisition(wave),
        policy(),
        reviewer_confirmed=True,
    )
    assert len(result.leads) == 1
    assert all(item.reasons == (Abstention.LEADS,) for item in result.measurements)


def test_missing_samples_are_not_interpolated_even_when_quality_policy_allows_gap():
    values = samples()
    values[100] = None
    result = extract(waveform(second=values))
    assert result.quality.measurement_allowed
    assert all(Abstention.MISSING in item.reasons for item in result.measurements)


def test_uncertain_threshold_boundaries_abstain():
    result = extract(
        interval=policy(
            boundary_high_mv=0.09, secondary_peak_mv=0.1, boundary_uncertainty_samples=1
        )
    )
    assert candidates(result)[TimingKind.QT].reasons == (Abstention.T,)


def test_equal_separated_ventricular_peaks_abstain():
    values = samples()
    values[102] = 1
    result = extract(waveform(second=values))
    assert all(Abstention.QRS in item.reasons for item in result.measurements)


def test_truncated_ventricular_excursion_is_not_silently_dropped():
    values = samples()
    values[:4] = [1, 0.75, 0.5, 0.25]
    result = extract(waveform(second=values))
    assert all(Abstention.QRS in item.reasons for item in result.measurements)


def test_large_slow_excursion_is_not_silently_ignored():
    values = samples()
    triangle(values, 360, 375, 390, 1)
    result = extract(waveform(second=values))
    assert all(Abstention.QRS in item.reasons for item in result.measurements)


def test_baseline_assumption_has_an_explicit_gate():
    wave = waveform()
    result = extract(wave, policy(baseline_fraction=0.99))
    assert all(item.reasons == (Abstention.BASELINE,) for item in result.measurements)


def test_fresh_quality_and_confirmation_cannot_be_bypassed():
    wave = waveform()
    with pytest.raises(SignalQualityError, match="reviewer confirmation"):
        extract_interval_candidates(wave, quality(wave), acquisition(wave), policy())
    changed = waveform(offset=18)
    with pytest.raises(SignalQualityError, match="acquisition quality"):
        extract_interval_candidates(
            changed,
            quality(changed),
            acquisition(wave),
            policy(),
            reviewer_confirmed=True,
        )
    with pytest.raises(SignalQualityError, match="acquisition quality"):
        extract_interval_candidates(
            wave,
            quality(wave),
            replace(acquisition(wave), calibrations=()),
            policy(),
            reviewer_confirmed=True,
        )


def test_value_free_reports_reprs_and_error_messages(capsys, caplog):
    result = extract()
    report = json.loads(result.to_report_json())
    assert "sample_rate_hz" not in report and "start_offset_seconds" not in report
    assert "quality" not in report
    assert all(
        "value" not in item
        and "lower" not in item
        and "upper" not in item
        and "agreement_score" not in item
        for item in report["measurements"]
    )
    assert "17.25" not in repr(result) + repr(result.leads) + repr(result.measurements)
    assert "260.0" not in repr(result.measurements)
    with pytest.raises(IntervalCandidateError) as caught:
        policy(version="synthetic secret sentinel")
    assert "sentinel" not in str(caught.value)
    assert caught.value.__context__ is None
    assert not caplog.records
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("boundary_low_mv", 0),
        ("boundary_high_mv", 0.005),
        ("secondary_peak_mv", 0.9),
        ("qrs_peak_mv", float("nan")),
        ("qrs_slope_mv_per_sample", float("inf")),
        ("rate_spread_bpm", 10**1000),
        ("interval_spread_ms", True),
        ("sample_rate_hz", -1),
        ("baseline_fraction", 0.4),
        ("baseline_fraction", 1.1),
        ("quiet_samples", 0),
        ("quiet_samples", True),
        ("join_samples", 2),
        ("minimum_beats", 1),
        ("minimum_beats", 257),
        ("minimum_leads", 1),
        ("alignment_samples", 0),
        ("boundary_uncertainty_samples", 1.5),
        ("qrs_width", (3, 15)),
        ("qrs_width", SampleRange(3, 70)),
        ("p_search", SampleRange(1, 40)),
        ("evidence_digest", "secret"),
    ],
)
def test_invalid_policy_declarations_are_value_free(key, value):
    with pytest.raises(
        IntervalCandidateError, match="^invalid ECG interval policy or input$"
    ):
        policy(**{key: value})


@pytest.mark.parametrize("bounds", [(0, 1), (2, 1), (True, 2), (1, 1_000_001)])
def test_invalid_sample_range(bounds):
    with pytest.raises(IntervalCandidateError):
        SampleRange(*bounds)


def test_wrong_sampling_rate_does_not_reinterpret_sample_domain_policy():
    with pytest.raises(IntervalCandidateError):
        extract(interval=policy(sample_rate_hz=250))


def test_exact_rate_conversion_uses_source_grid():
    original = waveform()
    wave = EcgWaveform(
        original.leads,
        250,
        original.provenance,
        start_offset_seconds=original.start_offset_seconds,
    )
    result = extract_interval_candidates(
        wave,
        quality(wave, sample_rate_hz=250),
        acquisition(wave),
        policy(sample_rate_hz=250),
        reviewer_confirmed=True,
    )
    items = candidates(result)
    assert items[TimingKind.RATE].value == 150
    assert items[TimingKind.QRS].lower <= 32 <= items[TimingKind.QRS].upper
    assert result.sample_rate_hz == 250


@pytest.mark.parametrize(
    "values",
    [
        [0, 0.02, 0] * 4100,
        [0, 0, 0, 1 / 3, 2 / 3, 1, 2 / 3, 1 / 3, 0, 0, 0, 0] * 257,
    ],
)
def test_resource_bounds_abstain_without_truncating_evidence(values):
    result = extract(waveform(values, values))
    assert all(item.reasons == (Abstention.LIMIT,) for item in result.measurements)


def test_biphasic_complex_is_joined_without_splitting_beats():
    values = samples()
    for peak in (100, 200, 300):
        values[peak - 6 : peak + 7] = [
            0,
            -0.2,
            -0.4,
            -0.2,
            0,
            0.5,
            1,
            0.5,
            0,
            -0.2,
            -0.4,
            -0.2,
            0,
        ]
    result = extract(waveform(values, values))
    assert len(result.leads[0].beats) == 3
    assert all(item.state is CandidateState.REVIEW for item in result.measurements)
    item = candidates(result)[TimingKind.QRS]
    assert item.lower <= 120 <= item.upper


def test_small_baseline_noise_preserves_known_truth_inside_reported_envelope():
    values = [
        value + (0.002 if index % 2 else -0.002)
        for index, value in enumerate(samples())
    ]
    result = extract(waveform(values, values))
    item = candidates(result)[TimingKind.QT]
    assert item.state is CandidateState.REVIEW
    assert item.lower <= 440 <= item.upper


def test_insufficient_beats_are_not_extrapolated():
    values = samples(anchors=(100, 300))
    result = extract(waveform(values, values))
    assert all(item.reasons == (Abstention.BEATS,) for item in result.measurements)


def test_one_excursion_cannot_be_both_t_and_next_p_even_with_permissive_spread():
    values = samples(anchors=(100, 170, 240), include_p=False, include_t=False)
    triangle(values, 70, 75, 80, 0.3)
    triangle(values, 135, 145, 155, 0.4)
    triangle(values, 205, 215, 225, 0.4)
    triangle(values, 260, 270, 280, 0.4)
    result = candidates(
        extract(waveform(values, values), policy(interval_spread_ms=1000))
    )
    assert result[TimingKind.PR].reasons == (Abstention.P,)
    assert result[TimingKind.QT].reasons == (Abstention.T,)
    assert result[TimingKind.QRS].state is CandidateState.REVIEW
    assert result[TimingKind.RATE].state is CandidateState.REVIEW
