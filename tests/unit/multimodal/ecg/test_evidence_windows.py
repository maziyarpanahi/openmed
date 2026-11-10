"""Synthetic waveform linkage, replay, refusal and privacy controls."""

import hashlib
import json
import socket
from dataclasses import replace

import pytest

from openmed.multimodal.ecg.evidence_windows import (
    WINDOW_SCHEMA,
    EvidenceWindowError,
    FiducialKind,
    WindowFiducial,
    WindowPolicy,
    build_evidence_windows,
    restore_evidence_windows,
)
from openmed.multimodal.ecg.interval_candidates import CandidateState, TimingKind
from openmed.multimodal.ecg.signal_quality import (
    SignalQualityError,
    waveform_fingerprint,
)

from .test_interval_candidates import (
    acquisition,
    quality,
    samples,
    waveform,
)
from .test_interval_candidates import (
    policy as interval_policy,
)


def window_policy(**changes):
    return replace(
        WindowPolicy(
            version="synthetic-windows-v1",
            evidence_digest="e" * 64,
            padding_samples=3,
            maximum_window_samples=200,
            maximum_total_samples=10000,
        ),
        **changes,
    )


def build(wave=None, *, wp=None, transforms=()):
    wave = waveform() if wave is None else wave
    return build_evidence_windows(
        wave,
        quality(wave),
        acquisition(wave),
        interval_policy(),
        wp or window_policy(),
        upstream_transform_digests=transforms,
        reviewer_confirmed=True,
    )


def restore(metadata, wave=None, *, wp=None, transforms=()):
    wave = waveform() if wave is None else wave
    return restore_evidence_windows(
        metadata,
        wave,
        quality(wave),
        acquisition(wave),
        interval_policy(),
        wp or window_policy(),
        upstream_transform_digests=transforms,
        reviewer_confirmed=True,
    )


def test_every_measurement_links_to_complete_lead_specific_original_samples():
    wave = waveform()
    result = build(wave)
    assert len(result.windows) == 22
    assert all(item.state is CandidateState.REVIEW for item in result.windows)
    for window in result.windows:
        assert 0 <= window.start_sample < window.end_sample <= 450
        assert all(
            window.start_sample
            <= point.lower_sample
            <= point.upper_sample
            < window.end_sample
            for point in window.fiducials
        )
        protected = result.samples_for(window.digest, wave, reviewer_confirmed=True)
        original = next(lead for lead in wave.leads if lead.lead_id == window.lead)
        assert (
            protected.samples_mv
            == original.samples_mv[window.start_sample : window.end_sample]
        )
        assert protected.source_start_sample == window.start_sample
        assert (
            protected.start_offset_seconds
            == wave.start_offset_seconds + window.start_sample / wave.sample_rate_hz
        )
        assert protected.sample_rate_hz == wave.sample_rate_hz
        assert protected.window_digest == window.digest
    assert waveform_fingerprint(wave) == result.candidates.waveform_digest


def test_rate_window_contains_two_consecutive_complexes_and_original_fiducials():
    result = build()
    window = next(item for item in result.windows if item.kind is TimingKind.RATE)
    peaks = [point for point in window.fiducials if point.kind is FiducialKind.QRS_PEAK]
    assert [(point.beat_index, point.lower_sample) for point in peaks] == [
        (0, 100),
        (1, 200),
    ]
    assert len(window.fiducials) == 6
    assert window.start_sample == 92 and window.end_sample == 209


def test_pr_and_qt_evidence_keep_observed_wave_boundaries_not_peak_substitutes():
    result = build()
    pr = next(item for item in result.windows if item.kind is TimingKind.PR)
    qt = next(item for item in result.windows if item.kind is TimingKind.QT)
    assert {point.kind for point in pr.fiducials} == {
        FiducialKind.P_ONSET,
        FiducialKind.P_PEAK,
        FiducialKind.P_OFFSET,
        FiducialKind.QRS_ONSET,
        FiducialKind.QRS_PEAK,
        FiducialKind.QRS_OFFSET,
    }
    p_onset = next(
        point for point in pr.fiducials if point.kind is FiducialKind.P_ONSET
    )
    t_offset = next(
        point for point in qt.fiducials if point.kind is FiducialKind.T_OFFSET
    )
    assert p_onset.lower_sample == 70
    assert t_offset.upper_sample == 140


def test_original_samples_not_median_centered_detector_values_are_returned():
    first = [value + 0.123456789 for value in samples()]
    second = [-value + 0.2 for value in samples()]
    wave = waveform(first, second)
    result = build(wave)
    window = next(item for item in result.windows if item.lead == "I")
    protected = result.samples_for(window.digest, wave, reviewer_confirmed=True)
    assert protected.samples_mv[0] == 0.123456789
    assert ".123456789" not in result.to_metadata_json()
    assert ".123456789" not in result.to_report_json() + repr(protected)


def test_metadata_roundtrip_recomputes_current_candidates_and_survives_reformatting():
    result = build(transforms=("a" * 64, "b" * 64))
    encoded = result.to_metadata_json()
    restored = restore(
        json.dumps(json.loads(encoded), indent=2), transforms=("a" * 64, "b" * 64)
    )
    assert restored == result
    assert restored is not result
    assert restored.candidates is not result.candidates
    assert restored.to_metadata_json() == encoded
    assert json.loads(encoded)["schema"] == WINDOW_SCHEMA


def test_transform_order_and_policies_are_bound_even_when_samples_are_unchanged():
    first = build(transforms=("a" * 64, "b" * 64))
    reordered = build(transforms=("b" * 64, "a" * 64))
    changed = build(
        wp=window_policy(padding_samples=4), transforms=("a" * 64, "b" * 64)
    )
    assert (
        len(
            {
                first.transform_digest,
                reordered.transform_digest,
                changed.transform_digest,
            }
        )
        == 3
    )
    assert first.windows[0].digest != reordered.windows[0].digest
    with pytest.raises(EvidenceWindowError):
        restore(first.to_metadata_json(), transforms=("b" * 64, "a" * 64))
    with pytest.raises(EvidenceWindowError):
        restore(
            first.to_metadata_json(),
            wp=window_policy(padding_samples=4),
            transforms=("a" * 64, "b" * 64),
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("source_digest",), "f" * 64),
        (("source_grid", "sample_count"), 451),
        (("source_grid", "start_offset_seconds"), 18),
        (("quality", "state"), "reacquire"),
        (("windows", 0, "start_sample"), -1),
        (("windows", 0, "end_sample"), 451),
        (("windows", 0, "lead"), "III"),
        (("windows", 0, "observation_index"), False),
        (("windows", 0, "start_sample"), 92.0),
        (("windows", 0, "fiducials", 0, "upper_sample"), 449),
        (("windows", 0, "fiducials", 0, "kind"), "p-onset"),
        (("windows", 0, "candidate_digest"), "0" * 64),
        (("windows", 0, "state"), "approved"),
        (("windows", 0, "window_digest"), "0" * 64),
        (("transform_digest",), "0" * 64),
        (("candidate_result_digest",), "0" * 64),
    ],
)
def test_tampering_cannot_pass_even_with_recomputed_bundle_digest(path, replacement):
    metadata = json.loads(build().to_metadata_json())
    target = metadata
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    metadata.pop("bundle_digest")
    metadata["bundle_digest"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(EvidenceWindowError):
        restore(json.dumps(metadata))


@pytest.mark.parametrize(
    "change", ["drop", "duplicate", "reorder", "raw", "unknown", "delete-fiducial"]
)
def test_incomplete_or_extra_evidence_is_not_silently_accepted(change):
    data = json.loads(build().to_metadata_json())
    if change == "drop":
        data["windows"].pop()
    elif change == "duplicate":
        data["windows"].append(data["windows"][0])
    elif change == "reorder":
        data["windows"].reverse()
    elif change == "raw":
        data["samples_mv"] = ["synthetic sensitive sentinel"]
    elif change == "unknown":
        data["path"] = "/synthetic/sensitive/sentinel"
    else:
        data["windows"][0]["fiducials"].pop()
    with pytest.raises(EvidenceWindowError) as caught:
        restore(json.dumps(data))
    assert "sentinel" not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "malformed",
    [
        '{"secret":"synthetic sentinel",',
        '["synthetic sentinel"]',
        '{"a":1,"a":2}',
        '{"a":{"b":1,"b":2}}',
        '{"a":NaN}',
        '{"a":Infinity}',
        '{"a":1e9999}',
        '{"a":' + "[" * 1200 + '"synthetic sentinel"' + "]" * 1200 + "}",
    ],
)
def test_malformed_json_has_no_input_or_parser_exception_context(malformed):
    with pytest.raises(EvidenceWindowError) as caught:
        restore(malformed)
    assert "sentinel" not in str(caught.value)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_stale_source_rejected_on_restore_and_sample_access():
    original = waveform()
    result = build(original)
    changed_values = samples()
    changed_values[400] = 0.005
    changed = waveform(second=changed_values)
    with pytest.raises(EvidenceWindowError):
        restore(result.to_metadata_json(), changed)
    with pytest.raises(EvidenceWindowError):
        result.samples_for(result.windows[0].digest, changed, reviewer_confirmed=True)
    with pytest.raises(EvidenceWindowError):
        result.samples_for(
            result.windows[0].digest, waveform(offset=18), reviewer_confirmed=True
        )


def test_fresh_acquisition_quality_and_consent_required_on_build_and_restore():
    wave = waveform()
    result = build(wave)
    with pytest.raises(SignalQualityError, match="reviewer confirmation"):
        build_evidence_windows(
            wave,
            quality(wave),
            acquisition(wave),
            interval_policy(),
            window_policy(),
            upstream_transform_digests=(),
        )
    with pytest.raises(SignalQualityError, match="reviewer confirmation"):
        restore_evidence_windows(
            result.to_metadata_json(),
            wave,
            quality(wave),
            acquisition(wave),
            interval_policy(),
            window_policy(),
            upstream_transform_digests=(),
        )
    with pytest.raises(SignalQualityError, match="acquisition quality"):
        restore_evidence_windows(
            result.to_metadata_json(),
            wave,
            quality(wave),
            replace(acquisition(wave), calibrations=()),
            interval_policy(),
            window_policy(),
            upstream_transform_digests=(),
            reviewer_confirmed=True,
        )


def test_viewing_requires_consent_and_exact_window_membership():
    wave = waveform()
    result = build(wave)
    with pytest.raises(EvidenceWindowError):
        result.samples_for(result.windows[0].digest, wave)
    with pytest.raises(EvidenceWindowError):
        result.samples_for("0" * 64, wave, reviewer_confirmed=True)
    with pytest.raises(EvidenceWindowError):
        result.samples_for("synthetic sentinel", wave, reviewer_confirmed=True)


def test_partial_fiducials_do_not_manufacture_windows_or_upgrade_abstention():
    wave = waveform(second=samples(include_p=False))
    result = build(wave)
    pr_windows = [item for item in result.windows if item.kind is TimingKind.PR]
    assert len(pr_windows) == 3
    assert all(
        item.lead == "I" and item.state is CandidateState.ABSTAINED
        for item in pr_windows
    )
    restored = restore(result.to_metadata_json(), wave)
    assert all(
        item.state is CandidateState.ABSTAINED
        for item in restored.windows
        if item.kind is TimingKind.PR
    )
    assert all(
        item.state is CandidateState.REVIEW
        for item in restored.windows
        if item.kind is not TimingKind.PR
    )


def test_available_windows_remain_reviewable_when_leads_disagree():
    wave = waveform(second=samples(anchors=(110, 210, 310)))
    result = build(wave)
    assert result.windows
    assert all(item.state is CandidateState.ABSTAINED for item in result.windows)
    assert result.samples_for(
        result.windows[0].digest, wave, reviewer_confirmed=True
    ).samples_mv
    assert all(item.value is None for item in result.candidates.measurements)


def test_no_fiducials_retains_abstentions_with_no_fabricated_windows():
    values = [value * 0.5 for value in samples()]
    wave = waveform(values, values)
    result = build(wave)
    assert result.windows == ()
    assert all(
        item.state is CandidateState.ABSTAINED
        for item in result.candidates.measurements
    )
    assert restore(result.to_metadata_json(), wave) == result


@pytest.mark.parametrize(
    "changes",
    [
        {"padding_samples": 100},
        {"maximum_window_samples": 5},
        {"maximum_window_samples": 150, "maximum_total_samples": 200},
    ],
)
def test_context_or_resource_overflow_refuses_instead_of_clipping(changes):
    with pytest.raises(EvidenceWindowError):
        build(wp=window_policy(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"padding_samples": -1},
        {"padding_samples": True},
        {"maximum_window_samples": 0},
        {"maximum_total_samples": 8_000_001},
        {"maximum_window_samples": 200, "maximum_total_samples": 100},
        {"version": "synthetic sensitive sentinel"},
        {"evidence_digest": "invalid"},
    ],
)
def test_invalid_window_policies_are_value_free(changes):
    with pytest.raises(EvidenceWindowError) as caught:
        window_policy(**changes)
    assert "sentinel" not in str(caught.value)


@pytest.mark.parametrize("transforms", [["a" * 64], ("invalid",), ("a" * 64,) * 33])
def test_transform_declarations_are_explicit_bounded_digests(transforms):
    with pytest.raises(EvidenceWindowError):
        build(transforms=transforms)


def test_invalid_bounds_cannot_form_a_descriptor():
    window = build().windows[0]
    for changes in (
        {"start_sample": -1},
        {"end_sample": 451},
        {"source_sample_count": 1},
        {"start_sample": True},
        {"fiducials": ()},
    ):
        with pytest.raises(EvidenceWindowError):
            replace(window, **changes)
    with pytest.raises(EvidenceWindowError):
        WindowFiducial(FiducialKind.QRS_PEAK, 0, 101, 100)
    with pytest.raises(EvidenceWindowError):
        replace(
            window, fiducials=(WindowFiducial(FiducialKind.QRS_PEAK, 0, 449, 449),) * 3
        )


def test_audit_and_reprs_omit_samples_timing_and_measurement_values(
    capsys, caplog, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("network access")

    monkeypatch.setattr(socket, "socket", forbidden)
    result = build()
    audit = result.to_report_json()
    protected_metadata = result.to_metadata_json()
    for name in (
        "start_sample",
        "sample_rate_hz",
        "source_grid",
        "samples_mv",
        "fiducials",
        "agreement_score",
    ):
        assert name not in audit
    assert "samples_mv" not in protected_metadata
    assert '"value"' not in protected_metadata
    assert (
        '"synthetic"' not in protected_metadata
    )  # No fabricated clinical packet marker.
    assert '"approved"' not in protected_metadata
    assert "17.25" not in repr(result) + repr(result.windows)
    assert not caplog.records
    assert capsys.readouterr() == ("", "")
