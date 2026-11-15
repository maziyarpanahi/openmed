"""Independent clock truth, discontinuities and actual modality reference controls."""

import json
import socket
from dataclasses import replace
from fractions import Fraction

import pytest

from openmed.multimodal.audio_windows import plan_audio_windows
from openmed.multimodal.ecg.evidence_windows import build_evidence_windows
from openmed.multimodal.ecg.interval_candidates import CandidateState
from openmed.multimodal.evidence.fused_graph import EvidenceNode, NodeKind, PageGeometry
from openmed.multimodal.provenance.clock_alignment import (
    AlignmentPolicy,
    AlignmentState,
    AudioChunkEvidence,
    AudioPlanEvidence,
    BreakKind,
    ClockAlignmentError,
    ClockAnchor,
    ClockBinding,
    ClockBreak,
    ClockEpoch,
    ClockReview,
    EcgEvidence,
    ImageEvidence,
    NanosecondBound,
    TranscriptEvidence,
    align_evidence,
    audio_chunk_digest,
    calibrate_clock,
    transcript_source_digest,
)
from openmed.multimodal.streaming.speech.providers import AudioChunk
from openmed.multimodal.streaming.transcript import RevisionLedger, TranscriptToken

from ..ecg.test_evidence_windows import build, window_policy
from ..ecg.test_interval_candidates import acquisition, quality, samples, waveform
from ..ecg.test_interval_candidates import policy as interval_policy

NS = 1_000_000_000
WALL = 1_700_000_000_000_000_000


def ref(index):
    return f"{index:032x}"


def epoch(index=1, *, offset=WALL, slope_ppb=0, error=0, **changes):
    return replace(
        ClockEpoch(
            clock_ref=ref(index),
            epoch_ref=ref(index + 100),
            wall_domain_digest="a" * 64,
            calibration_digest="b" * 64,
            anchors=(
                ClockAnchor(0, offset, error),
                ClockAnchor(10 * NS, offset + 10 * NS + 10 * slope_ppb, error),
            ),
            maximum_rate_error_ppb=1_000_000,
            nonlinearity_ns=0,
            breaks=(),
        ),
        **changes,
    )


def policy(**changes):
    return replace(
        AlignmentPolicy(
            version="synthetic-v1",
            evidence_digest="c" * 64,
            maximum_uncertainty_ns=10_000_000,
            maximum_anchor_gap_ns=10 * NS,
        ),
        **changes,
    )


def binding(clock, source="d" * 64, **changes):
    return replace(
        ClockBinding(
            source_digest=source,
            clock_ref=clock.clock_ref,
            epoch_ref=clock.epoch_ref,
            evidence_digest="e" * 64,
            local_start_ns=0,
            local_end_ns=10 * NS,
            monotonic_start_ns=0,
            timing_uncertainty_ns=0,
        ),
        **changes,
    )


def image(clock, start=2 * NS, end=3 * NS, *, error=0):
    return ImageEvidence(
        EvidenceNode(
            node_id=ref(400),
            kind=NodeKind.PIXELS,
            source_digest="d" * 64,
            geometry=PageGeometry(page=7, width=100, height=200),
            region=(10, 20, 30, 40),
        ),
        start,
        end,
        "f" * 64,
        binding(clock, timing_uncertainty_ns=error),
    )


def aligned(source=None, target=None, first=None, second=None, limit=None):
    first, second = first or epoch(), second or epoch(2)
    return align_evidence(
        source or image(first),
        target or image(second),
        first,
        second,
        limit or policy(),
    )


def held(result, reason):
    assert result.state is AlignmentState.REVIEW
    assert reason in result.reasons
    assert result.target_start_ns is result.target_end_ns is None
    assert result.guaranteed_overlap_ns is result.uncertainty_ns is None


@pytest.mark.parametrize("drift", [-999_999, -100, 0, 100, 999_999])
def test_exact_anchors_recover_independent_drift_without_float_cancellation(drift):
    fit = calibrate_clock(epoch(slope_ppb=drift)).fits[0]
    assert fit.reason is None
    assert fit.drift_estimate_ppb == Fraction(drift)
    assert fit.drift_bounds_ppb == (Fraction(drift), Fraction(drift))


def test_zero_rate_tolerance_supports_degenerate_segment_and_point_fits():
    for error in (0, 100):
        clock = epoch(maximum_rate_error_ppb=0, error=error)
        fit = calibrate_clock(clock).fits[0]
        assert fit.reason is None and fit.drift_bounds_ppb == (0, 0)
        result = aligned(first=clock)
        assert result.state is AlignmentState.ALIGNED
        assert result.target_start_ns == NanosecondBound(2 * NS - error, 2 * NS + error)


def test_real_offset_translates_into_target_clock_not_wall_or_asset_time():
    first, second = epoch(), epoch(2, offset=WALL - NS)
    result = aligned(target=image(second, 2 * NS, 5 * NS), first=first, second=second)
    assert result.state is AlignmentState.ALIGNED
    assert result.target_start_ns == NanosecondBound(3 * NS, 3 * NS)
    assert result.target_end_ns == NanosecondBound(4 * NS, 4 * NS)
    assert result.guaranteed_overlap_ns == (3 * NS, 4 * NS)


def test_large_monotonic_origins_retain_single_nanosecond_distinctions():
    source_origin, target_origin = 2**55 + 1, 2**56 + 3
    first, second = epoch(), epoch(2)
    first = replace(
        first,
        anchors=tuple(
            replace(a, monotonic_ns=a.monotonic_ns + source_origin)
            for a in first.anchors
        ),
    )
    second = replace(
        second,
        anchors=tuple(
            replace(a, monotonic_ns=a.monotonic_ns + target_origin)
            for a in second.anchors
        ),
    )
    source, target = image(first, 2 * NS + 1, 3 * NS + 1), image(second, NS, 4 * NS)
    source = replace(
        source, binding=replace(source.binding, monotonic_start_ns=source_origin)
    )
    target = replace(
        target, binding=replace(target.binding, monotonic_start_ns=target_origin)
    )
    result = aligned(source, target, first, second)
    assert result.state is AlignmentState.ALIGNED
    assert result.target_start_ns == NanosecondBound(
        target_origin + 2 * NS + 1, target_origin + 2 * NS + 1
    )


@pytest.mark.parametrize("source_ppb", [-400_000, 0, 700_000])
@pytest.mark.parametrize("target_ppb", [-500_000, 0, 800_000])
@pytest.mark.parametrize("perturbation", [-900, 0, 900])
def test_bounds_contain_independent_affine_truth_with_noisy_anchors(
    source_ppb, target_ppb, perturbation
):
    source_slope = 1 + Fraction(source_ppb, NS)
    target_slope = 1 + Fraction(target_ppb, NS)
    first = epoch(slope_ppb=source_ppb, error=1_000, nonlinearity_ns=700)
    second = epoch(
        2, offset=WALL - 123_456, slope_ppb=target_ppb, error=1_000, nonlinearity_ns=700
    )
    first = replace(
        first,
        anchors=tuple(
            replace(a, wall_ns=a.wall_ns + perturbation * (-1 if i else 1))
            for i, a in enumerate(first.anchors)
        ),
    )
    second = replace(
        second,
        anchors=tuple(
            replace(a, wall_ns=a.wall_ns - perturbation * (-1 if i else 1))
            for i, a in enumerate(second.anchors)
        ),
    )
    result = aligned(target=image(second, NS, 4 * NS), first=first, second=second)
    assert result.state is AlignmentState.ALIGNED
    # Independent true clocks: no access to fitter internals or its polygon.
    for point, bound in (
        (2 * NS, result.target_start_ns),
        (3 * NS, result.target_end_ns),
    ):
        for source_error in (-700, 700):
            for target_error in (-700, 700):
                truth = (
                    source_slope * point + 123_456 + source_error - target_error
                ) / target_slope
                assert bound.lower <= truth <= bound.upper


def test_all_anchor_constraints_retained_and_inconsistent_middle_not_dropped():
    clock = epoch(
        anchors=(
            ClockAnchor(0, WALL, 0),
            ClockAnchor(5 * NS, WALL + 5 * NS + 1, 0),
            ClockAnchor(10 * NS, WALL + 10 * NS, 0),
        )
    )
    assert calibrate_clock(clock).fits[0].reason is ClockReview.INCONSISTENT
    held(aligned(first=clock), ClockReview.INCONSISTENT)
    held(aligned(first=epoch(slope_ppb=1_000_001)), ClockReview.INCONSISTENT)


def test_perfect_residuals_do_not_remove_qualified_nonlinearity_or_boundary_error():
    first = epoch(nonlinearity_ns=100, error=50)
    second = epoch(2, nonlinearity_ns=200, error=75)
    result = aligned(image(first, error=10), image(second, error=20), first, second)
    assert result.state is AlignmentState.ALIGNED
    assert result.uncertainty_ns > 2 * (100 + 200 + 10 + 20)
    held(
        aligned(
            image(first, error=10),
            image(second, error=20),
            first,
            second,
            policy(maximum_uncertainty_ns=result.uncertainty_ns - 1),
        ),
        ClockReview.UNCERTAINTY,
    )
    assert (
        aligned(
            image(first, error=10),
            image(second, error=20),
            first,
            second,
            policy(maximum_uncertainty_ns=result.uncertainty_ns),
        ).state
        is AlignmentState.ALIGNED
    )


@pytest.mark.parametrize("kind", list(BreakKind))
@pytest.mark.parametrize("side", ["source", "target"])
def test_no_translation_across_declared_pause_drop_wall_step_or_reset(kind, side):
    broken = epoch(
        1 if side == "source" else 2,
        breaks=(ClockBreak(kind, 2 * NS + NS // 2, 2 * NS + NS // 2),),
    )
    held(
        aligned(first=broken) if side == "source" else aligned(second=broken),
        ClockReview.DISCONTINUITY,
    )


def test_wall_step_groups_fit_separately_and_do_not_hide_unaffected_intervals():
    first = epoch(
        anchors=(
            ClockAnchor(0, WALL, 0),
            ClockAnchor(4 * NS, WALL + 4 * NS, 0),
            ClockAnchor(6 * NS, WALL + 8 * NS, 0),
            ClockAnchor(10 * NS, WALL + 12 * NS, 0),
        ),
        breaks=(ClockBreak(BreakKind.WALL_STEP, 5 * NS, 5 * NS),),
    )
    second = epoch(
        2, anchors=(ClockAnchor(0, WALL, 0), ClockAnchor(14 * NS, WALL + 14 * NS, 0))
    )
    target = image(second, 8 * NS, 10 * NS)
    result = aligned(
        image(first, 7 * NS, 8 * NS),
        target,
        first,
        second,
        policy(maximum_anchor_gap_ns=14 * NS),
    )
    assert result.state is AlignmentState.ALIGNED
    assert result.target_start_ns == NanosecondBound(9 * NS, 9 * NS)
    assert result.target_end_ns == NanosecondBound(10 * NS, 10 * NS)
    assert len(calibrate_clock(first).fits) == 2
    assert aligned(first=first).state is AlignmentState.ALIGNED


def test_missing_sparse_and_unbracketed_anchors_require_review_without_extrapolation():
    held(aligned(first=epoch(anchors=())), ClockReview.ANCHORS)
    held(
        aligned(first=epoch(anchors=(ClockAnchor(2 * NS, WALL, 0),))),
        ClockReview.ANCHORS,
    )
    held(
        aligned(
            first=epoch(
                anchors=(
                    ClockAnchor(3 * NS, WALL, 0),
                    ClockAnchor(10 * NS, WALL + 7 * NS, 0),
                )
            )
        ),
        ClockReview.COVERAGE,
    )
    held(aligned(limit=policy(maximum_anchor_gap_ns=9 * NS)), ClockReview.SPARSE)
    held(aligned(second=epoch(2, offset=WALL - 9 * NS)), ClockReview.COVERAGE)


def test_extremely_loose_valid_inverse_is_reviewed_without_overflow_or_clipping():
    second = epoch(2, error=10**12, maximum_rate_error_ppb=NS - 1)
    held(aligned(second=second), ClockReview.COVERAGE)


def test_shared_clock_needs_no_wall_anchors_but_retains_binding_uncertainty():
    clock = epoch(anchors=())
    result = aligned(image(clock, error=10), image(clock, error=20), clock, clock)
    assert result.state is AlignmentState.ALIGNED
    assert result.target_start_ns == NanosecondBound(2 * NS - 10, 2 * NS + 10)
    assert result.guaranteed_overlap_ns == (2 * NS + 20, 3 * NS - 20)
    assert result.uncertainty_ns == 60


def test_shared_clock_cannot_mix_declarations_or_cross_discontinuities():
    clock = epoch(anchors=())
    held(
        aligned(first=clock, second=replace(clock, calibration_digest="f" * 64)),
        ClockReview.DECLARATION,
    )
    clock = replace(clock, breaks=(ClockBreak(BreakKind.DROPPED, 2 * NS, 2 * NS),))
    held(aligned(first=clock, second=clock), ClockReview.DISCONTINUITY)


def test_distinct_wall_domains_and_wrong_binding_epochs_are_not_numeric_matches():
    held(aligned(second=epoch(2, wall_domain_digest="f" * 64)), ClockReview.WALL_DOMAIN)
    source = image(epoch())
    held(
        aligned(
            source=replace(source, binding=replace(source.binding, epoch_ref=ref(999)))
        ),
        ClockReview.EPOCH,
    )


@pytest.mark.parametrize(
    ("target_start", "error", "reason"),
    [
        (4 * NS, 0, ClockReview.NO_OVERLAP),
        (3 * NS, 10, ClockReview.POSSIBLE_OVERLAP),
        (3 * NS, 0, ClockReview.NO_OVERLAP),
    ],
)
def test_uncertain_or_absent_overlap_withholds_all_translated_values(
    target_start, error, reason
):
    held(aligned(target=image(epoch(2), target_start, 5 * NS, error=error)), reason)


def test_audio_plan_resolves_exact_member_and_cannot_invent_a_window():
    clock = epoch()
    plan = plan_audio_windows(10_000, window_ms=1_000)
    source = AudioPlanEvidence(plan, plan.windows[2].window_id, binding(clock))
    result = aligned(source=source)
    assert result.state is AlignmentState.ALIGNED
    assert result.target_start_ns == NanosecondBound(2 * NS, 2 * NS)
    held(aligned(source=replace(source, window_id="unknown")), ClockReview.REFERENCE)
    with pytest.raises(ClockAlignmentError):
        aligned(source=replace(source, plan=replace(plan, windows=plan.windows[:-1])))


def test_encoded_chunk_uses_declared_time_not_payload_size_or_sequence():
    chunk = AudioChunk(999, b"synthetic-encoded-sentinel", 48_000, 2, 1_000)
    source = AudioChunkEvidence(
        chunk, 2 * NS, binding(epoch(), audio_chunk_digest(chunk))
    )
    result = aligned(source=source)
    assert result.state is AlignmentState.ALIGNED
    assert result.target_end_ns == NanosecondBound(3 * NS, 3 * NS)
    held(aligned(source=replace(source, local_start_ns=None)), ClockReview.TIMING)
    held(
        aligned(source=replace(source, chunk=replace(chunk, payload=b"changed"))),
        ClockReview.BINDING,
    )


def transcript(*, finalized=True):
    ledger = RevisionLedger(
        ref(50),
        ref(51),
        4,
        (
            TranscriptToken(
                token_ref=ref(52), start=0, end=4, start_ms=2000, end_ms=3000
            ),
        ),
    )
    reference = ledger.capture_span(ref(53), 0, 4)
    if finalized:
        ledger.finalize(expected_revision=0)
    return TranscriptEvidence(
        ledger, reference, binding(epoch(), transcript_source_digest(ledger))
    )


def test_transcript_uses_current_registered_lineage_after_timing_shift_and_finalize():
    source = transcript(finalized=False)
    held(aligned(source=source), ClockReview.UNFINALIZED)
    source.ledger.shift_timestamps((ref(52),), 100, expected_revision=0)
    source.ledger.finalize(expected_revision=1)
    result = aligned(source=source)
    assert result.state is AlignmentState.ALIGNED
    assert result.target_start_ns == NanosecondBound(2_100_000_000, 2_100_000_000)
    assert result.target_end_ns == NanosecondBound(3_100_000_000, 3_100_000_000)


def test_unregistered_or_invalidated_transcript_citations_require_review():
    source = transcript()
    held(
        aligned(
            source=replace(
                source, reference=replace(source.reference, span_ref=ref(99))
            )
        ),
        ClockReview.REFERENCE,
    )
    source = transcript(finalized=False)
    source.ledger.delete(1, 2, expected_revision=0)
    source.ledger.finalize(expected_revision=1)
    held(aligned(source=source), ClockReview.REFERENCE)


def ecg(*, rate=100):
    wave = replace(waveform(), sample_rate_hz=rate, sample_offsets_seconds=None)
    windows = build_evidence_windows(
        wave,
        quality(wave, sample_rate_hz=rate),
        acquisition(wave),
        interval_policy(sample_rate_hz=rate),
        window_policy(),
        upstream_transform_digests=(),
        reviewer_confirmed=True,
    )
    return EcgEvidence(
        windows,
        wave,
        windows.windows[0].digest,
        binding(
            epoch(),
            wave.provenance.source_digest,
            local_start_ns=17_250_000_000,
            local_end_ns=22 * NS,
        ),
    )


def test_ecg_uses_original_sample_grid_with_asset_offset_removed_only_by_binding():
    source = ecg()
    result = aligned(source=source, target=image(epoch(2), NS // 2, 2 * NS))
    assert result.state is AlignmentState.ALIGNED
    assert result.target_start_ns == NanosecondBound(920_000_000, 920_000_000)
    assert result.target_end_ns == NanosecondBound(1_090_000_000, 1_090_000_000)
    held(aligned(source=replace(source, window_digest="0" * 64)), ClockReview.REFERENCE)
    changed = samples()
    changed[400] = 0.005
    held(
        aligned(source=replace(source, waveform=waveform(second=changed))),
        ClockReview.BINDING,
    )


def test_fractional_sample_time_is_rounded_outwards_not_to_nearest_nanosecond():
    source = ecg(rate=300)
    result = aligned(source=source, target=image(epoch(2), 100_000_000, 900_000_000))
    assert result.state is AlignmentState.ALIGNED
    assert result.target_start_ns == NanosecondBound(306_666_666, 306_666_667)
    assert result.target_end_ns == NanosecondBound(363_333_333, 363_333_334)
    held(
        aligned(
            source=source,
            target=image(epoch(2), 100_000_000, 900_000_000),
            limit=policy(maximum_uncertainty_ns=0),
        ),
        ClockReview.UNCERTAINTY,
    )


def test_ecg_aggregate_abstention_not_promoted_by_temporal_agreement():
    wave = waveform(second=samples(anchors=(120, 220, 320)))
    windows = build(wave)
    source = replace(
        ecg(), waveform=wave, windows=windows, window_digest=windows.windows[0].digest
    )
    held(aligned(source=source), ClockReview.EVIDENCE)
    # An edited window descriptor cannot remove its aggregate candidate hold.
    edited = replace(windows.windows[0], state=CandidateState.REVIEW, reasons=())
    changed = replace(windows, windows=(edited,) + windows.windows[1:])
    held(
        aligned(source=replace(source, windows=changed, window_digest=edited.digest)),
        ClockReview.EVIDENCE,
    )


def test_image_capture_time_is_explicit_and_bound_to_source_not_page_number():
    source = image(epoch())
    held(
        aligned(source=replace(source, local_start_ns=None, local_end_ns=None)),
        ClockReview.TIMING,
    )
    held(
        aligned(
            source=replace(source, node=replace(source.node, source_digest="0" * 64))
        ),
        ClockReview.BINDING,
    )
    assert source.node.geometry.page == 7
    assert aligned(source=source).target_start_ns == NanosecondBound(2 * NS, 2 * NS)


def test_binding_must_cover_uncertain_endpoints_without_compressing_time():
    source = image(epoch(), error=10)
    held(
        aligned(
            source=replace(
                source, binding=replace(source.binding, local_start_ns=2 * NS)
            )
        ),
        ClockReview.BINDING,
    )
    held(
        aligned(source=image(epoch(), 2 * NS, 2 * NS + 10, error=10)),
        ClockReview.UNCERTAINTY,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"maximum_rate_error_ppb": True},
        {"maximum_rate_error_ppb": NS},
        {"nonlinearity_ns": float("nan")},
        {"nonlinearity_ns": -1},
        {"clock_ref": "private-sentinel"},
        {"anchors": []},
        {"anchors": (ClockAnchor(10, 0, 0), ClockAnchor(9, 1, 0))},
        {
            "breaks": (
                ClockBreak(BreakKind.PAUSE, 3, 5),
                ClockBreak(BreakKind.PAUSE, 5, 9),
            )
        },
        {"breaks": (ClockBreak(BreakKind.PAUSE, 0, 0),)},
    ],
)
def test_invalid_declarations_fail_with_fixed_value_free_error(change):
    with pytest.raises(ClockAlignmentError) as caught:
        epoch(**change)
    assert str(caught.value) == "invalid multimodal clock declaration"
    assert caught.value.__context__ is None and caught.value.__cause__ is None


def test_offline_deterministic_reports_do_not_expose_source_payload_or_times(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("clock alignment attempted network access")

    monkeypatch.setattr(socket, "socket", forbidden)
    chunk = AudioChunk(999, b"synthetic-encoded-sentinel", 48_000, 2, 1_000)
    source = AudioChunkEvidence(
        chunk, 2 * NS, binding(epoch(), audio_chunk_digest(chunk))
    )
    first, second = aligned(source=source), aligned(source=source)
    assert first == second and first.to_report_json() == second.to_report_json()
    report = first.to_report_json() + repr(first) + repr(source) + repr(epoch())
    report += calibrate_clock(epoch()).to_report_json()
    for forbidden_value in (
        "synthetic-encoded-sentinel",
        str(WALL),
        str(2 * NS),
        "start_sample",
    ):
        assert forbidden_value not in report
    assert "target_start_ns" not in json.loads(first.to_report_json())
    assert "clinical verification" in first.to_report_json()
