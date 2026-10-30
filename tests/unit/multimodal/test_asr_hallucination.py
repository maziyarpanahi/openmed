"""Synthetic tests for the ASR hallucination and repetition-loop check."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from openmed.multimodal.abstention import AbstentionReason, AbstentionStage
from openmed.multimodal.asr_hallucination import (
    ASR_HALLUCINATION_SCHEMA_VERSION,
    DEFAULT_HALLUCINATION_THRESHOLDS,
    FLAG_REASON_CODES,
    HALLUCINATION_REASON_CODES,
    INSUFFICIENT_EVIDENCE_REASON_CODES,
    AsrHallucinationError,
    AsrHypothesisSegment,
    AsrSegmentStatistics,
    HallucinationDecision,
    HallucinationOutcome,
    HallucinationReport,
    HallucinationThresholds,
    VoiceActivityInterval,
    VoiceActivityKind,
    VoiceActivityTimeline,
    check_asr_hallucinations,
    require_draft_eligible,
)
from openmed.multimodal.provider_result import (
    ProviderResultEnvelope,
    ProviderResultOutcome,
)

SENTINEL = "ZQXJ-SENTINEL-PATIENT-NAME-4471"
SENTINEL_WORDS = f"{SENTINEL} reports chest pain since monday"
DEFAULT = DEFAULT_HALLUCINATION_THRESHOLDS

# Pinned so that any silent edit of a default threshold fails this suite until
# the profile version is bumped and the digest re-pinned deliberately.
DEFAULT_DIGEST = "5b1319020ca30d81ae9760de42630b6b51553f1930aa13ce12d092e44d036412"

NORMAL_STATS = AsrSegmentStatistics(
    no_speech_probability=0.05,
    avg_logprob=-0.25,
    compression_ratio=1.4,
)


def _timeline(kind: VoiceActivityKind, duration_ms: int = 10_000):
    return VoiceActivityTimeline(
        duration_ms=duration_ms,
        intervals=(VoiceActivityInterval(0, duration_ms, kind),),
    )


def _segment(
    segment_id: str = "seg-0001",
    *,
    start_ms: int = 0,
    end_ms: int = 4_000,
    text: str | None = SENTINEL_WORDS,
    token_ids: tuple[int, ...] | None = None,
    statistics: AsrSegmentStatistics | None = NORMAL_STATS,
) -> AsrHypothesisSegment:
    return AsrHypothesisSegment(
        segment_id=segment_id,
        start_ms=start_ms,
        end_ms=end_ms,
        statistics=statistics,
        text=text,
        token_ids=token_ids,
    )


def _single(segment, timeline, thresholds=DEFAULT) -> HallucinationDecision:
    report = check_asr_hallucinations((segment,), timeline, thresholds=thresholds)
    assert len(report.decisions) == 1
    return report.decisions[0]


def _all_serialized(report: HallucinationReport) -> str:
    parts = [report.to_json(), repr(report)]
    for decision in report.decisions:
        parts.extend((decision.to_json(), repr(decision), str(decision.to_dict())))
    return "\n".join(parts)


# --- Non-speech timelines are flagged; normal speech passes. -----------------


def test_speech_timeline_with_normal_statistics_passes() -> None:
    decision = _single(_segment(), _timeline(VoiceActivityKind.SPEECH))

    assert decision.outcome is HallucinationOutcome.SUPPORTED
    assert decision.reason_codes == ()
    assert decision.final_eligible is True
    assert decision.review_required is False
    assert decision.abstention() is None
    assert decision.speech_ratio == 1.0
    assert decision.vad_coverage == 1.0
    assert decision.unit_kind == "word"
    # The hyphenated sentinel splits into five word units.
    assert decision.unit_count == 10


@pytest.mark.parametrize(
    ("kind", "dominant_code"),
    [
        (VoiceActivityKind.SILENCE, "silence_dominant"),
        (VoiceActivityKind.NOISE, "noise_dominant"),
        (VoiceActivityKind.MUSIC, "music_dominant"),
    ],
)
def test_non_speech_timelines_are_flagged(kind, dominant_code) -> None:
    decision = _single(_segment(), _timeline(kind))

    assert decision.outcome is HallucinationOutcome.FLAGGED
    assert decision.reason_codes == ("speech_coverage_low", dominant_code)
    assert decision.final_eligible is False
    assert decision.review_required is True
    assert decision.speech_ratio == 0.0


def test_provider_statistics_flag_even_over_speech() -> None:
    stats = AsrSegmentStatistics(
        no_speech_probability=0.9,
        avg_logprob=-1.7,
        compression_ratio=3.1,
    )
    decision = _single(_segment(statistics=stats), _timeline(VoiceActivityKind.SPEECH))

    assert decision.outcome is HallucinationOutcome.FLAGGED
    assert decision.reason_codes == (
        "no_speech_probability_high",
        "avg_logprob_low",
        "compression_ratio_high",
    )
    assert decision.no_speech_probability == 0.9
    assert decision.avg_logprob == -1.7
    assert decision.compression_ratio == 3.1


def test_threshold_boundaries_are_inclusive_passes() -> None:
    stats = AsrSegmentStatistics(
        no_speech_probability=DEFAULT.max_no_speech_probability,
        avg_logprob=DEFAULT.min_avg_logprob,
        compression_ratio=DEFAULT.max_compression_ratio,
    )
    decision = _single(_segment(statistics=stats), _timeline(VoiceActivityKind.SPEECH))
    assert decision.outcome is HallucinationOutcome.SUPPORTED


def test_mixed_timeline_reports_dominant_non_speech_kind() -> None:
    timeline = VoiceActivityTimeline(
        duration_ms=10_000,
        intervals=(
            VoiceActivityInterval(0, 1_000, VoiceActivityKind.SPEECH),
            VoiceActivityInterval(1_000, 2_000, VoiceActivityKind.SILENCE),
            VoiceActivityInterval(2_000, 10_000, VoiceActivityKind.MUSIC),
        ),
    )
    decision = _single(_segment(end_ms=10_000), timeline)

    # Ten words over one second of speech also exceed the unit rate.
    assert decision.reason_codes == (
        "speech_coverage_low",
        "music_dominant",
        "runaway_length",
    )
    assert decision.speech_ratio == 0.1
    assert decision.units_per_speech_second == 10.0


def test_segment_partially_over_speech_uses_overlap_only() -> None:
    timeline = VoiceActivityTimeline(
        duration_ms=10_000,
        intervals=(
            VoiceActivityInterval(0, 5_000, VoiceActivityKind.SILENCE),
            VoiceActivityInterval(5_000, 10_000, VoiceActivityKind.SPEECH),
        ),
    )
    decision = _single(_segment(start_ms=4_000, end_ms=9_000), timeline)

    assert decision.speech_ratio == 0.8
    assert decision.outcome is HallucinationOutcome.SUPPORTED


# --- Repetition loops and runaway length. -----------------------------------


def test_ngram_loop_is_detected_from_text() -> None:
    text = "please call the office " + "thank you so much " * 5
    decision = _single(
        _segment(text=text, end_ms=10_000),
        _timeline(VoiceActivityKind.SPEECH),
    )

    assert "ngram_loop" in decision.reason_codes
    assert decision.loop_ngram_size == 4
    assert decision.loop_repeats == 5
    assert decision.outcome is HallucinationOutcome.FLAGGED


def test_ngram_loop_is_detected_from_token_ids() -> None:
    token_ids = (11, 12, 13) + (7, 8) * 6
    decision = _single(
        _segment(text=None, token_ids=token_ids, end_ms=10_000),
        _timeline(VoiceActivityKind.SPEECH),
    )

    assert decision.unit_kind == "token"
    assert decision.unit_count == 15
    assert decision.loop_ngram_size == 2
    assert decision.loop_repeats == 6
    assert "ngram_loop" in decision.reason_codes


def test_token_ids_take_precedence_over_text() -> None:
    decision = _single(
        _segment(text="a a a a a a a a", token_ids=(1, 2, 3)),
        _timeline(VoiceActivityKind.SPEECH),
    )
    assert decision.unit_kind == "token"
    assert decision.unit_count == 3
    assert decision.outcome is HallucinationOutcome.SUPPORTED


def test_unigram_loop_needs_more_repeats_than_longer_ngrams() -> None:
    speech = _timeline(VoiceActivityKind.SPEECH)
    short = _single(_segment(text="no no no no no"), speech)
    long = _single(_segment(text="no no no no no no"), speech)

    assert "ngram_loop" not in short.reason_codes
    assert short.loop_repeats == 5
    assert "ngram_loop" in long.reason_codes
    assert long.loop_ngram_size == 1
    assert long.loop_repeats == 6


def test_repeated_ngram_fraction_is_flagged() -> None:
    # Interleaved repetition that is not a tandem loop but reuses trigrams.
    fixed = [f"f{index}" for index in range(10)]
    words = []
    for index in range(4):
        words.extend([*fixed, f"w{index}"])
    decision = _single(
        _segment(text=" ".join(words), end_ms=10_000),
        _timeline(VoiceActivityKind.SPEECH),
    )

    assert "ngram_loop" not in decision.reason_codes
    assert "repeated_ngram_fraction_high" in decision.reason_codes
    assert decision.repeated_ngram_fraction is not None
    assert decision.repeated_ngram_fraction > DEFAULT.max_repeated_ngram_fraction


def test_short_hypotheses_skip_repeated_fraction() -> None:
    decision = _single(_segment(text="no no no no no"), _timeline("speech"))
    assert decision.repeated_ngram_fraction is None


def test_runaway_length_relative_to_speech_duration() -> None:
    timeline = VoiceActivityTimeline(
        duration_ms=10_000,
        intervals=(
            VoiceActivityInterval(0, 9_000, VoiceActivityKind.SPEECH),
            VoiceActivityInterval(9_000, 10_000, VoiceActivityKind.SILENCE),
        ),
    )
    text = " ".join(f"w{index}" for index in range(60))
    # 60 words over 1 second of speech (segment covers 8000-10000).
    decision = _single(_segment(start_ms=8_000, end_ms=10_000, text=text), timeline)

    assert decision.units_per_speech_second == 60.0
    assert decision.reason_codes == ("runaway_length",)


def test_runaway_absolute_unit_cap() -> None:
    text = " ".join(f"w{index}" for index in range(DEFAULT.max_segment_units + 1))
    timeline = _timeline(VoiceActivityKind.SPEECH, duration_ms=600_000)
    decision = _single(_segment(end_ms=600_000, text=text), timeline)

    assert decision.reason_codes == ("runaway_length",)


def test_repetition_detection_is_deterministic() -> None:
    text = "start " + "we will see you " * 7 + "end"
    segments = (_segment(text=text, end_ms=10_000),)
    timeline = _timeline(VoiceActivityKind.SPEECH)
    first = check_asr_hallucinations(segments, timeline)
    second = check_asr_hallucinations(segments, timeline)

    assert first.to_json() == second.to_json()
    assert first.digest == second.digest
    assert first.decisions[0].loop_repeats == 7


# --- Missing evidence is never a pass. --------------------------------------


@pytest.mark.parametrize(
    ("statistics", "expected"),
    [
        (None, ("no_speech_probability_missing", "avg_logprob_missing")),
        (
            AsrSegmentStatistics(avg_logprob=-0.2, compression_ratio=1.2),
            ("no_speech_probability_missing",),
        ),
        (
            AsrSegmentStatistics(no_speech_probability=0.1, compression_ratio=1.2),
            ("avg_logprob_missing",),
        ),
        (
            AsrSegmentStatistics(no_speech_probability=0.1, avg_logprob=-0.2),
            ("compression_ratio_missing",),
        ),
    ],
)
def test_missing_statistics_yield_insufficient_evidence(statistics, expected) -> None:
    decision = _single(
        _segment(statistics=statistics), _timeline(VoiceActivityKind.SPEECH)
    )

    assert decision.outcome is HallucinationOutcome.INSUFFICIENT_EVIDENCE
    for code in expected:
        assert code in decision.reason_codes
    assert set(decision.reason_codes) <= set(INSUFFICIENT_EVIDENCE_REASON_CODES)
    assert decision.final_eligible is False
    assert decision.review_required is True


def test_missing_statistics_list_all_three_codes() -> None:
    decision = _single(_segment(statistics=None), _timeline("speech"))
    assert decision.reason_codes == (
        "no_speech_probability_missing",
        "avg_logprob_missing",
        "compression_ratio_missing",
    )


def test_missing_timeline_yields_insufficient_evidence() -> None:
    decision = _single(_segment(), None)

    assert decision.outcome is HallucinationOutcome.INSUFFICIENT_EVIDENCE
    assert decision.reason_codes == ("vad_coverage_missing",)
    assert decision.vad_coverage is None
    assert decision.speech_ratio is None


def test_partial_timeline_coverage_is_insufficient() -> None:
    timeline = VoiceActivityTimeline(
        duration_ms=10_000,
        intervals=(VoiceActivityInterval(0, 2_000, VoiceActivityKind.SPEECH),),
    )
    decision = _single(_segment(end_ms=4_000), timeline)

    assert decision.outcome is HallucinationOutcome.INSUFFICIENT_EVIDENCE
    assert decision.reason_codes == ("vad_coverage_missing",)
    assert decision.vad_coverage == 0.5


def test_segment_beyond_timeline_counts_as_unlabeled() -> None:
    decision = _single(
        _segment(start_ms=8_000, end_ms=12_000),
        _timeline(VoiceActivityKind.SPEECH),
    )
    assert decision.reason_codes == ("vad_coverage_missing",)


def test_missing_units_yield_insufficient_evidence() -> None:
    decision = _single(_segment(text=None), _timeline("speech"))

    assert decision.outcome is HallucinationOutcome.INSUFFICIENT_EVIDENCE
    assert decision.reason_codes == ("token_units_missing",)
    assert decision.unit_kind == "none"
    assert decision.unit_count is None


def test_flag_wins_over_missing_but_lists_both() -> None:
    decision = _single(_segment(statistics=None), _timeline("silence"))

    assert decision.outcome is HallucinationOutcome.FLAGGED
    assert decision.reason_codes == (
        "speech_coverage_low",
        "silence_dominant",
        "no_speech_probability_missing",
        "avg_logprob_missing",
        "compression_ratio_missing",
    )


# --- Keeping flagged segments out of drafts. --------------------------------


def _mixed_report(thresholds=DEFAULT) -> HallucinationReport:
    timeline = VoiceActivityTimeline(
        duration_ms=12_000,
        intervals=(
            VoiceActivityInterval(0, 4_000, VoiceActivityKind.SPEECH),
            VoiceActivityInterval(4_000, 8_000, VoiceActivityKind.NOISE),
            VoiceActivityInterval(8_000, 12_000, VoiceActivityKind.SPEECH),
        ),
    )
    segments = (
        _segment("seg-a", start_ms=0, end_ms=4_000),
        _segment("seg-b", start_ms=4_000, end_ms=8_000),
        _segment("seg-c", start_ms=8_000, end_ms=12_000, statistics=None),
    )
    return check_asr_hallucinations(segments, timeline, thresholds=thresholds)


def test_guard_allows_only_supported_segments() -> None:
    report = _mixed_report()

    assert report.eligible_segment_ids == ("seg-a",)
    assert report.review_segment_ids == ("seg-b", "seg-c")
    assert report.outcome_counts == {
        "supported": 1,
        "flagged": 1,
        "insufficient_evidence": 1,
    }
    assert require_draft_eligible(report, ["seg-a"]) == ("seg-a",)


@pytest.mark.parametrize("segment_id", ["seg-b", "seg-c"])
def test_guard_blocks_flagged_and_insufficient_segments(segment_id) -> None:
    report = _mixed_report()
    with pytest.raises(AsrHallucinationError) as caught:
        require_draft_eligible(report, ["seg-a", segment_id])

    assert caught.value.category == "asr_segment_review_required"
    assert str(caught.value) == "asr_segment_review_required"
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True or caught.value.__context__ is None


def test_guard_rejects_unknown_segments() -> None:
    with pytest.raises(AsrHallucinationError) as caught:
        require_draft_eligible(_mixed_report(), ["seg-z"])
    assert caught.value.category == "asr_segment_unknown"


def test_guard_rejects_non_report_input() -> None:
    with pytest.raises(AsrHallucinationError) as caught:
        require_draft_eligible({"decisions": []}, ["seg-a"])  # type: ignore[arg-type]
    assert caught.value.category == "asr_report_type_invalid"


def test_non_final_decisions_map_to_post_process_abstention() -> None:
    report = _mixed_report()
    for decision in report.decisions[1:]:
        record = decision.abstention()
        assert record is not None
        assert record.stage is AbstentionStage.POST_PROCESS
        assert record.reason is AbstentionReason.LOW_QUALITY


def test_decision_invariants_prevent_forged_eligibility() -> None:
    decision = _mixed_report().decisions[1]
    with pytest.raises(FrozenInstanceError):
        decision.outcome = HallucinationOutcome.SUPPORTED  # type: ignore[misc]
    with pytest.raises(AsrHallucinationError) as caught:
        replace(decision, outcome=HallucinationOutcome.SUPPORTED)
    assert caught.value.category == "asr_decision_inconsistent"
    with pytest.raises(AsrHallucinationError):
        replace(decision, reason_codes=("vad_coverage_missing",))
    with pytest.raises(AsrHallucinationError):
        replace(decision, reason_codes=("made_up_code",))


# --- Versioned thresholds and negative controls. ----------------------------


def test_default_threshold_profile_is_pinned() -> None:
    assert DEFAULT.profile_id == "openmed-asr-hallucination-default"
    assert DEFAULT.version == "1.0.0"
    assert DEFAULT.digest == DEFAULT_DIGEST
    assert len(DEFAULT.digest) == 64
    payload = json.loads(DEFAULT.to_json())
    assert payload["version"] == "1.0.0"
    assert "digest" not in payload


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_speech_ratio", 0.0),
        ("max_no_speech_probability", 1.0),
        ("min_avg_logprob", -50.0),
        ("max_compression_ratio", 100.0),
        ("min_loop_repeats", 50),
        ("max_units_per_speech_second", 500.0),
    ],
)
def test_mutated_thresholds_change_digest_and_are_rejected(field, value) -> None:
    mutated = replace(DEFAULT, **{field: value})

    assert mutated.version == DEFAULT.version
    assert mutated.digest != DEFAULT.digest
    report = _mixed_report(thresholds=mutated)
    assert report.thresholds_digest == mutated.digest
    with pytest.raises(AsrHallucinationError) as caught:
        require_draft_eligible(report, report.eligible_segment_ids)
    assert caught.value.category == "asr_threshold_profile_mismatch"
    assert (
        require_draft_eligible(report, report.eligible_segment_ids, thresholds=mutated)
        == report.eligible_segment_ids
    )


def test_loosened_thresholds_would_let_a_silent_hallucination_pass() -> None:
    # Negative control: the silence flag depends on the threshold value, so
    # a loosened profile changes the outcome and must also change the digest.
    loose = replace(DEFAULT, min_speech_ratio=0.0)
    decision = _single(_segment(), _timeline("silence"), thresholds=loose)

    assert decision.outcome is HallucinationOutcome.SUPPORTED
    assert decision.thresholds_digest != DEFAULT_DIGEST


def test_version_bump_changes_digest() -> None:
    assert replace(DEFAULT, version="1.0.1").digest != DEFAULT.digest


@pytest.mark.parametrize(
    ("changes", "category"),
    [
        ({"profile_id": "Bad Id"}, "asr_thresholds_id_invalid"),
        ({"version": "v1"}, "asr_thresholds_version_invalid"),
        ({"min_vad_coverage": 1.5}, "asr_thresholds_ratio_invalid"),
        ({"min_speech_ratio": float("nan")}, "asr_thresholds_ratio_invalid"),
        ({"max_no_speech_probability": True}, "asr_thresholds_ratio_invalid"),
        ({"min_avg_logprob": 0.5}, "asr_thresholds_logprob_invalid"),
        ({"max_compression_ratio": 0.0}, "asr_thresholds_rate_invalid"),
        ({"loop_max_ngram": 0}, "asr_thresholds_count_invalid"),
        ({"min_loop_repeats": 1}, "asr_thresholds_count_invalid"),
        ({"min_unigram_loop_repeats": 1.0}, "asr_thresholds_count_invalid"),
        ({"repetition_ngram": 0}, "asr_thresholds_count_invalid"),
        ({"max_units_per_speech_second": -1.0}, "asr_thresholds_rate_invalid"),
        ({"max_segment_units": 0}, "asr_thresholds_count_invalid"),
    ],
)
def test_threshold_validation_categories(changes, category) -> None:
    with pytest.raises(AsrHallucinationError) as caught:
        replace(DEFAULT, **changes)
    assert caught.value.category == category


# --- Input validation and privacy. ------------------------------------------


@pytest.mark.parametrize(
    ("build", "category"),
    [
        (lambda: VoiceActivityInterval(5, 5, "speech"), "asr_vad_interval_invalid"),
        (lambda: VoiceActivityInterval(-1, 5, "speech"), "asr_vad_interval_invalid"),
        (lambda: VoiceActivityInterval(0, 5, "laughter"), "asr_vad_kind_unsupported"),
        (
            lambda: VoiceActivityTimeline(
                duration_ms=10,
                intervals=(
                    VoiceActivityInterval(0, 6, "speech"),
                    VoiceActivityInterval(5, 10, "silence"),
                ),
            ),
            "asr_vad_intervals_unordered",
        ),
        (
            lambda: VoiceActivityTimeline(
                duration_ms=10, intervals=(VoiceActivityInterval(0, 11, "speech"),)
            ),
            "asr_vad_interval_out_of_range",
        ),
        (
            lambda: VoiceActivityTimeline(duration_ms=10, intervals=[]),
            "asr_vad_intervals_invalid",
        ),
        (lambda: AsrSegmentStatistics(no_speech_probability=1.5), "asr_stat_invalid"),
        (lambda: AsrSegmentStatistics(avg_logprob=0.1), "asr_stat_invalid"),
        (lambda: AsrSegmentStatistics(compression_ratio=0.0), "asr_stat_invalid"),
        (lambda: AsrSegmentStatistics(avg_logprob=True), "asr_stat_invalid"),
        (lambda: _segment(segment_id="Bad Id"), "asr_segment_id_invalid"),
        (lambda: _segment(segment_id=SENTINEL), "asr_segment_id_invalid"),
        (lambda: _segment(start_ms=5, end_ms=5), "asr_segment_bounds_invalid"),
        (lambda: _segment(text=b"bytes"), "asr_segment_text_invalid"),
        (lambda: _segment(token_ids=[1, 2]), "asr_segment_tokens_invalid"),
        (lambda: _segment(token_ids=(1, -2)), "asr_segment_tokens_invalid"),
        (lambda: _segment(statistics={}), "asr_segment_statistics_invalid"),
    ],
)
def test_input_validation_categories(build, category) -> None:
    with pytest.raises(AsrHallucinationError) as caught:
        build()
    assert caught.value.category == category
    assert SENTINEL not in str(caught.value)
    assert caught.value.__cause__ is None


def test_check_rejects_duplicate_and_mistyped_inputs() -> None:
    with pytest.raises(AsrHallucinationError) as caught:
        check_asr_hallucinations((_segment("dup"), _segment("dup")), None)
    assert caught.value.category == "asr_segment_id_duplicate"
    with pytest.raises(AsrHallucinationError) as caught:
        check_asr_hallucinations((SENTINEL,), None)  # type: ignore[arg-type]
    assert caught.value.category == "asr_segment_type_invalid"
    assert SENTINEL not in str(caught.value)
    with pytest.raises(AsrHallucinationError) as caught:
        check_asr_hallucinations((_segment(),), SENTINEL)  # type: ignore[arg-type]
    assert caught.value.category == "asr_timeline_type_invalid"
    with pytest.raises(AsrHallucinationError) as caught:
        check_asr_hallucinations((_segment(),), None, thresholds=None)  # type: ignore
    assert caught.value.category == "asr_thresholds_type_invalid"


def test_sentinel_text_never_reaches_records_reprs_or_errors() -> None:
    segment = _segment(text=SENTINEL_WORDS + " " + "loop again " * 8)
    assert SENTINEL not in repr(segment)
    report = check_asr_hallucinations(
        (segment, _segment("seg-0002", text=SENTINEL)),
        _timeline(VoiceActivityKind.SILENCE),
    )
    serialized = _all_serialized(report)

    assert SENTINEL not in serialized
    assert "loop again" not in serialized
    assert "chest" not in serialized


def test_records_are_fixed_shape_and_value_free() -> None:
    report = _mixed_report()
    payload = json.loads(report.to_json())

    assert payload["schema_version"] == ASR_HALLUCINATION_SCHEMA_VERSION
    assert payload["thresholds_version"] == "1.0.0"
    assert payload["thresholds_digest"] == DEFAULT.digest
    assert len(payload["timeline_digest"]) == 64
    assert payload["provider_output_digest"] is None
    decision = payload["decisions"][0]
    assert sorted(decision) == sorted(
        [
            "schema_version",
            "segment_id",
            "start_ms",
            "end_ms",
            "outcome",
            "reason_codes",
            "final_eligible",
            "review_required",
            "unit_kind",
            "unit_count",
            "vad_coverage",
            "speech_ratio",
            "no_speech_probability",
            "avg_logprob",
            "compression_ratio",
            "loop_ngram_size",
            "loop_repeats",
            "repeated_ngram_fraction",
            "units_per_speech_second",
            "thresholds_version",
            "thresholds_digest",
        ]
    )
    for value in decision.values():
        assert value is None or isinstance(value, (bool, int, float, str, list))


def test_reason_code_vocabulary_is_partitioned_and_ordered() -> None:
    assert HALLUCINATION_REASON_CODES == FLAG_REASON_CODES + (
        INSUFFICIENT_EVIDENCE_REASON_CODES
    )
    assert not set(FLAG_REASON_CODES) & set(INSUFFICIENT_EVIDENCE_REASON_CODES)
    decision = _single(_segment(statistics=None), _timeline("music"))
    indexes = [HALLUCINATION_REASON_CODES.index(c) for c in decision.reason_codes]
    assert indexes == sorted(indexes)


# --- End to end with a provider envelope. -----------------------------------


def _envelope(outcome=ProviderResultOutcome.SUCCESS) -> ProviderResultEnvelope:
    return ProviderResultEnvelope(
        provider_id="local-asr",
        model_id="synthetic-model",
        input_digest="a" * 64,
        output_digest="b" * 64 if outcome is ProviderResultOutcome.SUCCESS else None,
        outcome=outcome,
        duration_ms=120.0,
        count_metadata={"segment_count": 4},
    )


@pytest.mark.integration
def test_end_to_end_ambient_visit_draft_gate() -> None:
    timeline = VoiceActivityTimeline(
        duration_ms=40_000,
        intervals=(
            VoiceActivityInterval(0, 10_000, VoiceActivityKind.SPEECH),
            VoiceActivityInterval(10_000, 20_000, VoiceActivityKind.SILENCE),
            VoiceActivityInterval(20_000, 30_000, VoiceActivityKind.SPEECH),
            VoiceActivityInterval(30_000, 40_000, VoiceActivityKind.MUSIC),
        ),
    )
    segments = (
        _segment("s1", start_ms=0, end_ms=10_000),
        _segment("s2", start_ms=10_000, end_ms=20_000, text=SENTINEL),
        _segment(
            "s3",
            start_ms=20_000,
            end_ms=30_000,
            text=SENTINEL + " thank you for watching" * 6,
        ),
        _segment("s4", start_ms=30_000, end_ms=40_000, text=SENTINEL),
    )
    report = check_asr_hallucinations(segments, timeline, provider_result=_envelope())

    assert report.provider_input_digest == "a" * 64
    assert report.provider_output_digest == "b" * 64
    assert report.eligible_segment_ids == ("s1",)
    assert report.review_segment_ids == ("s2", "s3", "s4")
    assert "ngram_loop" in report.decisions[2].reason_codes
    assert report.decisions[3].reason_codes[:2] == (
        "speech_coverage_low",
        "music_dominant",
    )
    assert require_draft_eligible(report, ("s1",)) == ("s1",)
    for blocked in ("s2", "s3", "s4"):
        with pytest.raises(AsrHallucinationError):
            require_draft_eligible(report, (blocked,))
    assert SENTINEL not in _all_serialized(report)
    again = check_asr_hallucinations(segments, timeline, provider_result=_envelope())
    assert again.to_json() == report.to_json()


def test_unsuccessful_provider_result_is_rejected() -> None:
    envelope = ProviderResultEnvelope(
        provider_id="local-asr",
        model_id="synthetic-model",
        input_digest="a" * 64,
        outcome=ProviderResultOutcome.PROVIDER_UNAVAILABLE,
        duration_ms=1.0,
    )
    with pytest.raises(AsrHallucinationError) as caught:
        check_asr_hallucinations((_segment(),), None, provider_result=envelope)
    assert caught.value.category == "asr_provider_result_not_success"


def test_records_are_frozen() -> None:
    report = _mixed_report()
    with pytest.raises(FrozenInstanceError):
        report.thresholds_digest = "0" * 64  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        DEFAULT.min_speech_ratio = 0.0  # type: ignore[misc]
