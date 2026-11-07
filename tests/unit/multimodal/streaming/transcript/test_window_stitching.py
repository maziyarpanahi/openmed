"""Exact overlap, replay, incomplete evidence and protected alternative controls."""

import json
from dataclasses import replace

import pytest

from openmed.multimodal.audio_windows import TailPolicy, plan_audio_windows
from openmed.multimodal.streaming.privacy import PhiResolution, WithholdingPolicy
from openmed.multimodal.streaming.transcript import RevisionKind
from openmed.multimodal.streaming.transcript.window_stitching import (
    SeamReason,
    StitchingError,
    reconcile_windows,
)
from tests.fixtures.multimodal.asr_overlap import (
    PLAN,
    PRIVATE,
    PROVIDER,
    SESSION,
    SOURCE,
    SyntheticScreener,
    negation_windows,
    reconcile,
    window,
)


def protected_text(result):
    return "".join(operation.text for operation in result.operations)


def test_exact_timing_match_preserves_negation_and_emits_existing_ledger_edits():
    result = reconcile()
    assert not result.review_required
    assert protected_text(result) == "No allergy to penicillin reported"
    assert [op.kind for op in result.operations] == [
        RevisionKind.INSERT,
        RevisionKind.INSERT,
        RevisionKind.FINALIZE,
    ]
    assert [op.expected_revision for op in result.operations] == [0, 1, 2]
    assert [op.position for op in result.operations] == [0, 24, 33]
    assert len(result.operations[1].tokens) == 1
    assert result.operations[1].tokens[0].start == 1
    assert result == reconcile()
    public = result.to_json() + repr(result)
    for word in ("penicillin", "allergy", "reported"):
        assert word not in public


def test_repeated_phrase_at_distinct_times_is_not_deleted_by_substring_matching():
    result = reconcile(
        (
            window(0, (("drug", 500, 700), ("drug", 1500, 1700))),
            window(1, (("drug", 1500, 1700), ("drug", 2500, 2700))),
        )
    )
    assert protected_text(result) == "drug drug drug"
    assert not result.review_required
    assert (
        len({token.token_ref for op in result.operations for token in op.tokens}) == 3
    )


@pytest.mark.parametrize(
    "variant", ["missing_negation", "changed_medication", "shifted_timing"]
)
def test_conflicting_overlap_remains_provisional_with_all_original_alternatives(
    variant,
):
    first, second = negation_windows()
    if variant == "missing_negation":
        first = window(0, (("No", 1100, 1200), ("allergy", 1500, 1800)))
        second = window(1, (("allergy", 1500, 1800), ("reported", 2200, 2500)))
    elif variant == "changed_medication":
        second = window(
            1,
            (
                ("allergy", 1100, 1250),
                ("to", 1300, 1350),
                ("aspirin", 1500, 1800),
                ("reported", 2200, 2500),
            ),
        )
    else:
        second = window(
            1,
            (
                ("allergy", 1101, 1250),
                ("to", 1300, 1350),
                ("penicillin", 1500, 1800),
                ("reported", 2200, 2500),
            ),
        )
    result = reconcile((first, second))
    assert SeamReason.CONFLICT in result.reasons
    assert result.alternatives == (first, second)
    assert all(op.kind is not RevisionKind.FINALIZE for op in result.operations)
    screener = SyntheticScreener()
    assert result.release_with(screener, clock=lambda: 0) == ()
    assert screener.calls == 0


@pytest.mark.parametrize(
    "kind", ["missing_first", "missing_last", "reordered", "duplicate", "empty"]
)
def test_missing_and_out_of_order_windows_have_explicit_gaps_or_duplicate_reason(kind):
    first, second = negation_windows()
    windows = {
        "missing_first": (second,),
        "missing_last": (first,),
        "reordered": (second, first),
        "duplicate": (first, first, second),
        "empty": (),
    }[kind]
    result = reconcile(windows)
    assert result.review_required
    assert result.gaps or SeamReason.DUPLICATE_WINDOW in result.reasons
    assert result.release_with(SyntheticScreener(), clock=lambda: 0) == ()
    assert result.segment_ref == reconcile().segment_ref


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"final": False}, SeamReason.PROVISIONAL),
        ({"complete": False}, SeamReason.INCOMPLETE),
    ],
)
def test_provider_stability_and_coverage_are_required(changes, reason):
    first, second = negation_windows()
    first = window(
        0,
        (
            ("No", 100, 300),
            ("allergy", 1100, 1250),
            ("to", 1300, 1350),
            ("penicillin", 1500, 1800),
        ),
        **changes,
    )
    result = reconcile((first, second))
    assert reason in result.reasons
    assert result.release_with(SyntheticScreener(), clock=lambda: 0) == ()


def test_unaligned_negation_or_punctuation_cannot_disappear_from_transcript():
    first, second = negation_windows()
    for text, offsets in (
        (
            "NOT " + first.hypothesis.text,
            tuple((a + 4, b + 4) for a, b in first.character_offsets),
        ),
        (first.hypothesis.text + "?", first.character_offsets),
    ):
        changed = replace(
            first,
            hypothesis=replace(first.hypothesis, text=text),
            character_offsets=offsets,
        )
        result = reconcile((changed, second))
        assert SeamReason.UNALIGNED in result.reasons
        assert result.release_with(SyntheticScreener(), clock=lambda: 0) == ()
        assert result.alternatives[0].hypothesis.text == text


def test_zero_duration_and_duplicate_timing_force_review():
    for words in ((("No", 100, 100),), (("No", 100, 300), ("No", 100, 300))):
        result = reconcile((window(0, words), negation_windows()[1]))
        assert SeamReason.TIMING in result.reasons


def test_uncovered_audio_tail_is_not_silently_finalized():
    plan = plan_audio_windows(
        4100, window_ms=2000, overlap_ms=0, min_tail_ms=500, tail_policy=TailPolicy.DROP
    )
    result = reconcile(
        (window(0, (("First", 100, 300),)), window(1, (("Second", 2200, 2400),))),
        plan=plan,
    )
    assert SeamReason.UNCOVERED_TAIL in result.reasons
    assert (4000, 4100) in result.gaps
    assert result.release_with(SyntheticScreener(), clock=lambda: 0) == ()


def test_untimed_boundary_does_not_invent_word_continuity():
    plan = plan_audio_windows(4000, window_ms=2000, overlap_ms=0)
    result = reconcile(
        (window(0, (("met", 100, 300),)), window(1, (("formin", 2200, 2500),))),
        plan=plan,
    )
    assert SeamReason.BOUNDARY in result.reasons
    assert protected_text(result) == "met"
    assert result.alternatives[1].hypothesis.text == "formin"


@pytest.mark.parametrize("change", ["source", "provider", "plan", "shape", "unicode"])
def test_invalid_source_provider_plan_or_alignment_has_value_free_refusal(change):
    first, second = negation_windows()
    with pytest.raises(StitchingError) as caught:
        if change == "source":
            first = replace(first, source_digest="c" * 64)
        elif change == "provider":
            first = replace(first, provider=replace(PROVIDER, provider_revision="v2"))
        elif change == "plan":
            reconcile((first, second), plan=replace(PLAN, covered_ms=1))
        elif change == "shape":
            replace(first, character_offsets=((0, 1),) + first.character_offsets[1:])
        else:
            replace(
                first, hypothesis=replace(first.hypothesis, text=PRIVATE + "\ud800")
            )
        reconcile((first, second))
    assert PRIVATE not in str(caught.value)


def test_bounded_plan_and_window_payloads_refuse_before_reconciliation():
    large = plan_audio_windows(65000, window_ms=1000)
    with pytest.raises(StitchingError):
        reconcile((), plan=large)
    first = negation_windows()[0]
    with pytest.raises(StitchingError):
        reconcile((first,) * 65)


def test_forged_finalization_or_changed_edit_cannot_bypass_fresh_reconciliation():
    first, second = negation_windows()
    result = reconcile((first, replace(second, coverage_complete=False)))
    forged = replace(
        result, review_required=False, reasons=(), operations=reconcile().operations
    )
    with pytest.raises(StitchingError):
        forged.release_with(SyntheticScreener(), clock=lambda: 0)
    valid = reconcile()
    altered = replace(
        valid,
        operations=(replace(valid.operations[0], text=PRIVATE),) + valid.operations[1:],
    )
    with pytest.raises(StitchingError):
        altered.release_with(SyntheticScreener(), clock=lambda: 0)


@pytest.mark.parametrize(
    "resolution", [PhiResolution.UNRESOLVED, PhiResolution.CRITICAL_UNRESOLVED]
)
def test_alignment_success_never_substitutes_for_phi_clearance(resolution):
    result = reconcile()
    screener = SyntheticScreener(resolution)
    assert result.release_with(screener, clock=lambda: 0) == ()
    assert screener.calls == 1


def test_existing_holdback_character_and_latency_limits_still_apply():
    result = reconcile()
    assert (
        result.release_with(
            SyntheticScreener(), policy=WithholdingPolicy(max_chars=10), clock=lambda: 0
        )
        == ()
    )
    ticks = iter((0, 0, 100))
    assert (
        result.release_with(SyntheticScreener(), clock=lambda: next(ticks, 100)) == ()
    )


def test_new_model_revision_changes_segment_binding_but_replay_does_not():
    provider = replace(PROVIDER, provider_revision="v2")
    windows = tuple(replace(item, provider=provider) for item in negation_windows())
    result = reconcile_windows(
        PLAN, windows, session_ref=SESSION, source_digest=SOURCE, provider=provider
    )
    assert result.segment_ref != reconcile().segment_ref
    assert json.loads(result.to_json())["received_windows"] == 2
