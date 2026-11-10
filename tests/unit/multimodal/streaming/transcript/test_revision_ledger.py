"""Revision algebra, stable identities and ambiguous citation invalidation."""

from dataclasses import replace

import pytest

from openmed.multimodal.streaming.transcript import (
    RevisionLedger,
    RevisionLedgerError,
    SpanInvalidation,
    TranscriptToken,
)
from openmed.multimodal.streaming.transcript.revision_ledger import (
    MAX_REVISIONS,
    MAX_TEXT_LENGTH,
    MAX_TIME_MS,
)


def ref(index):
    return f"{index:032x}"


def token(index, start, end, start_ms=0, end_ms=100):
    return TranscriptToken(
        token_ref=ref(index), start=start, end=end, start_ms=start_ms, end_ms=end_ms
    )


@pytest.fixture
def ledger():
    return RevisionLedger(
        ref(100),
        ref(200),
        14,
        (token(1, 0, 3, 0, 100), token(2, 4, 8, 100, 200), token(3, 9, 14, 200, 300)),
    )


def test_insert_before_span_remaps_offsets_and_retains_original_identity(ledger):
    citation = ledger.capture_span(ref(10), 4, 8)
    ledger.insert(0, 2, (token(4, 0, 1, 0, 0),), expected_revision=0)
    result = ledger.remap_span(citation)
    assert result.valid and (result.start, result.end) == (6, 10)
    assert (result.start_ms, result.end_ms) == (100, 200)
    assert result.reference is citation
    assert citation.revision == 0 and result.revision == 1
    assert ledger.tokens[2].token_ref == ref(2)


def test_delete_before_span_moves_it_left_without_reassigning_tokens(ledger):
    citation = ledger.capture_span(ref(10), 9, 14)
    ledger.delete(0, 4, expected_revision=0)
    assert (ledger.remap_span(citation).start, ledger.remap_span(citation).end) == (
        5,
        10,
    )
    assert ledger.tokens[-1].token_ref == ref(3)


def test_insert_boundary_affinity_is_explicit(ledger):
    citation = ledger.capture_span(ref(10), 4, 8)
    ledger.insert(4, 1, (), expected_revision=0)
    assert (ledger.remap_span(citation).start, ledger.remap_span(citation).end) == (
        5,
        9,
    )
    ledger.insert(9, 1, (), expected_revision=1)
    assert (ledger.remap_span(citation).start, ledger.remap_span(citation).end) == (
        5,
        9,
    )


@pytest.mark.parametrize("operation", ["insert", "delete", "replace"])
def test_edits_inside_a_cited_span_invalidate_it(ledger, operation):
    citation = ledger.capture_span(ref(10), 4, 8)
    if operation == "insert":
        ledger.insert(6, 1, (), expected_revision=0)
    elif operation == "delete":
        ledger.delete(5, 7, expected_revision=0)
    else:
        ledger.replace(4, 8, 4, (token(4, 0, 4, 100, 200),), expected_revision=0)
    result = ledger.remap_span(citation)
    assert not result.valid and result.invalidation is SpanInvalidation.TEXT_CHANGED
    assert result.start is result.end is result.start_ms is result.end_ms is None


def test_edit_outside_partial_span_still_invalidates_its_changed_token(ledger):
    citation = ledger.capture_span(ref(10), 6, 8)
    ledger.delete(4, 5, expected_revision=0)
    assert ledger.remap_span(citation).invalidation is SpanInvalidation.TOKEN_CHANGED


def test_invalidated_span_never_revives_after_similar_text_is_reinserted(ledger):
    citation = ledger.capture_span(ref(10), 4, 8)
    ledger.delete(4, 8, expected_revision=0)
    ledger.insert(4, 4, (token(4, 0, 4, 100, 200),), expected_revision=1)
    assert not ledger.remap_span(citation).valid
    fresh = ledger.capture_span(ref(11), 4, 8)
    assert ledger.remap_span(fresh).valid
    assert fresh.token_refs == (ref(4),)


def test_new_tokens_cannot_reuse_retired_or_live_identities(ledger):
    ledger.delete(4, 8, expected_revision=0)
    before = ledger.to_json()
    for index in (1, 2):
        with pytest.raises(RevisionLedgerError):
            ledger.insert(4, 4, (token(index, 0, 4, 100, 200),), expected_revision=1)
    assert ledger.to_json() == before


def test_timestamp_shift_updates_citations_without_text_or_identity_changes(ledger):
    citation = ledger.capture_span(ref(10), 4, 8)
    ledger.shift_timestamps((ref(2),), 25, expected_revision=0)
    result = ledger.remap_span(citation)
    assert result.valid and (result.start, result.end) == (4, 8)
    assert (result.start_ms, result.end_ms) == (125, 225)
    assert result.reference.token_refs == (ref(2),)


def test_finalization_seals_text_and_timing_but_preserves_citations(ledger):
    citation = ledger.capture_span(ref(10), 4, 8)
    ledger.finalize(expected_revision=0)
    assert ledger.remap_span(citation).finalized
    assert ledger.remap_span(citation).valid
    before = ledger.to_json()
    for action in (
        lambda: ledger.insert(0, 1, (), expected_revision=1),
        lambda: ledger.shift_timestamps((ref(2),), 1, expected_revision=1),
        lambda: ledger.finalize(expected_revision=1),
    ):
        with pytest.raises(RevisionLedgerError):
            action()
    assert ledger.to_json() == before


def test_timing_change_rejects_out_of_order_or_out_of_bounds_updates_atomically(ledger):
    citation = ledger.capture_span(ref(10), 4, 8)
    before = ledger.to_json()
    for delta in (-101, 201, MAX_TIME_MS):
        with pytest.raises(RevisionLedgerError):
            ledger.shift_timestamps((ref(2),), delta, expected_revision=0)
    assert ledger.to_json() == before
    assert ledger.remap_span(citation).revision == 0


def test_replacement_validates_new_token_timing_before_any_state_changes(ledger):
    citation = ledger.capture_span(ref(10), 4, 8)
    before = ledger.to_json()
    with pytest.raises(RevisionLedgerError):
        ledger.replace(4, 8, 4, (token(4, 0, 4, 500, 600),), expected_revision=0)
    assert ledger.to_json() == before
    assert ledger.remap_span(citation).valid
    ledger.replace(4, 8, 4, (token(4, 0, 4, 100, 200),), expected_revision=0)
    assert not ledger.remap_span(citation).valid


def test_stale_and_cross_session_span_references_fail_closed(ledger):
    citation = ledger.capture_span(ref(10), 4, 8)
    for changed in (
        replace(citation, session_ref=ref(999)),
        replace(citation, segment_ref=ref(999)),
        replace(citation, revision=1),
        replace(citation, start=5),
    ):
        with pytest.raises(RevisionLedgerError):
            ledger.remap_span(changed)
    ledger.insert(0, 1, (), expected_revision=0)
    with pytest.raises(RevisionLedgerError):
        ledger.delete(0, 1, expected_revision=0)


def test_multistep_remap_composes_without_losing_segment_identity(ledger):
    citation = ledger.capture_span(ref(10), 9, 14)
    ledger.insert(0, 2, (), expected_revision=0)
    ledger.replace(2, 5, 1, (token(4, 0, 1, 0, 100),), expected_revision=1)
    ledger.delete(0, 2, expected_revision=2)
    result = ledger.remap_span(citation)
    assert result.valid and (result.start, result.end) == (7, 12)
    assert result.reference.segment_ref == ref(200)


def test_offsets_are_unicode_code_points_without_storing_text():
    synthetic = "é🙂 ok"
    ledger = RevisionLedger(
        ref(100), ref(200), len(synthetic), (token(1, 0, 2), token(2, 3, 5, 100, 200))
    )
    citation = ledger.capture_span(ref(10), 3, 5)
    ledger.insert(0, len("中"), (token(3, 0, 1, 0, 0),), expected_revision=0)
    assert (ledger.remap_span(citation).start, ledger.remap_span(citation).end) == (
        4,
        6,
    )
    assert synthetic not in ledger.to_json()


@pytest.mark.parametrize(
    "start,end,length",
    [
        (-1, 1, 1),
        (0, 15, 1),
        (5, 4, 1),
        (0, 1, 0),
        (0, 1, MAX_TEXT_LENGTH),
        (True, 1, 1),
    ],
)
def test_invalid_edit_bounds_are_atomic(ledger, start, end, length):
    before = ledger.to_json()
    with pytest.raises(RevisionLedgerError, match="^invalid transcript lineage$"):
        ledger.replace(start, end, length, (), expected_revision=0)
    assert ledger.to_json() == before


@pytest.mark.parametrize(
    "change",
    [
        dict(token_ref="synthetic-private-value"),
        dict(start=-1),
        dict(end=0),
        dict(start_ms=float("nan")),
        dict(end_ms=MAX_TIME_MS + 1),
        dict(start=True),
    ],
)
def test_invalid_tokens_never_echo_values(change):
    with pytest.raises(RevisionLedgerError) as caught:
        replace(token(1, 0, 3), **change)
    assert str(caught.value) == "invalid transcript lineage"


def test_empty_segment_can_receive_an_insert_and_citations_require_tokens():
    ledger = RevisionLedger(ref(100), ref(200), 0)
    ledger.insert(0, 3, (token(1, 0, 3),), expected_revision=0)
    assert ledger.remap_span(ledger.capture_span(ref(10), 0, 3)).valid
    ledger.insert(3, 2, (), expected_revision=1)
    with pytest.raises(RevisionLedgerError):
        ledger.capture_span(ref(11), 3, 5)


def test_report_copies_cannot_modify_ledger_state(ledger):
    ledger.capture_span(ref(10), 4, 8)
    ledger.insert(0, 1, (), expected_revision=0)
    before = ledger.to_json()
    report = ledger.to_dict()
    report["operations"][0]["start"] = 1000
    report["tokens"][0]["start"] = 1000
    assert ledger.to_json() == before


def test_total_citation_and_history_links_are_bounded_atomically(ledger, monkeypatch):
    from openmed.multimodal.streaming.transcript import revision_ledger

    monkeypatch.setattr(revision_ledger, "MAX_TOKEN_LINKS", 2)
    ledger.capture_span(ref(10), 4, 8)
    before = ledger.to_json()
    with pytest.raises(RevisionLedgerError):
        ledger.capture_span(ref(11), 0, 8)
    assert ledger.to_json() == before
    ledger.shift_timestamps((ref(1), ref(2)), 1, expected_revision=0)
    before = ledger.to_json()
    with pytest.raises(RevisionLedgerError):
        ledger.shift_timestamps((ref(1),), 1, expected_revision=1)
    assert ledger.to_json() == before


def test_revision_limit_fails_without_overwriting_history():
    ledger = RevisionLedger(ref(100), ref(200), 1, (token(1, 0, 1),))
    for revision in range(MAX_REVISIONS):
        ledger.shift_timestamps(
            (ref(1),), 1 if revision % 2 == 0 else -1, expected_revision=revision
        )
    before = ledger.to_json()
    with pytest.raises(RevisionLedgerError):
        ledger.finalize(expected_revision=MAX_REVISIONS)
    assert ledger.to_json() == before


def test_serialization_is_deterministic_and_content_free(ledger, caplog):
    citation = ledger.capture_span(ref(10), 4, 8)
    ledger.finalize(expected_revision=0)
    report = ledger.to_json()
    assert report == ledger.to_json()
    assert all(key not in report for key in ('"text":', '"path":', '"audio":'))
    assert ledger.remap_span(citation).valid
    assert caplog.records == []
