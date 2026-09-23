"""Synthetic tests for deterministic, value-safe summary claim segmentation."""

import json
import socket

import pytest

from openmed.clinical.summary_claim_segments import (
    SummaryClaimSegment,
    SummaryClaimSegmentationError,
    segment_summary_claims,
)


def test_segments_sentences_with_exact_output_offsets() -> None:
    summary = "Signal alpha is stable. Marker beta improved."

    result = segment_summary_claims(summary)

    assert [segment.text for segment in result.segments] == [
        "Signal alpha is stable.",
        "Marker beta improved.",
    ]
    assert [segment.offset for segment in result.segments] == [(0, 23), (24, 45)]
    assert all(
        summary[start:end] == segment.text
        for segment in result.segments
        for start, end in [segment.offset]
    )
    assert result.review_required is False


def test_splits_stable_semicolon_and_newline_boundaries() -> None:
    summary = (
        "Signal alpha is stable; marker beta improved.\nItem gamma remains present."
    )

    result = segment_summary_claims(summary)

    assert [segment.text for segment in result.segments] == [
        "Signal alpha is stable",
        "marker beta improved.",
        "Item gamma remains present.",
    ]
    assert [segment.offset for segment in result.segments] == [
        (0, 22),
        (24, 45),
        (46, 73),
    ]


def test_splits_high_confidence_independent_clauses() -> None:
    summary = "Signal alpha improved, but marker beta persisted."

    result = segment_summary_claims(summary)

    assert [segment.text for segment in result.segments] == [
        "Signal alpha improved",
        "marker beta persisted.",
    ]
    assert [segment.offset for segment in result.segments] == [(0, 21), (27, 49)]
    assert result.review_required is False


def test_splits_high_confidence_comma_splice() -> None:
    summary = "Signal alpha is stable, marker beta improved."

    result = segment_summary_claims(summary)

    assert [segment.text for segment in result.segments] == [
        "Signal alpha is stable",
        "marker beta improved.",
    ]
    assert [segment.offset for segment in result.segments] == [(0, 22), (24, 45)]
    assert result.review_required is False


@pytest.mark.parametrize(
    "summary",
    [
        "Signals alpha and beta are stable.",
        "Signal alpha is stable because marker beta improved.",
        "No signal alpha or marker beta is present.",
    ],
)
def test_flags_unsegmentable_compounds_before_verification(summary: str) -> None:
    result = segment_summary_claims(summary)

    assert len(result.segments) == 1
    assert result.segments[0].text == summary
    assert result.segments[0].review_required is True
    assert result.segments[0].review_reason == "unsegmentable_compound"


def test_non_english_clause_analysis_fails_closed() -> None:
    summary = "信号稳定。标记改善。"

    result = segment_summary_claims(summary, language="zh-CN")

    assert [segment.text for segment in result.segments] == ["信号稳定。", "标记改善。"]
    assert result.review_required_count == 2
    assert {segment.review_reason for segment in result.segments} == {
        "unsupported_language"
    }


def test_metadata_serialization_omits_claim_values() -> None:
    summary = "Synthetic sentinel is stable and marker delta is present."

    result = segment_summary_claims(summary)
    serialized = result.to_json()
    payload = json.loads(serialized)

    assert summary not in serialized
    assert "Synthetic sentinel" not in serialized
    assert payload["segment_count"] == 2
    assert payload["review_required_count"] == 0
    assert set(payload["segments"][0]) == {
        "output_offset",
        "review_reason",
        "review_required",
    }
    assert "Synthetic sentinel" not in repr(result.segments[0])


def test_deterministic_without_network_access(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = "Signal alpha is stable. Marker beta improved."

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is not permitted")

    monkeypatch.setattr(socket, "create_connection", reject_network)

    first = segment_summary_claims(summary)
    second = segment_summary_claims(summary)

    assert first == second
    assert first.to_json() == second.to_json()


def test_errors_do_not_echo_submitted_values() -> None:
    sentinel = "private-sentinel-value"

    with pytest.raises(SummaryClaimSegmentationError) as exc_info:
        segment_summary_claims(sentinel, language=f"en-{sentinel}")

    assert sentinel not in str(exc_info.value)


def test_empty_summary_produces_empty_segmentation() -> None:
    result = segment_summary_claims("")

    assert result.segments == ()
    assert result.review_required is False
    assert result.to_dict()["segment_count"] == 0


def test_segment_validation_rejects_inconsistent_safe_metadata() -> None:
    with pytest.raises(
        SummaryClaimSegmentationError,
        match="inconsistent claim review metadata",
    ):
        SummaryClaimSegment(
            text="synthetic",
            start=0,
            end=9,
            review_required=False,
            review_reason="unsegmentable_compound",
        )
