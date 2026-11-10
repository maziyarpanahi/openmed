"""Tests for deterministic, metadata-only processing summary differences."""

from __future__ import annotations

from dataclasses import replace

import pytest

from openmed.multimodal.abstention import (
    AbstentionReason,
    AbstentionRecord,
    AbstentionStage,
)
from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.digest import AssetDigest
from openmed.multimodal.processing_diff import (
    ProcessingDiffError,
    diff_processing_summaries,
    render_processing_diff_markdown,
)
from openmed.multimodal.processing_summary import (
    AssetProcessingResult,
    ProcessingOutcome,
    summarize_processing_run,
)


def _result(
    asset_id: str,
    media_type: str,
    input_byte: str,
    *,
    byte_size: int,
    duration: float,
    pages: int | None = None,
    frames: int | None = None,
    outcome: ProcessingOutcome = ProcessingOutcome.SUCCESS,
    output_byte: str | None = None,
    abstention: AbstentionRecord | None = None,
) -> AssetProcessingResult:
    manifest = AssetManifest(
        asset_id=asset_id,
        media_type=media_type,
        sha256=input_byte * 64,
        byte_size=byte_size,
        pages=pages,
        frames=frames,
    )
    return AssetProcessingResult(
        manifest=manifest,
        outcome_code=outcome,
        duration_seconds=duration,
        input_digest=AssetDigest(sha256=manifest.sha256, byte_count=byte_size),
        output_digest=(
            AssetDigest(sha256=output_byte * 64, byte_count=1) if output_byte else None
        ),
        abstention=abstention,
    )


def test_equal_empty_runs_have_no_changes() -> None:
    empty = summarize_processing_run([])

    difference = diff_processing_summaries(empty, empty)

    assert difference.total_assets_delta == 0
    assert difference.added_digests == ()
    assert difference.removed_digests == ()
    assert difference.to_json() == diff_processing_summaries(empty, empty).to_json()


def test_hand_calculated_changes_and_reverse() -> None:
    before = summarize_processing_run(
        [
            _result(
                "opaque-a",
                "image/png",
                "a",
                byte_size=100,
                pages=2,
                duration=1.25,
                output_byte="b",
            ),
            _result(
                "opaque-b",
                "audio/wav",
                "c",
                byte_size=200,
                frames=4,
                duration=2.0,
                outcome=ProcessingOutcome.ABSTAINED,
                abstention=AbstentionRecord(
                    stage=AbstentionStage.DECODE,
                    reason=AbstentionReason.LOW_QUALITY,
                ),
            ),
        ]
    )
    after = summarize_processing_run(
        [
            _result(
                "opaque-a",
                "image/png",
                "a",
                byte_size=100,
                pages=2,
                duration=1.5,
                output_byte="b",
            ),
            _result(
                "opaque-c",
                "image/png",
                "d",
                byte_size=150,
                pages=3,
                duration=0.5,
                outcome=ProcessingOutcome.ERROR,
                output_byte="e",
            ),
        ]
    )

    difference = diff_processing_summaries(before, after)
    assert difference.total_assets_delta == 0
    assert difference.total_bytes_delta == -50
    assert difference.total_duration_seconds_delta == -1.25
    assert difference.asset_count_with_output_digest_delta == 1
    assert [entry.to_dict() for entry in difference.by_media_type] == [
        {
            "media_type": "audio/wav",
            "count": -1,
            "total_bytes": -200,
            "total_pages": 0,
            "total_frames": -4,
        },
        {
            "media_type": "image/png",
            "count": 1,
            "total_bytes": 150,
            "total_pages": 3,
            "total_frames": 0,
        },
    ]
    assert [
        (entry.outcome.value, entry.count) for entry in difference.outcome_counts
    ] == [
        ("success", 0),
        ("abstained", -1),
        ("error", 1),
    ]
    assert [
        (entry.stage, entry.reason, entry.count)
        for entry in difference.abstention_counts
    ] == [("decode", "low_quality", -1)]
    assert [entry.to_dict() for entry in difference.added_digests] == [
        {"input_sha256": "d" * 64, "output_sha256": "e" * 64, "count": 1}
    ]
    assert [entry.to_dict() for entry in difference.removed_digests] == [
        {"input_sha256": "c" * 64, "count": 1}
    ]

    reverse = diff_processing_summaries(after, before)
    assert reverse.total_assets_delta == -difference.total_assets_delta
    assert reverse.total_bytes_delta == -difference.total_bytes_delta
    assert (
        reverse.total_duration_seconds_delta == -difference.total_duration_seconds_delta
    )
    assert (
        reverse.asset_count_with_output_digest_delta
        == -difference.asset_count_with_output_digest_delta
    )
    assert [entry.count for entry in reverse.by_media_type] == [
        -entry.count for entry in difference.by_media_type
    ]
    for field in ("total_bytes", "total_pages", "total_frames"):
        assert [getattr(entry, field) for entry in reverse.by_media_type] == [
            -getattr(entry, field) for entry in difference.by_media_type
        ]
    assert [entry.count for entry in reverse.outcome_counts] == [
        -entry.count for entry in difference.outcome_counts
    ]
    assert [entry.count for entry in reverse.abstention_counts] == [
        -entry.count for entry in difference.abstention_counts
    ]
    assert reverse.added_digests == difference.removed_digests
    assert reverse.removed_digests == difference.added_digests

    json_output = difference.to_json()
    markdown_output = render_processing_diff_markdown(difference)
    assert '"total_bytes_delta":-50' in json_output
    assert "| image/png | +1 | +150 | +3 | +0 |" in markdown_output
    assert "- Duration (seconds): -1.25" in markdown_output
    assert "opaque-" not in json_output + markdown_output


def test_reordered_assets_and_duplicate_digests_are_stable() -> None:
    first = _result("first", "image/png", "a", byte_size=10, duration=1.0)
    second = _result("second", "image/png", "a", byte_size=10, duration=1.0)
    third = _result("third", "image/png", "a", byte_size=10, duration=1.0)
    before = summarize_processing_run([first, second])
    after = summarize_processing_run([second, first, third])
    difference = diff_processing_summaries(before, after)

    assert difference.total_assets_delta == 1
    assert difference.added_digests[0].to_dict() == {
        "input_sha256": "a" * 64,
        "count": 1,
    }
    assert difference.removed_digests == ()
    assert (
        difference.to_json()
        == diff_processing_summaries(
            summarize_processing_run([second, first]),
            summarize_processing_run([third, first, second]),
        ).to_json()
    )


def test_changed_output_digest_is_removed_and_added() -> None:
    before = summarize_processing_run(
        [_result("same", "image/png", "a", byte_size=10, duration=1.0, output_byte="b")]
    )
    after = summarize_processing_run(
        [_result("same", "image/png", "a", byte_size=10, duration=1.0, output_byte="c")]
    )

    difference = diff_processing_summaries(before, after)

    assert difference.total_assets_delta == 0
    assert difference.added_digests[0].to_dict() == {
        "input_sha256": "a" * 64,
        "output_sha256": "c" * 64,
        "count": 1,
    }
    assert difference.removed_digests[0].to_dict() == {
        "input_sha256": "a" * 64,
        "output_sha256": "b" * 64,
        "count": 1,
    }


def test_unsupported_summary_version_rejected_without_values() -> None:
    valid = summarize_processing_run([])
    with pytest.raises(ProcessingDiffError, match="schema version"):
        diff_processing_summaries(valid, replace(valid, schema_version=999))
    with pytest.raises(ProcessingDiffError, match="both inputs"):
        diff_processing_summaries(valid, object())  # type: ignore[arg-type]
