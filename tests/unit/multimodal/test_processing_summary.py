"""Tests for deterministic, PHI-safe multimodal processing summaries."""

from __future__ import annotations

import random

import pytest

from openmed.multimodal.abstention import AbstentionReason, AbstentionRecord, AbstentionStage
from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.digest import AssetDigest
from openmed.multimodal.processing_summary import (
    PROCESSING_SUMMARY_SCHEMA_VERSION,
    AssetProcessingResult,
    ProcessingOutcome,
    ProcessingSummaryError,
    render_processing_summary_markdown,
    summarize_processing_run,
)


def _digest(byte: int) -> AssetDigest:
    return AssetDigest(sha256=f"{byte:02x}" * 32, byte_count=byte)


def _manifest(asset_id: str, media_type: str, **kwargs) -> AssetManifest:
    return AssetManifest(
        asset_id=asset_id,
        media_type=media_type,
        sha256="a" * 64,
        byte_size=kwargs.pop("byte_size", 1024),
        **kwargs,
    )


def _success(asset_id: str, media_type: str, **kwargs) -> AssetProcessingResult:
    return AssetProcessingResult(
        manifest=_manifest(asset_id, media_type, **kwargs),
        outcome_code=ProcessingOutcome.SUCCESS,
        duration_seconds=1.5,
        input_digest=_digest(10),
        output_digest=_digest(5),
    )


def _abstained(asset_id: str, media_type: str, stage: AbstentionStage, reason: AbstentionReason) -> AssetProcessingResult:
    return AssetProcessingResult(
        manifest=_manifest(asset_id, media_type),
        outcome_code=ProcessingOutcome.ABSTAINED,
        duration_seconds=0.2,
        input_digest=_digest(3),
        abstention=AbstentionRecord(stage=stage, reason=reason),
    )


def test_empty_run() -> None:
    summary = summarize_processing_run([])
    assert summary.total_assets == 0
    assert summary.total_bytes == 0
    assert summary.total_duration_seconds == 0.0
    assert summary.by_media_type == ()
    assert summary.outcome_counts == ()
    assert summary.abstention_counts == ()
    assert summary.asset_digests == ()
    assert summary.schema_version == PROCESSING_SUMMARY_SCHEMA_VERSION


def test_single_modality_run() -> None:
    results = [
        _success("a1", "image/png", byte_size=100, pages=1),
        _success("a2", "image/png", byte_size=200, pages=2),
    ]
    summary = summarize_processing_run(results)
    assert summary.total_assets == 2
    assert len(summary.by_media_type) == 1
    entry = summary.by_media_type[0]
    assert entry.media_type == "image/png"
    assert entry.count == 2
    assert entry.total_bytes == 300
    assert entry.total_pages == 3


def test_mixed_modality_run() -> None:
    results = [
        _success("a1", "image/png", byte_size=100),
        _success("a2", "application/pdf", byte_size=500, pages=10),
        _success("a3", "application/dicom", byte_size=900),
        _success("a4", "audio/wav", byte_size=300),
    ]
    summary = summarize_processing_run(results)
    media_types = [entry.media_type for entry in summary.by_media_type]
    assert media_types == sorted(media_types)
    assert {"image/png", "application/pdf", "application/dicom", "audio/wav"} == set(
        media_types
    )
    by_type = {entry.media_type: entry for entry in summary.by_media_type}
    assert by_type["application/pdf"].total_bytes == 500
    assert by_type["image/png"].total_bytes == 100


def test_stable_ordering_regardless_of_input_order() -> None:
    results = [
        _success("a3", "image/png", byte_size=1),
        _abstained("a1", "audio/wav", AbstentionStage.DECODE, AbstentionReason.LOW_QUALITY),
        _success("a2", "application/pdf", byte_size=2),
    ]
    forward = summarize_processing_run(results).to_json()
    reversed_run = summarize_processing_run(list(reversed(results))).to_json()
    shuffled = list(results)
    random.Random(42).shuffle(shuffled)
    shuffled_run = summarize_processing_run(shuffled).to_json()
    assert forward == reversed_run == shuffled_run


def test_unknown_outcome_code_fails_closed() -> None:
    result = _success("a1", "image/png")
    object.__setattr__(result, "outcome_code", "not_a_real_outcome")
    with pytest.raises(ProcessingSummaryError):
        summarize_processing_run([result])


def test_schema_version_present() -> None:
    for summary in (
        summarize_processing_run([]),
        summarize_processing_run([_success("a1", "image/png")]),
    ):
        assert summary.to_dict()["schema_version"] == PROCESSING_SUMMARY_SCHEMA_VERSION


def test_sentinel_phi_leak() -> None:
    sentinel = "SENTINEL-PHI-LEAK-TOKEN"
    manifest = _manifest("a1", "image/png")
    result = AssetProcessingResult(
        manifest=manifest,
        outcome_code=ProcessingOutcome.SUCCESS,
        duration_seconds=1.0,
        input_digest=_digest(1),
        output_digest=_digest(2),
    )
    # Attempt to smuggle raw content past the frozen/slotted type.
    with pytest.raises(AttributeError):
        object.__setattr__(result, "ocr_text", sentinel)

    summary = summarize_processing_run([result])
    dumped_json = summary.to_json()
    dumped_markdown = render_processing_summary_markdown(summary)
    assert sentinel not in dumped_json
    assert sentinel not in dumped_markdown

    with pytest.raises(ProcessingSummaryError) as excinfo:
        AssetProcessingResult(
            manifest=manifest,
            outcome_code=ProcessingOutcome.ABSTAINED,
            duration_seconds=1.0,
            input_digest=_digest(1),
            abstention=None,
        )
    assert sentinel not in str(excinfo.value)


def test_abstention_aggregation() -> None:
    results = [
        _success("a1", "image/png"),
        _abstained("a2", "image/png", AbstentionStage.DECODE, AbstentionReason.LOW_QUALITY),
        _abstained("a3", "image/png", AbstentionStage.DECODE, AbstentionReason.LOW_QUALITY),
        _abstained(
            "a4", "audio/wav", AbstentionStage.PREFLIGHT, AbstentionReason.UNSUPPORTED_MEDIA
        ),
    ]
    summary = summarize_processing_run(results)
    counts = {(e.stage, e.reason): e.count for e in summary.abstention_counts}
    assert counts[(AbstentionStage.DECODE, AbstentionReason.LOW_QUALITY)] == 2
    assert counts[(AbstentionStage.PREFLIGHT, AbstentionReason.UNSUPPORTED_MEDIA)] == 1
    outcome_counts = {e.outcome: e.count for e in summary.outcome_counts}
    assert outcome_counts[ProcessingOutcome.SUCCESS] == 1
    assert outcome_counts[ProcessingOutcome.ABSTAINED] == 3


def test_asset_processing_result_post_init_validation() -> None:
    manifest = _manifest("a1", "image/png")
    with pytest.raises(ProcessingSummaryError):
        AssetProcessingResult(
            manifest=manifest,
            outcome_code=ProcessingOutcome.ABSTAINED,
            duration_seconds=1.0,
            input_digest=_digest(1),
            abstention=None,
        )
    with pytest.raises(ProcessingSummaryError):
        AssetProcessingResult(
            manifest=manifest,
            outcome_code=ProcessingOutcome.SUCCESS,
            duration_seconds=1.0,
            input_digest=_digest(1),
            abstention=AbstentionRecord(
                stage=AbstentionStage.DECODE, reason=AbstentionReason.LOW_QUALITY
            ),
        )
    with pytest.raises(ProcessingSummaryError):
        AssetProcessingResult(
            manifest=manifest,
            outcome_code=ProcessingOutcome.SUCCESS,
            duration_seconds=-1.0,
            input_digest=_digest(1),
        )
    with pytest.raises(ProcessingSummaryError):
        AssetProcessingResult(
            manifest=manifest,
            outcome_code=ProcessingOutcome.ABSTAINED,
            duration_seconds=1.0,
            input_digest=_digest(1),
            abstention=AbstentionRecord(
                stage=AbstentionStage.DECODE, reason=AbstentionReason.LOW_QUALITY
            ),
            output_digest=_digest(2),
        )
