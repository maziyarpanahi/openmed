"""Tests for PHI-safe multimodal abstention records."""

from __future__ import annotations

import json

import pytest

from openmed.multimodal.abstention import (
    AbstentionReason,
    AbstentionRecord,
    AbstentionStage,
)

VALID_RECORDS = (
    (AbstentionStage.PREFLIGHT, AbstentionReason.UNSUPPORTED_MEDIA),
    (AbstentionStage.PREFLIGHT, AbstentionReason.RESOURCE_LIMIT),
    (AbstentionStage.PREFLIGHT, AbstentionReason.PROVIDER_UNAVAILABLE),
    (AbstentionStage.DECODE, AbstentionReason.MALFORMED_MEDIA),
    (AbstentionStage.DECODE, AbstentionReason.RESOURCE_LIMIT),
    (AbstentionStage.DECODE, AbstentionReason.LOW_QUALITY),
    (AbstentionStage.INFERENCE, AbstentionReason.RESOURCE_LIMIT),
    (AbstentionStage.INFERENCE, AbstentionReason.LOW_QUALITY),
    (AbstentionStage.INFERENCE, AbstentionReason.PHI_UNCERTAINTY),
    (AbstentionStage.INFERENCE, AbstentionReason.SPEAKER_UNCERTAINTY),
    (AbstentionStage.INFERENCE, AbstentionReason.TEMPORAL_INSTABILITY),
    (AbstentionStage.INFERENCE, AbstentionReason.PROVIDER_UNAVAILABLE),
    (AbstentionStage.POST_PROCESS, AbstentionReason.RESOURCE_LIMIT),
    (AbstentionStage.POST_PROCESS, AbstentionReason.LOW_QUALITY),
    (AbstentionStage.POST_PROCESS, AbstentionReason.PHI_UNCERTAINTY),
    (AbstentionStage.POST_PROCESS, AbstentionReason.SPEAKER_UNCERTAINTY),
    (AbstentionStage.POST_PROCESS, AbstentionReason.TEMPORAL_INSTABILITY),
)

INVALID_RECORDS = tuple(
    (stage, reason)
    for stage in AbstentionStage
    for reason in AbstentionReason
    if (stage, reason) not in VALID_RECORDS
)


def test_abstention_contract_is_available_from_public_multimodal_api() -> None:
    from openmed.multimodal import (
        ABSTENTION_SCHEMA_VERSION,
        AbstentionValidationError,
    )
    from openmed.multimodal import (
        AbstentionReason as PublicReason,
    )
    from openmed.multimodal import (
        AbstentionRecord as PublicRecord,
    )
    from openmed.multimodal import (
        AbstentionStage as PublicStage,
    )

    assert ABSTENTION_SCHEMA_VERSION == 1
    assert PublicReason is AbstentionReason
    assert PublicRecord is AbstentionRecord
    assert PublicStage is AbstentionStage
    assert issubclass(AbstentionValidationError, ValueError)


@pytest.mark.parametrize(("stage", "reason"), VALID_RECORDS)
def test_every_stage_and_reason_round_trips(
    stage: AbstentionStage,
    reason: AbstentionReason,
) -> None:
    record = AbstentionRecord(stage=stage, reason=reason)

    assert AbstentionRecord.from_json(record.to_json()) == record


def test_json_is_deterministic_and_metadata_only() -> None:
    record = AbstentionRecord(
        stage=AbstentionStage.PREFLIGHT,
        reason=AbstentionReason.UNSUPPORTED_MEDIA,
    )

    assert record.to_json() == (
        '{"schema_version":1,"stage":"preflight","reason":"unsupported_media"}'
    )
    assert set(json.loads(record.to_json())) == {
        "schema_version",
        "stage",
        "reason",
    }


@pytest.mark.parametrize(
    ("stage", "reason"),
    INVALID_RECORDS,
)
def test_invalid_stage_reason_combinations_fail_closed(
    stage: AbstentionStage,
    reason: AbstentionReason,
) -> None:
    with pytest.raises(ValueError, match="reason is not valid for stage"):
        AbstentionRecord(stage=stage, reason=reason)


@pytest.mark.parametrize(
    "payload",
    (
        {
            "schema_version": 1,
            "stage": "secret-stage-raw-ocr-text",
            "reason": "unsupported_media",
        },
        {
            "schema_version": 1,
            "stage": "preflight",
            "reason": "secret-reason-dicom-value",
        },
        {
            "schema_version": 1,
            "stage": "preflight",
            "reason": "unsupported_media",
            "transcript": "secret-transcript",
        },
    ),
)
def test_invalid_payloads_do_not_echo_submitted_content(payload: object) -> None:
    submitted = json.dumps(payload)

    with pytest.raises(ValueError) as exc_info:
        AbstentionRecord.from_json(submitted)

    assert "secret" not in str(exc_info.value)
    assert "raw-ocr-text" not in str(exc_info.value)
    assert "dicom-value" not in str(exc_info.value)
    assert "secret-transcript" not in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    (
        "not-json",
        "[]",
        '{"schema_version":1,"stage":"preflight","stage":"decode",'
        '"reason":"unsupported_media"}',
        '{"schema_version":2,"stage":"preflight","reason":"unsupported_media"}',
        '{"schema_version":1,"stage":"preflight"}',
    ),
)
def test_malformed_or_incomplete_payloads_fail_closed(payload: str) -> None:
    with pytest.raises(ValueError):
        AbstentionRecord.from_json(payload)
