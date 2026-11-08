"""Offline compatibility checks across the shared vocabulary boundaries."""

import json
from pathlib import Path

import pytest

from openmed.multimodal.abstention import (
    AbstentionReason,
    AbstentionRecord,
    AbstentionStage,
)
from openmed.multimodal.provider_result import (
    ProviderAbstentionCode,
    ProviderResultEnvelope,
)

pytestmark = pytest.mark.integration


def test_baseline_v1_golden_bytes_are_preserved() -> None:
    fixture = Path(__file__).parents[1] / "fixtures/multimodal/abstention_v1.json"
    golden = json.loads(fixture.read_text())
    for encoded in golden["abstention_records"]:
        assert (
            AbstentionRecord.from_json(encoded).to_json().encode() == encoded.encode()
        )
    for encoded in golden["provider_results"]:
        assert (
            ProviderResultEnvelope.from_json(encoded).to_json().encode()
            == encoded.encode()
        )


def test_shared_vocabulary_differs_only_for_provider_unavailability() -> None:
    assert {code.value for code in ProviderAbstentionCode} == {
        reason.value for reason in AbstentionReason
    } - {"provider_unavailable"}


@pytest.mark.parametrize(
    ("stage", "code"),
    [
        ("preflight", "consent_unavailable"),
        ("decode", "consent_withdrawn"),
        ("decode", "capture_interrupted"),
        ("preflight", "language_unsupported"),
        ("inference", "language_uncertain"),
        ("post_process", "policy_refused"),
    ],
)
def test_new_codes_round_trip_between_content_free_boundaries(stage, code) -> None:
    record = AbstentionRecord(AbstentionStage(stage), AbstentionReason(code))
    result = ProviderResultEnvelope(
        provider_id="synthetic-provider",
        model_id="synthetic-model",
        input_digest="1" * 64,
        outcome="abstention",
        abstention_code=record.reason.value,
        duration_ms=0,
    )
    restored = ProviderResultEnvelope.from_json(result.to_json())
    assert record.schema_version == 2
    assert restored.schema_version == "openmed.multimodal.provider_result.v2"
    assert restored.abstention_code.value == record.reason.value
    assert AbstentionRecord.from_json(record.to_json()) == record
